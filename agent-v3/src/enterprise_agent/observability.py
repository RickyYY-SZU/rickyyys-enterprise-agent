"""Structured tracing and fault localisation for agent requests.

Every request produces one trace composed of timed spans. Spans capture node
latency, tool and LLM failures, and the attributes needed to reproduce a
failure. Traces are emitted as structured JSON logs, kept in a bounded
in-memory store served by ``/api/traces``, and mirrored to an OTLP collector
when one is configured.

Two failure modes motivated this module:

* ``EnterpriseAgent._answer`` used to swallow LLM exceptions with a bare
  ``except Exception: pass``, so provider outages silently degraded answers.
* LangGraph nodes reported neither latency nor errors, so a slow or failing
  node was invisible unless the whole request failed.

Tracing must never break the request path, so export failures are logged and
ignored at every boundary.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("enterprise_agent.trace")

_CURRENT_TRACE: ContextVar[RequestTrace | None] = ContextVar(
    "enterprise_agent_trace", default=None
)
_CURRENT_SPAN: ContextVar[str | None] = ContextVar("enterprise_agent_span", default=None)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _flat_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """OTel only accepts scalar attribute values; encode the rest as JSON."""
    flat: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False, default=str)[:1024]
    return flat


@dataclass(slots=True)
class SpanRecord:
    name: str
    span_id: str
    parent_id: str | None
    started_at: str
    duration_ms: float
    status: str = "ok"
    error_type: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attributes": self.attributes,
        }


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    name: str
    started_at: str
    duration_ms: float
    status: str
    error_type: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: list[SpanRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "spans": [span.as_dict() for span in self.spans],
        }

    def error_spans(self) -> list[SpanRecord]:
        return [span for span in self.spans if span.status == "error"]

    def summary(self) -> dict[str, Any]:
        errors = self.error_spans()
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error_type": self.error_type,
            "attributes": self.attributes,
            "span_count": len(self.spans),
            "error_span_count": len(errors),
            "error_spans": [span.name for span in errors],
        }


class TraceStore:
    """Bounded in-memory trace buffer so recent failures stay inspectable."""

    def __init__(self, max_records: int = 200) -> None:
        self._records: deque[TraceRecord] = deque(maxlen=max(1, max_records))
        self._lock = threading.Lock()

    def add(self, record: TraceRecord) -> None:
        with self._lock:
            self._records.append(record)

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            for record in reversed(self._records):
                if record.trace_id == trace_id:
                    return record
        return None

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        with self._lock:
            records = list(self._records)[-max(1, limit) :]
        return list(reversed(records))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class _OtlpBridge:
    """Optional OTLP exporter; absent dependencies degrade to local tracing."""

    def __init__(self, endpoint: str, service_name: str) -> None:
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import Status, StatusCode

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        otel_trace.set_tracer_provider(provider)

        self._otel = otel_trace
        self._tracer = provider.get_tracer("enterprise_agent")
        self._status = Status
        self._status_code = StatusCode

    def start(self, name: str, attributes: dict[str, Any], parent: Any = None) -> Any:
        context = self._otel.set_span_in_context(parent) if parent is not None else None
        return self._tracer.start_span(
            name, context=context, attributes=_flat_attributes(attributes)
        )

    def finish(self, span: Any, status: str, error: BaseException | None) -> None:
        if span is None:
            return
        if error is not None:
            span.record_exception(error)
            span.set_status(self._status(self._status_code.ERROR, str(error)[:200]))
        elif status == "error":
            span.set_status(self._status(self._status_code.ERROR))
        span.end()


@dataclass(slots=True)
class _Runtime:
    enabled: bool = True
    log_spans: bool = True
    include_in_response: bool = False
    max_records: int = 200
    service_name: str = "enterprise-agent"
    endpoint: str | None = None


_RUNTIME = _Runtime()
_STORE = TraceStore(_RUNTIME.max_records)
_BRIDGE: _OtlpBridge | None = None
_BRIDGE_LOCK = threading.Lock()


def configure_observability(
    *,
    enabled: bool = True,
    log_spans: bool = True,
    include_in_response: bool = False,
    max_records: int = 200,
    service_name: str = "enterprise-agent",
    otlp_endpoint: str | None = None,
) -> None:
    """Apply runtime settings; safe to call repeatedly (tests, app startup)."""
    global _RUNTIME, _STORE, _BRIDGE
    _RUNTIME = _Runtime(
        enabled=enabled,
        log_spans=log_spans,
        include_in_response=include_in_response,
        max_records=max_records,
        service_name=service_name,
        endpoint=otlp_endpoint,
    )
    _STORE = TraceStore(max_records)

    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None
    with _BRIDGE_LOCK:
        _BRIDGE = None
        if enabled and endpoint:
            try:
                _BRIDGE = _OtlpBridge(endpoint, service_name)
                logger.info("OTLP trace export enabled endpoint=%s", endpoint)
            except Exception:
                logger.warning(
                    "OTLP trace export unavailable; keeping local traces", exc_info=True
                )


def langsmith_enabled() -> bool:
    """LangChain exports traces to LangSmith when these env vars are set.

    LangGraph nodes are LangChain runnables, so no code change is needed:
    setting the variables is enough to get per-node runs in LangSmith.
    """
    flag = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    return flag in {"1", "true", "yes", "on"} and bool(api_key)


def trace_store() -> TraceStore:
    return _STORE


def tracing_config() -> _Runtime:
    return _RUNTIME


def current_trace() -> RequestTrace | None:
    return _CURRENT_TRACE.get()


class _SpanContext:
    """Timed span; records latency, status and error details on exit."""

    def __init__(
        self,
        trace: RequestTrace,
        name: str,
        attributes: dict[str, Any],
        parent_id: str | None,
    ) -> None:
        self.trace = trace
        self.name = name
        self.span_id = _new_id()
        self.parent_id = parent_id
        self.attributes = dict(attributes)
        self.started_at = _now()
        self.duration_ms = 0.0
        self.status = "ok"
        self.error_type: str | None = None
        self.error_message: str | None = None
        self._started = 0.0
        self._token: Any = None
        self._otel_span: Any = None

    def __enter__(self) -> _SpanContext:
        self._started = time.perf_counter()
        self._token = _CURRENT_SPAN.set(self.span_id)
        if _BRIDGE is not None and _RUNTIME.enabled:
            try:
                parent = self.trace.otel_spans().get(self.parent_id) if self.parent_id else self.trace.otel_root()
                self._otel_span = _BRIDGE.start(self.name, self.attributes, parent)
                if self._otel_span is not None:
                    self.trace.otel_spans()[self.span_id] = self._otel_span
            except Exception:
                logger.debug("failed to start OTLP span", exc_info=True)
        return self

    def set(self, **attributes: Any) -> _SpanContext:
        self.attributes.update(attributes)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.duration_ms = (time.perf_counter() - self._started) * 1000.0
        record = SpanRecord(
            name=self.name,
            span_id=self.span_id,
            parent_id=self.parent_id,
            started_at=self.started_at,
            duration_ms=self.duration_ms,
            attributes=self.attributes,
        )
        if exc is not None:
            record.status = "error"
            record.error_type = type(exc).__name__
            record.error_message = str(exc)[:2000]
            self.status = "error"
            self.error_type = record.error_type
            self.error_message = record.error_message

        self.trace.spans.append(record)
        if _BRIDGE is not None:
            try:
                _BRIDGE.finish(self._otel_span, record.status, exc)
            except Exception:
                logger.debug("failed to finish OTLP span", exc_info=True)

        if _RUNTIME.enabled and _RUNTIME.log_spans:
            try:
                logger.info(
                    json.dumps(
                        {"event": "span", "trace_id": self.trace.trace_id, **record.as_dict()},
                        ensure_ascii=False,
                        default=str,
                    )
                )
            except Exception:
                logger.debug("failed to serialize span", exc_info=True)

        if self._token is not None:
            _CURRENT_SPAN.reset(self._token)
        return False


class _NoopSpan:
    def set(self, **attributes: Any) -> _NoopSpan:
        return self


class RequestTrace:
    """One request worth of spans sharing a trace id."""

    def __init__(self, name: str, attributes: dict[str, Any]) -> None:
        self.trace_id = _new_id()
        self.name = name
        self.attributes = dict(attributes)
        self.spans: list[SpanRecord] = []
        self.started_at = _now()
        self.duration_ms = 0.0
        self.status = "ok"
        self.error_type: str | None = None
        self.error_message: str | None = None
        self._started = 0.0
        self._trace_token: Any = None
        self._span_token: Any = None
        self._otel_root_span: Any = None
        self._otel_spans: dict[str, Any] = {}

    def otel_root(self) -> Any:
        return self._otel_root_span

    def otel_spans(self) -> dict[str, Any]:
        return self._otel_spans

    def __enter__(self) -> RequestTrace:
        self._started = time.perf_counter()
        self._trace_token = _CURRENT_TRACE.set(self)
        self._span_token = _CURRENT_SPAN.set(None)
        if _BRIDGE is not None and _RUNTIME.enabled:
            try:
                self._otel_root_span = _BRIDGE.start(self.name, self.attributes)
            except Exception:
                logger.debug("failed to start OTLP root span", exc_info=True)
        return self

    def set(self, **attributes: Any) -> RequestTrace:
        self.attributes.update(attributes)
        return self

    def record_error(self, exc: BaseException) -> None:
        self.status = "error"
        self.error_type = type(exc).__name__
        self.error_message = str(exc)[:2000]

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_SpanContext]:
        context = _SpanContext(self, name, attributes, _CURRENT_SPAN.get())
        with context:
            yield context

    def as_record(self) -> TraceRecord:
        return TraceRecord(
            trace_id=self.trace_id,
            name=self.name,
            started_at=self.started_at,
            duration_ms=self.duration_ms,
            status=self.status,
            error_type=self.error_type,
            error_message=self.error_message,
            attributes=self.attributes,
            spans=list(self.spans),
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.duration_ms = (time.perf_counter() - self._started) * 1000.0
        if exc is not None:
            self.record_error(exc)
        elif self.status == "ok" and any(span.status == "error" for span in self.spans):
            # The request completed, but through a fallback path. Surfacing this
            # separately keeps silent degradation visible in /api/traces.
            self.status = "degraded"

        if _BRIDGE is not None:
            try:
                _BRIDGE.finish(self._otel_root_span, self.status, exc)
            except Exception:
                logger.debug("failed to finish OTLP root span", exc_info=True)

        if _RUNTIME.enabled:
            record = self.as_record()
            _STORE.add(record)
            try:
                logger.info(
                    json.dumps(
                        {"event": "trace", **record.summary()},
                        ensure_ascii=False,
                        default=str,
                    )
                )
            except Exception:
                logger.debug("failed to serialize trace", exc_info=True)

        if self._span_token is not None:
            _CURRENT_SPAN.reset(self._span_token)
        if self._trace_token is not None:
            _CURRENT_TRACE.reset(self._trace_token)
        return False


@contextmanager
def start_trace(name: str, **attributes: Any) -> Iterator[RequestTrace]:
    """Open a trace, or nest into the active one when a request already has it."""
    existing = _CURRENT_TRACE.get()
    if existing is not None:
        with existing.span(name, **attributes):
            yield existing
        return

    trace = RequestTrace(name, attributes)
    with trace:
        yield trace


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Record a span inside the active trace; a no-op when tracing is off."""
    trace = _CURRENT_TRACE.get()
    if trace is None or not _RUNTIME.enabled:
        yield _NoopSpan()
        return
    with trace.span(name, **attributes) as context:
        yield context