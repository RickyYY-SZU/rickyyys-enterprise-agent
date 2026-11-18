"""Cross-encoder rerankers for second-stage retrieval refinement."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from .config import PROJECT_ROOT, Settings

DEFAULT_RERANK_MODEL = "bge-reranker-base"
DEFAULT_RERANK_REPO = "Xenova/bge-reranker-base"
DEFAULT_MODEL_ENDPOINT = (
    "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
)
# Local cache filename -> path inside the ModelScope repository.
MODEL_FILES = {
    "config.json": "config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "special_tokens_map.json": "special_tokens_map.json",
    "model_int8.onnx": "onnx/model_int8.onnx",
}


class Reranker(ABC):
    """Scores how well each candidate document answers a query."""

    model_name: str

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document; higher means more relevant."""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """ONNX cross-encoder reranker, defaulting to BAAI bge-reranker-base int8.

    Unlike the bi-encoder used for retrieval, a cross-encoder reads the query and
    the document together, so it orders the candidate set returned by vector
    search far more sharply than cosine similarity alone can.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        model_name: str = DEFAULT_RERANK_MODEL,
        repo: str = DEFAULT_RERANK_REPO,
        endpoint: str = DEFAULT_MODEL_ENDPOINT,
        max_length: int = 512,
        batch_size: int = 16,
        threads: int | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "RERANK_ENABLED=true requires onnxruntime and tokenizers"
            ) from exc

        self._np = np
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.model_dir = model_dir or default_model_dir(model_name)
        download_model(self.model_dir, repo=repo, endpoint=endpoint)

        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.device = self._resolve_device(device, ort)
        if self.device == "cuda":
            _preload_cuda_libraries()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        try:
            self._session = ort.InferenceSession(
                str(self.model_dir / "model_int8.onnx"),
                sess_options=options,
                providers=providers,
            )
        except Exception:
            if self.device != "cuda":
                raise
            self.device = "cpu"
            self._session = ort.InferenceSession(
                str(self.model_dir / "model_int8.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self._input_names = {item.name for item in self._session.get_inputs()}

    @staticmethod
    def _resolve_device(device: str, ort: object) -> str:
        requested = (device or "auto").strip().lower()
        if requested in {"cpu", "false", "off"}:
            return "cpu"
        available = getattr(ort, "get_available_providers", lambda: [])()
        if "CUDAExecutionProvider" in available:
            return "cuda"
        if requested in {"cuda", "gpu"}:
            raise RuntimeError(
                "RERANK_DEVICE=cuda requested but CUDAExecutionProvider is unavailable"
            )
        return "cpu"

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            encodings = self._tokenizer.encode_batch([(query, text) for text in batch])
            feed = {
                "input_ids": self._np.array(
                    [item.ids for item in encodings], dtype=self._np.int64
                ),
                "attention_mask": self._np.array(
                    [item.attention_mask for item in encodings], dtype=self._np.int64
                ),
            }
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = self._np.array(
                    [item.type_ids for item in encodings], dtype=self._np.int64
                )
            logits = self._session.run(None, feed)[0][:, 0]
            scores.extend((1.0 / (1.0 + self._np.exp(-logits))).tolist())
        return scores


def _preload_cuda_libraries() -> None:
    """Make pip-installed CUDA/cuDNN wheels discoverable by onnxruntime.

    onnxruntime-gpu expects libcublas*/libcudnn* on the loader path. When CUDA
    comes from nvidia-*-cu12 wheels rather than a system toolkit, the libraries
    live under site-packages/nvidia/*/lib and are not searched by default.
    """
    import ctypes
    import site
    import sys

    candidates: list[Path] = []
    for base in {*site.getsitepackages(), *sys.path}:
        nvidia_root = Path(base) / "nvidia"
        if nvidia_root.is_dir():
            candidates.extend(sorted(nvidia_root.glob("*/lib")))
    if not candidates:
        return

    existing = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    for directory in candidates:
        text_dir = str(directory)
        if text_dir not in existing:
            existing.append(text_dir)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(existing)

    for directory in candidates:
        for library in sorted(directory.glob("*.so*")):
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def default_model_dir(model_name: str = DEFAULT_RERANK_MODEL) -> Path:
    override = os.getenv("RERANK_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / "models" / model_name


def download_model(
    target: Path,
    repo: str = DEFAULT_RERANK_REPO,
    endpoint: str = DEFAULT_MODEL_ENDPOINT,
    timeout: float = 300.0,
) -> None:
    """Fetch the ONNX export from ModelScope once and cache it on disk."""
    import urllib.request
    from urllib.parse import quote

    for name, remote_path in MODEL_FILES.items():
        destination = target / name
        if destination.exists() and destination.stat().st_size > 1024:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = endpoint.format(repo=repo, path=quote(remote_path, safe=""))
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
        temporary.replace(destination)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(
        model_name=settings.rerank_model,
        max_length=settings.rerank_max_length,
        batch_size=settings.rerank_batch_size,
        device=settings.rerank_device,
    )
