from __future__ import annotations

import secrets
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .schemas import TicketCreate, TicketNote, TicketUpdate

TERMINAL_STATUSES = {"resolved", "closed"}
ALLOWED_TRANSITIONS = {
    "open": {"in_progress", "resolved", "closed"},
    "in_progress": {"pending", "resolved", "closed"},
    "pending": {"in_progress", "resolved", "closed"},
    "resolved": {"reopened", "closed"},
    "closed": {"reopened"},
    "reopened": {"in_progress", "resolved", "closed"},
}
SLA_HOURS = {"critical": 2, "high": 8, "medium": 24, "low": 72}


class TicketNotFoundError(KeyError):
    pass


class InvalidTransitionError(ValueError):
    pass


class TicketStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._schema_lock = threading.Lock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_schema(self) -> None:
        with self._schema_lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    ticket_no TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    requester TEXT NOT NULL,
                    email TEXT,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    due_at TEXT,
                    resolution TEXT
                );
                CREATE TABLE IF NOT EXISTS ticket_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(ticket_id) REFERENCES tickets(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
                CREATE INDEX IF NOT EXISTS idx_ticket_events_ticket ON ticket_events(ticket_id);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _ticket_number() -> str:
        day = datetime.now(UTC).strftime("%Y%m%d")
        return f"ENT-{day}-{secrets.token_hex(3).upper()}"

    def create(self, payload: TicketCreate) -> dict[str, Any]:
        ticket_id = secrets.token_hex(16)
        ticket_no = self._ticket_number()
        now = datetime.now(UTC)
        due_at = (now + timedelta(hours=SLA_HOURS[payload.priority])).isoformat()
        values = (
            ticket_id,
            ticket_no,
            payload.title,
            payload.description,
            payload.requester,
            payload.email,
            payload.priority,
            "open",
            payload.category,
            payload.source,
            now.isoformat(),
            now.isoformat(),
            due_at,
            None,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tickets (
                    id, ticket_no, title, description, requester, email, priority,
                    status, category, source, created_at, updated_at, due_at, resolution
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._insert_event(
                connection, ticket_id, "created", payload.requester, "工单已创建"
            )
        return self.get(ticket_id)

    def seed_examples(self) -> None:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        if count:
            return
        examples = [
            TicketCreate(
                title="VPN 客户端连接超时",
                description="在家办公时 VPN 连续连接超时，已尝试重启客户端。",
                requester="张明",
                email="zhangming@example.com",
                priority="high",
                category="网络与设备",
                source="demo",
            ),
            TicketCreate(
                title="报销审批流程未同步",
                description="上周提交的差旅报销仍显示待同步。",
                requester="李倩",
                email="liqian@example.com",
                priority="medium",
                category="财务报销",
                source="demo",
            ),
        ]
        created = [self.create(item) for item in examples]
        self.update(
            created[0]["id"],
            TicketUpdate(status="in_progress", note="网络组正在排查出口网关。", actor="王工"),
        )

    def get(self, ticket_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
            ).fetchone()
            if not row:
                raise TicketNotFoundError(ticket_id)
            events = connection.execute(
                """
                SELECT id, event_type, actor, note, created_at
                FROM ticket_events WHERE ticket_id = ? ORDER BY id ASC
                """,
                (ticket_id,),
            ).fetchall()
        result = dict(row)
        result["events"] = [dict(event) for event in events]
        return result

    def list(
        self,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM tickets {where}
                ORDER BY CASE priority
                    WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2 ELSE 3 END,
                    created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def update(self, ticket_id: str, payload: TicketUpdate) -> dict[str, Any]:
        current = self.get(ticket_id)
        target_status = payload.status or current["status"]
        if target_status != current["status"]:
            allowed = ALLOWED_TRANSITIONS.get(current["status"], set())
            if target_status not in allowed:
                raise InvalidTransitionError(
                    f"Cannot move ticket from {current['status']} to {target_status}"
                )
        priority = payload.priority or current["priority"]
        resolution = payload.resolution if payload.resolution is not None else current["resolution"]
        updated_at = self._now()
        due_at = current["due_at"]
        if payload.priority and payload.priority != current["priority"]:
            due_at = (
                datetime.now(UTC) + timedelta(hours=SLA_HOURS[payload.priority])
            ).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tickets
                SET status = ?, priority = ?, resolution = ?, updated_at = ?, due_at = ?
                WHERE id = ?
                """,
                (target_status, priority, resolution, updated_at, due_at, ticket_id),
            )
            if target_status != current["status"]:
                self._insert_event(
                    connection,
                    ticket_id,
                    "status_changed",
                    payload.actor,
                    f"{current['status']} -> {target_status}",
                )
            if payload.note:
                self._insert_event(connection, ticket_id, "note", payload.actor, payload.note)
        return self.get(ticket_id)

    def add_note(self, ticket_id: str, payload: TicketNote) -> dict[str, Any]:
        self.get(ticket_id)
        with self._connect() as connection:
            self._insert_event(connection, ticket_id, "note", payload.actor, payload.note)
            connection.execute(
                "UPDATE tickets SET updated_at = ? WHERE id = ?", (self._now(), ticket_id)
            )
        return self.get(ticket_id)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        ticket_id: str,
        event_type: str,
        actor: str,
        note: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO ticket_events(ticket_id, event_type, actor, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticket_id, event_type, actor, note, datetime.now(UTC).isoformat()),
        )

    def metrics(self) -> dict[str, Any]:
        now = self._now()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, priority, due_at FROM tickets"
            ).fetchall()
        total = len(rows)
        open_count = sum(row["status"] not in TERMINAL_STATUSES for row in rows)
        resolved_count = sum(row["status"] in TERMINAL_STATUSES for row in rows)
        overdue = sum(
            bool(row["due_at"])
            and row["due_at"] < now
            and row["status"] not in TERMINAL_STATUSES
            for row in rows
        )
        return {
            "total": total,
            "open": open_count,
            "resolved": resolved_count,
            "overdue": overdue,
            "resolution_rate": round(resolved_count / total, 3) if total else 0.0,
        }

