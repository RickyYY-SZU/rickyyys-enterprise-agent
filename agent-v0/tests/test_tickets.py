from pathlib import Path

import pytest

from enterprise_agent.schemas import TicketCreate, TicketNote, TicketUpdate
from enterprise_agent.tickets import InvalidTransitionError, TicketStore


def test_ticket_lifecycle_and_sla(tmp_path: Path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    ticket = store.create(
        TicketCreate(
            title="无法连接 VPN",
            description="客户端提示超时",
            requester="tester",
            priority="high",
            category="网络与设备",
        )
    )
    assert ticket["status"] == "open"
    assert ticket["due_at"]
    updated = store.update(
        ticket["id"],
        TicketUpdate(status="in_progress", note="正在排查", actor="operator"),
    )
    assert updated["status"] == "in_progress"
    assert updated["events"][-1]["note"] == "正在排查"
    noted = store.add_note(ticket["id"], TicketNote(note="已联系用户"))
    assert noted["events"][-1]["event_type"] == "note"
    resolved = store.update(ticket["id"], TicketUpdate(status="resolved"))
    assert resolved["status"] == "resolved"
    assert store.metrics()["resolution_rate"] == 1.0


def test_invalid_status_transition_is_rejected(tmp_path: Path) -> None:
    store = TicketStore(tmp_path / "tickets.db")
    ticket = store.create(TicketCreate(title="测试", description="测试描述"))
    store.update(ticket["id"], TicketUpdate(status="resolved"))
    with pytest.raises(InvalidTransitionError):
        store.update(ticket["id"], TicketUpdate(status="in_progress"))

