import json
from pathlib import Path
from typing import Any

from enterprise_agent.agent import EnterpriseAgent
from enterprise_agent.config import Settings
from enterprise_agent.knowledge import RetrievedChunk
from enterprise_agent.schemas import ChatRequest
from enterprise_agent.tickets import TicketStore


class FakeKnowledge:
    def search(self, query: str, limit: int = 4) -> list[RetrievedChunk]:
        if "VPN" in query:
            return [
                RetrievedChunk(
                    id="1",
                    title="VPN 使用与故障排查",
                    source="internal://it/vpn",
                    content="检查客户端版本和网络后重新认证。",
                    score=0.88,
                    document_id="doc-1",
                )
            ]
        return []

    @staticmethod
    def serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "title": chunk.title,
            "source": chunk.source,
            "content": chunk.content,
            "score": chunk.score,
            "tags": chunk.tags,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
        }


def _agent(tmp_path: Path) -> EnterpriseAgent:
    settings = Settings(llm_provider="local", min_relevance=0.2)
    store = TicketStore(tmp_path / "tickets.db")
    return EnterpriseAgent(settings, FakeKnowledge(), store)  # type: ignore[arg-type]


def test_vpn_issue_retrieves_and_creates_ticket(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.run(
        ChatRequest(query="VPN 无法连接，请帮我处理", requester="tester")
    )
    assert result["intent"] == "support_request"
    assert result["sources"]
    assert result["ticket"]["ticket_no"].startswith("ENT-")
    assert "已创建工单" in result["answer"]


def test_ticket_can_be_disabled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.run(
        ChatRequest(
            query="VPN 无法连接，请帮我处理",
            auto_create_ticket=False,
        )
    )
    assert result["ticket"] is None
    assert result["intent"] == "support_request"


def test_smalltalk_does_not_search_or_create_ticket(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.run(ChatRequest(query="你好"))
    assert result["intent"] == "smalltalk"
    assert result["ticket"] is None
    assert result["sources"] == []


def test_printer_fault_is_treated_as_support_request(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.run(
        ChatRequest(query="打印机一直卡纸，请帮我处理", auto_create_ticket=True)
    )
    assert result["intent"] == "support_request"
    assert result["category"] == "网络与设备"


def test_sla_question_is_classified_as_knowledge_question(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.run(
        ChatRequest(query="P1 工单的响应时间是多少？", auto_create_ticket=False)
    )
    assert result["intent"] == "knowledge_question"


def test_ticket_tool_returns_json(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = json.loads(
        agent.ticket_tool.invoke(
            {
                "title": "账号锁定",
                "description": "登录后提示锁定",
                "requester": "tester",
                "priority": "high",
                "category": "账号与权限",
            }
        )
    )
    assert result["status"] == "open"

