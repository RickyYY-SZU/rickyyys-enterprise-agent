from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Priority = Literal["low", "medium", "high", "critical"]
TicketStatus = Literal["open", "in_progress", "pending", "resolved", "closed", "reopened"]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    requester: str = Field(default="anonymous", max_length=80)
    email: str | None = Field(default=None, max_length=160)
    session_id: str | None = Field(default=None, max_length=100)
    auto_create_ticket: bool = True
    history: list[ChatTurn] = Field(default_factory=list)


class SourceRef(BaseModel):
    title: str
    source: str
    score: float
    excerpt: str


class TicketRead(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    ticket_no: str
    title: str
    description: str
    requester: str
    email: str | None = None
    priority: Priority
    status: TicketStatus
    category: str
    source: str
    created_at: datetime | str
    updated_at: datetime | str
    due_at: datetime | str | None = None
    resolution: str | None = None


class ChatResponse(BaseModel):
    answer: str
    intent: str
    category: str
    priority: Priority
    confidence: float
    sources: list[SourceRef] = Field(default_factory=list)
    ticket: TicketRead | None = None
    actions: list[str] = Field(default_factory=list)
    mode: Literal["local", "llm"] = "local"
    session_id: str | None = None
    trace_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=4, ge=1, le=20)


class DocumentIngest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=500_000)
    source: str = Field(min_length=1, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=20)


class DocumentRead(BaseModel):
    document_id: str
    title: str
    source: str
    tags: list[str] = Field(default_factory=list)
    chunks: int
    created_at: str


class TicketCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=10_000)
    requester: str = Field(default="anonymous", max_length=80)
    email: str | None = Field(default=None, max_length=160)
    priority: Priority = "medium"
    category: str = Field(default="其他", max_length=100)
    source: str = Field(default="manual", max_length=100)


class TicketUpdate(BaseModel):
    status: TicketStatus | None = None
    priority: Priority | None = None
    resolution: str | None = Field(default=None, max_length=10_000)
    note: str | None = Field(default=None, max_length=10_000)
    actor: str = Field(default="operator", max_length=80)


class TicketNote(BaseModel):
    note: str = Field(min_length=1, max_length=10_000)
    actor: str = Field(default="operator", max_length=80)

