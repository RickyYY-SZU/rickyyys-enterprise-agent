from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import __version__
from .agent import EnterpriseAgent
from .config import PROJECT_ROOT, Settings
from .embeddings import build_embedder
from .knowledge import KnowledgeBase, read_text_file
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIngest,
    DocumentRead,
    SearchRequest,
    TicketCreate,
    TicketNote,
    TicketRead,
    TicketUpdate,
)
from .tickets import InvalidTransitionError, TicketNotFoundError, TicketStore

logger = logging.getLogger("enterprise_agent")
STATIC_DIR = PROJECT_ROOT / "static"


def _services(request: Request) -> tuple[KnowledgeBase, TicketStore, EnterpriseAgent]:
    return (
        request.app.state.knowledge,
        request.app.state.tickets,
        request.app.state.agent,
    )


def _knowledge(request: Request) -> KnowledgeBase:
    return request.app.state.knowledge


def _tickets(request: Request) -> TicketStore:
    return request.app.state.tickets


def _agent(request: Request) -> EnterpriseAgent:
    return request.app.state.agent


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        knowledge = KnowledgeBase(settings, build_embedder(settings))
        knowledge.connect()
        seeded = knowledge.bootstrap()

        tickets = TicketStore(settings.database_path)
        if settings.seed_tickets:
            tickets.seed_examples()

        app.state.knowledge = knowledge
        app.state.tickets = tickets
        app.state.agent = EnterpriseAgent(settings, knowledge, tickets)
        logger.info(
            "EnterpriseAgent ready mode=%s collection=%s seeded_documents=%s",
            "llm" if settings.llm_provider == "openai" and settings.llm_api_key else "local",
            settings.qdrant_collection,
            seeded,
        )
        yield
        knowledge.client.close()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Enterprise knowledge retrieval and ticket handling agent powered by "
            "FastAPI, LangGraph, and Qdrant."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(TicketNotFoundError)
    async def ticket_not_found_handler(
        _request: Request, exc: TicketNotFoundError
    ) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": f"Ticket {exc.args[0]} not found"})

    @app.exception_handler(InvalidTransitionError)
    async def transition_handler(_request: Request, exc: InvalidTransitionError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz", tags=["system"])
    async def healthz(request: Request) -> dict[str, Any]:
        try:
            points = request.app.state.knowledge.count()
            qdrant_ready = True
        except Exception:
            points = 0
            qdrant_ready = False
        return {
            "status": "ok" if qdrant_ready else "degraded",
            "version": __version__,
            "environment": settings.app_env,
            "agent_mode": (
                "llm"
                if settings.llm_provider == "openai" and settings.llm_api_key
                else "local"
            ),
            "embedding_provider": settings.embedding_provider,
            "qdrant_ready": qdrant_ready,
            "knowledge_points": points,
        }

    @app.get("/api", tags=["system"])
    async def api_root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "endpoints": {
                "chat": "/api/chat",
                "knowledge": "/api/documents",
                "tickets": "/api/tickets",
                "dashboard": "/api/dashboard",
            },
        }

    @app.post("/api/chat", response_model=ChatResponse, tags=["agent"])
    async def chat(
        payload: ChatRequest,
        agent: EnterpriseAgent = Depends(_agent),
    ) -> ChatResponse:
        result = agent.run(payload)
        return ChatResponse(**result)

    @app.get("/api/agent/tools", tags=["agent"])
    async def agent_tools(agent: EnterpriseAgent = Depends(_agent)) -> list[dict[str, Any]]:
        return agent.tool_descriptions()

    @app.get("/api/agent/graph", tags=["agent"])
    async def agent_graph(agent: EnterpriseAgent = Depends(_agent)) -> dict[str, str]:
        return {"mermaid": agent.graph.get_graph().draw_mermaid()}

    @app.post("/api/search", tags=["knowledge"])
    async def search(
        payload: SearchRequest,
        knowledge: KnowledgeBase = Depends(_knowledge),
    ) -> list[dict[str, Any]]:
        chunks = knowledge.search(payload.query, limit=payload.limit)
        return [knowledge.serialize_chunk(chunk) for chunk in chunks]

    @app.post("/api/documents", response_model=DocumentRead, tags=["knowledge"])
    async def ingest_document(
        payload: DocumentIngest,
        knowledge: KnowledgeBase = Depends(_knowledge),
    ) -> DocumentRead:
        try:
            result = knowledge.ingest(
                title=payload.title,
                content=payload.content,
                source=payload.source,
                tags=payload.tags,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DocumentRead(**result)

    @app.post("/api/documents/upload", response_model=DocumentRead, tags=["knowledge"])
    async def upload_document(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        tags: str = Form(default=""),
        knowledge: KnowledgeBase = Depends(_knowledge),
    ) -> DocumentRead:
        if not file.filename:
            raise HTTPException(status_code=422, detail="A file is required")
        if not file.filename.lower().endswith((".txt", ".md", ".csv")):
            raise HTTPException(
                status_code=415,
                detail="Only .txt, .md, and .csv text files are supported",
            )
        raw = await file.read()
        if len(raw) > 2_000_000:
            raise HTTPException(status_code=413, detail="File exceeds the 2 MB limit")
        safe_filename = Path(file.filename).name
        temp_path = settings.data_dir / f".upload-{uuid.uuid4().hex}"
        temp_path.write_bytes(raw)
        try:
            content = read_text_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        document_title = (title or Path(safe_filename).stem).strip()
        source = f"upload://{safe_filename}"
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        result = knowledge.ingest(
            title=document_title,
            content=content,
            source=source,
            tags=tag_list,
        )
        return DocumentRead(**result)

    @app.get("/api/documents", response_model=list[DocumentRead], tags=["knowledge"])
    async def list_documents(
        knowledge: KnowledgeBase = Depends(_knowledge),
    ) -> list[DocumentRead]:
        return [DocumentRead(**item) for item in knowledge.list_documents()]

    @app.post("/api/tickets", response_model=TicketRead, tags=["tickets"])
    async def create_ticket(
        payload: TicketCreate,
        tickets: TicketStore = Depends(_tickets),
    ) -> TicketRead:
        return TicketRead(**tickets.create(payload))

    @app.get("/api/tickets", response_model=list[TicketRead], tags=["tickets"])
    async def list_tickets(
        status: str | None = Query(default=None),
        priority: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        tickets: TicketStore = Depends(_tickets),
    ) -> list[TicketRead]:
        return [
            TicketRead(**ticket)
            for ticket in tickets.list(status=status, priority=priority, limit=limit)
        ]

    @app.get("/api/tickets/{ticket_id}", response_model=TicketRead, tags=["tickets"])
    async def get_ticket(
        ticket_id: str,
        tickets: TicketStore = Depends(_tickets),
    ) -> TicketRead:
        return TicketRead(**tickets.get(ticket_id))

    @app.patch("/api/tickets/{ticket_id}", response_model=TicketRead, tags=["tickets"])
    async def update_ticket(
        ticket_id: str,
        payload: TicketUpdate,
        tickets: TicketStore = Depends(_tickets),
    ) -> TicketRead:
        return TicketRead(**tickets.update(ticket_id, payload))

    @app.post(
        "/api/tickets/{ticket_id}/notes",
        response_model=TicketRead,
        tags=["tickets"],
    )
    async def add_ticket_note(
        ticket_id: str,
        payload: TicketNote,
        tickets: TicketStore = Depends(_tickets),
    ) -> TicketRead:
        return TicketRead(**tickets.add_note(ticket_id, payload))

    @app.get("/api/dashboard", tags=["tickets"])
    async def dashboard(
        services: tuple[KnowledgeBase, TicketStore, EnterpriseAgent] = Depends(_services),
    ) -> dict[str, Any]:
        knowledge, tickets, _agent_service = services
        metrics = tickets.metrics()
        metrics["knowledge_documents"] = len(knowledge.list_documents())
        metrics["knowledge_chunks"] = knowledge.count()
        return metrics

    return app


app = create_app()

