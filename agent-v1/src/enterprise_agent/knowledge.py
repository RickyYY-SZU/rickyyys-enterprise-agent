from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import Settings
from .embeddings import Embedder


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    title: str
    source: str
    content: str
    score: float
    tags: list[str] = field(default_factory=list)
    document_id: str = ""
    chunk_index: int = 0


def chunk_text(text: str, chunk_size: int = 520, overlap: int = 80) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    soft_boundaries = "\n。！？.!?；;"
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary_start = start + chunk_size // 2
            boundary = max(text.rfind(char, boundary_start, end) for char in soft_boundaries)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class KnowledgeBase:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10,
        )

    def connect(self, retries: int = 20, delay_seconds: float = 1.0) -> None:
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                self.client.get_collections()
                return
            except Exception as exc:  # pragma: no cover - depends on external service
                last_error = exc
                time.sleep(delay_seconds)
        raise RuntimeError(f"Unable to connect to Qdrant at {self.settings.qdrant_url}") from last_error

    def ensure_collection(self) -> None:
        name = self.settings.qdrant_collection
        if self.client.collection_exists(name):
            info = self.client.get_collection(name)
            vectors = info.config.params.vectors
            existing_size = getattr(vectors, "size", None)
            if existing_size != self.embedder.dimension:
                self.client.delete_collection(name)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def bootstrap(self) -> int:
        self.ensure_collection()
        count = self.client.count(self.settings.qdrant_collection, exact=True).count
        if count or not self.settings.auto_seed:
            return 0
        path = self.settings.seed_documents_path
        if not path.exists():
            return 0
        documents = json.loads(path.read_text(encoding="utf-8"))
        indexed = 0
        for document in documents:
            self.ingest(
                title=document["title"],
                content=document["content"],
                source=document["source"],
                tags=document.get("tags", ["seed"]),
            )
            indexed += 1
        return indexed

    def ingest(
        self,
        title: str,
        content: str,
        source: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        tags = tags or []
        chunks = chunk_text(content)
        if not chunks:
            raise ValueError("Document content is empty")
        document_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{self.settings.qdrant_collection}:{source}:{title}")
        )
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        vectors = self.embedder.embed_documents(chunks)
        created_at = datetime.now(UTC).isoformat()
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            point_id = str(uuid.uuid5(uuid.UUID(document_id), str(index)))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "chunk_index": index,
                        "title": title,
                        "source": source,
                        "tags": tags,
                        "content": chunk,
                        "created_at": created_at,
                    },
                )
            )
        self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=points,
            wait=True,
        )
        return {
            "document_id": document_id,
            "title": title,
            "source": source,
            "tags": tags,
            "chunks": len(chunks),
            "created_at": created_at,
        }

    def search(self, query: str, limit: int = 4) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        vector = self.embedder.embed_query(query)
        result = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=max(1, min(limit, 20)),
            with_payload=True,
        )
        chunks: list[RetrievedChunk] = []
        for point in result.points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    id=str(point.id),
                    title=str(payload.get("title", "Untitled")),
                    source=str(payload.get("source", "")),
                    content=str(payload.get("content", "")),
                    score=round(float(point.score), 4),
                    tags=list(payload.get("tags", [])),
                    document_id=str(payload.get("document_id", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )
        return chunks

    def list_documents(self) -> list[dict[str, Any]]:
        records, _ = self.client.scroll(
            collection_name=self.settings.qdrant_collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        documents: dict[str, dict[str, Any]] = {}
        for point in records:
            payload = point.payload or {}
            document_id = str(payload.get("document_id", point.id))
            current = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "title": payload.get("title", "Untitled"),
                    "source": payload.get("source", ""),
                    "tags": payload.get("tags", []),
                    "chunks": 0,
                    "created_at": payload.get("created_at", ""),
                },
            )
            current["chunks"] += 1
        return sorted(documents.values(), key=lambda item: item["created_at"], reverse=True)

    def count(self) -> int:
        return self.client.count(self.settings.qdrant_collection, exact=True).count

    @staticmethod
    def serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
        return asdict(chunk)


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unsupported text encoding; please upload UTF-8 or GB18030 text")

