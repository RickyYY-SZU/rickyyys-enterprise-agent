from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    app_name: str = "EnterpriseAgent"
    app_env: str = "development"
    log_level: str = "INFO"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "enterprise_knowledge"
    embedding_provider: str = "local"
    embedding_dimension: int = 384
    embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "local"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    auto_seed: bool = True
    seed_tickets: bool = True
    min_relevance: float = 0.1
    rerank_enabled: bool = True
    rerank_model: str = "bge-reranker-base"
    rerank_candidates: int = 20
    rerank_max_length: int = 512
    rerank_batch_size: int = 16
    rerank_device: str = "auto"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    seed_documents_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "seed_documents.json"
    )
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def database_path(self) -> Path:
        return self.data_dir / "enterprise.db"

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
        seed_path = Path(
            os.getenv("SEED_DOCUMENTS_PATH", str(PROJECT_ROOT / "data" / "seed_documents.json"))
        )
        origins = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]
        api_key = os.getenv("QDRANT_API_KEY") or None
        llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
        return cls(
            app_name=os.getenv("APP_NAME", "EnterpriseAgent"),
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=api_key,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "enterprise_knowledge"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_provider=os.getenv("LLM_PROVIDER", "local").lower(),
            llm_api_key=llm_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            auto_seed=_as_bool(os.getenv("AUTO_SEED"), True),
            seed_tickets=_as_bool(os.getenv("SEED_TICKETS"), True),
            min_relevance=float(os.getenv("MIN_RELEVANCE", "0.1")),
            rerank_enabled=_as_bool(os.getenv("RERANK_ENABLED"), True),
            rerank_model=os.getenv("RERANK_MODEL", "bge-reranker-base"),
            rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "20")),
            rerank_max_length=int(os.getenv("RERANK_MAX_LENGTH", "512")),
            rerank_batch_size=int(os.getenv("RERANK_BATCH_SIZE", "16")),
            rerank_device=os.getenv("RERANK_DEVICE", "auto").lower(),
            data_dir=data_dir,
            seed_documents_path=seed_path,
            cors_origins=origins or ["*"],
        )

