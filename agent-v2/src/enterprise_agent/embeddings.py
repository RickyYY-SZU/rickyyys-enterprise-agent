from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from abc import ABC, abstractmethod

from .config import Settings


class Embedder(ABC):
    dimension: int

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Dependency-free multilingual feature hashing.

    It is intentionally local so the demo can start without downloading a model.
    Character n-grams make Chinese retrieval useful while still supporting English.
    For production quality, switch EMBEDDING_PROVIDER to openai.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("Embedding dimension must be at least 64")
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> list[tuple[str, float]]:
        text = unicodedata.normalize("NFKC", text).lower()
        words = re.findall(r"[a-z0-9_+.-]+", text)
        cjk = re.findall(r"[\u3400-\u9fff]", text)
        bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        trigrams = [cjk[i] + cjk[i + 1] + cjk[i + 2] for i in range(len(cjk) - 2)]
        tokens: list[tuple[str, float]] = []
        tokens.extend((f"w:{word}", 1.8) for word in words)
        tokens.extend((f"c:{char}", 0.7) for char in cjk)
        tokens.extend((f"b:{gram}", 1.35) for gram in bigrams)
        tokens.extend((f"t:{gram}", 1.05) for gram in trigrams)
        return tokens or [("empty", 1.0)]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token, weight in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 9) & 1 else -1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(item * item for item in vector))
        if norm:
            vector = [item / norm for item in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class OpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required when EMBEDDING_PROVIDER=openai")
        from langchain_openai import OpenAIEmbeddings

        self.dimension = settings.embedding_dimension
        self._client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            check_embedding_ctx_length=False,
        )

    def embed_query(self, text: str) -> list[float]:
        vector = self._client.embed_query(text)
        if len(vector) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: configured={self.dimension}, actual={len(vector)}"
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._client.embed_documents(texts)
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: configured={self.dimension}, actual={len(vector)}"
                )
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "local":
        return HashingEmbedder(settings.embedding_dimension)
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")

