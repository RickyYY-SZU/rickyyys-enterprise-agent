from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

from .config import PROJECT_ROOT, Settings


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


DEFAULT_PRETRAINED_MODEL = "bge-small-zh-v1.5"
DEFAULT_MODEL_REPO = "Xenova/bge-small-zh-v1.5"
DEFAULT_MODEL_ENDPOINT = (
    "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
)
# Local cache filename -> path inside the ModelScope repository.
MODEL_FILES = {
    "config.json": "config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "special_tokens_map.json": "special_tokens_map.json",
    "model.onnx": "onnx/model.onnx",
}
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class PretrainedChineseEmbedder(Embedder):
    """Chinese pretrained embedding model served locally through ONNX.

    Defaults to BAAI bge-small-zh-v1.5 exported to ONNX. The model is fetched once
    from ModelScope into a local cache and then loaded offline, so retrieval keeps
    working without an API key or a Python ML framework.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        model_name: str = DEFAULT_PRETRAINED_MODEL,
        repo: str = DEFAULT_MODEL_REPO,
        endpoint: str = DEFAULT_MODEL_ENDPOINT,
        max_length: int = 512,
        batch_size: int = 16,
        use_instruction: bool = True,
        threads: int | None = None,
    ) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "EMBEDDING_PROVIDER=pretrained requires onnxruntime and tokenizers"
            ) from exc

        self._np = np
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.use_instruction = use_instruction
        self.model_dir = model_dir or default_model_dir(model_name)
        download_model(self.model_dir, repo=repo, endpoint=endpoint)

        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        # The upstream vocab only contains lowercase English pieces, while
        # tokenizer_config.json disables lowercasing, which turns acronyms such as
        # VPN or SLA into [UNK]. Lowercasing keeps those domain terms searchable.
        self._tokenizer.normalizer.lowercase = True
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        self.dimension = int(self._session.get_outputs()[0].shape[-1])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encodings = self._tokenizer.encode_batch(batch)
            feed = {
                "input_ids": self._np.array([item.ids for item in encodings], dtype=self._np.int64),
                "attention_mask": self._np.array(
                    [item.attention_mask for item in encodings], dtype=self._np.int64
                ),
            }
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = self._np.array(
                    [item.type_ids for item in encodings], dtype=self._np.int64
                )
            hidden = self._session.run(None, feed)[0]
            cls = hidden[:, 0, :]
            norms = self._np.linalg.norm(cls, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            outputs.extend((cls / norms).tolist())
        return outputs

    def embed_query(self, text: str) -> list[float]:
        if self.use_instruction and text.strip():
            text = QUERY_INSTRUCTION + text
        return self._encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)


def default_model_dir(model_name: str = DEFAULT_PRETRAINED_MODEL) -> Path:
    override = os.getenv("EMBEDDING_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / "models" / model_name


def download_model(
    target: Path,
    repo: str = DEFAULT_MODEL_REPO,
    endpoint: str = DEFAULT_MODEL_ENDPOINT,
    timeout: float = 120.0,
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
    if settings.embedding_provider in {"pretrained", "bge"}:
        return PretrainedChineseEmbedder(
            model_name=settings.embedding_model,
            max_length=settings.embedding_max_length,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")

