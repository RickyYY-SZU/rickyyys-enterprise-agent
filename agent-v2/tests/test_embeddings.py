from enterprise_agent.embeddings import HashingEmbedder
from enterprise_agent.knowledge import chunk_text


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_local_embedding_ranks_related_chinese_text_higher() -> None:
    embedder = HashingEmbedder(384)
    query = embedder.embed_query("VPN 无法连接")
    related = embedder.embed_query("VPN 客户端连接超时，请检查网络和账号")
    unrelated = embedder.embed_query("差旅报销发票需要公司抬头")
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_chunking_preserves_content_and_honors_size() -> None:
    text = "第一段说明。\n\n" + ("这是一个较长的企业流程说明。" * 50)
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 2
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "第一段说明" in chunks[0]
    assert chunks[-1].endswith("说明。")


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text(" \n ") == []

