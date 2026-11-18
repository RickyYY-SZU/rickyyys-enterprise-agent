# Retrieval Evaluation

- Dataset: Bank RAG Benchmark
- Cases evaluated: 450
- Documents indexed: 100 (599 chunks)
- Top-K retrieved per query: 10
- Embedding provider: local (dim=384)
- Reranker: bge-reranker-base
- Chunking: project defaults (chunk_size=520, overlap=80)

## Overall

| Metric | Value |
| --- | ---: |
| Recall@5 | 0.6813 |
| Recall@10 | 0.7235 |
| MRR@10 | 0.7657 |
| nDCG@5 | 0.6442 |
| nDCG@10 | 0.6618 |
| P@5 | 0.3107 |
| P95 latency | 2284.825 ms |
| Mean latency | 1910.126 ms |
| Hit rate@10 | 0.9156 |

## By question type

| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基础事实 | 50 | 0.8700 | 0.9200 | 0.6494 | 0.6969 | 0.7136 | 0.1800 | 2158.302 |
| 多跳检索 | 100 | 0.6407 | 0.6682 | 0.7832 | 0.6361 | 0.6472 | 0.3160 | 2302.283 |
| 版本更新 | 100 | 0.5983 | 0.6448 | 0.8343 | 0.6220 | 0.6426 | 0.3680 | 2095.317 |
| 相似文档 | 100 | 0.7979 | 0.8420 | 0.8223 | 0.7317 | 0.7516 | 0.3560 | 2198.085 |
| 逻辑推理与边界 | 100 | 0.5938 | 0.6408 | 0.6812 | 0.5609 | 0.5801 | 0.2680 | 2638.951 |

## By difficulty

| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 中等 | 200 | 0.6981 | 0.7434 | 0.8283 | 0.6768 | 0.6971 | 0.3620 | 2175.856 |
| 困难 | 200 | 0.6173 | 0.6545 | 0.7322 | 0.5985 | 0.6136 | 0.2920 | 2434.561 |
| 简单 | 50 | 0.8700 | 0.9200 | 0.6494 | 0.6969 | 0.7136 | 0.1800 | 2158.302 |

## Notes

- Metrics are document-level: chunks are collapsed to their source document, and the first occurrence defines the document rank.
- Relevance is binary. Recall@K divides by the full gold document count, so cases with many gold documents cannot reach 1.0 at small K.
- P@5 divides by 5 in the usual way.
- Latency covers only KnowledgeBase.search (query embedding plus vector search); prompt assembly and generation are excluded.
- min_relevance is applied later in the answer node, so ranking is measured unfiltered, matching the retrieval layer as it is today.
- Settings are the project defaults. The environment is not read.
