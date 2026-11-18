# Retrieval Evaluation

- Dataset: Bank RAG Benchmark
- Cases evaluated: 450
- Documents indexed: 100 (599 chunks)
- Top-K retrieved per query: 10
- Embedding provider: pretrained (dim=384)
- Chunking: project defaults (chunk_size=520, overlap=80)

## Overall

| Metric | Value |
| --- | ---: |
| Recall@5 | 0.7144 |
| Recall@10 | 0.7692 |
| MRR@10 | 0.7794 |
| nDCG@5 | 0.6627 |
| nDCG@10 | 0.6865 |
| P@5 | 0.3284 |
| P95 latency | 8.639 ms |
| Mean latency | 7.214 ms |
| Hit rate@10 | 0.9489 |

## By question type

| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基础事实 | 50 | 0.8700 | 0.9300 | 0.7017 | 0.7329 | 0.7527 | 0.1800 | 8.883 |
| 多跳检索 | 100 | 0.6904 | 0.7565 | 0.8243 | 0.6629 | 0.6917 | 0.3500 | 8.118 |
| 版本更新 | 100 | 0.6225 | 0.6833 | 0.8400 | 0.6324 | 0.6613 | 0.3820 | 8.561 |
| 相似文档 | 100 | 0.7961 | 0.8451 | 0.7664 | 0.7025 | 0.7247 | 0.3540 | 8.753 |
| 逻辑推理与边界 | 100 | 0.6710 | 0.7113 | 0.7258 | 0.6179 | 0.6350 | 0.3020 | 8.529 |

## By difficulty

| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 中等 | 200 | 0.7093 | 0.7642 | 0.8032 | 0.6675 | 0.6930 | 0.3680 | 8.725 |
| 困难 | 200 | 0.6807 | 0.7339 | 0.7750 | 0.6404 | 0.6634 | 0.3260 | 8.427 |
| 简单 | 50 | 0.8700 | 0.9300 | 0.7017 | 0.7329 | 0.7527 | 0.1800 | 8.883 |

## Notes

- Metrics are document-level: chunks are collapsed to their source document, and the first occurrence defines the document rank.
- Relevance is binary. Recall@K divides by the full gold document count, so cases with many gold documents cannot reach 1.0 at small K.
- P@5 divides by 5 in the usual way.
- Latency covers only KnowledgeBase.search (query embedding plus vector search); prompt assembly and generation are excluded.
- min_relevance is applied later in the answer node, so ranking is measured unfiltered, matching the retrieval layer as it is today.
- Settings are the project defaults. The environment is not read.
