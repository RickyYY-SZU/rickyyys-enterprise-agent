# Retrieval Evaluation

- Dataset: Bank RAG Benchmark
- Cases evaluated: 450
- Documents indexed: 100 (599 chunks)
- Top-K retrieved per query: 10
- Embedding provider: local (dim=384)
- Chunking: project defaults (chunk_size=520, overlap=80)

## Overall

| Metric | Value |
| --- | ---: |
| Recall@5 | 0.5970 |
| Recall@10 | 0.6741 |
| MRR@10 | 0.6541 |
| nDCG@5 | 0.5381 |
| nDCG@10 | 0.5701 |
| P@5 | 0.2733 |
| P95 latency | 0.909 ms |
| Mean latency | 0.640 ms |
| Hit rate@10 | 0.8822 |

## By question type

| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基础事实 | 50 | 0.7500 | 0.8800 | 0.5267 | 0.5692 | 0.6115 | 0.1560 | 0.842 |
| 多跳检索 | 100 | 0.5615 | 0.6349 | 0.6840 | 0.5268 | 0.5578 | 0.2780 | 1.060 |
| 版本更新 | 100 | 0.5403 | 0.6087 | 0.7585 | 0.5560 | 0.5864 | 0.3340 | 0.655 |
| 相似文档 | 100 | 0.7094 | 0.7888 | 0.6914 | 0.6074 | 0.6423 | 0.3160 | 0.891 |
| 逻辑推理与边界 | 100 | 0.5003 | 0.5610 | 0.5459 | 0.4467 | 0.4731 | 0.2240 | 0.912 |

## By difficulty

| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 中等 | 200 | 0.6249 | 0.6987 | 0.7250 | 0.5817 | 0.6143 | 0.3250 | 0.857 |
| 困难 | 200 | 0.5309 | 0.5979 | 0.6150 | 0.4868 | 0.5154 | 0.2510 | 0.953 |
| 简单 | 50 | 0.7500 | 0.8800 | 0.5267 | 0.5692 | 0.6115 | 0.1560 | 0.842 |

## Notes

- Metrics are document-level: chunks are collapsed to their source document, and the first occurrence defines the document rank.
- Relevance is binary. Recall@K divides by the full gold document count, so cases with many gold documents cannot reach 1.0 at small K.
- P@5 divides by 5 in the usual way.
- Latency covers only KnowledgeBase.search (query embedding plus vector search); prompt assembly and generation are excluded.
- min_relevance is applied later in the answer node, so ranking is measured unfiltered, matching the retrieval layer as it is today.
- Settings are the project defaults. The environment is not read.
