# Retrieval Evaluation: v0 → v3

Retrieval quality of the four versions, measured on the same frozen setup so the
numbers are directly comparable. `v0` is the baseline; `v1`/`v2`/`v3` are reported
as deltas against it.

## Evaluation setup

| Item | Value |
| --- | --- |
| Dataset | Bank RAG Benchmark (derived JSON) |
| Cases | 450 |
| Documents / chunks | 100 / 599 |
| Top-K | 10 |
| Chunking | `chunk_size=520`, `overlap=80` (project defaults) |
| Relevance | document-level, binary |
| Latency scope | `KnowledgeBase.search` only (query embedding + vector search) |

The evaluator is deliberately frozen: `scripts/evaluate_retrieval.py` uses
`Settings()` with **code defaults** (`locked_settings()`), does not read
environment variables, and runs against an isolated in-memory Qdrant instance.
Each version therefore benchmarks exactly the configuration baked into its own
`config.py` defaults.

## What each version changes

| Version | Retrieval change | Code default |
| --- | --- | --- |
| `agent-v0` | Baseline: zero-dependency hashed n-gram embedding | `embedding_provider="local"` |
| `agent-v1` | Pretrained Chinese embedding (ONNX, bge-small-zh-v1.5) | `embedding_provider="pretrained"` |
| `agent-v2` | Cross-encoder reranker (ONNX, bge-reranker-base) on the local embedder | `rerank_enabled=True`, `rerank_candidates=20` |
| `agent-v3` | No retrieval change; adds structured tracing and OTLP export | `embedding_provider="local"`, no reranker |

> `agent-v2` is **not** `v1 + reranker`. It keeps the hash embedder of `v0` and adds
> reranking, which is why it improves on `v0` but stays below `v1` overall.

## Overall metrics

Deltas are percentage points (pp) on a 0–1 scale, with relative change in
parentheses.

| Metric | v0 (baseline) | v1 | v1 Δ vs v0 | v2 | v2 Δ vs v0 | v3 | v3 Δ vs v0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall@5 | 0.5970 | 0.7144 | **+11.74 pp** (+19.7%) | 0.6813 | +8.43 pp (+14.1%) | 0.5970 | 0 pp |
| Recall@10 | 0.6741 | 0.7692 | **+9.51 pp** (+14.1%) | 0.7235 | +4.94 pp (+7.3%) | 0.6741 | 0 pp |
| MRR@10 | 0.6541 | 0.7794 | **+12.53 pp** (+19.2%) | 0.7657 | +11.16 pp (+17.1%) | 0.6541 | 0 pp |
| nDCG@5 | 0.5381 | 0.6627 | **+12.46 pp** (+23.2%) | 0.6442 | +10.61 pp (+19.7%) | 0.5381 | 0 pp |
| nDCG@10 | 0.5701 | 0.6865 | **+11.64 pp** (+20.4%) | 0.6618 | +9.17 pp (+16.1%) | 0.5701 | 0 pp |
| P@5 | 0.2733 | 0.3284 | +5.51 pp (+20.2%) | 0.3107 | +3.74 pp (+13.7%) | 0.2733 | 0 pp |
| Hit rate@10 | 0.8822 | 0.9489 | +6.67 pp (+7.6%) | 0.9156 | +3.34 pp (+3.8%) | 0.8822 | 0 pp |

**Takeaway:** `v1` is the quality winner on every metric. `v2` recovers part of the
gap on ranking metrics (MRR/nDCG) but not on recall, because its base embedding is
still the hash vector.

## Latency and cost

| Metric | v0 | v1 | v2 | v3 |
| --- | ---: | ---: | ---: | ---: |
| Mean search latency | 0.64 ms | 7.21 ms (×11.3) | 1910.13 ms (×2985) | 0.64 ms |
| P95 search latency | 0.91 ms | 8.64 ms (×9.5) | 2284.83 ms (×2513) | 0.91 ms |
| Index build | 0.68 s | 31.89 s* | 0.83 s | 0.68 s |

\* `v1` includes one-time model download/initialization. Steady-state search is
~7 ms. `v2` pays roughly 1.9 s per query at P95 2.3 s, which is the main cost of
cross-encoder reranking with 20 candidates on CPU.

## By question type

### Recall@10

| Question type (cases) | v0 | v1 | v1 Δ | v2 | v2 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic facts (50) | 0.8800 | 0.9300 | +5.00 pp | 0.9200 | +4.00 pp |
| Multi-hop retrieval (100) | 0.6349 | 0.7565 | **+12.16 pp** | 0.6682 | +3.33 pp |
| Version updates (100) | 0.6087 | 0.6833 | +7.46 pp | 0.6448 | +3.61 pp |
| Similar documents (100) | 0.7888 | 0.8451 | +5.63 pp | 0.8420 | +5.32 pp |
| Reasoning & boundaries (100) | 0.5610 | 0.7113 | **+15.03 pp** | 0.6408 | +7.98 pp |

### MRR@10

| Question type | v0 | v1 | v1 Δ | v2 | v2 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic facts | 0.5267 | 0.7017 | **+17.50 pp** | 0.6494 | +12.27 pp |
| Multi-hop retrieval | 0.6840 | 0.8243 | +14.03 pp | 0.7832 | +9.92 pp |
| Version updates | 0.7585 | 0.8400 | +8.15 pp | 0.8343 | +7.58 pp |
| Similar documents | 0.6914 | 0.7664 | +7.50 pp | 0.8223 | **+13.09 pp** |
| Reasoning & boundaries | 0.5459 | 0.7258 | **+17.99 pp** | 0.6812 | +13.53 pp |

### nDCG@10

| Question type | v0 | v1 | v1 Δ | v2 | v2 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic facts | 0.6115 | 0.7527 | +14.12 pp | 0.7136 | +10.21 pp |
| Multi-hop retrieval | 0.5578 | 0.6917 | +13.39 pp | 0.6472 | +8.94 pp |
| Version updates | 0.5864 | 0.6613 | +7.49 pp | 0.6426 | +5.62 pp |
| Similar documents | 0.6423 | 0.7247 | +8.24 pp | 0.7516 | +10.93 pp |
| Reasoning & boundaries | 0.4731 | 0.6350 | **+16.19 pp** | 0.5801 | +10.70 pp |

## By difficulty (Recall@10)

| Difficulty (cases) | v0 | v1 | v1 Δ | v2 | v2 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy (50) | 0.8800 | 0.9300 | +5.00 pp | 0.9200 | +4.00 pp |
| Medium (200) | 0.6987 | 0.7642 | +6.55 pp | 0.7434 | +4.47 pp |
| Hard (200) | 0.5979 | 0.7339 | **+13.60 pp** | 0.6545 | +5.66 pp |

Hard questions gain the most, which is consistent with a stronger semantic
embedding replacing hashed n-grams.

## About v3

`v3` mainly adds **observability**, not retrieval quality:

- per-request trace trees over graph nodes, tools and LLM calls
- JSON structured logs, `/api/traces` endpoints
- optional OTLP export and LangSmith compatibility

Its retrieval path is unchanged from `v0` (local hash embedder, no reranker), so
the metrics are **identical by design**. In fact `agent-v3/result/*` is
byte-identical to `agent-v0/result/*`, which means the report was copied rather
than regenerated. A 0 Δ is therefore expected and does **not** indicate a
regression. If `v3` needs `v1`/`v2` retrieval quality, the corresponding
embedder/reranker code has to be ported forward.

## Reproducing

Each version is self-contained. The evaluator uses code defaults, so no
environment variables are needed:

```bash
cd agent-v0
pip install -r requirements.txt
python scripts/evaluate_retrieval.py
```

```bash
cd agent-v1          # downloads bge-small-zh-v1.5 on first run
pip install -r requirements.txt
python scripts/evaluate_retrieval.py
```

```bash
cd agent-v2          # downloads bge-reranker-base on first run
pip install -r requirements.txt
python scripts/evaluate_retrieval.py
```

```bash
cd agent-v3
pip install -r requirements.txt
python scripts/evaluate_retrieval.py
```

Useful options:

```bash
# quick smoke run
python scripts/evaluate_retrieval.py --max-cases 50

# only one question type
python scripts/evaluate_retrieval.py --types 版本更新

# write somewhere else
python scripts/evaluate_retrieval.py --output-dir /tmp/eval
```

Outputs are written to `result/` as `retrieval_metrics.json`,
`retrieval_metrics.md` and `retrieval_per_case.csv`.

## Caveats

- Metrics are document-level: retrieved chunks are collapsed to their source
  document, and the first occurrence defines the document rank.
- Relevance is binary. `Recall@K` divides by the full gold document count, so
  cases with many gold documents cannot reach 1.0 at small K.
- Latency excludes prompt assembly and LLM generation; it only measures the
  retrieval layer.
- `min_relevance` is applied later in the answer node, so ranking is measured
  unfiltered.
