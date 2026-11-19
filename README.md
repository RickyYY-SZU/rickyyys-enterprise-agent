# ricky yys enterprise agent

An enterprise knowledge-base and ticket-handling agent built with FastAPI,
LangGraph and Qdrant. The agent decides whether retrieval is needed, grounds
answers in retrieved documents, and escalates to a trackable support ticket when
it cannot answer.

This repository keeps the full iteration history as four self-contained versions:

| Version | Focus |
| --- | --- |
| `agent-v0` | Baseline MVP: FastAPI + LangGraph + Qdrant, local zero-key RAG, ticket lifecycle |
| `agent-v1` | Adds a local pretrained Chinese embedding model (ONNX) |
| `agent-v2` | Adds a cross-encoder reranker (ONNX) on top of v1 |
| `agent-v3` | Latest: structured tracing/observability, OTLP export, production hardening notes |

Each version is independently runnable with its own `README.md`, Dockerfile and
Docker Compose file. Start with `agent-v3` unless you want to compare the
evolution of the design.

## Quick start

```bash
cd agent-v3
docker compose up -d --build
docker compose ps
```

Then open:

- Console: <http://localhost:8000>
- OpenAPI: <http://localhost:8000/docs>
- Qdrant dashboard: <http://localhost:6333/dashboard>

The default configuration runs fully local without an API key. To use an
OpenAI-compatible chat model, copy `.env.example` to `.env` and set
`LLM_PROVIDER=openai` with your key.

## Model weights

The ONNX model weights used by `agent-v1` and `agent-v2` are **not committed**.
They are downloaded from ModelScope on first use into each version's `models/`
directory. See the version README for details.

## Benchmark data

Retrieval evaluation uses the third-party
[Bank RAG Benchmark](https://github.com/sjtu-suzhou-ai-zhiwei-tech/bank-rag-benchmark)
(CC BY 4.0). The upstream PDFs and test-case workbook are **not redistributed**
here; `data/bank-rag-benchmark/README.md` documents how to fetch them and
`data/BANK_RAG_ATTRIBUTION.md` records attribution.

## Retrieval evaluation

All four versions are benchmarked on the same 450-case setup. Recall, MRR, nDCG,
precision, hit rate and latency per version, with deltas against `agent-v0`, are
in [docs/EVALUATION_COMPARISON.md](docs/EVALUATION_COMPARISON.md).

## Notes

These are MVP implementations intended for local demonstration. They are not
hardened for public exposure; see each README's security section for the
production checklist.
