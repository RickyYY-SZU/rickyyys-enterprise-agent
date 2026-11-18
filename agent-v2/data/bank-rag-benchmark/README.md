# Bank RAG Benchmark (upstream reference)

This folder intentionally contains **no benchmark data**. The original dataset is
sourced from the upstream repository and is **not redistributed** in this project.

- Upstream: <https://github.com/sjtu-suzhou-ai-zhiwei-tech/bank-rag-benchmark>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Upstream commit used locally: `228f44edf12daf51b84015fa30210fd842fa167c`

## What was here

The upstream repository ships 100 PDF documents (`documents/A01`-`A100`) and an
`bank-rag-benchmark-test-cases.xlsx` workbook with 450 evaluation cases. Both were
removed from this public repository to avoid redistributing third-party data.

## How to restore it locally

```bash
git clone https://github.com/sjtu-suzhou-ai-zhiwei-tech/bank-rag-benchmark
cp -r bank-rag-benchmark/documents .
cp bank-rag-benchmark/bank-rag-benchmark-test-cases.xlsx .
```

The derived JSON files used by this project (`data/bank_rag_documents.json` and
`data/bank_rag_cases.json`) are still included so retrieval evaluation works out of
the box.

## Attribution

See [`data/BANK_RAG_ATTRIBUTION.md`](../BANK_RAG_ATTRIBUTION.md). The upstream also
states the benchmark is for testing only, does not constitute real banking policy,
and must not be used for production business decisions.
