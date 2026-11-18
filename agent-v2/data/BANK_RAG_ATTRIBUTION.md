# Bank RAG Benchmark Attribution

This project includes data derived from [Bank RAG Benchmark](https://github.com/sjtu-suzhou-ai-zhiwei-tech/bank-rag-benchmark), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Source commit: `228f44edf12daf51b84015fa30210fd842fa167c`
- Original data: 100 PDF documents and 450 evaluation cases.
- Local modifications: PDF text extraction, metadata normalization, and conversion to JSON for ingestion and retrieval evaluation.
- The upstream README states that the benchmark is for testing only, does not constitute real banking policy, and must not be used for production business decisions.

## What is and is not redistributed

The upstream raw data (100 PDF documents and the `bank-rag-benchmark-test-cases.xlsx`
workbook) is **not** redistributed in this repository. Fetch it from the upstream
link above; see `data/bank-rag-benchmark/README.md` for the exact commands.

The complete upstream license is retained at `data/bank-rag-benchmark/LICENSE`.
The normalized JSON derived from the upstream data (`data/bank_rag_documents.json`
and `data/bank_rag_cases.json`) is kept so retrieval evaluation runs out of the box.
