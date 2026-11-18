"""Retrieval evaluation for the current EnterpriseAgent pipeline.

The evaluation is intentionally frozen: it uses the project default Settings
values and the project own KnowledgeBase.search path, and overrides nothing
except the Qdrant backend (an isolated in-memory instance) so the evaluation
cannot mutate or depend on a live deployment.

Metrics are document-level and binary: a retrieved chunk counts as relevant
when its document code (for example A42) is one of the gold documents.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --max-cases 50 --types \u7248\u672c\u66f4\u65b0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from enterprise_agent.config import Settings  # noqa: E402
from enterprise_agent.embeddings import build_embedder  # noqa: E402
from enterprise_agent.knowledge import KnowledgeBase  # noqa: E402

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "bank_rag_documents.json"
CASES_PATH = PROJECT_ROOT / "data" / "bank_rag_cases.json"
DOC_CODE_RE = re.compile(r"^(A\d+)")
DEFAULT_TOP_K = 10

KEY_ID = "\u6848\u4f8b\u7f16\u53f7"
KEY_TYPE = "\u95ee\u9898\u7c7b\u578b"
KEY_QUESTION = "\u6d4b\u8bd5\u95ee\u9898"
KEY_DIFFICULTY = "\u96be\u5ea6"
KEY_GOLD = "\u5e94\u547d\u4e2d\u6587\u6863"


def parse_doc_codes(raw: str) -> list[str]:
    codes: list[str] = []
    for line in str(raw).replace("\r", "\n").split("\n"):
        match = DOC_CODE_RE.match(line.strip())
        if match and match.group(1) not in codes:
            codes.append(match.group(1))
    return codes


def chunk_doc_code(chunk: Any) -> str | None:
    match = DOC_CODE_RE.match(str(getattr(chunk, "title", "")).strip())
    return match.group(1) if match else None


def ranked_doc_codes(chunks: list[Any]) -> list[str]:
    """Collapse a chunk ranking into a first-occurrence document ranking."""
    seen: list[str] = []
    for chunk in chunks:
        code = chunk_doc_code(chunk)
        if code and code not in seen:
            seen.append(code)
    return seen


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def precision_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / k


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    for index, code in enumerate(ranked[:k], start=1):
        if code in gold_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, code in enumerate(ranked[:k], start=1)
        if code in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_dataset(types: list[str] | None, max_cases: int | None) -> tuple[list[dict], list[dict]]:
    documents = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if types:
        wanted = set(types)
        cases = [case for case in cases if case.get(KEY_TYPE) in wanted]
    if max_cases:
        cases = cases[:max_cases]
    return documents, cases


def build_knowledge(settings: Settings) -> KnowledgeBase:
    client = QdrantClient(location=":memory:")
    knowledge = KnowledgeBase(settings, build_embedder(settings), client=client)
    knowledge.ensure_collection()
    return knowledge


def index_documents(knowledge: KnowledgeBase, documents: list[dict]) -> dict[str, Any]:
    started = time.perf_counter()
    chunk_counts: list[int] = []
    for document in documents:
        result = knowledge.ingest(
            title=document["title"],
            content=document["content"],
            source=document["source"],
            tags=document.get("tags", []),
        )
        chunk_counts.append(int(result["chunks"]))
    elapsed = time.perf_counter() - started
    return {
        "documents": len(documents),
        "chunks": knowledge.count(),
        "avg_chunks_per_document": round(statistics.fmean(chunk_counts), 2)
        if chunk_counts
        else 0.0,
        "max_chunks_per_document": max(chunk_counts) if chunk_counts else 0,
        "min_chunks_per_document": min(chunk_counts) if chunk_counts else 0,
        "index_build_seconds": round(elapsed, 3),
    }


def evaluate_case(knowledge: KnowledgeBase, case: dict, top_k: int) -> dict[str, Any]:
    query = str(case[KEY_QUESTION]).strip()
    gold = parse_doc_codes(case[KEY_GOLD])
    started = time.perf_counter()
    chunks = knowledge.search(query, limit=top_k)
    latency_ms = (time.perf_counter() - started) * 1000.0
    ranked = ranked_doc_codes(chunks)
    gold_set = set(gold)
    ranks = {code: index for index, code in enumerate(ranked, start=1) if code in gold_set}
    return {
        "case_id": case[KEY_ID],
        "type": case[KEY_TYPE],
        "difficulty": case[KEY_DIFFICULTY],
        "question": query,
        "gold_documents": gold,
        "gold_count": len(gold),
        "retrieved_documents": ranked,
        "retrieved_chunks": len(chunks),
        "first_relevant_rank": min(ranks.values()) if ranks else None,
        "recall_at_5": recall_at_k(ranked, gold, 5),
        "recall_at_10": recall_at_k(ranked, gold, 10),
        "mrr_at_10": mrr_at_k(ranked, gold, 10),
        "ndcg_at_5": ndcg_at_k(ranked, gold, 5),
        "ndcg_at_10": ndcg_at_k(ranked, gold, 10),
        "precision_at_5": precision_at_k(ranked, gold, 5),
        "latency_ms": latency_ms,
    }


METRIC_KEYS = (
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "precision_at_5",
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in METRIC_KEYS:
        summary[key] = round(statistics.fmean(row[key] for row in rows), 4)
    latencies = [row["latency_ms"] for row in rows]
    summary["p95_ms"] = round(percentile(latencies, 95), 3)
    summary["mean_ms"] = round(statistics.fmean(latencies), 3)
    summary["median_ms"] = round(statistics.median(latencies), 3)
    summary["p99_ms"] = round(percentile(latencies, 99), 3)
    summary["cases"] = len(rows)
    summary["hit_rate_at_10"] = round(
        sum(1 for row in rows if row["first_relevant_rank"] is not None) / len(rows), 4
    )
    return summary


def group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {name: summarize(items) for name, items in sorted(grouped.items())}


def settings_snapshot(settings: Settings) -> dict[str, Any]:
    snapshot = asdict(settings)
    snapshot["data_dir"] = str(settings.data_dir)
    snapshot["seed_documents_path"] = str(settings.seed_documents_path)
    return snapshot


def locked_settings() -> Settings:
    """Project defaults, unmodified, with secrets cleared for reproducibility."""
    settings = Settings()
    if settings.qdrant_api_key or settings.llm_api_key:
        raise RuntimeError("Refusing to run: frozen settings must not carry credentials")
    return settings


def write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["overall"]
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Dataset: {report['dataset']['name']}",
        f"- Cases evaluated: {overall['cases']}",
        f"- Documents indexed: {report['index']['documents']} ({report['index']['chunks']} chunks)",
        f"- Top-K retrieved per query: {report['top_k']}",
        f"- Embedding provider: {report['settings']['embedding_provider']} "
        f"(dim={report['settings']['embedding_dimension']})",
        "- Chunking: project defaults (chunk_size=520, overlap=80)",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Recall@5 | {overall['recall_at_5']:.4f} |",
        f"| Recall@10 | {overall['recall_at_10']:.4f} |",
        f"| MRR@10 | {overall['mrr_at_10']:.4f} |",
        f"| nDCG@5 | {overall['ndcg_at_5']:.4f} |",
        f"| nDCG@10 | {overall['ndcg_at_10']:.4f} |",
        f"| P@5 | {overall['precision_at_5']:.4f} |",
        f"| P95 latency | {overall['p95_ms']:.3f} ms |",
        f"| Mean latency | {overall['mean_ms']:.3f} ms |",
        f"| Hit rate@10 | {overall['hit_rate_at_10']:.4f} |",
        "",
    ]
    for title, key in (("By question type", "by_type"), ("By difficulty", "by_difficulty")):
        lines += [
            f"## {title}",
            "",
            "| Group | Cases | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | P95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, metrics in report[key].items():
            lines.append(
                f"| {name} | {metrics['cases']} | {metrics['recall_at_5']:.4f} "
                f"| {metrics['recall_at_10']:.4f} | {metrics['mrr_at_10']:.4f} "
                f"| {metrics['ndcg_at_5']:.4f} | {metrics['ndcg_at_10']:.4f} "
                f"| {metrics['precision_at_5']:.4f} | {metrics['p95_ms']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- Metrics are document-level: chunks are collapsed to their source document, "
        "and the first occurrence defines the document rank.",
        "- Relevance is binary. Recall@K divides by the full gold document count, so cases "
        "with many gold documents cannot reach 1.0 at small K.",
        "- P@5 divides by 5 in the usual way.",
        "- Latency covers only KnowledgeBase.search (query embedding plus vector search); "
        "prompt assembly and generation are excluded.",
        "- min_relevance is applied later in the answer node, so ranking is measured "
        "unfiltered, matching the retrieval layer as it is today.",
        "- Settings are the project defaults. The environment is not read.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id",
        "type",
        "difficulty",
        "gold_count",
        "first_relevant_rank",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "precision_at_5",
        "latency_ms",
        "gold_documents",
        "retrieved_documents",
        "question",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["gold_documents"] = " ".join(row["gold_documents"])
            record["retrieved_documents"] = " ".join(row["retrieved_documents"])
            writer.writerow({key: record[key] for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate current retrieval quality")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--types", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result")
    args = parser.parse_args()

    settings = locked_settings()
    documents, cases = load_dataset(args.types, args.max_cases)
    if not cases:
        raise SystemExit("No evaluation cases matched the filters")

    knowledge = build_knowledge(settings)
    index_stats = index_documents(knowledge, documents)

    for case in cases[: min(5, len(cases))]:
        knowledge.search(str(case[KEY_QUESTION]), limit=args.top_k)

    rows = [evaluate_case(knowledge, case, args.top_k) for case in cases]
    knowledge.client.close()

    report = {
        "dataset": {
            "name": json.loads(CASES_PATH.read_text(encoding="utf-8"))["dataset"]["name"],
            "cases_path": str(CASES_PATH.relative_to(PROJECT_ROOT)),
            "documents_path": str(DOCUMENTS_PATH.relative_to(PROJECT_ROOT)),
        },
        "top_k": args.top_k,
        "settings": settings_snapshot(settings),
        "index": index_stats,
        "gold_count_distribution": dict(sorted(Counter(row["gold_count"] for row in rows).items())),
        "overall": summarize(rows),
        "by_type": group_summary(rows, "type"),
        "by_difficulty": group_summary(rows, "difficulty"),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "retrieval_metrics.md")
    write_csv(rows, output_dir / "retrieval_per_case.csv")

    overall = report["overall"]
    print(
        f"cases={overall['cases']} docs={index_stats['documents']} chunks={index_stats['chunks']}"
    )
    print(
        "Recall@5={recall_at_5:.4f} Recall@10={recall_at_10:.4f} MRR@10={mrr_at_10:.4f} "
        "nDCG@5={ndcg_at_5:.4f} nDCG@10={ndcg_at_10:.4f} P@5={precision_at_5:.4f} "
        "P95={p95_ms:.3f}ms".format(**overall)
    )
    print(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
