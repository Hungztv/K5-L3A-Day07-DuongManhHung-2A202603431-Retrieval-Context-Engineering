"""Evaluate retrieval methods and context-packing configs on Day 7 benchmark."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from baseline import pack_context  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    RerankRetriever,
    TfidfRetriever,
)


def norm_text(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


def is_hit(chunk: dict[str, Any], gold_list: list[dict[str, Any]]) -> bool:
    ctext = norm_text(chunk.get("text", ""))
    for g in gold_list:
        ev = norm_text(g["evidence"]).rstrip(".")
        if (
            chunk.get("source_id") == g.get("source_id")
            and chunk.get("page") == g.get("page")
            and ev in ctext
        ):
            return True
    return False


def context_has_evidence(context: str, gold_list: list[dict[str, Any]]) -> bool:
    ctx_norm = norm_text(context)
    for g in gold_list:
        ev = norm_text(g["evidence"]).rstrip(".")
        if ev in ctx_norm:
            return True
    return False


def evaluate_config(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    split: str,
    top_k: int = 3,
    budget: int = 800,
    out_dir: Path,
) -> dict[str, Any]:
    answerable = [q for q in questions if q["split"] == split and q.get("answerable", True)]
    traces: list[dict[str, Any]] = []
    hit1 = 0
    hit3 = 0
    ctx_hit = 0
    mrr_sum = 0.0
    total_ms = 0.0

    for q in answerable:
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=max(top_k, 3))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms

        top3 = results[:3]
        packed_candidates = results[:top_k]
        context, ctx_ids = pack_context(packed_candidates, budget=budget)

        ranks = [i + 1 for i, r in enumerate(top3) if is_hit(r, q["gold"])]
        h1 = 1 in ranks
        h3 = bool(ranks)
        ch = context_has_evidence(context, q["gold"])

        hit1 += int(h1)
        hit3 += int(h3)
        ctx_hit += int(ch)
        if ranks:
            mrr_sum += 1.0 / ranks[0]

        traces.append(
            {
                "id": q["id"],
                "question": q["question"],
                "latency_ms": round(elapsed_ms, 2),
                "hit_at_1": h1,
                "evidence_hit_at_3": h3,
                "context_evidence": ch,
                "first_hit_rank": ranks[0] if ranks else None,
                "top3": [
                    {
                        "id": r["id"],
                        "page": r["page"],
                        "score": r["score"],
                        "is_hit": is_hit(r, q["gold"]),
                    }
                    for r in top3
                ],
                "context_ids": ctx_ids,
                "context_chars": len(context),
            }
        )

    n = len(answerable)
    summary = {
        "config": name,
        "split": split,
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "hit_at_1": f"{hit1}/{n} ({hit1 / n:.1%})" if n else "0/0",
        "hit_at_1_rate": round(hit1 / n, 4) if n else 0.0,
        "evidence_hit_at_3": f"{hit3}/{n} ({hit3 / n:.1%})" if n else "0/0",
        "evidence_hit_at_3_rate": round(hit3 / n, 4) if n else 0.0,
        "context_evidence_rate": f"{ctx_hit}/{n} ({ctx_hit / n:.1%})" if n else "0/0",
        "context_evidence_rate_num": round(ctx_hit / n, 4) if n else 0.0,
        "mrr_at_3": round(mrr_sum / n, 4) if n else 0.0,
        "total_time_s": round(total_ms / 1000.0, 3),
        "avg_latency_ms": round(total_ms / n, 2) if n else 0.0,
    }

    cfg_dir = out_dir / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cfg_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--with-rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "starter" / "results" / "retrieval")
    args = parser.parse_args()

    chunks = json.loads((ROOT / "starter" / "data" / "converted" / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "starter" / "data" / "converted" / "questions_day7.json").read_text(encoding="utf-8"))

    dense = DenseRetriever()
    try:
        tfidf = TfidfRetriever(chunks)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRRFRetriever(dense, bm25, fetch_k=20)

        configs: list[tuple[str, Any, int, int]] = [
            ("dense_day6_k3_b800", dense, 3, 800),
            ("tfidf_k3_b800", tfidf, 3, 800),
            ("bm25_k3_b800", bm25, 3, 800),
            ("hybrid_rrf_k3_b800", hybrid, 3, 800),
            # Improvement 1: increase context budget from 800 -> 2400 chars (since strategy_B chunks are up to 800 chars!)
            ("dense_day6_k3_b2400", dense, 3, 2400),
            ("hybrid_rrf_k3_b2400", hybrid, 3, 2400),
            ("hybrid_rrf_k5_b3200", hybrid, 5, 3200),
        ]

        if args.with_rerank:
            rerank_hybrid = RerankRetriever(hybrid, fetch_k=15)
            configs.append(("hybrid_rerank_k3_b2400", rerank_hybrid, 3, 2400))

        results = []
        for name, ret, top_k, budget in configs:
            s = evaluate_config(
                name, ret, questions, split=args.split, top_k=top_k, budget=budget, out_dir=args.output
            )
            results.append(s)
            print(
                f"{name:26s} | Hit@1={s['hit_at_1']:14s} | EvidHit@3={s['evidence_hit_at_3']:14s} "
                f"| CtxEvid={s['context_evidence_rate']:14s} | MRR@3={s['mrr_at_3']:.3f} | {s['avg_latency_ms']} ms/q"
            )
    finally:
        dense.close()


if __name__ == "__main__":
    main()
