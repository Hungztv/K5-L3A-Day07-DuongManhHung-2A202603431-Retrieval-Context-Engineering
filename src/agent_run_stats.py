"""Aggregate agent-run metrics from traces (no grading here; grading lives in reviews.json).

Per run it reports:
- tool use, distribution of search calls, LLM calls
- Agent Context Evidence Rate: gold evidence present in the FINAL context sent to the model
  (context of the last search event), over answerable questions
- tokens, USD cost (gpt-4o-mini on OpenRouter: $0.15/M input, $0.60/M output; checked 2026-11-14)
- wall time
Gold is only read here, after the run, for scoring.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000


def norm(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


def final_context(trace: dict) -> tuple[str, list[str]]:
    searches = [e for e in trace.get("events", []) if e.get("event") == "search"]
    if not searches:
        return "", []
    return searches[-1].get("context", ""), searches[-1].get("context_ids", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    qs = {q["id"]: q for q in json.loads((ROOT / "starter/data/converted/questions_day7.json").read_text(encoding="utf-8"))}

    all_rows = []
    for run in args.runs:
        traces = json.loads((Path(run) / "traces.json").read_text(encoding="utf-8"))
        ok = [t for t in traces if t.get("status") != "error"]
        answerable = [t for t in ok if qs[t["query_id"]].get("answerable", True)]
        ctx_hits, per_q = 0, {}
        for t in answerable:
            ctx, ids = final_context(t)
            nctx = norm(ctx)
            hit = any(norm(g["evidence"]).rstrip(".") in nctx for g in qs[t["query_id"]]["gold"])
            ctx_hits += hit
            per_q[t["query_id"]] = {"context_has_gold": hit, "context_ids": ids, "context_chars": len(ctx)}
        n_search = [sum(e.get("event") == "search" for e in t.get("events", [])) for t in ok]
        p_tok = sum(u.get("prompt_tokens", 0) for t in traces for u in t.get("usage", []))
        c_tok = sum(u.get("completion_tokens", 0) for t in traces for u in t.get("usage", []))
        llm_calls = sum(len(t.get("usage", [])) for t in traces)
        row = {
            "run": run,
            "model": traces[0].get("model") if traces else None,
            "cases": len(traces),
            "errors": len(traces) - len(ok),
            "tool_use": f"{sum(1 for s in n_search if s > 0)}/{len(ok)}",
            "search_calls_distribution": dict(sorted(Counter(n_search).items())),
            "llm_calls": llm_calls,
            "agent_context_evidence": f"{ctx_hits}/{len(answerable)} ({ctx_hits / max(len(answerable), 1):.1%})",
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "cost_usd": round(p_tok * PRICE_IN + c_tok * PRICE_OUT, 5),
            "wall_time_s": round(sum(t.get("elapsed_s", 0) for t in traces), 2),
            "avg_time_s": round(sum(t.get("elapsed_s", 0) for t in traces) / max(len(traces), 1), 2),
            "per_question": per_q,
        }
        all_rows.append(row)
        print(f"\n{run}")
        for k, v in row.items():
            if k not in ("run", "per_question"):
                print(f"  {k}: {v}")
        missing = [q for q, v in per_q.items() if not v["context_has_gold"]]
        print(f"  answerable without gold in final context: {missing}")
    if args.json_out:
        args.json_out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
