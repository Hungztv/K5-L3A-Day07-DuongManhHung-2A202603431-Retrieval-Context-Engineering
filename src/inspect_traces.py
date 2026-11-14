"""Inspect agent traces against benchmark questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dirs", nargs="+")
    parser.add_argument("--detail", nargs="*", default=[])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    chunks = {
        c["id"]: c
        for c in json.loads(
            (ROOT / "starter" / "data" / "converted" / "chunks.json").read_text(encoding="utf-8")
        )
    }

    qs = {
        q["id"]: q
        for q in json.loads(
            (ROOT / "starter" / "data" / "converted" / "questions_day7.json").read_text(encoding="utf-8")
        )
    }

    for d in args.trace_dirs:
        p = Path(d) / "traces.json"
        traces = json.loads(p.read_text(encoding="utf-8"))
        total_tok = sum(t.get("total_tokens", 0) for t in traces)
        total_s = sum(t.get("elapsed_s", 0) for t in traces)
        model = traces[0].get("model", "") if traces else ""
        print(f"\n=== {d} (model={model}, total_tokens={total_tok}, total_time={total_s:.2f}s) ===")
        for t in traces:
            qid = t["query_id"]
            q = qs[qid]
            searches = [e["query"] for e in t["events"] if e["event"] == "search"]
            ans_short = t["answer"].replace("\n", " ")[:110]
            print(
                f"  {qid} (ans={q.get('answerable', True)}): status={t['status']} "
                f"searches={len(searches)} cites={t['citations']} | {ans_short}"
            )
            if args.verify:
                print(f"    Q: {q['question']}")
                if q.get("answerable", True):
                    print(f"    Expected: {q.get('expected_answer')}")
                    print(f"    Gold: page={q['gold'][0]['page']} | {q['gold'][0]['evidence'][:110]}")
                print(f"    Answer: {t['answer']}")
                for cid in t["citations"]:
                    c = chunks.get(cid, {})
                    snip = c.get("text", "").replace("\n", " ")[:220]
                    print(f"    Cited {cid} (p.{c.get('page')}): {snip}")
            if qid in args.detail:
                print(f"    Question: {q['question']}")
                if q.get("answerable", True):
                    print(f"    Expected: {q.get('expected_answer')}")
                    print(f"    Gold: {q.get('gold')}")
                for idx, ev in enumerate(t["events"], 1):
                    if ev["event"] == "search":
                        top_ids = [(r["id"], r["page"], round(r["score"], 4)) for r in ev["results"]]
                        print(f"    Search#{idx} q={ev['query']!r} -> {top_ids}")
                        print(f"      ctx_ids={ev['context_ids']}")
                    else:
                        print(f"    Event#{idx}: {ev['event']}")
                print(f"    Full Answer: {t['answer']}")


if __name__ == "__main__":
    main()
