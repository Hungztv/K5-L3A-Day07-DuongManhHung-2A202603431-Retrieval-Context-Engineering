"""Dump a human-readable review sheet per run: question, expected, gold, every search,
full final context, raw model decisions, final answer and full text of each cited chunk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    chunks = {c["id"]: c for c in json.loads((ROOT / "starter/data/converted/chunks.json").read_text(encoding="utf-8"))}
    qs = {q["id"]: q for q in json.loads((ROOT / "starter/data/converted/questions_day7.json").read_text(encoding="utf-8"))}
    traces = json.loads((Path(args.run) / "traces.json").read_text(encoding="utf-8"))
    out = []
    for t in traces:
        if args.ids and t["query_id"] not in args.ids:
            continue
        q = qs[t["query_id"]]
        out.append(f"\n################ {t['query_id']} | status={t['status']} | answerable={q.get('answerable', True)}")
        out.append(f"Q: {q['question']}")
        if q.get("answerable", True):
            out.append(f"EXPECTED: {q['expected_answer']}")
            for g in q["gold"]:
                out.append(f"GOLD p.{g['page']}: {g['evidence']}")
        for i, e in enumerate(t.get("events", []), 1):
            if e.get("event") == "search":
                out.append(f"  SEARCH {i}: {e['query']!r} -> {[(r['id'].split('-3e-')[-1], round(r['score'], 4)) for r in e['results']]}")
                out.append(f"    context_ids: {[c.split('-3e-')[-1] for c in e['context_ids']]}")
            else:
                out.append(f"  EVENT {i}: {e.get('event')}")
        for d in t.get("model_decisions", []):
            if d.get("action") == "answer":
                out.append(f"  RAW MODEL ANSWER: {d['answer']}")
        out.append(f"FINAL ANSWER: {t['answer']}")
        for cid in t.get("citations", []):
            c = chunks.get(cid)
            out.append(f"  CITED {cid} (p.{c['page'] if c else '?'}):\n    {' '.join(c['text'].split()) if c else 'NOT FOUND'}")
    text = "\n".join(out)
    dest = Path(args.run) / "review_sheet.txt"
    dest.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
