"""Verify unanswerable questions by keyword search over the FULL corpus (not just top-k).

For each unanswerable question we list the key entities that an answer would require
and count chunks containing them (case-insensitive, whole-word). The question is kept
as 'unanswerable' only if no chunk contains the required co-occurring entities.
Output: starter/results/absence_check.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each check: question id -> (single keywords to report, required co-occurrence groups)
CHECKS = {
    "q22": {
        "keywords": ["Vietnam", "Vietnamese", "social insurance", "pension", "2024"],
        "required_together": [["Vietnam", "social insurance"], ["Vietnam", "pension"], ["Vietnam", "2024"]],
    },
    "q23": {
        "keywords": ["Gini", "World Bank", "2025"],
        "required_together": [["Gini", "2025"], ["World Bank", "2025"]],
    },
    "q24": {
        "keywords": ["AI Act", "artificial intelligence", "emotion recognition", "European Union", "2024"],
        "required_together": [["AI Act"], ["emotion recognition"], ["European Union", "artificial intelligence"]],
    },
    "q25": {
        "keywords": ["Singapore", "burnout", "remote work", "Ministry of Manpower", "2025"],
        "required_together": [["Singapore", "burnout"], ["Ministry of Manpower"], ["Singapore", "2025"]],
    },
}


def has(text: str, kw: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", text) is not None


def main() -> None:
    chunks = json.loads((ROOT / "starter/data/converted/chunks.json").read_text(encoding="utf-8"))
    lowered = [(c["id"], c["page"], re.sub(r"\s+", " ", c["text"]).lower()) for c in chunks]
    report = {}
    for qid, spec in CHECKS.items():
        kw_counts = {}
        for kw in spec["keywords"]:
            pages = sorted({p for _, p, t in lowered if has(t, kw)})
            kw_counts[kw] = {"n_chunks": sum(1 for _, _, t in lowered if has(t, kw)), "pages_sample": pages[:10]}
        together = {}
        for group in spec["required_together"]:
            hits = [(cid, p) for cid, p, t in lowered if all(has(t, k) for k in group)]
            together[" + ".join(group)] = hits
        verdict = all(len(v) == 0 for v in together.values())
        report[qid] = {"keywords": kw_counts, "co_occurrence_hits": together, "absent_verified": verdict}
        print(f"{qid}: absent_verified={verdict}")
        for k, v in kw_counts.items():
            print(f"   '{k}': {v['n_chunks']} chunks, pages {v['pages_sample']}")
        for k, v in together.items():
            print(f"   [{k}] co-occur: {v[:5]}")
    out = ROOT / "starter/results/absence_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
