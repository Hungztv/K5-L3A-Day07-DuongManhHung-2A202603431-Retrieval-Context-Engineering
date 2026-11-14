"""Build Day 7 benchmark (21 Day 6 answerable + 2 dev unanswerable + 2 test unanswerable)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def norm_text(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


UNANSWERABLE_QUESTIONS = [
    {
        "id": "q22",
        "split": "dev",
        "question": "What are the main changes in Vietnam's 2024 Law on Social Insurance regarding voluntary pension contributions?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 2,726 chunks (src/verify_absence.py): 'Vietnam' appears in 16 chunks, 'pension' in 1 chunk (p.383), 'social insurance' in 0 chunks, '2024' in 0 chunks, and co-occurrence of (Vietnam+social insurance / Vietnam+pension / Vietnam+2024) is 0.",
    },
    {
        "id": "q23",
        "split": "dev",
        "question": "What was the exact global Gini coefficient reported by the World Bank in 2025?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 2,726 chunks (src/verify_absence.py): 'Gini' appears in 4 chunks, 'World Bank' in 27 chunks, '2025' in 1 chunk (p.383), and co-occurrence of (Gini+2025 / World Bank+2025) is 0.",
    },
    {
        "id": "q24",
        "split": "test",
        "question": "How did the European Union's 2024 AI Act classify workplace emotion-recognition systems?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 2,726 chunks (src/verify_absence.py): 'AI Act' (0 chunks), 'emotion recognition' (0 chunks), '2024' (0 chunks), and co-occurrence of (European Union+artificial intelligence) is 0.",
    },
    {
        "id": "q25",
        "split": "test",
        "question": "What percentage of remote workers in Singapore reported burnout in the 2025 Ministry of Manpower survey?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 2,726 chunks (src/verify_absence.py): 'burnout' (0 chunks), 'remote work' (0 chunks), 'Ministry of Manpower' (0 chunks), and co-occurrence of (Singapore+burnout / Singapore+2025) is 0.",
    },
]


def main() -> None:
    chunks_path = ROOT / "starter" / "data" / "converted" / "chunks.json"
    questions_path = ROOT / "starter" / "data" / "converted" / "questions.json"
    out_path = ROOT / "starter" / "data" / "converted" / "questions_day7.json"

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    # Verify all 21 answerable questions resolve against chunks
    by_page: dict[tuple[str, int], list[dict]] = {}
    for c in chunks:
        by_page.setdefault((c["source_id"], c["page"]), []).append(c)

    for q in questions:
        q["answerable"] = True
        # Fix q14 trailing period before citation '(Smith 1991)' in source PDF
        if q["id"] == "q14":
            q["gold"][0]["evidence"] = "The ying-yang symbol and the concept of polar forces are central Taoist ideas"
        matched = False
        for g in q["gold"]:
            ev = norm_text(g["evidence"]).rstrip(".")
            for c in by_page.get((g["source_id"], g["page"]), []):
                if ev in norm_text(c["text"]):
                    matched = True
                    break
        if not matched:
            raise ValueError(f"Question {q['id']} gold evidence did not resolve in chunks.json")

    all_questions = questions + UNANSWERABLE_QUESTIONS
    out_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dev_ans = sum(1 for q in all_questions if q["split"] == "dev" and q.get("answerable", True))
    dev_unans = sum(1 for q in all_questions if q["split"] == "dev" and not q.get("answerable", True))
    test_ans = sum(1 for q in all_questions if q["split"] == "test" and q.get("answerable", True))
    test_unans = sum(1 for q in all_questions if q["split"] == "test" and not q.get("answerable", True))
    print(f"Saved {len(all_questions)} questions to {out_path}")
    print(f"  dev : {dev_ans} answerable + {dev_unans} unanswerable = {dev_ans + dev_unans}")
    print(f"  test: {test_ans} answerable + {test_unans} unanswerable = {test_ans + test_unans}")


if __name__ == "__main__":
    main()
