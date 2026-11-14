"""Export Day 6 Qdrant strategy_B chunks and benchmark.json to Day 7 JSON schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from convert_day6 import convert_question  # noqa: E402


def export_qdrant_chunks(qdrant_dir: Path, collection: str = "strategy_B") -> list[dict]:
    client = QdrantClient(path=str(qdrant_dir))
    chunks: list[dict] = []
    offset = None
    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                p = pt.payload or {}
                meta = {
                    k: v
                    for k, v in p.items()
                    if k not in {"chunk_id", "document_id", "source", "page", "text"}
                }
                chunks.append(
                    {
                        "id": p.get("chunk_id", str(pt.id)),
                        "source_id": p.get("document_id"),
                        "source": p.get("source"),
                        "page": p.get("page"),
                        "text": p.get("text", ""),
                        "metadata": meta,
                    }
                )
            if next_offset is None:
                break
            offset = next_offset
    finally:
        client.close()

    chunks.sort(key=lambda c: (c.get("page") or 0, c["id"]))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Day 6 Qdrant chunks & benchmark to Day 7 schema")
    parser.add_argument("--qdrant-dir", type=Path, default=ROOT / "starter" / "qdrant_db")
    parser.add_argument("--collection", default="strategy_B")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "starter" / "data" / "day6_benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "starter" / "data" / "converted")
    args = parser.parse_args()

    chunks = export_qdrant_chunks(args.qdrant_dir, args.collection)
    raw_questions = json.loads(args.benchmark.read_text(encoding="utf-8"))
    questions = [convert_question(q) for q in raw_questions]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(chunks)} chunks from {args.collection} and {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
