"""Real OpenRouter agent controller + runner connecting to Day 6 Qdrant / Hybrid retriever."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from agentic import run_agent  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from retrievers import BM25Retriever, DenseRetriever, HybridRRFRetriever  # noqa: E402

SYSTEM_PROMPT = """You are a careful sociology research assistant answering questions strictly from the provided textbook context (OpenStax Introduction to Sociology 3e).

Rules:
1. Always call `search_documents` before answering if you do not yet have sufficient evidence in the context. You may search multiple times with refined queries.
2. Every factual claim in your final answer MUST be grounded in the retrieved context and cite the exact chunk ID in square brackets, e.g. [openstax-introduction-sociology-3e-p0031-paragraph-0000].
3. Only cite chunk IDs that actually appear in the latest retrieved context.
4. If the retrieved context does not contain enough evidence to answer the question (for example, the topic is outside the textbook, or lacks the requested year/country/statistic), DO NOT guess or use outside knowledge. Respond clearly that there is insufficient evidence in the provided documents and do NOT invent citations.
5. Keep your final answer concise and directly supported by the cited chunks."""

SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the sociology textbook chunks for relevant evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query with key terms or concepts.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class OpenRouterController:
    """Controller calling a real OpenRouter model with tool calling."""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Set it in environment or .env before running the real agent."
            )
        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.usage_log: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []

    def reset_usage(self) -> None:
        self.usage_log = []
        self.decision_log = []

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        if history:
            latest = history[-1]
            prev_queries = ", ".join(latest.get("queries", []))
            ctx = latest.get("context", "")
            ctx_ids = latest.get("context_ids", [])
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Previous search queries: {prev_queries}\n"
                        f"Available chunk IDs in context: {ctx_ids}\n\n"
                        f"Retrieved Context:\n{ctx}\n\n"
                        "If this context is sufficient, answer now with [chunk-id] citations. "
                        "If you need more specific evidence, call `search_documents` with a better query. "
                        "If the textbook clearly does not cover this question, state that there is insufficient evidence."
                    ),
                }
            )

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=SEARCH_TOOL,
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = getattr(resp, "usage", None)
        self.usage_log.append(
            {
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = json.loads(tc.function.arguments or "{}")
            query = str(args.get("query", "")).strip() or question
            action = {"action": "search", "query": query}
            self.decision_log.append({**action, "n_tool_calls_returned": len(tool_calls)})
            return action

        answer = (msg.content or "").strip()
        # Extract [chunk-id] citations; preserve order & deduplicate
        raw_cites = re.findall(r"\[([A-Za-z0-9_.:-]+)\]", answer)
        citations = list(dict.fromkeys(raw_cites))
        action = {"action": "answer", "answer": answer, "citations": citations}
        # Keep the raw model answer: run_agent replaces invalid answers with a fixed
        # abstention string, so without this log a hallucinated answer would be hidden.
        self.decision_log.append(dict(action))
        return action


def build_retriever(method: str, chunks: list[dict[str, Any]]) -> tuple[Any, DenseRetriever | None]:
    if method == "dense":
        dense = DenseRetriever()
        return dense, dense
    if method == "bm25":
        return BM25Retriever(chunks), None
    if method == "hybrid":
        dense = DenseRetriever()
        bm25 = BM25Retriever(chunks)
        return HybridRRFRetriever(dense, bm25, fetch_k=20), dense
    raise ValueError(f"Unknown retriever method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenRouter agent on Day 7 benchmark split")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--method", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of questions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads((ROOT / "starter" / "data" / "converted" / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "starter" / "data" / "converted" / "questions_day7.json").read_text(encoding="utf-8"))
    selected = [q for q in questions if q["split"] == args.split]
    if args.limit > 0:
        selected = selected[: args.limit]

    controller = OpenRouterController(model=args.model)
    retriever, dense_to_close = build_retriever(args.method, chunks)

    traces: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    try:
        for q in selected:
            cases.append({"id": q["id"], "split": q["split"], "answerable": q.get("answerable", True)})
            controller.reset_usage()
            t0 = time.perf_counter()
            try:
                res = run_agent(
                    q["question"],
                    controller,
                    retriever,
                    filters=q.get("filter"),
                    max_searches=args.max_searches,
                    top_k=args.top_k,
                    budget=args.budget,
                )
                elapsed_s = round(time.perf_counter() - t0, 3)
                traces.append(
                    {
                        "query_id": q["id"],
                        "question": q["question"],
                        "answerable": q.get("answerable", True),
                        "model": controller.model,
                        "method": args.method,
                        "elapsed_s": elapsed_s,
                        "usage": controller.usage_log,
                        "total_tokens": sum(u["total_tokens"] for u in controller.usage_log),
                        "prompt_tokens": sum(u["prompt_tokens"] for u in controller.usage_log),
                        "completion_tokens": sum(u["completion_tokens"] for u in controller.usage_log),
                        "llm_calls": len(controller.usage_log),
                        "model_decisions": controller.decision_log,
                        "config": {"top_k": args.top_k, "budget": args.budget, "max_searches": args.max_searches},
                        **res,
                    }
                )
                print(
                    f"[{q['id']}] status={res['status']} searches={sum(e['event']=='search' for e in res['events'])} "
                    f"cites={len(res['citations'])} time={elapsed_s}s"
                )
            except Exception as exc:
                elapsed_s = round(time.perf_counter() - t0, 3)
                traces.append(
                    {
                        "query_id": q["id"],
                        "question": q["question"],
                        "answerable": q.get("answerable", True),
                        "model": controller.model,
                        "method": args.method,
                        "elapsed_s": elapsed_s,
                        "status": "error",
                        "error": str(exc),
                        "events": [],
                        "citations": [],
                    }
                )
                print(f"[{q['id']}] ERROR: {exc}")
    finally:
        if dense_to_close is not None:
            dense_to_close.close()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()
