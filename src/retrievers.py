"""Retrieval implementations for Day 7: Dense (Day 6 Qdrant), TF-IDF, BM25, Hybrid RRF, and Rerank."""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from baseline import TfidfIndex  # noqa: E402


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class DenseRetriever:
    """Day 6 Qdrant PersistentClient + all-MiniLM-L6-v2 (384-dim cosine)."""

    ALLOWED_FILTER_KEYS = {"chapter", "document_type", "language"}

    def __init__(
        self,
        qdrant_dir: Path | str = ROOT / "starter" / "qdrant_db",
        collection: str = "strategy_B",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.qdrant_dir = Path(qdrant_dir)
        self.collection = collection
        self.model_name = model_name
        self._embedder = SentenceTransformer(model_name, local_files_only=True)
        self.dimensions = self._embedder.get_embedding_dimension()
        self._client = QdrantClient(path=str(self.qdrant_dir))
        pts, _ = self._client.scroll(collection_name=self.collection, limit=1, with_payload=True)
        self.available_keys = set(pts[0].payload.keys()) if pts and pts[0].payload else set()

    def close(self) -> None:
        self._client.close()

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        qdrant_filter = None
        if filters:
            unknown = set(filters) - self.ALLOWED_FILTER_KEYS
            if unknown:
                raise ValueError("unsupported metadata filter: " + ", ".join(sorted(unknown)))
            active = {k: v for k, v in filters.items() if k in self.available_keys}
            if active:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=str(v)))
                    for k, v in active.items()
                ]
                qdrant_filter = Filter(must=conditions)

        vec = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].tolist()

        response = self._client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits: list[dict[str, Any]] = []
        for pt in response.points:
            p = pt.payload or {}
            meta = {
                k: v
                for k, v in p.items()
                if k not in {"chunk_id", "document_id", "source", "page", "text"}
            }
            hits.append(
                {
                    "id": p.get("chunk_id", str(pt.id)),
                    "source_id": p.get("document_id"),
                    "source": p.get("source", ""),
                    "page": p.get("page"),
                    "text": p.get("text", ""),
                    "metadata": meta,
                    "score": float(pt.score),
                }
            )
        return hits


class TfidfRetriever:
    """Lexical TF-IDF retriever reusing starter/baseline.py TfidfIndex."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._index = TfidfIndex(chunks)
        self.available_keys = {k for c in chunks for k in c.get("metadata", {}).keys()}

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        active = {k: v for k, v in (filters or {}).items() if k in self.available_keys}
        if not active:
            return self._index.search(query, top_k=top_k)
        filtered = [
            c
            for c in self._chunks
            if all(c.get("metadata", {}).get(k) == v for k, v in active.items())
        ]
        return TfidfIndex(filtered).search(query, top_k=top_k) if filtered else []


class BM25Retriever:
    """Pure-Python BM25 lexical retriever (k1=1.5, b=0.75)."""

    def __init__(
        self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.available_keys = {k for c in chunks for k in c.get("metadata", {}).keys()}
        self.doc_tokens = [tokenize_bm25(c["text"]) for c in chunks]
        self.doc_tfs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        n = len(chunks)
        self.idf = {
            t: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for t, freq in df.items()
        }

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_toks = tokenize_bm25(query)
        active = {k: v for k, v in (filters or {}).items() if k in self.available_keys}
        scored: list[tuple[float, int]] = []
        for idx, (chunk, tf_map, dl) in enumerate(
            zip(self.chunks, self.doc_tfs, self.doc_lens)
        ):
            if active and any(
                chunk.get("metadata", {}).get(k) != v for k, v in active.items()
            ):
                continue
            score = 0.0
            for qt in q_toks:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(qt, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda x: (-x[0], self.chunks[x[1]]["id"]))
        return [{**self.chunks[idx], "score": round(score, 6)} for score, idx in scored[:top_k]]


class HybridRRFRetriever:
    """Reciprocal Rank Fusion combining Dense (Qdrant) + BM25."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        *,
        rrf_k: int = 60,
        fetch_k: int = 20,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        k_fetch = max(self.fetch_k, top_k)
        dense_hits = self.dense.search(query, filters=filters, top_k=k_fetch)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=k_fetch)

        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, item in enumerate(bm25_hits, start=1):
            cid = item["id"]
            by_id.setdefault(cid, item)
            scores[cid] = scores.get(cid, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [{**by_id[cid], "score": round(sc, 6)} for cid, sc in ranked]


class RerankRetriever:
    """Two-stage retriever: fetch candidates with base retriever, rerank with CrossEncoder."""

    def __init__(
        self,
        base_retriever: Any,
        *,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 15,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.base = base_retriever
        self.fetch_k = fetch_k
        self.rerank_model = rerank_model
        self._cross = CrossEncoder(rerank_model)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        candidates = self.base.search(query, filters=filters, top_k=max(self.fetch_k, top_k))
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self._cross.predict(pairs)
        reranked = sorted(
            zip(ce_scores, candidates),
            key=lambda x: (-float(x[0]), x[1]["id"]),
        )[:top_k]
        return [{**c, "score": round(float(sc), 6)} for sc, c in reranked]
