"""Retrievers: BM25 (sparse), dense sentence embeddings, hybrid RRF, random.

All retrievers share one interface::

    retriever.retrieve(query, top_k) -> list[(chunk_id, score)]

ordered best-first. Dense retrievers lazily import sentence-transformers (the
heavy stack lives behind the ``rag`` extra; unit tests skip when absent).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from fin_rag_research_assistant.chunk import Chunk, tokenize
from fin_rag_research_assistant.metrics import reciprocal_rank_fusion

DENSE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Retrieved:
    chunk_id: str
    score: float


class BM25Retriever:
    """Sparse lexical retrieval (Okapi BM25 over whitespace tokens)."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._ids = [chunk.chunk_id for chunk in chunks]
        self._corpus = [tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[Retrieved]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")
        return [Retrieved(self._ids[i], float(scores[i])) for i in order[:top_k]]


class DenseRetriever:
    """Dense retrieval with sentence-transformers cosine similarity."""

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        model_id: str = DENSE_MODEL_ID,
        cache_dir: Path | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - CI has no heavy stack
            raise ImportError(
                "DenseRetriever requires the 'rag' extra: pip install -e .[rag]"
            ) from exc
        kwargs = {} if cache_dir is None else {"cache_folder": str(cache_dir)}
        self._model = SentenceTransformer(model_id, **kwargs)
        self.chunks = chunks
        self._ids = [chunk.chunk_id for chunk in chunks]
        self._embeddings = self._model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Retrieved]:
        query_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        scores = self._embeddings @ query_vec  # cosine (both sides normalized)
        order = np.argsort(-scores, kind="stable")
        return [Retrieved(self._ids[i], float(scores[i])) for i in order[:top_k]]


class HybridRetriever:
    """Reciprocal rank fusion of BM25 + dense rankings (k = 60)."""

    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, k: int = 60) -> None:
        if len(bm25.chunks) != len(dense.chunks) or bm25._ids != dense._ids:
            raise ValueError("BM25 and dense retrievers must index the same chunks")
        self._bm25 = bm25
        self._dense = dense
        self.k = k

    def retrieve(self, query: str, top_k: int = 5) -> list[Retrieved]:
        depth = max(top_k * 5, 20)  # fusion input depth; documented choice
        bm25_ranking = [r.chunk_id for r in self._bm25.retrieve(query, depth)]
        dense_ranking = [r.chunk_id for r in self._dense.retrieve(query, depth)]
        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=self.k)
        # Report RRF scores; keep the dense cosine for the refusal-policy layer.
        rrf_scores = _rrf_score_map([bm25_ranking, dense_ranking], self.k)
        return [Retrieved(cid, rrf_scores[cid]) for cid in fused[:top_k]]


def _rrf_score_map(rankings: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class RandomRetriever:
    """Seeded random-order control: the floor every retriever must beat."""

    name = "random"

    def __init__(self, chunks: list[Chunk], seed: int = 42) -> None:
        self.chunks = chunks
        self._ids = [chunk.chunk_id for chunk in chunks]
        self._rng = random.Random(seed)

    def retrieve(self, query: str, top_k: int = 5) -> list[Retrieved]:  # noqa: ARG002
        ids = list(self._ids)
        self._rng.shuffle(ids)
        return [Retrieved(cid, float("nan")) for cid in ids[:top_k]]
