"""Retriever tests: BM25 ordering, RRF fusion wiring, random control."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant.metrics import reciprocal_rank_fusion
from fin_rag_research_assistant.retrievers import (
    BM25Retriever,
    HybridRetriever,
    RandomRetriever,
    _rrf_score_map,
)
from tests.conftest import make_chunks

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Apple reported record iPhone revenue for the fiscal year.",
    "A local bakery sells fresh sourdough bread every single morning.",
]


def test_bm25_ranks_relevant_document_first():
    bm25 = BM25Retriever(make_chunks(TEXTS))
    results = bm25.retrieve("Apple iPhone record revenue", top_k=3)
    assert results[0].chunk_id == "d1"
    assert len(results) == 3
    assert results[0].score > 0.0
    assert all(a.score >= b.score for a, b in zip(results[:-1], results[1:], strict=True))


def test_bm25_scores_are_sparse_across_topics():
    bm25 = BM25Retriever(make_chunks(TEXTS))
    scores = bm25.retrieve("sourdough bakery bread", top_k=3)
    assert scores[0].chunk_id == "d2"


def test_rrf_score_map_matches_hand_computation():
    rankings = [["d0", "d1", "d2"], ["d1", "d2", "d0"]]
    scores = _rrf_score_map(rankings, k=60)
    assert scores["d0"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["d1"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["d2"] == pytest.approx(1 / 63 + 1 / 62)


class _StubRetriever:
    """Duck-typed retriever with a fixed ranking per query (for fusion tests)."""

    name = "stub"

    def __init__(self, chunks, ranking: list[str]):
        self.chunks = chunks
        self._ids = [c.chunk_id for c in chunks]
        self._ranking = ranking

    def retrieve(self, query: str, top_k: int = 5):  # noqa: ARG002
        from fin_rag_research_assistant.retrievers import Retrieved

        return [Retrieved(cid, 1.0 - 0.01 * rank) for rank, cid in enumerate(self._ranking)]


def test_hybrid_fuses_two_rankings_with_rrf():
    chunks = make_chunks(TEXTS)
    first = _StubRetriever(chunks, ["d0", "d1", "d2"])
    second = _StubRetriever(chunks, ["d1", "d2", "d0"])
    hybrid = HybridRetriever(first, second, k=60)  # type: ignore[arg-type]
    results = hybrid.retrieve("anything", top_k=3)
    fused = reciprocal_rank_fusion([["d0", "d1", "d2"], ["d1", "d2", "d0"]], k=60)
    assert [r.chunk_id for r in results] == fused
    assert results[0].chunk_id == "d1"


def test_hybrid_rejects_mismatched_chunk_sets():
    chunks_a = make_chunks(TEXTS)
    chunks_b = make_chunks(TEXTS[:2])
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(_StubRetriever(chunks_a, ["d0"]), _StubRetriever(chunks_b, ["d1"]))  # type: ignore[arg-type]


def test_random_retriever_deterministic_and_seed_sensitive():
    n = 20
    chunks = make_chunks([f"document number {i} about topic {i}" for i in range(n)])
    r1, r2 = RandomRetriever(chunks, seed=42), RandomRetriever(chunks, seed=42)
    r3 = RandomRetriever(chunks, seed=7)
    ids1 = [x.chunk_id for x in r1.retrieve("query", top_k=5)]
    ids2 = [x.chunk_id for x in r2.retrieve("query", top_k=5)]
    ids3 = [x.chunk_id for x in r3.retrieve("query", top_k=5)]
    assert ids1 == ids2  # same seed -> identical control ranking
    assert ids1 != ids3  # different seed -> different control ranking
