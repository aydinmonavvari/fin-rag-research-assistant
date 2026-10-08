"""Ranking metrics (pure Python) and reciprocal rank fusion.

Implementations are deliberately dependency-free and unit-tested against hand
computed examples; nDCG additionally cross-checked against scikit-learn's
``ndcg_score`` in the test suite (binary relevance, single gold chunk).
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    """Fraction of the (single) gold chunk present in the top-k of the ranking."""
    if k <= 0 or not ranked_ids:
        return 0.0
    return float(gold_id in ranked_ids[:k])


def reciprocal_rank(ranked_ids: list[str], gold_id: str) -> float:
    """1 / rank of the gold chunk (0 if absent); the building block of MRR."""
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id == gold_id:
            return 1.0 / rank
    return 0.0


def mrr(ranked_lists: list[list[str]], gold_ids: list[str]) -> float:
    """Mean Reciprocal Rank over questions."""
    if not ranked_lists:
        return 0.0
    values = [
        reciprocal_rank(ranked, gold)
        for ranked, gold in zip(ranked_lists, gold_ids, strict=True)
    ]
    return sum(values) / len(values)


def ndcg_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    """nDCG@k with binary relevance and a single gold chunk.

    DCG  = sum over ranks i<=k of rel_i / log2(i + 1), rel_i = 1 iff ranked_ids[i-1] == gold_id.
    IDCG = 1 / log2(2) = 1 (the gold chunk ranked first), so nDCG@k collapses to
    1 / log2(rank + 1) when the gold chunk is inside the top-k, else 0.
    """
    if k <= 0 or not ranked_ids:
        return 0.0
    idcg = 1.0 / math.log2(2.0)  # ideal: gold chunk at rank 1
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id == gold_id:
            dcg += 1.0 / math.log2(rank + 1.0)
            break  # single relevant document; later hits are impossible
    return dcg / idcg


def evaluate_ranking(
    ranked_lists: list[list[str]], gold_ids: list[str], ks: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    """Recall@k for each k, MRR and nDCG@5 over a set of questions."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(
            recall_at_k(ranked, gold, k)
            for ranked, gold in zip(ranked_lists, gold_ids, strict=True)
        ) / len(ranked_lists)
    out["mrr"] = mrr(ranked_lists, gold_ids)
    out["ndcg@5"] = sum(
        ndcg_at_k(ranked, gold, 5)
        for ranked, gold in zip(ranked_lists, gold_ids, strict=True)
    ) / len(ranked_lists)
    return out


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    top_n: int | None = None,
) -> list[str]:
    """RRF of multiple id rankings (Cormack, Clarke & Buettcher 2009).

    score(d) = sum over rankings of 1 / (k + rank_i(d)); ties broken by the
    lexicographically smaller id for determinism.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused = [chunk_id for chunk_id, _ in ordered]
    return fused[:top_n] if top_n is not None else fused
