"""Ranking metrics verified against hand-computed values."""

from __future__ import annotations

import math

from fin_rag_research_assistant.metrics import (
    evaluate_ranking,
    mrr,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


def test_recall_at_k_basics():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, "c", 5) == 1.0
    assert recall_at_k(ranked, "c", 2) == 0.0
    assert recall_at_k(ranked, "a", 1) == 1.0
    assert recall_at_k(ranked, "e", 4) == 0.0


def test_mrr_hand_computed():
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, "a") == 1.0
    assert reciprocal_rank(ranked, "c") == 1.0 / 3.0
    assert reciprocal_rank(ranked, "zzz") == 0.0
    # MRR over two questions: ranks 1 and 2 -> (1 + 1/2) / 2
    assert mrr([["a", "b"], ["b", "c"]], ["a", "c"]) == (1.0 + 0.5) / 2.0


def test_ndcg_at_k_hand_computed():
    # binary relevance, single gold: nDCG = 1/log2(rank+1); ideal = 1/log2(2) = 1
    assert ndcg_at_k(["a", "b", "c"], "a", 5) == 1.0
    assert ndcg_at_k(["a", "b", "c"], "c", 5) == 1.0 / math.log2(4)
    assert ndcg_at_k(["a", "b", "c"], "b", 5) == 1.0 / math.log2(3)
    assert ndcg_at_k(["a", "b", "c"], "z", 5) == 0.0
    assert ndcg_at_k(["a", "b", "c"], "c", 2) == 0.0  # outside the cutoff


def test_evaluate_ranking_aggregates():
    ranked_lists = [["a", "b", "c"], ["b", "c", "d"]]
    gold_ids = ["a", "c"]
    out = evaluate_ranking(ranked_lists, gold_ids, ks=(1, 5))
    assert out["recall@1"] == 0.5  # question 1 hit at rank 1, question 2 missed
    assert out["recall@5"] == 1.0
    assert out["mrr"] == (1.0 + 0.5) / 2.0
    assert out["ndcg@5"] == (1.0 + 1.0 / math.log2(3)) / 2.0


def test_rrf_hand_computed_scores():
    # rankings A: d0 > d1 > d2 ; B: d1 > d2 > d0 (k = 60)
    rankings = [["d0", "d1", "d2"], ["d1", "d2", "d0"]]
    expected = {
        "d0": 1 / 61 + 1 / 63,
        "d1": 1 / 62 + 1 / 61,
        "d2": 1 / 63 + 1 / 62,
    }
    fused = reciprocal_rank_fusion(rankings, k=60)
    assert fused[0] == "d1"  # 0.032522 > d0 0.032266 > d2 0.032002
    assert fused[1] == "d0"
    assert fused[2] == "d2"
    assert abs(expected["d1"] - max(expected.values())) < 1e-12
