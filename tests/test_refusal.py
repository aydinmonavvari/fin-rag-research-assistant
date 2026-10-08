"""Refusal-policy logic on hand-built score distributions."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant import evaluation
from fin_rag_research_assistant.evaluation import (
    RetrievalRun,
    refusal_eval_report,
    select_refusal_threshold,
)
from fin_rag_research_assistant.policy import (
    pick_refusal_threshold,
    refusal_threshold_report,
    should_refuse,
)


def _run(scores: dict[str, float]) -> RetrievalRun:
    """Stub dense RetrievalRun: one top score per qid."""
    return RetrievalRun(
        retriever_name="dense",
        ranked_lists={qid: ["c0"] for qid in scores},
        score_lists={qid: [score] for qid, score in scores.items()},
    )


def test_should_refuse_boundary_logic():
    assert should_refuse(0.10, 0.30) is True
    assert should_refuse(0.50, 0.30) is False
    assert should_refuse(0.30, 0.30) is False  # score == tau answers (>= comparison)
    assert should_refuse(None, 0.30) is True  # no retrieval at all -> refuse


def test_pick_threshold_separates_clean_gap():
    in_scope = [0.55, 0.62, 0.70, 0.48]
    probes = [0.10, 0.18, 0.25, 0.31]
    tau = pick_refusal_threshold(in_scope, probes)
    assert 0.31 < tau < 0.48
    report = refusal_threshold_report(in_scope, probes, tau)
    assert report["in_scope_passed"] == 4
    assert report["probes_refused"] == 4
    assert report["in_scope_false_refusal_rate"] == 0.0
    assert report["probe_refusal_rate"] == 1.0


def test_pick_threshold_overlap_fallback_keeps_in_scope():
    in_scope = [0.50, 0.60]
    probes = [0.55, 0.58]  # distributions overlap
    tau = pick_refusal_threshold(in_scope, probes)
    assert tau == pytest.approx(0.48)  # just below the in-scope minimum (margin 0.02)
    report = refusal_threshold_report(in_scope, probes, tau)
    assert report["in_scope_passed"] == 2  # no in-scope question refused (documented priority)
    assert report["probes_refused"] == 0  # probes score above tau: overlap defeats the gate


def test_pick_threshold_requires_both_lists():
    with pytest.raises(ValueError, match="required"):
        pick_refusal_threshold([0.5], [])


def test_pick_threshold_is_not_a_percentile_rule():
    """The implemented rule is the clean-gap rule / in_min-margin fallback —
    NOT the 5th percentile the docs once claimed. For in-scope [0.50, 0.60]
    with probes overlapping, the 5th percentile of in-scope scores would be
    ~0.5025, but the rule returns in_min - margin = 0.48."""
    tau = pick_refusal_threshold([0.50, 0.60], [0.55, 0.58])
    assert tau == pytest.approx(0.48)


class _Question:
    """Minimal QAItem stand-in (only .qid is read by the evaluation helpers)."""

    def __init__(self, qid: str):
        self.qid = qid


def test_select_refusal_threshold_uses_only_dev_scores(monkeypatch):
    """The selection step must see ONLY the dev distribution: feed it a stub
    pick_refusal_threshold that records what it was given."""
    seen: dict[str, list] = {}

    def _spy(in_scope, probes, margin=0.02):  # noqa: ARG001
        seen["in_scope"], seen["probes"] = list(in_scope), list(probes)
        return 0.42

    monkeypatch.setattr(evaluation, "pick_refusal_threshold", _spy)
    questions = [_Question(f"Q-{i}") for i in range(3)]
    run = _run({"Q-0": 0.55, "Q-1": 0.61, "Q-2": 0.49})
    report = select_refusal_threshold(run, questions, {"OOS-A": 0.44})
    assert seen["in_scope"] == [0.55, 0.61, 0.49]
    assert seen["probes"] == [0.44]
    assert report["tau"] == 0.42
    assert report["selection_set"] == "dev (SET A)"


def test_refusal_eval_report_applies_tau_without_repicking(monkeypatch):
    """Applying tau to a split must NEVER re-pick it (held-out semantics)."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("tau must not be re-selected outside the dev set")

    monkeypatch.setattr(evaluation, "pick_refusal_threshold", _boom)
    questions = [_Question(f"B-{i}") for i in range(2)]
    run = _run({"B-0": 0.51, "B-1": 0.44})
    report = refusal_eval_report(run, questions, {"B-OOS-1": 0.58}, tau=0.42, split="heldout")
    assert report["split"] == "heldout"
    assert report["tau"] == 0.42
    assert report["in_scope_passed"] == 2  # 0.51 and 0.44 both >= tau
    assert report["probes_refused"] == 0   # 0.58 >= tau -> answered
    assert report["in_scope_scores"] == [0.51, 0.44]


def test_refusal_eval_report_headline_refusal_behaviour():
    questions = [_Question(f"B-{i}") for i in range(2)]
    run = _run({"B-0": 0.40, "B-1": 0.45})  # B-0 falls below tau
    report = refusal_eval_report(run, questions, {"B-OOS-1": 0.30, "B-OOS-2": 0.55},
                                 tau=0.42, split="heldout")
    assert report["in_scope_passed"] == 1
    assert report["in_scope_false_refusal_rate"] == pytest.approx(0.5)
    assert report["probes_refused"] == 1
    assert report["probe_refusal_rate"] == pytest.approx(0.5)
