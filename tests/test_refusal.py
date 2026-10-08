"""Refusal-policy logic on hand-built score distributions."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant.policy import (
    pick_refusal_threshold,
    refusal_threshold_report,
    should_refuse,
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
