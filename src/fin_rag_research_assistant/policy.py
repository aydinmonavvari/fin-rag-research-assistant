"""Refusal policy: refuse when retrieval confidence falls below a threshold.

The gate uses the best available DENSE cosine score of a query (dense scores
are bounded in [-1, 1], unlike BM25 scores, and are the weakest link of the
hybrid stack). tau is selected from the observed score distribution of the
in-scope questions versus the out-of-scope probes — not from a textbook value.

Since the enforcement fix, this module also produces the DECISION the
generation stage binds to its output (:func:`gate_decision`): the same
calibrated rule, with fail-closed semantics on any gate failure and an
explicit DISABLED state when the gate is not armed.
"""

from __future__ import annotations

import math


def should_refuse(top_dense_score: float | None, tau: float) -> bool:
    """True when the system should refuse to answer.

    A ``None`` score (no candidates retrieved at all) always refuses.
    """
    if top_dense_score is None:
        return True
    return bool(top_dense_score < tau)


# Gate decision outcomes. GATE_REFUSE and GATE_ANSWER bind the emitted answer
# at the generation output boundary; GATE_DISABLED records that the gate was
# not armed (--no-dense, or no tau available) and must never be rendered as
# either an approval or a safety refusal.
GATE_REFUSE = "refuse"
GATE_ANSWER = "answer"
GATE_DISABLED = "disabled"


def _finite_float_or_none(value) -> float | None:
    """Return ``value`` as a finite float, or None when it is malformed.

    Malformed means: not a real number (str, None, container, ...), a bool
    (bool is an int subclass but is never a measurement), NaN, or +/-inf.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def gate_decision(top_dense_score, tau, armed: bool) -> str:
    """Decide the gate outcome for one request: refuse, answer, or disabled.

    The armed rule is EXACTLY :func:`should_refuse` (refuse iff the top dense
    cosine score < tau; ``None`` = no candidates -> refuse). Enforcement does
    not change the calibrated rule — it only binds the decision to the answer
    that is actually emitted.

    - ``armed=False`` (e.g. --no-dense) or ``tau=None``: ``GATE_DISABLED``.
      Recorded as such; never rendered as approval or as a safety refusal.
    - FAIL CLOSED on any gate failure: malformed tau (non-numeric, bool, NaN,
      inf), malformed or non-finite score, an exception from the rule, or a
      non-bool rule output all return ``GATE_REFUSE`` — an unmeasurable gate
      state must never emit model text.
    """
    if not armed or tau is None:
        return GATE_DISABLED
    try:
        tau_value = _finite_float_or_none(tau)
        if tau_value is None:  # malformed tau -> fail closed
            return GATE_REFUSE
        if top_dense_score is None:
            score = None  # no dense candidates; the rule itself refuses
        else:
            score = _finite_float_or_none(top_dense_score)
            if score is None:  # malformed/non-finite score -> fail closed
                return GATE_REFUSE
        refuse = should_refuse(score, tau_value)
        if not isinstance(refuse, bool):  # non-bool rule output -> fail closed
            return GATE_REFUSE
    except Exception:  # noqa: BLE001 - any gate failure refuses (fail closed)
        return GATE_REFUSE
    return GATE_REFUSE if refuse else GATE_ANSWER


def pick_refusal_threshold(
    in_scope_scores: list[float],
    out_of_scope_scores: list[float],
    margin: float = 0.02,
) -> float:
    """Pick tau separating the two score distributions (the implemented rule).

    Refusals must trigger for LOW dense scores, so tau must sit below the
    in-scope scores. The rule, exactly as implemented:

    - **Clean gap** (``min(in_scope) > max(out_of_scope)``): place tau inside
      the gap, biased toward in-scope safety —
      ``tau = min(in_min - margin, max(probe_max + 0.01, (in_min + probe_max) / 2))``.
    - **Fallback** (distributions overlap, i.e. ``in_min <= probe_max``): there
      is no threshold that both passes every in-scope question and refuses
      every probe; tau is placed just below the in-scope minimum —
      ``tau = in_min - margin`` — which keeps in-scope recall at 100% BY
      CONSTRUCTION while probes above tau are still answered (documented
      failure mode, not a solved separation).

    This is NOT a percentile rule. (On the SET A dev distribution the two
    groups overlap — probe max 0.583 > in-scope min 0.389 — so the fallback
    applied: tau = 0.389 - 0.02 = 0.369.)
    """
    if not in_scope_scores or not out_of_scope_scores:
        raise ValueError("both score lists are required")
    in_min = min(in_scope_scores)
    probe_max = max(out_of_scope_scores)
    if in_min > probe_max:
        # clean gap: tau anywhere in (probe_max, in_min); bias toward in-scope
        # safety (margin below in_min) so no in-scope question is refused.
        return round(min(in_min - margin, max(probe_max + 0.01, (in_min + probe_max) / 2)), 3)
    # overlapping distributions: fall back just below the in-scope minimum,
    # which keeps in-scope recall at 100% and is documented as a limitation.
    return round(in_min - margin, 3)


def refusal_threshold_report(
    in_scope_scores: list[float],
    out_of_scope_scores: list[float],
    tau: float,
) -> dict:
    """Summarize how tau partitions the two observed score groups."""
    passed = sum(1 for s in in_scope_scores if not should_refuse(s, tau))
    refused_probes = sum(1 for s in out_of_scope_scores if should_refuse(s, tau))
    return {
        "tau": tau,
        "in_scope_n": len(in_scope_scores),
        "in_scope_passed": passed,
        "in_scope_false_refusal_rate": 1.0 - passed / len(in_scope_scores),
        "probe_n": len(out_of_scope_scores),
        "probes_refused": refused_probes,
        "probe_refusal_rate": refused_probes / len(out_of_scope_scores),
        "in_scope_score_min": min(in_scope_scores),
        "in_scope_score_max": max(in_scope_scores),
        "probe_score_min": min(out_of_scope_scores),
        "probe_score_max": max(out_of_scope_scores),
    }
