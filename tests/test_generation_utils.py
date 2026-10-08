"""Generation-side logic tests (no model needed): prompts, citations, F1."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant import config
from fin_rag_research_assistant.generation import (
    REFUSAL_STRING,
    answer_precision,
    build_prompt,
    evaluate_generation_result,
    find_citations,
    is_refusal,
    lexical_groundedness,
    parse_citations,
)


def test_instruction_matches_specification_verbatim():
    assert config.GEN_INSTRUCTION == (
        "Answer using ONLY the provided excerpts; cite excerpt numbers; "
        "if the excerpts do not contain the answer, reply exactly INSUFFICIENT_CONTEXT"
    )
    assert REFUSAL_STRING == "INSUFFICIENT_CONTEXT"


def test_build_prompt_includes_excerpts_question_and_instruction():
    messages = build_prompt("What was total net sales?", ["Alpha excerpt.", "Beta excerpt."])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert "[1] Alpha excerpt." in user
    assert "[2] Beta excerpt." in user
    assert "Question: What was total net sales?" in user
    assert config.GEN_INSTRUCTION in messages[0]["content"]


def test_parse_citations_valid_in_order_of_appearance():
    assert parse_citations("Revenue grew per [2] and [1].", 3) == [2, 1]
    assert parse_citations("[1] [3]", 3) == [1, 3]


def test_parse_citations_dedups_and_drops_out_of_range():
    assert parse_citations("[1] [1] [3]", 2) == [1]  # [3] out of range
    assert parse_citations("no citations here", 3) == []
    assert parse_citations("[0] [4] [-1]", 3) == []


def test_find_citations_counts_fabricated_before_filtering():
    valid, fabricated = find_citations("Revenue grew [1] per [99] and [2]; see also [0] [99].", 3)
    assert valid == [1, 2]           # in range, deduplicated, in order
    assert fabricated == [99, 0]     # out of range, counted BEFORE filtering, deduplicated


def test_find_citations_all_fabricated_or_none():
    assert find_citations("[7] [7]", 2) == ([], [7])
    assert find_citations("no brackets", 2) == ([], [])


def test_parse_citations_backward_compatible_with_fabricated_present():
    # wrapper keeps pre-existing behavior: only valid ids are returned/used
    assert parse_citations("[1] [5]", 3) == [1]


def test_answer_precision_hand_computed():
    excerpt = "Apple's total net sales were $391,035 million in fiscal 2024."
    precision = answer_precision("Total net sales rose.", [excerpt], [1])
    assert precision == pytest.approx(0.75)  # overlap {total, net, sales} / 4 answer tokens


def test_answer_precision_robust_to_long_excerpts():
    """Precision keeps its denominator = answer tokens, so padding the cited
    excerpt with unrelated words does not change it (unlike the F1, whose
    recall denominator is the whole excerpt)."""
    filler = " " + " ".join(f"zz{i}" for i in range(500))
    short = ["Apple's total net sales were $391,035 million in fiscal 2024."]
    long_ex = [short[0] + filler]
    p_short = answer_precision("Total net sales rose.", short, [1])
    p_long = answer_precision("Total net sales rose.", long_ex, [1])
    assert p_short == pytest.approx(0.75)
    assert p_long == pytest.approx(p_short)
    # sanity: the F1 does drop when the excerpt grows (documented limitation)
    f1_short = lexical_groundedness("Total net sales rose.", short, [1])
    f1_long = lexical_groundedness("Total net sales rose.", long_ex, [1])
    assert f1_long < f1_short


def test_is_refusal_detection():
    assert is_refusal("INSUFFICIENT_CONTEXT") is True
    assert is_refusal("The excerpts show INSUFFICIENT_CONTEXT.") is True
    assert is_refusal("Total net sales were $416,161 million [1].") is False


def test_lexical_groundedness_hand_computed():
    excerpt = "Apple's total net sales were $391,035 million in fiscal 2024."
    f1 = lexical_groundedness("Total net sales rose.", [excerpt], [1])
    # overlap {total, net, sales}: P = 3/4, R = 3/10 -> F1 = 3/7
    assert f1 == pytest.approx(3 / 7)


def test_lexical_groundedness_edges():
    assert lexical_groundedness("answer text", ["excerpt"], []) == 0.0  # nothing cited
    assert lexical_groundedness("", ["excerpt"], [1]) == 0.0
    assert lexical_groundedness("Revenue rose", ["revenue rose"], [1]) == 1.0


def test_evaluate_result_fabricated_citation_invalid():
    result = evaluate_generation_result(
        "Q", "q?", ["a", "b"], ["id1", "id2"], "Sales rose [5].", 0.1
    )
    assert result.citations == []
    assert result.fabricated_citations == [5]  # recorded, counted before filtering
    assert result.citation_existence is False
    assert result.citation_valid is False  # cited id outside the retrieved range
    assert result.used_refusal is False


def test_evaluate_result_mixed_valid_and_fabricated_citations():
    result = evaluate_generation_result(
        "Q", "q?", ["a", "b"], ["id1", "id2"], "Sales rose [1]; also [9].", 0.1
    )
    assert result.citations == [1]  # valid id still usable as evidence
    assert result.fabricated_citations == [9]
    assert result.citation_existence is True   # cites at least one provided excerpt
    assert result.citation_valid is False      # ...but also emitted a fabricated id


def test_evaluate_result_refusal_with_fabricated_id_is_invalid():
    result = evaluate_generation_result(
        "Q", "q?", ["a", "b"], ["id1", "id2"], "INSUFFICIENT_CONTEXT [9]", 0.1
    )
    assert result.used_refusal is True
    assert result.fabricated_citations == [9]
    assert result.citation_valid is False  # contradictory: refusal + fake citation


def test_evaluate_result_missing_citation_invalid():
    result = evaluate_generation_result(
        "Q", "q?", ["a", "b"], ["id1", "id2"], "Sales rose a lot.", 0.1
    )
    assert result.citations == []
    assert result.citation_valid is False  # asserted a fact with zero citations


def test_evaluate_result_refusal_needs_no_citation():
    result = evaluate_generation_result(
        "Q", "q?", ["a", "b"], ["id1", "id2"], "INSUFFICIENT_CONTEXT", 0.1
    )
    assert result.used_refusal is True
    assert result.citation_valid is True  # a refusal makes no factual claim


def test_evaluate_result_valid_grounded_answer():
    result = evaluate_generation_result(
        "Q", "q?", ["Apple total net sales rose."], ["id1"], "Apple total net sales rose [1].", 0.2
    )
    assert result.citation_valid is True
    assert result.citations == [1]
    assert result.fabricated_citations == []
    assert result.groundedness_f1 > 0.9
    # answer tokens {apple,total,net,sales,rose,1} -> the bare citation id "1"
    # counts as a token, so precision is 5/6, not 1.0
    assert result.answer_precision == pytest.approx(5 / 6)
