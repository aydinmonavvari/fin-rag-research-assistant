"""Generation-side logic tests (no model needed): prompts, citations, F1."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant import config
from fin_rag_research_assistant.generation import (
    REFUSAL_STRING,
    build_prompt,
    evaluate_generation_result,
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
    assert result.citation_valid is False  # cited id outside the retrieved range
    assert result.used_refusal is False


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
    assert result.groundedness_f1 > 0.9
