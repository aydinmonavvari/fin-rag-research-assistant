"""Heavy-model tests: gated on the local HF cache; skipped in CI / offline runs.

These verify the real dense retriever and generation wrapper end-to-end once
the models are cached (they are cached by the eval stages of run_study).
"""

from __future__ import annotations

import pytest
from tests.conftest import make_chunks, skip_if_model_not_cached

from fin_rag_research_assistant.config import GEN_MODEL_ID
from fin_rag_research_assistant.retrievers import DENSE_MODEL_ID, DenseRetriever

SEMANTIC_TEXTS = [
    "The company's fiscal year ends on the last Saturday of September.",
    "The board declared a quarterly dividend of twenty-five cents per share.",
    "A feline rested quietly on the woven rug beside the fireplace.",
    "The auditor issued an unqualified opinion on the financial statements.",
    "Heavy rain flooded the streets downtown late last night.",
]


def test_dense_retriever_ranks_semantic_match_first():
    skip_if_model_not_cached(DENSE_MODEL_ID)
    pytest.importorskip("sentence_transformers")
    dense = DenseRetriever(make_chunks(SEMANTIC_TEXTS))
    results = dense.retrieve("the cat is sleeping on the carpet", top_k=5)
    assert results[0].chunk_id == "d2"  # paraphrase, zero lexical overlap in query
    assert -1.0 <= results[0].score <= 1.0  # cosine similarity


def test_generation_wrapper_returns_nonempty_string():
    skip_if_model_not_cached(GEN_MODEL_ID)
    pytest.importorskip("transformers")
    from fin_rag_research_assistant.generation import generate_answer, load_generation_model

    model, tokenizer = load_generation_model()
    raw, latency = generate_answer(
        "What is two plus two?", ["Two plus two equals four."], model, tokenizer, max_new_tokens=24
    )
    assert isinstance(raw, str) and len(raw.strip()) > 0
    assert latency > 0.0
