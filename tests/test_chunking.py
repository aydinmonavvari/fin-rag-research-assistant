"""Chunker tests: sizes, overlap, provenance spans, no truncation loss."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant.chunk import chunk_text, count_tokens


def _unique_token_text(n_tokens: int) -> tuple[str, list[str]]:
    words = [f"t{i:05d}" for i in range(n_tokens)]
    return " ".join(words), words


def test_chunks_respect_max_tokens():
    text, _ = _unique_token_text(3000)
    chunks = chunk_text(text, "TEST", chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 800 for c in chunks)
    assert all(c.filing_id == "TEST" for c in chunks)


def test_no_truncation_loss_and_exact_overlap():
    text, words = _unique_token_text(1500)
    chunks = chunk_text(text, "T", chunk_size=800, overlap=100)
    covered: set[str] = set()
    for chunk in chunks:
        covered |= set(chunk.text.split())
    assert covered == set(words)  # every token survives in at least one chunk
    # stride = 800 - 100 = 700 -> consecutive windows share exactly 100 tokens
    token_sets = [set(c.text.split()) for c in chunks]
    for a, b in zip(token_sets[:-1], token_sets[1:], strict=True):
        assert len(a & b) == 100


def test_provenance_char_spans_align_with_text():
    paragraphs = [
        f"Paragraph {i:02d} " + " ".join(f"p{i}w{j}" for j in range(45)) for i in range(30)
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, "P", chunk_size=800, overlap=100)
    assert len(chunks) >= 2
    for chunk in chunks:
        span_tokens = text[chunk.start_char : chunk.end_char].split()
        assert span_tokens == chunk.text.split()  # same token window, char-exact provenance


def test_single_chunk_for_short_text():
    text, _ = _unique_token_text(50)
    chunks = chunk_text(text, "T", chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].n_tokens == 50
    assert chunks[0].position == 0
    assert chunks[0].start_char == 0 and chunks[0].end_char == len(text)


def test_chunk_ids_deterministic_and_unique():
    text, _ = _unique_token_text(3000)
    first = chunk_text(text, "AAPL", chunk_size=800, overlap=100)
    second = chunk_text(text, "AAPL", chunk_size=800, overlap=100)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].chunk_id.startswith("AAPL:c0000:")


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError, match="exceed"):
        chunk_text("a b c", "T", chunk_size=100, overlap=100)


def test_count_tokens_whitespace_definition():
    assert count_tokens("one two  three\n\nfour") == 4
