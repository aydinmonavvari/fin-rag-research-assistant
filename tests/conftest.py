"""Shared test helpers. Heavy-model tests are gated on the local HF cache so
the suite stays offline and fast in CI (no torch / transformers needed)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def hub_cache_path(repo_id: str) -> Path:
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub" / f"models--{repo_id.replace('/', '--')}"


def skip_if_model_not_cached(repo_id: str) -> None:
    """Skip unless the model snapshot is already in the local HF cache."""
    if os.environ.get("FIN_RAG_FORCE_HEAVY") == "1":
        return
    if not hub_cache_path(repo_id).exists():
        pytest.skip(f"{repo_id} not in local HF cache; heavy test skipped (offline suite)")


def make_chunks(texts: list[str]):
    """Build Chunk records for retriever tests (ids d0, d1, ...)."""
    from fin_rag_research_assistant.chunk import Chunk

    return [
        Chunk(
            chunk_id=f"d{i}",
            doc_id="TEST",
            position=i,
            text=text,
            start_char=0,
            end_char=len(text),
            n_tokens=len(text.split()),
        )
        for i, text in enumerate(texts)
    ]
