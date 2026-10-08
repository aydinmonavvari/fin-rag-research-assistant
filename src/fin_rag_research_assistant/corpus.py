"""Corpus construction: cached filings -> parsed text -> canonical corpus.

The parsed corpus is cached under data/processed/ (git-ignored) together with
a provenance JSON recording the SEC accession numbers, fetch dates and URLs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fin_rag_research_assistant import config
from fin_rag_research_assistant.chunk import Chunk, chunk_corpus, save_chunks
from fin_rag_research_assistant.edgar import fetch_all_filings
from fin_rag_research_assistant.parse import extract_paragraphs, full_text


def build_corpus(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> tuple[dict[str, str], dict]:
    """Parse every cached filing into {filing_id: corpus_text} + provenance."""
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    fetched = fetch_all_filings(raw_dir)
    corpus: dict[str, str] = {}
    provenance: dict = {"fetched_at": time.strftime("%Y-%m-%d"), "filings": {}}
    for ticker, (ref, html_path) in sorted(fetched.items()):
        paragraphs = extract_paragraphs(html_path.read_text(encoding="utf-8", errors="replace"))
        corpus[ticker] = full_text(paragraphs)
        provenance["filings"][ticker] = {
            "company": config.TARGET_FILINGS[ticker]["name"],
            "form": ref.form,
            "cik": ref.cik,
            "accession": ref.accession,
            "filing_date": ref.filing_date,
            "report_date": ref.report_date,
            "document_url": ref.document_url,
            "submissions_url": config.SEC_SUBMISSIONS_URL.format(cik10=ref.cik),
            "user_agent": config.SEC_USER_AGENT,
            "n_paragraphs": len(paragraphs),
            "n_chars": len(corpus[ticker]),
            "cached_html_bytes": html_path.stat().st_size,
        }
    (processed_dir / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False), encoding="utf-8"
    )
    (processed_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return corpus, provenance


def load_corpus(processed_dir: Path | None = None) -> tuple[dict[str, str], dict]:
    """Load the cached parsed corpus (raises if the corpus was never built)."""
    processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    corpus_path = processed_dir / "corpus.json"
    prov_path = processed_dir / "provenance.json"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found — run the `fetch` + `index` stages first"
        )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    return corpus, provenance


def build_and_save_chunks(
    corpus: dict[str, str],
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
    processed_dir: Path | None = None,
) -> list[Chunk]:
    """Chunk the corpus and persist to data/processed/chunks_{size}.jsonl."""
    processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    chunks = chunk_corpus(corpus, chunk_size=chunk_size, overlap=overlap)
    save_chunks(chunks, processed_dir / f"chunks_{chunk_size}.jsonl")
    return chunks
