"""Token-window chunking with per-chunk provenance.

A "token" is a whitespace-delimited word (documented approximation — keeps
chunking deterministic and dependency-free). Chunks slide with a fixed overlap
so no corpus text is lost between window boundaries; provenance records the
source doc and the character span in the canonical corpus text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # deterministic: {doc_id}:c{position:04d}
    doc_id: str  # ticker, e.g. "AAPL"
    position: int  # 0-based chunk index within the doc
    text: str
    start_char: int  # span in the doc's canonical corpus text
    end_char: int
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization used for chunking and for BM25."""
    return text.split()


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def _chunk_id(doc_id: str, position: int) -> str:
    digest = hashlib.sha1(f"{doc_id}:{position}".encode()).hexdigest()[:8]
    return f"{doc_id}:c{position:04d}:{digest}"


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Split ``text`` into overlapping token windows.

    The stride is ``chunk_size - overlap``; the final window is always aligned
    to the end of the text (no truncation loss). Overlap windows are trimmed to
    actual token counts so tiny docs produce a single chunk, never empty or
    duplicated chunks.
    """
    tokens = tokenize(text)
    if not tokens:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    n = len(tokens)
    if n <= chunk_size:
        return [
            Chunk(
                chunk_id=_chunk_id(doc_id, 0),
                doc_id=doc_id,
                position=0,
                text=text,
                start_char=0,
                end_char=len(text),
                n_tokens=n,
            )
        ]

    # Token start offsets -> character offsets for provenance spans.
    offsets: list[int] = []
    pos = 0
    for tok in tokens:
        offsets.append(text.index(tok, pos))
        pos = offsets[-1] + len(tok)

    starts: list[int] = list(range(0, n - overlap, chunk_size - overlap))
    if starts[-1] + chunk_size < n:  # ensure the tail is covered
        starts.append(n - chunk_size)
    starts = sorted(set(starts))

    chunks: list[Chunk] = []
    for position, tok_start in enumerate(starts):
        tok_end = min(tok_start + chunk_size, n)
        char_start = offsets[tok_start]
        char_end = offsets[tok_end - 1] + len(tokens[tok_end - 1])
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, position),
                doc_id=doc_id,
                position=position,
                text=" ".join(tokens[tok_start:tok_end]),
                start_char=char_start,
                end_char=char_end,
                n_tokens=tok_end - tok_start,
            )
        )
    return chunks


def chunk_corpus(
    corpus: dict[str, str],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk a {doc_id: corpus_text} mapping into a single chunk list."""
    chunks: list[Chunk] = []
    for doc_id in sorted(corpus):
        chunks.extend(
            chunk_text(corpus[doc_id], doc_id, chunk_size, overlap)
        )
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
