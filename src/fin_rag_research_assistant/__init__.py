"""fin-rag-research-assistant — retrieval-augmented QA study over SEC 10-K filings.

An educational research project comparing sparse (BM25), dense (sentence
embeddings) and hybrid (reciprocal rank fusion) retrieval on a 20-question QA
set authored over two real SEC 10-K filings, with citation-grounded generation
and an explicit refusal policy. NOT financial advice; NOT a production system.
"""

from __future__ import annotations

__version__ = "1.0.0"
