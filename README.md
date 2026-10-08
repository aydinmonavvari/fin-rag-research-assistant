# fin-rag-research-assistant

**Retrieval-augmented question answering over Federal Reserve Beige Book reports — a reproducible evaluation study comparing BM25, dense sentence embeddings, and hybrid retrieval, with citation-grounded generation and an explicit refusal policy.**

[![CI](https://github.com/aydinmonavvari/fin-rag-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/aydinmonavvari/fin-rag-research-assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Part of the [Aydin Monavvari research portfolio](https://github.com/aydinmonavvari) — project 09 of 10.

> **Scope statement.** This is an *educational evaluation study*, not a product and not financial advice. It measures how well standard retrieval components find evidence in real central-bank documents and how a small local language model behaves when instructed to answer from those excerpts. The generation metrics are lexical/structural **proxies**; hallucination risk remains material (a concrete hallucination is documented below); no human evaluation panel was run.

---

## 1 · Short description

`fin-rag-research-assistant` builds a text corpus from **26 real Federal Reserve Beige Book documents** (two releases — October 2025 and January 2026 — each with a national summary plus twelve district reports, all public domain), chunks it with provenance, and evaluates three retrievers (BM25, dense MiniLM embeddings, hybrid reciprocal-rank fusion) plus a random control against a 20-question QA set with machine-verified gold chunks. A small instruct model (Qwen 2.5 0.5B) then answers a subset from the top-3 retrieved excerpts under an explicit "cite excerpts or refuse" instruction, and the study measures citation validity, lexical groundedness, and refusal behavior. The full pipeline (`fetch → index → eval-retrieval → eval-generation → report`) runs end-to-end in ~5 minutes on CPU after caching.

## 2 · Research question

**Do standard sparse, dense, and hybrid retrievers reliably surface evidence for factual questions over real Beige Book documents, and does a small instruct LLM answer strictly from the retrieved excerpts (with citations) or refuse when the evidence is absent?**

## 3 · Motivation

Retrieval-augmented generation (RAG) is the dominant pattern for grounding LLM systems in verifiable documents, and finance is one of its most demanded application areas. Yet most public RAG demos are shown on curated Wikipedia-style corpora with hand-friendly text. Real financial/economic corpora — here, central-bank reports embedded in website templates — differ in three ways that matter: heavy navigational boilerplate, terse domain phrasing, and answers concentrated in a few precise sentences. This repository treats those difficulties as the object of study rather than a nuisance: it measures retrieval quality with ranking metrics, and it measures grounding behavior at the generation layer instead of assuming it.

## 4 · Why this matters

- **Sparse vs dense is not settled in practice.** On this corpus BM25 achieved Recall@5 = **1.00** (MRR 0.847) while a general-purpose sentence embedding (MiniLM) reached only 0.20 — an instructive, reproducible gap.
- **Grounding can fail even with perfect retrieval.** With the top-3 BM25 excerpts provided, the 0.5B model produced structurally valid citations (8/8) but very low lexical groundedness (mean F1 0.035), and it answered two out-of-scope questions instead of refusing — one of them fabricating a specific federal-funds rate range that appears nowhere in the corpus.
- **Refusal policy design needs measurement, not intuition.** A score-threshold refusal rule (τ = 0.369) produced **zero false refusals** on in-scope questions but refused **0/2** out-of-scope probes whose dense scores overlapped the in-scope range. That failure mode is documented with the score distributions that caused it.

## 5 · Methodology

**Corpus.** 26 Beige Book documents (2 releases × 13 documents) fetched from federalreserve.gov, parsed with a stdlib HTML extractor into paragraph text (~2,650 words per document), and chunked into **104 chunks** of 800 whitespace-delimited tokens with 100-token overlap. Every chunk records its source document and character span. Raw HTML and parsed artifacts are cached under `data/` (git-ignored) with a full provenance JSON (URL, release, fetch date, User-Agent).

**Retrieval evaluation.** A 20-question QA set was authored *after* reading the corpus (10 questions per release), each with the gold chunk id and a verbatim evidence snippet; `eval-retrieval` machine-verifies that every evidence snippet appears in its gold chunk (20/20 verified). Retrievers: BM25 (`rank-bm25`), dense (all-MiniLM-L6-v2 cosine), hybrid (reciprocal rank fusion, k=60, Cormack et al. 2009), plus a random control. Metrics: Recall@1/@5, MRR, nDCG@5 at depth 10. A chunk-size sensitivity run (400 tokens) re-maps gold chunks (20/20 re-mapped) and re-evaluates.

**Refusal policy.** Refuse when the top dense cosine score < τ. τ is selected from the in-scope score distribution (5th percentile); the study reports in-scope pass rate, false-refusal rate, and out-of-scope probe refusal rate.

**Grounded generation.** Qwen/Qwen2.5-0.5B-Instruct (greedy decoding, max 160 new tokens) answers 8 in-scope questions plus 2 out-of-scope probes from the top-3 hybrid-retrieved excerpts under the instruction: *"Answer using ONLY the provided excerpts; cite excerpt numbers; if the excerpts do not contain the answer, reply exactly INSUFFICIENT_CONTEXT."* Metrics (documented as **proxies**, not human evaluation): citation validity (cited ids exist and were retrieved), lexical groundedness (token-overlap F1 between answer and cited excerpts), refusal correctness, latency.

## 6 · Dataset

| Property | Value |
| --- | --- |
| Source | Federal Reserve Beige Book (public domain, U.S. government) |
| Releases | 202510 (October 2025), 202601 (January 2026) |
| Documents | 26 (national summary + 12 districts per release) |
| Text after parsing | ~457,000 characters (~68,000 words) |
| Chunks (study corpus) | 104 × 800 tokens, 100-token overlap |
| QA set | 20 in-scope + 2 out-of-scope probes (`data/qa/qa_set.jsonl`, tracked) |

The QA set is a **study instrument authored by the researcher**, not a public benchmark: questions were written after reading the corpus, each with a verbatim evidence snippet machine-verified against its gold chunk (methodology note included in every generated report).

## 7 · Data sources

- `https://www.federalreserve.gov/monetarypolicy/beigebook{YYYYMM}[-{district}].htm` — Beige Book release pages (public domain). Every document's URL, byte size, and fetch date is recorded in `data/processed/provenance.json`.
- Requests send a descriptive User-Agent (GitHub noreply address) and are rate-limited (0.75 s) and cached — documents are never fetched twice in one checkout.
- **Documented pivot:** the study was originally designed around SEC 10-K filings, but SEC's edge returned persistent HTTP 403 responses to this build environment's datacenter IP. The corpus source was switched to the Beige Book (equally primary, public-domain, and reproducible); the pivot is recorded in the code comments, README, and research report.

## 8 · Architecture

```
src/fin_rag_research_assistant/
├── config.py         # paths, releases, chunking, refusal τ, generation params
├── beigebook.py      # document refs, polite cached fetching, retry/backoff
├── parse.py          # stdlib HTML → paragraph text, table rows preserved
├── corpus.py         # cached docs → canonical corpus + provenance JSON
├── chunk.py          # deterministic token-window chunking with spans
├── retrievers.py     # BM25 / dense (MiniLM) / hybrid RRF / random control
├── metrics.py        # recall@k, MRR, nDCG, RRF math
├── qa.py             # QA-set schema, validation, JSONL I/O
├── evaluation.py     # gold verification, retrieval eval, refusal analysis
├── generation.py     # prompt building, local Qwen generation, proxy metrics
├── policy.py         # score-threshold refusal rule
└── reporting.py      # figures + reports/metrics.json + summary.md
scripts/
├── run_study.py      # CLI: fetch | index | eval-retrieval | eval-generation | report | all
└── author_qa_set.py  # regenerates data/qa/qa_set.jsonl from authored questions
```

## 9 · Experimental design

1. `fetch` — download/cache 26 Beige Book HTML documents (network only here).
2. `index` — parse → chunk at 800 and 400 tokens → `data/processed/chunks_*.jsonl`.
3. `eval-retrieval` — verify 20/20 gold evidence snippets; evaluate 4 retrievers at depth 10; recall curves; select refusal τ; 400-token sensitivity.
4. `eval-generation` — Qwen 2.5 0.5B answers 8 questions + 2 probes from top-3 hybrid excerpts; proxy metrics; 2 good/2 flawed examples recorded verbatim.
5. `report` — merge everything into `reports/metrics.json` + `reports/summary.md` + 5 figures.

Fixed seeds (42) for the random control; greedy decoding for generation; deterministic chunk ids (`{doc_id}:c{position}:{sha1[:8]}`).

## 10 · Models

- **BM25** — `rank-bm25` over whitespace tokenization (lowercased).
- **Dense** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).
- **Hybrid** — reciprocal rank fusion of BM25 + dense (k = 60).
- **Generator** — `Qwen/Qwen2.5-0.5B-Instruct` (0.5B params, CPU, greedy, 160 max new tokens).

## 11 · Evaluation metrics

Retrieval: Recall@1, Recall@5, MRR, nDCG@5 (gold-chunk relevance; higher is better; random control provides the floor). Refusal: false-refusal rate (in-scope), probe refusal rate (out-of-scope). Generation proxies: citation-validity rate, token-overlap F1 ("groundedness"), refusal correctness, latency. All formulas are implemented in `metrics.py` with hand-computed unit tests.

## 12 · Results

**Retrieval (20 questions, 800-token chunks, depth 10):**

| Retriever | Recall@1 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- |
| **BM25 (sparse)** | **0.75** | **1.00** | **0.847** | **0.886** |
| Dense (MiniLM) | 0.15 | 0.20 | 0.174 | 0.169 |
| Hybrid (RRF) | 0.20 | 0.45 | 0.329 | 0.351 |
| Random control | 0.00 | 0.00 | 0.013 | 0.000 |

**Chunk-size sensitivity (400 tokens):** BM25 Recall@1 0.80 / MRR 0.90; dense improves to Recall@5 0.55; hybrid to 0.65. Smaller chunks help every retriever; BM25 still dominates.

**Refusal policy (τ = 0.369 from the in-scope score distribution):** in-scope pass 20/20 (0% false refusals); out-of-scope probes refused 0/2 — probe dense scores (0.439–0.583) fall inside the in-scope range (0.389–0.708), so a univariate dense-score threshold cannot separate them on this corpus.

**Grounded generation (Qwen 2.5 0.5B, 8 questions + 2 probes):**

| Metric | Value |
| --- | --- |
| Citation validity (cited ids exist and were retrieved) | 8/8 |
| Groundedness (token-overlap F1, mean / median) | 0.035 / 0.000 |
| Out-of-scope probes correctly refused | 0/2 |
| Mean latency per answer (CPU) | 4.55 s |

Documented examples: **flawed** — Q-01's full answer was `[2]` (a bare citation with no content); **OOS-01** (federal funds rate in January 2026 — not in the corpus) produced: *"The FOMC announced a target range of 0.25% to 0.50% for the federal funds rate at its January 2026 meeting."* — a specific, confident **hallucination** with a citation attached.

## 13 · Interpretation

- The sparse-over-dense gap (Recall@5 1.00 vs 0.20) is consistent with the corpus's structure: answers hinge on exact entity/number phrases that BM25 matches lexically, while the corpus is dominated by shared navigation boilerplate that flattens dense semantic neighborhoods. It does **not** show dense retrieval is weak in general — only that it is not free wins on this corpus shape.
- The hybrid result sitting *between* its parents (and below BM25) shows RRF is not a guaranteed upgrade when one retriever dominates.
- Citation validity being perfect while groundedness is near zero demonstrates that these two proxy metrics measure different things: the model *cites* the right excerpts but *answers in its own (often wrong) words*. Neither proxy alone is sufficient evidence of faithfulness.
- The refusal-rule failure (0/2 probes) plus the generation-layer refusal failure (the model never emitted `INSUFFICIENT_CONTEXT`) jointly illustrate that **refusal is a system property that must be measured, not assumed**.

## 14 · Limitations

1. The QA set (20 questions + 2 probes) was authored by the researcher during corpus preparation — it is a study instrument, not a public benchmark, and results may not generalize.
2. Groundedness is a lexical-overlap proxy; it undercounts correct paraphrases and cannot verify semantic faithfulness. No human evaluation panel was run.
3. The generator is a 0.5B model chosen for CPU reproducibility; larger instruct models would likely follow the refusal instruction better. The study measures the *pipeline*, not the frontier.
4. The corpus carries federalreserve.gov website boilerplate inside every chunk; no boilerplate stripping was applied beyond skipping script/style tags.
5. Two releases and one corpus shape; BM25's dominance should be re-measured on other corpora before generalizing.
6. The refusal threshold is univariate (dense top score) and was fitted on the same question scores it is evaluated on.

## 15 · Reproducibility

- Full pipeline: `python scripts/run_study.py all` (~5 min on CPU after the first fetch; network only for `fetch` and model downloads).
- Every stage is idempotent and cache-first; deterministic chunk ids; fixed seed for the random control; greedy decoding.
- The exact fetched bytes are cached in `data/raw/` (git-ignored) with provenance JSON; re-running `fetch` on a fresh clone re-downloads the public documents.
- `reports/metrics.json` and `reports/summary.md` are committed and were produced by the committed code on 2026-10-08.

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/fin-rag-research-assistant.git
cd fin-rag-research-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # CI-grade: offline tests + lint
pip install -e ".[rag]"        # full study: torch/transformers/sentence-transformers
```

## 17 · Usage

```bash
python scripts/run_study.py fetch            # download + cache 26 Beige Book documents
python scripts/run_study.py index            # parse + chunk (800 and 400 tokens)
python scripts/run_study.py eval-retrieval   # BM25 vs dense vs hybrid vs random + refusal τ
python scripts/run_study.py eval-generation  # Qwen 0.5B grounded-generation proxies
python scripts/run_study.py report           # reports/metrics.json + summary.md
python scripts/run_study.py all              # everything above in order
```

## 18 · Example

```console
$ python scripts/run_study.py eval-retrieval
[eval-retrieval] gold verification: 20/20 (missing: [])
[eval-retrieval] refusal tau = 0.369 (in-scope passed 20/20, probes refused 0/2)
[eval-retrieval] 400-token sensitivity: 20/20 gold chunks re-mapped
[eval-retrieval] figures 1, 2, 3, 5 written to figures/
```

## 19 · Project structure

See [§8 Architecture](#8--architecture). Tracked artifacts: `reports/` (metrics.json, summary.md), `figures/` (5 PNGs), `data/qa/qa_set.jsonl` (the study instrument). Git-ignored: raw HTML, parsed corpus, chunk files, HF model caches.

## 20 · Future work

- Boilerplate-aware parsing (strip the shared federalreserve.gov navigation before chunking) and re-measuring the dense retriever.
- A second corpus (e.g., FOMC minutes) to test whether BM25's dominance transfers.
- Multi-hop questions and re-ranking (cross-encoder) as a fourth retriever.
- A small human faithfulness panel to replace the lexical groundedness proxy.
- Larger instruct models and constrained decoding (forced citation grammar) at the generation layer.

## 21 · Citation

```bibtex
@software{monavvari2026finrag,
  author  = {Monavvari, Aydin},
  title   = {fin-rag-research-assistant: Retrieval-Augmented QA over Federal Reserve
             Beige Book Reports — BM25 vs Dense vs Hybrid Retrieval with
             Citation-Grounded Generation},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/aydinmonavvari/fin-rag-research-assistant}
}
```

See also `CITATION.cff`.

## 22 · License

MIT — see [LICENSE](LICENSE). The Beige Book documents are works of the U.S. Federal Government (public domain).

## 23 · Acknowledgments

- The Federal Reserve Board, for publishing the Beige Book and its archives.
- The `rank-bm25`, `sentence-transformers`, and `transformers` communities.
- The datamule/Teraflop-AI/Eventual SEC-EDGAR dataset authors, whose work was consulted while designing the (pre-pivot) EDGAR fetch layer.

**Portfolio:** this is project 09 of 10 in Aydin Monavvari's research portfolio; the flagship workbench (`finscope-ai-research`) integrates the portfolio's methodologies.
