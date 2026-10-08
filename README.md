# fin-rag-research-assistant

**Retrieval-augmented question answering over Federal Reserve Beige Book reports — a reproducible evaluation study comparing BM25, dense sentence embeddings, and hybrid retrieval, with citation-grounded generation and an explicit refusal policy.**

[![CI](https://github.com/aydinmonavvari/fin-rag-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/aydinmonavvari/fin-rag-research-assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Part of the [Aydin Monavvari research portfolio](https://github.com/aydinmonavvari) — project 09 of 10.

> **Scope statement.** This is an *educational evaluation study*, not a product and not financial advice. It measures how well standard retrieval components find evidence in real central-bank documents and how a small local language model behaves when instructed to answer from those excerpts. The generation metrics are lexical/structural **proxies**; hallucination risk remains material (a concrete hallucination is documented below); no human evaluation panel was run.

---

## 1 · Short description

`fin-rag-research-assistant` builds a text corpus from **26 real Federal Reserve Beige Book documents** (two releases — October 2025 and January 2026 — each with a national summary plus twelve district reports, all public domain), chunks it with provenance, and evaluates three retrievers (BM25, dense MiniLM embeddings, hybrid reciprocal-rank fusion) plus a random control against a **two-part, self-authored QA design**: SET A (20 dev questions + 2 probes) used for gold verification, debugging, and refusal-threshold selection, and SET B (12 held-out questions + 2 probes, paraphrased, never used for any tuning) used for the headline evaluation. A small instruct model (Qwen 2.5 0.5B) then answers the held-out questions from the top-3 retrieved excerpts under an explicit "cite excerpts or refuse" instruction, and the study measures citation existence, lexical groundedness, and refusal behavior. The full pipeline (`fetch → index → eval-retrieval → eval-generation → report`) runs end-to-end in ~5–10 minutes on CPU after caching.

## 2 · Research question

**Do standard sparse, dense, and hybrid retrievers reliably surface evidence for factual questions over real Beige Book documents, and does a small instruct LLM answer strictly from the retrieved excerpts (with citations) or refuse when the evidence is absent?**

## 3 · Motivation

Retrieval-augmented generation (RAG) is the dominant pattern for grounding LLM systems in verifiable documents, and finance is one of its most demanded application areas. Yet most public RAG demos are shown on curated Wikipedia-style corpora with hand-friendly text. Real financial/economic corpora — here, central-bank reports embedded in website templates — differ in three ways that matter: heavy navigational boilerplate, terse domain phrasing, and answers concentrated in a few precise sentences. This repository treats those difficulties as the object of study rather than a nuisance: it measures retrieval quality with ranking metrics, and it measures grounding behavior at the generation layer instead of assuming it.

## 4 · Why this matters

- **Sparse vs dense is not settled in practice.** On dev questions (SET A) BM25 achieved Recall@5 = **1.00** (MRR 0.847) while a general-purpose sentence embedding (MiniLM) reached only 0.20; on paraphrased held-out questions (SET B) BM25 drops to MRR **0.586** — dev-set numbers overstate real retrieval quality, which is exactly why SET B exists.
- **Grounding can fail even with perfect retrieval.** With the top-3 hybrid excerpts provided, the 0.5B model produced structurally valid citations (citation-existence 12/12 on held-out questions) but very low lexical groundedness (mean F1 0.020; answer-precision 0.425), and it answered the two out-of-scope probes instead of refusing — one of them asserting a specific national unemployment figure attributed to the BLS that appears nowhere in the corpus in that form.
- **Refusal policy design needs measurement, not intuition.** A score-threshold refusal rule (τ = 0.369, selected on SET A only) produced **zero false refusals** on in-scope questions in both splits (20/20 dev, 12/12 held-out) but refused **0/2** out-of-scope probes in both splits — probe dense scores overlap the in-scope range. That failure mode is documented with the score distributions that caused it.

## 5 · Methodology

**Corpus.** 26 Beige Book documents (2 releases × 13 documents) fetched from federalreserve.gov, parsed with a stdlib HTML extractor into paragraph text (~2,650 words per document), and chunked into **104 chunks** of 800 whitespace-delimited tokens with 100-token overlap. Every chunk records its source document and character span. Raw HTML and parsed artifacts are cached under `data/` (git-ignored) with a full provenance JSON (URL, release, fetch date, User-Agent) **and a committed `data/corpus_manifest.json` that pins the SHA-256 of each fetched document** — `fetch` verifies hashes and warns loudly if federalreserve.gov revises a page, so the snapshot behind every reported number is detectable, not silent.

**Two-part QA design (both sets self-authored by the researcher).** SET A (development, `data/qa/qa_set_dev.jsonl`) — 20 in-scope questions (10 per release) + 2 out-of-scope probes — was authored after reading the corpus, each with the gold chunk id and a verbatim evidence snippet that `eval-retrieval` machine-verifies (20/20). It is used for gold verification, debugging, and **refusal-threshold selection**. SET B (held-out, `data/qa/qa_set_eval.jsonl`) — 12 in-scope questions (paraphrased; wording deliberately differs from both the corpus and SET A) + 2 out-of-scope probes + temporal/citation-trap variants — was authored after SET A was frozen and is **never used for threshold selection or tuning**. There are no external annotators; both sets are study instruments, disclosed as such. Gold evidence is machine-verified for both sets (20/20 dev, 12/12 held-out).

**Retrieval evaluation.** Retrievers: BM25 (`rank-bm25`), dense (all-MiniLM-L6-v2 cosine), hybrid (reciprocal rank fusion, k=60, Cormack et al. 2009), plus a random control. Metrics: Recall@1/@5, MRR, nDCG@5 at depth 10, reported **separately for SET A (dev) and SET B (held-out)**. A chunk-size sensitivity run (400 tokens) re-maps gold chunks and re-evaluates.

**Refusal policy.** Refuse when the top dense cosine score < τ. τ is selected **on SET A only**, by an explicit clean-gap rule: `τ = min(in_scope_min − margin, max(probe_max + 0.01, (in_scope_min + probe_max)/2))` with margin 0.02 (falling back to `in_scope_min − margin` when the gap rule degenerates), which yields τ = 0.369 on SET A (in-scope score range 0.389–0.708; probe scores 0.439–0.583 overlap that range, so the fallback branch fired). The rule is *not* a percentile of the in-scope distribution. The study reports in-scope pass rate, false-refusal rate, and out-of-scope probe refusal rate **on both splits** — SET B is the headline.

**Grounded generation.** Qwen/Qwen2.5-0.5B-Instruct (greedy decoding, max 160 new tokens) answers **all 12 held-out (SET B) in-scope questions** plus the 2 SET B probes from the top-3 hybrid-retrieved excerpts under the instruction: *"Answer using ONLY the provided excerpts; cite excerpt numbers; if the excerpts do not contain the answer, reply exactly INSUFFICIENT_CONTEXT."* Metrics (documented as **proxies**, not human evaluation): citation-existence rate (every non-refusal answer cites ≥1 provided excerpt — existence only, **not** entailment), fabricated-citation rate (bracketed ids outside the provided range), lexical groundedness (token-overlap F1 between answer and cited excerpts, reported alongside answer-precision because the F1 recall denominator spans the full 800-token excerpts), refusal correctness, latency.

## 6 · Dataset

| Property | Value |
| --- | --- |
| Source | Federal Reserve Beige Book (public domain, U.S. government) |
| Releases | 202510 (October 2025), 202601 (January 2026) |
| Documents | 26 (national summary + 12 districts per release) |
| Text after parsing | ~457,000 characters (~68,000 words) |
| Chunks (study corpus) | 104 × 800 tokens, 100-token overlap |
| QA SET A (dev) | 20 in-scope + 2 out-of-scope probes (`data/qa/qa_set_dev.jsonl`, tracked) |
| QA SET B (held-out) | 12 in-scope + 2 out-of-scope probes (`data/qa/qa_set_eval.jsonl`, tracked) |

Both QA sets are **study instruments authored by the researcher** (single builder; no external annotators), not public benchmarks: SET A questions were written after reading the corpus; SET B questions were written after SET A was frozen, with paraphrased wording to reduce overlap. Every question carries a verbatim evidence snippet machine-verified against its gold chunk (methodology note included in every generated report).

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
└── author_qa_set.py  # regenerates data/qa/qa_set_dev.jsonl + qa_set_eval.jsonl from authored questions
```

## 9 · Experimental design

1. `fetch` — download/cache 26 Beige Book HTML documents (network only here) and verify each against the committed SHA-256 manifest.
2. `index` — parse → chunk at 800 and 400 tokens → `data/processed/chunks_*.jsonl`.
3. `eval-retrieval` — verify gold evidence snippets (SET A 20/20, SET B 12/12); evaluate 4 retrievers at depth 10 **per split**; recall curves; select refusal τ **on SET A only**; 400-token sensitivity.
4. `eval-generation` — Qwen 2.5 0.5B answers **all 12 SET B questions** + 2 SET B probes from top-3 hybrid excerpts; proxy metrics; 2 good/2 flawed examples recorded verbatim.
5. `report` — merge everything into `reports/metrics.json` + `reports/summary.md` + 5 figures.

Fixed seeds (42) for the random control; greedy decoding for generation; deterministic chunk ids (`{doc_id}:c{position}:{sha1[:8]}`).

## 10 · Models

- **BM25** — `rank-bm25` over whitespace tokenization (lowercased).
- **Dense** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).
- **Hybrid** — reciprocal rank fusion of BM25 + dense (k = 60).
- **Generator** — `Qwen/Qwen2.5-0.5B-Instruct` (0.5B params, CPU, greedy, 160 max new tokens).

## 11 · Evaluation metrics

Retrieval: Recall@1, Recall@5, MRR, nDCG@5 (gold-chunk relevance; higher is better; random control provides the floor), reported per split. Refusal: false-refusal rate (in-scope), probe refusal rate (out-of-scope), per split. Generation proxies: citation-**existence** rate (cited ids exist and were retrieved — this does **not** verify that the cited passage supports the claim; no entailment model is run), fabricated-citation rate (out-of-range bracketed ids), token-overlap F1 ("groundedness") and answer-precision (same numerator, answer-token denominator — robust to the long-excerpt recall denominator), refusal correctness, latency. All formulas are implemented in `metrics.py`/`generation.py` with hand-computed unit tests.

## 12 · Results

**Retrieval (800-token chunks, depth 10). SET A = 20 dev questions; SET B = 12 held-out paraphrased questions (never used for tuning):**

| Retriever | SET A R@1 | SET A R@5 | SET A MRR | SET A nDCG@5 | SET B R@1 | SET B R@5 | SET B MRR | SET B nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **BM25 (sparse)** | **0.75** | **1.00** | **0.847** | **0.886** | **0.417** | **0.833** | **0.586** | **0.641** |
| Dense (MiniLM) | 0.15 | 0.20 | 0.174 | 0.169 | 0.167 | 0.333 | 0.211 | 0.241 |
| Hybrid (RRF) | 0.20 | 0.45 | 0.329 | 0.351 | 0.333 | 0.417 | 0.364 | 0.366 |
| Random control | 0.00 | 0.00 | 0.013 | 0.000 | 0.000 | 0.083 | 0.021 | 0.036 |

Held-out performance is lower across the board — SET A numbers are optimistic because the questions were written while reading the corpus; SET B is the honest estimate.

**Chunk-size sensitivity (400 tokens, SET A):** BM25 Recall@1 0.80 / MRR 0.90; dense and hybrid improve (dense Recall@5 0.55, hybrid 0.65). Smaller chunks help every retriever; BM25 still dominates.

**Refusal policy (τ = 0.369, selected on SET A by the clean-gap rule; SET B never touched during selection):** SET A in-scope pass 20/20 (0% false refusals), probes refused 0/2; **SET B (headline) in-scope pass 12/12 (0% false refusals), probes refused 0/2** — probe dense scores overlap the in-scope range in both splits (SET B: probes 0.409–0.434 vs in-scope 0.388–0.669), so a univariate dense-score threshold cannot separate them on this corpus. The τ selected on dev generalizes to held-out questions (zero false refusals in both), but it provides no out-of-scope protection.

**Grounded generation (Qwen 2.5 0.5B, all 12 SET B questions + 2 SET B probes):**

| Metric | Value |
| --- | --- |
| Citation-existence rate (cites ≥1 provided excerpt) | 12/12 (1.00) |
| Fabricated-citation rate (out-of-range ids) | 0/12 (0.00) |
| Groundedness (token-overlap F1, mean / median) | 0.020 / 0.006 |
| Answer-precision (mean / median) | 0.425 / 0.345 |
| Out-of-scope probes correctly refused | 0/2 |
| Mean latency per answer (CPU) | 7.22 s |

Citation-**existence** is a structural check only — it does not verify that the cited passage supports the claim (no entailment model is run). Groundedness F1 is small partly because its recall denominator spans the full 800-token excerpts; answer-precision (answer-token denominator) is the more comparable figure. Documented examples (SET B ids): **flawed** — B-02/B-03. The probe hallucination persists in a new form: B-OOS-01 ("national unemployment rate for December 2025 according to the Bureau of Labor Statistics" — not in the corpus) produced: *"[1] The national unemployment rate for December 2025 according to the Bureau of Labor Statistics was 4.7%."* — a confident, specific figure with a citation attached; the only "4.7 percent" in the corpus is Philadelphia firms' third-quarter inflation expectations.

## 13 · Interpretation

- The sparse-over-dense gap replicates on held-out questions (BM25 R@5 0.833 vs dense 0.333 on SET B) but narrows in absolute terms — dev-only evaluation would have overstated BM25's advantage (R@5 1.00 vs 0.20). Single-split RAG evaluations overestimate retrieval quality.
- The hybrid result sitting *between* its parents (and below BM25 on both splits) shows RRF is not a guaranteed upgrade when one retriever dominates.
- Citation existence being perfect (12/12) while groundedness is near zero demonstrates that these proxies measure different things: the model *cites* provided excerpts but *answers in its own (often wrong) words*. Neither proxy alone is sufficient evidence of faithfulness, and existence is not entailment.
- The refusal-rule failure (0/2 probes on both splits) plus the generation-layer refusal failure (the model never emitted `INSUFFICIENT_CONTEXT`) jointly illustrate that **refusal is a system property that must be measured, not assumed** — and that a threshold calibrated on dev data can generalize to in-scope behavior while still failing out-of-scope.

## 14 · Limitations

1. Both QA sets were authored by the researcher (single builder; no external annotators) — they are study instruments, not public benchmarks, and results may not generalize. The SET A/SET B split removes the *threshold-circularity* (τ is selected on SET A and evaluated on SET B) but not the self-authorship bias; SET B paraphrases reduce, and cannot eliminate, wording overlap with the corpus.
2. Groundedness is a lexical-overlap proxy; it undercounts correct paraphrases and cannot verify semantic faithfulness (its recall denominator spans the full 800-token excerpts, so F1 is structurally small — read it alongside answer-precision). Citation existence does not verify entailment. No human evaluation panel was run.
3. The generator is a 0.5B model chosen for CPU reproducibility; larger instruct models would likely follow the refusal instruction better. The study measures the *pipeline*, not the frontier.
4. The corpus carries federalreserve.gov website boilerplate inside every chunk; no boilerplate stripping was applied beyond skipping script/style tags.
5. Two releases and one corpus shape; BM25's dominance should be re-measured on other corpora before generalizing. 12 held-out questions is a small evaluation set; confidence intervals are not reported because per-question relevance is deterministic and the sample is too small for stable resampling claims.

## 15 · Reproducibility

- Full pipeline: `python scripts/run_study.py all` (~5–10 min on CPU after the first fetch; network only for `fetch` and model downloads).
- Every stage is idempotent and cache-first; deterministic chunk ids; fixed seed for the random control; greedy decoding.
- **Snapshot pinning:** `data/corpus_manifest.json` (committed) records the SHA-256, size, URL, and fetch date of each of the 26 source documents. `fetch` re-verifies hashes on every run and reports mismatches loudly, so a silently revised federalreserve.gov page cannot change results undetected. The exact fetched bytes are cached in `data/raw/` (git-ignored) with provenance JSON; a fresh clone re-downloads the public documents and must pass hash verification for its evaluation to be comparable with the numbers reported here.
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

See [§8 Architecture](#8--architecture). Tracked artifacts: `reports/` (metrics.json, summary.md), `figures/` (5 PNGs), `data/qa/qa_set_dev.jsonl` + `data/qa/qa_set_eval.jsonl` (the study instruments), `data/corpus_manifest.json` (SHA-256 snapshot manifest). Git-ignored: raw HTML, parsed corpus, chunk files, HF model caches.

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
