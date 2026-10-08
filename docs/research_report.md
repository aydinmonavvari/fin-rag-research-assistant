# Research Report — fin-rag-research-assistant

**Retrieval-augmented question answering over Federal Reserve Beige Book reports: BM25 vs dense vs hybrid retrieval, citation-grounded generation, and refusal behavior**

Aydin Monavvari · October 2026 · v1.0.0

---

## Abstract

This study evaluates retrieval and grounded-generation components for question answering over a real financial-document corpus: 26 Federal Reserve Beige Book documents from the October 2025 and January 2026 releases (public domain), pinned by a committed SHA-256 manifest. Two self-authored QA splits are used: SET A (20 dev questions + 2 probes) for gold verification and refusal-threshold selection, and SET B (12 held-out paraphrased questions + 2 probes), never used for tuning. On dev questions BM25 achieved Recall@5 = 1.00 (MRR 0.847, nDCG@5 0.886), substantially outperforming dense retrieval with all-MiniLM-L6-v2 (Recall@5 0.20) and hybrid reciprocal-rank fusion (0.45); on held-out questions BM25 retains the lead (Recall@5 0.833, MRR 0.586) with a materially smaller margin. A score-threshold refusal rule (τ = 0.369, selected on SET A only) produced no false refusals on in-scope questions in either split but refused neither of two out-of-scope probes in either split. Qwen 2.5 0.5B-Instruct, answering all held-out questions from top-3 retrieved excerpts, cited provided excerpts in 12/12 cases (zero fabricated ids) but with near-zero lexical groundedness (mean F1 0.020; answer-precision 0.425), emitted bare citations for 8 of 14 answers, and answered both out-of-scope probes rather than refusing, in one case fabricating a specific BLS-attributed unemployment figure. The results document that (i) sparse lexical retrieval dominates on boilerplate-heavy institutional corpora and that dev-only evaluation overstates the advantage, and (ii) grounding and refusal are system properties that fail independently and must each be measured.

## Introduction

Retrieval-augmented generation (RAG) couples a retriever, which selects evidence passages from a document collection, with a generator, which is instructed to answer using that evidence. In financial research settings, RAG systems are attractive because answers can carry citations to primary sources. However, published demonstrations frequently use curated corpora whose text is clean and self-contained. Real institutional corpora — central-bank reports served inside website templates — contain navigational boilerplate, terse domain phrasing, and answers concentrated in individual sentences. This repository treats those properties as the subject of measurement: it builds a corpus from primary documents, evaluates three retrievers against a verified gold set, and then measures how a small local instruct model behaves when instructed to answer only from retrieved excerpts or refuse.

The study is deliberately small, deterministic, and CPU-reproducible: every reported number is an output of the committed pipeline (`scripts/run_study.py`), and every fetch is documented with URL and date in a provenance file.

## Research Question

1. How well do sparse (BM25), dense (sentence embeddings), and hybrid (reciprocal rank fusion) retrievers surface the evidence chunk for factual questions over the Beige Book corpus?
2. Can a score-threshold refusal rule separate answerable questions from out-of-scope probes at retrieval time?
3. Does a small instruct LLM, given the correct excerpts and an explicit citation-or-refuse instruction, (a) cite valid excerpts, (b) stay lexically grounded in them, and (c) refuse when the evidence is absent?

## Related Work

Dense passage retrieval established bi-encoder retrieval over fine-tuned embeddings (Karpukhin et al., 2020); RAG combined such retrievers with sequence-to-sequence generators (Lewis et al., 2020). BM25 remains a strong sparse baseline (Robertson & Zaragoza, 2009), and reciprocal rank fusion is a simple, parameter-light way to merge ranked lists (Cormack, Clarke & Buettcher, 2009). Sentence-BERT made inference-time dense retrieval cheap for frozen encoders (Reimers & Gurevych, 2019). The present study does not propose a new method; it measures how these off-the-shelf components behave on a specific real institutional corpus and reports failure modes (citation-without-grounding, refusal failure) that are usually invisible in demos.

## Data

The corpus comprises both releases of the Federal Reserve Beige Book published on federalreserve.gov at build time: **202510** (October 2025) and **202601** (January 2026), each with a national summary and twelve district reports — 26 documents in total, works of the U.S. Federal Government (public domain).

- Fetching: `beigebook.py` downloads each document once, sends a descriptive User-Agent (GitHub noreply address), sleeps 0.75 s between requests, and retries with bounded backoff on 5xx/429.
- Parsing: a stdlib HTML extractor keeps paragraph text (table rows joined with a separator), skips script/style; each document yields ~2,600 words of study text (~457k characters corpus-wide).
- Chunking: 800-token windows with 100-token overlap (a "token" is a whitespace-delimited word — a documented approximation), yielding **104 chunks**; a 400-token variant (209 chunks) supports sensitivity analysis. Chunk ids are deterministic (`{doc_id}:c{position}:{sha1[:8]}`).
- Provenance: `data/processed/provenance.json` records every document's title, release, URL, byte size, fetch date, and User-Agent; the committed `data/corpus_manifest.json` additionally pins each document's SHA-256 so snapshot drift is detectable.

**QA sets.** SET A (development): twenty in-scope questions (10 per release) and 2 out-of-scope probes, authored by the researcher after reading the parsed corpus, each with a verbatim evidence snippet machine-verified against the chunked corpus (20/20 pass). SET B (held-out): twelve in-scope questions (paraphrased; including temporal and citation-trap variants) and 2 out-of-scope probes, authored after SET A was frozen, verified 12/12. Both sets are study instruments authored by a single builder — there are no external annotators, and this is disclosed rather than claimed away. SET A is tracked in `data/qa/qa_set_dev.jsonl` and SET B in `data/qa/qa_set_eval.jsonl` for transparency; SET B is never used for threshold selection or tuning.

**Documented pivot.** The corpus was originally planned as SEC 10-K filings. SEC's edge servers returned persistent HTTP 403 responses to this build environment's egress IP, which survived long backoffs; the corpus source was therefore switched to the Beige Book. The EDGAR fetch layer was removed (its design is acknowledged in the acknowledgments), and this pivot is recorded here and in the code.

## Methodology

**Retrievers.** BM25 (`rank-bm25`, lowercased whitespace tokenization); dense cosine similarity over frozen `all-MiniLM-L6-v2` embeddings; hybrid reciprocal rank fusion of the two (k = 60); and a seeded random control. All retrievers return ranked chunk lists; evaluation depth is 10.

**Ranking metrics.** Recall@1, Recall@5, MRR (reciprocal rank of the gold chunk), and nDCG@5, computed in `metrics.py` and verified against hand-computed examples in the unit tests.

**QA design.** Both question sets are study instruments authored by the researcher (single builder; there are no external annotators). SET A (development; `data/qa/qa_set_dev.jsonl`) contains 20 in-scope questions (10 per release) plus 2 out-of-scope probes, each with a machine-verified verbatim evidence snippet; it is used for gold verification, debugging, and refusal-threshold selection. SET B (held-out; `data/qa/qa_set_eval.jsonl`) contains 12 in-scope questions — authored after SET A was frozen, with wording deliberately paraphrased away from both the corpus and SET A, including temporal and citation-trap variants — plus 2 out-of-scope probes; it is **never used for threshold selection or tuning** and provides the headline evaluation. The split removes the threshold-circularity of a single-set design; it does not remove self-authorship bias, which is disclosed rather than claimed away.

**Refusal analysis.** τ is selected **on SET A only** by an explicit clean-gap rule: `τ = min(in_scope_min − margin, max(probe_max + 0.01, (in_scope_min + probe_max)/2))` with margin 0.02, falling back to `in_scope_min − margin` when the gap rule degenerates; on SET A this yields τ = 0.369 (in-scope score range 0.389–0.708; probe scores 0.439–0.583 overlap the range, so the fallback branch fired). The rule is not a percentile of the in-scope distribution. In-scope pass rate, false-refusal rate, and probe refusal rate are reported **separately for both splits**.

**Grounded generation.** For **all 12 held-out (SET B) in-scope questions and the 2 SET B probes**, the top-3 hybrid-retrieved excerpts are placed in the prompt with the instruction to answer only from them, cite excerpt numbers in brackets, and reply exactly `INSUFFICIENT_CONTEXT` otherwise. Qwen/Qwen2.5-0.5B-Instruct runs locally (CPU, greedy decoding, ≤160 new tokens). Proxy metrics: citation-**existence** rate (a non-refusal answer cites ≥1 provided excerpt — a structural check that does **not** verify the cited passage supports the claim; no entailment model is run), fabricated-citation rate (bracketed ids outside the provided range), token-overlap F1 between the answer and cited excerpts ("groundedness", whose recall denominator spans the full 800-token excerpts) reported alongside answer-precision (same numerator, answer-token denominator), refusal correctness on probes, and latency. Two good and two flawed runs are recorded verbatim.

## Experimental Design

The pipeline runs as five idempotent, cache-first stages (`fetch → index → eval-retrieval → eval-generation → report`), deterministically ordered, with fixed seed 42 for the random control and greedy decoding for generation. Gold evidence is verified before any retrieval scoring (SET A 20/20, SET B 12/12). `fetch` verifies every document against the committed SHA-256 manifest (`data/corpus_manifest.json`), so a silently revised federalreserve.gov page cannot change results undetected. The 400-token sensitivity run re-maps gold chunks and re-runs all retrievers. All artifacts (metrics JSON, summary, five figures) are regenerated by one command and committed.

## Results

**Retrieval (800-token chunks). SET A = 20 dev questions; SET B = 12 held-out paraphrased questions:**

| Retriever | A R@1 | A R@5 | A MRR | A nDCG@5 | B R@1 | B R@5 | B MRR | B nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.75 | 1.00 | 0.847 | 0.886 | 0.417 | 0.833 | 0.586 | 0.641 |
| Dense (MiniLM) | 0.15 | 0.20 | 0.174 | 0.169 | 0.167 | 0.333 | 0.211 | 0.241 |
| Hybrid (RRF) | 0.20 | 0.45 | 0.329 | 0.351 | 0.333 | 0.417 | 0.364 | 0.366 |
| Random control | 0.00 | 0.00 | 0.013 | 0.000 | 0.000 | 0.083 | 0.021 | 0.036 |

Held-out performance is uniformly lower than dev performance — the dev questions were written while reading the corpus, so dev-only evaluation overstates retrieval quality. BM25's dominance persists on SET B but with a materially smaller margin.

**Chunk-size sensitivity (400 tokens, SET A).** BM25: Recall@1 0.80, Recall@5 1.00, MRR 0.90, nDCG@5 0.926. Dense improves (Recall@5 0.55); hybrid improves (0.65). BM25 remains dominant at both chunk sizes.

**Refusal threshold.** τ = 0.369, selected on SET A by the clean-gap rule (see Methodology). SET A: in-scope 20/20 passed, 0% false refusals; probes 0/2 refused (probe scores 0.439–0.583 inside the in-scope range 0.389–0.708). **SET B (headline): in-scope 12/12 passed, 0% false refusals; probes 0/2 refused** (probe scores 0.409–0.434 vs in-scope range 0.388–0.669). The dev-selected τ generalizes to held-out in-scope behavior while providing no out-of-scope protection in either split.

**Grounded generation (all 12 SET B questions + 2 SET B probes).** Citation existence 12/12; fabricated-citation rate 0/12. Groundedness: mean F1 0.020, median 0.006 (recall denominator = full 800-token excerpts); answer-precision mean 0.425, median 0.345. Refusal probes: 0/2 correct — the model never emitted `INSUFFICIENT_CONTEXT`. 8 of 14 answers were bare citations (`[k]` with no content). Mean latency 7.22 s/answer (CPU).

Recorded examples (verbatim, from `reports/metrics_generation.json`):

- *Flawed — bare citation:* B-02 ("One tourism business in the Fourth District described a steep annual drop in gue[st]…") was answered with `[1]` and no content; B-03 likewise with `[3]`.
- *Flawed — hallucination:* B-OOS-01 asked for the national unemployment rate for December 2025 "according to the Bureau of Labor Statistics". That figure is not in the corpus. The model answered: *"[1] The national unemployment rate for December 2025 according to the Bureau of Labor Statistics was 4.7%."* — a confident, specific fabrication with a citation attached; the only "4.7 percent" in the corpus is Philadelphia firms' third-quarter inflation expectations.
- *Good (by proxy, with a caveat):* B-09 and B-10 (temporal questions) were answered with a bare `[3]` whose single emitted token also appears in the cited excerpt, so answer-precision is recorded as 1.0 while groundedness F1 is 0.006 — the proxy metrics agree the answer contains no substantive content.

## Discussion

**Sparse dominance on this corpus is structural.** Gold evidence in the Beige Book is identified by exact entities and numbers ("three-quarters", "January 5, 2026", "low $60 per barrel range"), which lexical matching captures directly. Meanwhile roughly the first third of every chunk consists of site navigation text shared across all 26 documents; this shared mass compresses dense-embedding distinctions between documents and plausibly explains much of MiniLM's weakness here. The gap replicates on held-out questions but narrows (BM25 Recall@5 0.833 vs dense 0.333 on SET B, versus 1.00 vs 0.20 on SET A): dev-only evaluation would have overstated the sparse advantage. The finding is corpus-conditional, and the boilerplate-stripping experiment (future work) would test that explanation.

**RRF is not a free upgrade.** Hybrid fusion placed *between* its parents and well below BM25. When one retriever dominates, rank fusion averages away its advantage. Hybrid retrieval needs re-evaluation per corpus, not assumption.

**Grounding fails at the generation layer, not the retrieval layer.** Even where retrieval succeeded, the model emitted a bare `[k]` for 8 of 14 answers, and fabricated a specific BLS-attributed unemployment figure on an out-of-scope probe. Citation existence (12/12) and groundedness (0.020 F1; 0.425 answer-precision) measure different constructs; a system can look properly cited while being unfaithful, and existence is not entailment. This is the study's central caution.

**Refusal is a system property.** Neither refusal mechanism worked on the probes in either split: the dense-score threshold could not separate them (overlapping score distributions), and the 0.5B model ignored the refusal instruction entirely. Notably, the τ calibrated on dev data generalized perfectly to held-out in-scope questions (0% false refusals on both SET A and SET B) while still failing out-of-scope — a threshold can be well-calibrated for what it was selected on and still be useless against the failure mode it exists for. Any claim that a RAG system "refuses when uncertain" requires exactly this kind of adversarial measurement.

## Limitations

1. Both QA sets are small (SET A 20 + 2; SET B 12 + 2) and were authored by the researcher (single builder; no external annotators); they are study instruments, not benchmarks. The SET A/SET B split removes the threshold-circularity but not self-authorship bias; SET B paraphrasing reduces, and cannot eliminate, wording overlap with the corpus.
2. Groundedness is a lexical-overlap proxy: it undercounts legitimate paraphrase and cannot establish semantic faithfulness; its recall denominator spans the full 800-token excerpts, so F1 is structurally small and should be read alongside answer-precision. Citation existence does not verify entailment. No human evaluation panel was conducted.
3. The generator (0.5B) was chosen for CPU reproducibility; larger models would likely comply better with the refusal instruction. The study measures pipeline behavior, not frontier capability.
4. Site boilerplate remains inside chunks; results may differ substantially after boilerplate stripping or section-aware parsing.
5. One corpus, two releases; the BM25-over-dense gap should be re-measured before any generalization. Twelve held-out questions is a small evaluation set; no confidence intervals are reported because per-question relevance is deterministic and resampling claims would not be stable at this n.

## Conclusion

On a real, boilerplate-heavy institutional corpus, a classical sparse retriever solved evidence identification on development questions (Recall@5 = 1.00) and retained a clear but narrower lead on held-out paraphrases (Recall@5 = 0.833), where a frozen dense encoder and a standard fusion strategy lagged behind; a small local instruct model, given the correct excerpts, still produced unfaithful answers — mostly bare citations — and fabricated specifics on out-of-scope questions while citing excerpts either way. The practical lesson for financial RAG research is to separate development from held-out evaluation before believing any threshold or ranking number, to measure retrieval and grounding separately, to include adversarial out-of-scope probes by design, and to treat refusal as an empirical property rather than a prompt-level assumption.

## Future Research

- Boilerplate-aware parsing and section-aware chunking, then re-running the full study (hypothesis: dense retrieval narrows the gap).
- Cross-encoder re-ranking as a fourth retriever, and multi-hop questions requiring evidence from two documents.
- Corpus replication on FOMC minutes and on a shareholder-letter corpus to test transfer of the sparse-over-dense result.
- Constrained decoding (grammar-enforced citations) and larger instruct models for the generation layer; a held-out calibration set for the refusal threshold.
- A small human faithfulness panel to replace the lexical proxy and quantify proxy bias.

## References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR '09.
- Karpukhin, V., et al. (2020). *Dense passage retrieval for open-domain question answering.* EMNLP 2020.
- Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 2020.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* EMNLP 2019.
- Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond.* Foundations and Trends in Information Retrieval.

*Board of Governors of the Federal Reserve System, Beige Book, October 2025 and January 2026 releases (public domain).*
