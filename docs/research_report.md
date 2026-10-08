# Research Report — fin-rag-research-assistant

**Retrieval-augmented question answering over Federal Reserve Beige Book reports: BM25 vs dense vs hybrid retrieval, citation-grounded generation, and refusal behavior**

Aydin Monavvari · October 2026 · v1.1.0

---

## Abstract

This study evaluates retrieval and grounded-generation components for question answering over a real financial-document corpus: 26 Federal Reserve Beige Book documents from the October 2025 and January 2026 releases (public domain). A 20-question QA set with machine-verified gold chunks was authored during corpus preparation. BM25 achieved Recall@5 = 1.00 (MRR 0.847, nDCG@5 0.886), substantially outperforming dense retrieval with all-MiniLM-L6-v2 (Recall@5 0.20) and hybrid reciprocal-rank fusion (0.45); a random control scored at chance. A score-threshold refusal rule (τ = 0.369) produced no false refusals in-domain but refused neither of two out-of-scope probes. Qwen 2.5 0.5B-Instruct, answering from top-3 retrieved excerpts, produced structurally valid citations in 8/8 cases but near-zero lexical groundedness (mean F1 0.035) and answered both out-of-scope probes rather than refusing, in one case fabricating a specific federal funds rate range. The results document that (i) sparse lexical retrieval can dominate on boilerplate-heavy institutional corpora, and (ii) grounding and refusal are system properties that fail independently and must each be measured.

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
- Provenance: `data/processed/provenance.json` records every document's title, release, URL, byte size, fetch date, and User-Agent.

**QA set.** Twenty in-scope questions (10 per release; 7 numeric, 9 categorical, 4 definitional) and 2 out-of-scope probes were authored by the researcher after reading the parsed corpus. Each in-scope question records its gold chunk id and a verbatim evidence snippet; the pipeline machine-verifies all 20 snippets against the chunked corpus (20/20 pass). The set is a study instrument, not a public benchmark, and is tracked in `data/qa/qa_set.jsonl` for transparency.

**Documented pivot.** The corpus was originally planned as SEC 10-K filings. SEC's edge servers returned persistent HTTP 403 responses to this build environment's egress IP, which survived long backoffs; the corpus source was therefore switched to the Beige Book. The EDGAR fetch layer was removed (its design is acknowledged in the acknowledgments), and this pivot is recorded here and in the code.

## Methodology

**Retrievers.** BM25 (`rank-bm25`, lowercased whitespace tokenization); dense cosine similarity over frozen `all-MiniLM-L6-v2` embeddings; hybrid reciprocal rank fusion of the two (k = 60); and a seeded random control. All retrievers return ranked chunk lists; evaluation depth is 10.

**Ranking metrics.** Recall@1, Recall@5, MRR (reciprocal rank of the gold chunk), and nDCG@5, computed in `metrics.py` and verified against hand-computed examples in the unit tests.

**Refusal analysis.** τ is selected as the 5th percentile of in-scope top dense cosine scores; the analysis reports in-scope pass rate, false-refusal rate, and the refusal rate on out-of-scope probes, alongside both score distributions.

**Grounded generation.** For 8 in-scope questions and 2 probes, the top-3 hybrid-retrieved excerpts are placed in the prompt with the instruction to answer only from them, cite excerpt numbers in brackets, and reply exactly `INSUFFICIENT_CONTEXT` otherwise. Qwen/Qwen2.5-0.5B-Instruct runs locally (CPU, greedy decoding, ≤160 new tokens). Proxy metrics: citation validity (cited ids exist and were retrieved), token-overlap F1 between the answer and cited excerpts ("groundedness"), refusal correctness on probes, and latency. Two good and two flawed runs are recorded verbatim.

## Experimental Design

The pipeline runs as five idempotent, cache-first stages (`fetch → index → eval-retrieval → eval-generation → report`), deterministically ordered, with fixed seed 42 for the random control and greedy decoding for generation. Gold evidence is verified before any retrieval scoring. The 400-token sensitivity run re-maps gold chunks (20/20 re-mapped) and re-runs all retrievers. All artifacts (metrics JSON, summary, five figures) are regenerated by one command and committed.

## Results

**Retrieval (800-token chunks).**

| Retriever | Recall@1 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- |
| BM25 | 0.75 | 1.00 | 0.847 | 0.886 |
| Dense (MiniLM) | 0.15 | 0.20 | 0.174 | 0.169 |
| Hybrid (RRF) | 0.20 | 0.45 | 0.329 | 0.351 |
| Random control | 0.00 | 0.00 | 0.013 | 0.000 |

**Chunk-size sensitivity (400 tokens).** BM25: Recall@1 0.80, Recall@5 1.00, MRR 0.90, nDCG@5 0.926. Dense improves (Recall@5 0.55); hybrid improves (0.65). BM25 remains dominant at both chunk sizes.

**Refusal threshold.** τ = 0.369 (5th percentile of in-scope top dense scores; in-scope range 0.389–0.708). In-scope: 20/20 passed, 0% false refusals. Out-of-scope probes: 0/2 refused; probe scores 0.439–0.583 lie inside the in-scope range.

**Grounded generation.** Citation validity 8/8. Groundedness: mean F1 0.035, median 0.000. Refusal probes: 0/2 correct — the model never emitted `INSUFFICIENT_CONTEXT`. Mean latency 4.55 s/answer (CPU).

Recorded examples (verbatim, from `reports/metrics_generation.json`):

- *Flawed — bare citation:* Q-01 ("how many Districts reported slight to modest growth…") was answered with `[2]` and no content.
- *Flawed — hallucination:* OOS-01 asked for the federal funds rate target range announced at the January 2026 FOMC meeting. That fact is not in the corpus. The model answered: *"The FOMC announced a target range of 0.25% to 0.50% for the federal funds rate at its January 2026 meeting."* — a confident, specific fabrication, with a citation attached.
- *Good (by proxy):* Q-08 and Q-06 produced answers whose cited excerpts matched the answer text (overlap F1 0.148 and 0.131 — still low in absolute terms).

## Discussion

**Sparse dominance on this corpus is structural.** Gold evidence in the Beige Book is identified by exact entities and numbers ("three-quarters", "January 5, 2026", "low $60 per barrel range"), which lexical matching captures directly. Meanwhile roughly the first third of every chunk consists of site navigation text shared across all 26 documents; this shared mass compresses dense-embedding distinctions between documents and plausibly explains much of MiniLM's weakness here. The finding is corpus-conditional, and the boilerplate-stripping experiment (future work) would test that explanation.

**RRF is not a free upgrade.** Hybrid fusion placed *between* its parents and well below BM25. When one retriever dominates, rank fusion averages away its advantage. Hybrid retrieval needs re-evaluation per corpus, not assumption.

**Grounding fails at the generation layer, not the retrieval layer.** With BM25 providing perfect Recall@5, the model still paraphrased freely, answered a bare `[2]`, and fabricated a rate range on an out-of-scope question. Citation validity (8/8) and groundedness (0.035) measure different constructs; a system can look properly cited while being unfaithful. This is the study's central caution.

**Refusal is a system property.** Neither refusal mechanism worked on the probes: the dense-score threshold could not separate them (overlapping score distributions), and the 0.5B model ignored the refusal instruction entirely. Any claim that a RAG system "refuses when uncertain" requires exactly this kind of adversarial measurement.

## Limitations

1. The QA set is small (20 + 2) and was authored by the researcher during corpus preparation; it is a study instrument, not a benchmark, and authoring choices (e.g., question phrasing echoing corpus vocabulary) may favor lexical retrieval.
2. Groundedness is a lexical-overlap proxy: it undercounts legitimate paraphrase and cannot establish semantic faithfulness. No human evaluation panel was conducted.
3. The generator (0.5B) was chosen for CPU reproducibility; larger models would likely comply better with the refusal instruction. The study measures pipeline behavior, not frontier capability.
4. Site boilerplate remains inside chunks; results may differ substantially after boilerplate stripping or section-aware parsing.
5. One corpus, two releases; the BM25-over-dense gap should be re-measured before any generalization.
6. The refusal threshold is fitted and evaluated on the same in-scope scores; a held-out calibration would be required for an unbiased false-refusal estimate.

## Conclusion

On a real, boilerplate-heavy institutional corpus, a classical sparse retriever solved evidence identification (Recall@5 = 1.00) where a frozen dense encoder and a standard fusion strategy did not; a small local instruct model, given the correct excerpts, still produced unfaithful answers and fabricated specifics on out-of-scope questions while citing excerpts either way. The practical lesson for financial RAG research is to measure retrieval and grounding separately, to include adversarial out-of-scope probes by design, and to treat refusal as an empirical property rather than a prompt-level assumption.

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
