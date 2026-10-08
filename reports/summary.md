# fin-rag-research-assistant — study summary

_Generated 2026-10-08 22:12 UTC by `scripts/run_study.py report`. Every number below is an actual output of the committed pipeline run._

## Corpus provenance (Federal Reserve Beige Book, public domain)

Snapshot pin: `data/corpus_manifest.json` (sha256, 26 documents) — verified against the cache before every evaluation stage. If federalreserve.gov revises a page, the hash check fails loudly.

| Document | Title | Release | Cached HTML (bytes) | Source |
| --- | --- | --- | --- | --- |
| 202510-atlanta | Beige Book 202510 — Sixth District (Atlanta) (district report) | 202510 | 88353 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-atlanta.htm) |
| 202510-boston | Beige Book 202510 — First District (Boston) (district report) | 202510 | 88604 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-boston.htm) |
| 202510-chicago | Beige Book 202510 — Seventh District (Chicago) (district report) | 202510 | 88189 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-chicago.htm) |
| 202510-cleveland | Beige Book 202510 — Fourth District (Cleveland) (district report) | 202510 | 88529 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-cleveland.htm) |
| 202510-dallas | Beige Book 202510 — Eleventh District (Dallas) (district report) | 202510 | 88420 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-dallas.htm) |
| 202510-kansas-city | Beige Book 202510 — Tenth District (Kansas City) (district report) | 202510 | 88653 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-kansas-city.htm) |
| 202510-minneapolis | Beige Book 202510 — Ninth District (Minneapolis) (district report) | 202510 | 87955 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-minneapolis.htm) |
| 202510-new-york | Beige Book 202510 — Second District (New York) (district report) | 202510 | 88866 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-new-york.htm) |
| 202510-philadelphia | Beige Book 202510 — Third District (Philadelphia) (district report) | 202510 | 89015 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-philadelphia.htm) |
| 202510-richmond | Beige Book 202510 — Fifth District (Richmond) (district report) | 202510 | 88162 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-richmond.htm) |
| 202510-san-francisco | Beige Book 202510 — Twelfth District (San Francisco) (district report) | 202510 | 89014 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-san-francisco.htm) |
| 202510-st-louis | Beige Book 202510 — Eighth District (St. Louis) (district report) | 202510 | 88317 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-st-louis.htm) |
| 202510-summary | Beige Book 202510 — National Summary | 202510 | 89461 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202510-summary.htm) |
| 202601-atlanta | Beige Book 202601 — Sixth District (Atlanta) (district report) | 202601 | 88054 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-atlanta.htm) |
| 202601-boston | Beige Book 202601 — First District (Boston) (district report) | 202601 | 88327 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-boston.htm) |
| 202601-chicago | Beige Book 202601 — Seventh District (Chicago) (district report) | 202601 | 88057 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-chicago.htm) |
| 202601-cleveland | Beige Book 202601 — Fourth District (Cleveland) (district report) | 202601 | 88871 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-cleveland.htm) |
| 202601-dallas | Beige Book 202601 — Eleventh District (Dallas) (district report) | 202601 | 88830 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-dallas.htm) |
| 202601-kansas-city | Beige Book 202601 — Tenth District (Kansas City) (district report) | 202601 | 88704 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-kansas-city.htm) |
| 202601-minneapolis | Beige Book 202601 — Ninth District (Minneapolis) (district report) | 202601 | 88102 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-minneapolis.htm) |
| 202601-new-york | Beige Book 202601 — Second District (New York) (district report) | 202601 | 88595 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-new-york.htm) |
| 202601-philadelphia | Beige Book 202601 — Third District (Philadelphia) (district report) | 202601 | 88846 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-philadelphia.htm) |
| 202601-richmond | Beige Book 202601 — Fifth District (Richmond) (district report) | 202601 | 87813 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-richmond.htm) |
| 202601-san-francisco | Beige Book 202601 — Twelfth District (San Francisco) (district report) | 202601 | 89293 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-san-francisco.htm) |
| 202601-st-louis | Beige Book 202601 — Eighth District (St. Louis) (district report) | 202601 | 87361 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-st-louis.htm) |
| 202601-summary | Beige Book 202601 — National Summary | 202601 | 88925 | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/beigebook202601-summary.htm) |

## Corpus & chunking

| Document | Paragraphs | Characters | Chunks (800 tok) | Tokens in chunks |
| --- | --- | --- | --- | --- |
| 202510-atlanta | 311 | 17413 | 4 | 2643 |
| 202510-boston | 306 | 17721 | 4 | 2707 |
| 202510-chicago | 308 | 17296 | 4 | 2647 |
| 202510-cleveland | 309 | 17633 | 4 | 2694 |
| 202510-dallas | 311 | 17460 | 4 | 2675 |
| 202510-kansas-city | 308 | 17696 | 4 | 2715 |
| 202510-minneapolis | 308 | 17067 | 4 | 2642 |
| 202510-new-york | 311 | 17944 | 4 | 2738 |
| 202510-philadelphia | 318 | 18096 | 4 | 2780 |
| 202510-richmond | 307 | 17269 | 4 | 2662 |
| 202510-san-francisco | 309 | 18017 | 4 | 2702 |
| 202510-st-louis | 308 | 17427 | 4 | 2672 |
| 202510-summary | 319 | 18312 | 4 | 2747 |
| 202601-atlanta | 310 | 17117 | 4 | 2598 |
| 202601-boston | 304 | 17530 | 4 | 2685 |
| 202601-chicago | 308 | 17156 | 4 | 2620 |
| 202601-cleveland | 309 | 17975 | 4 | 2703 |
| 202601-dallas | 311 | 17862 | 4 | 2758 |
| 202601-kansas-city | 308 | 17778 | 4 | 2682 |
| 202601-minneapolis | 308 | 17169 | 4 | 2634 |
| 202601-new-york | 311 | 17697 | 4 | 2700 |
| 202601-philadelphia | 316 | 17937 | 4 | 2803 |
| 202601-richmond | 307 | 16944 | 4 | 2644 |
| 202601-san-francisco | 309 | 18339 | 4 | 2732 |
| 202601-st-louis | 307 | 16462 | 4 | 2561 |
| 202601-summary | 319 | 17774 | 4 | 2700 |

Corpus total: **104 chunks** at 800-token windows with 100-token overlap (a 'token' is a whitespace-delimited word — documented approximation).

## QA design (authorship disclosed)

> Both QA sets were authored by the researcher (single builder; there are no external annotators). SET A (dev, data/qa/qa_set_dev.jsonl: 20 in-scope + 2 probes) was written after reading the parsed Beige Book corpus and is used ONLY for refusal-threshold selection and debugging. SET B (held-out, data/qa/qa_set_eval.jsonl: 12 in-scope + 2 probes) was authored after SET A was frozen, using paraphrased wording to reduce overlap; it is never used for threshold selection. Answers are verifiable in the cited Beige Book documents (each in-scope item records a gold chunk id plus a verbatim evidence snippet, machine-verified by the eval-retrieval stage). These are study instruments, not public benchmarks.

## Retrieval evaluation — SET A (dev) — used for tau selection and debugging only; **NOT a final evaluation**.

Questions: **20** in-scope + 2 out-of-scope probes (`data/qa/qa_set_dev.jsonl`). Gold verification: 20/20 evidence snippets verified.

| Retriever | Recall@1 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- |
| BM25 (sparse) | 0.750 | 1.000 | 0.847 | 0.886 |
| Dense (MiniLM) | 0.150 | 0.200 | 0.174 | 0.169 |
| Hybrid (RRF) | 0.200 | 0.450 | 0.329 | 0.351 |
| Random control | 0.000 | 0.000 | 0.013 | 0.000 |

## Retrieval evaluation — SET B (held-out) — **final reported retrieval evaluation**; never used for threshold selection.

Questions: **12** in-scope + 2 out-of-scope probes (`data/qa/qa_set_eval.jsonl`). Gold verification: 12/12 evidence snippets verified.

| Retriever | Recall@1 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- |
| BM25 (sparse) | 0.417 | 0.833 | 0.586 | 0.641 |
| Dense (MiniLM) | 0.167 | 0.333 | 0.211 | 0.241 |
| Hybrid (RRF) | 0.333 | 0.417 | 0.364 | 0.366 |
| Random control | 0.000 | 0.083 | 0.021 | 0.036 |

## Refusal policy

Chosen threshold **tau = 0.369** — selected ONCE on the SET A (dev) top-1 dense cosine distribution (20 in-scope questions vs 2 out-of-scope probes), via the clean-gap rule in `policy.py` (overlap fallback: tau = in-scope min − margin 0.02; NOT a percentile rule). The same tau is then applied unchanged to each split:

- **SET A (dev, in-sample sanity)**: false refusals 20/20 (rate 0.000); probes refused 0/2 (rate 0.000).
- **SET B (held-out, headline)**: false refusals 12/12 (rate 0.000); probes refused 0/2 (rate 0.000).

## Chunk-size sensitivity (400 vs 800 tokens) — dev

| Retriever | Recall@1 (400) | Recall@5 (400) | MRR (400) | nDCG@5 (400) |
| --- | --- | --- | --- | --- |
| BM25 (sparse) | 0.800 | 1.000 | 0.900 | 0.926 |
| Dense (MiniLM) | 0.250 | 0.550 | 0.348 | 0.397 |
| Hybrid (RRF) | 0.400 | 0.650 | 0.502 | 0.539 |
| Random control | 0.000 | 0.000 | 0.000 | 0.000 |

Gold chunks re-mapped to the 400-token granularity: 20/20 (exact evidence substring: 20; unmapped: 0).

## Chunk-size sensitivity (400 vs 800 tokens) — heldout

| Retriever | Recall@1 (400) | Recall@5 (400) | MRR (400) | nDCG@5 (400) |
| --- | --- | --- | --- | --- |
| BM25 (sparse) | 0.417 | 0.917 | 0.646 | 0.715 |
| Dense (MiniLM) | 0.250 | 0.333 | 0.292 | 0.303 |
| Hybrid (RRF) | 0.333 | 0.500 | 0.367 | 0.398 |
| Random control | 0.000 | 0.083 | 0.042 | 0.053 |

Gold chunks re-mapped to the 400-token granularity: 12/12 (exact evidence substring: 12; unmapped: 0).

## Grounded generation (Qwen2.5-0.5B-Instruct, CPU) — SET B held-out only

Metrics below are **documented PROXIES**, not human evaluation. Citation metrics measure EXISTENCE/range only — **not entailment**: no NLI/entailment model is run, so a cited passage is never verified to support the claim. Groundedness F1's recall denominator is ALL unique non-stopword tokens of the cited 800-token excerpts, so read F1 alongside answer_precision (same numerator, answer-token denominator).

- Evaluation set: **heldout (SET B, data/qa/qa_set_eval.jsonl)** — ALL 12 in-scope questions (no subsetting) + 2 refusal probes
- Citation existence: **1.000** (12/12 non-refusal answers cite >= 1 provided excerpt)
- Fabricated citations (bracketed id outside the provided range, counted before filtering): **0/12** answers (rate 0.000)
- Lexical groundedness F1 (answer vs cited excerpts): mean **0.020**, median 0.006
- Answer precision (answer-token denominator): mean **0.425**, median 0.345
- Model-layer false refusals on in-scope questions: 0/12
- Retrieval-layer gate on held-out probes: 0/2 refused (B-OOS-01: ANSWERED; B-OOS-02: ANSWERED)
- Total generation wall time: 101.0 s (mean 7.2 s / answer, greedy decoding)

### Verbatim examples (from the results table in metrics_generation.json)

**GOOD — B-09** — *In the most recent Beige Book release included in this corpus, how many of the twelve Federal Reserve Districts described overall activity as expanding at a slight-to-modest pace, and which release is that?*

> [3]

citations=[3], fabricated=[], citation_valid=True, refusal=False, groundedness_f1=0.006, answer_precision=1.000


**GOOD — B-10** — *What does the newest Beige Book report in the corpus say about whether the current pickup in activity breaks from the pattern of the preceding three report cycles?*

> [3]

citations=[3], fabricated=[], citation_valid=True, refusal=False, groundedness_f1=0.006, answer_precision=1.000


**FLAWED — B-02** — *One tourism business in the Fourth District described a steep annual drop in guests arriving from a neighboring country. How large was the decline it reported?*

> [1]

citations=[1], fabricated=[], citation_valid=True, refusal=False, groundedness_f1=0.000, answer_precision=0.000


**FLAWED — B-03** — *Which disruption outside the housing market did First District contacts in January 2026 partly blame for weaker home purchases?*

> [3]

citations=[3], fabricated=[], citation_valid=True, refusal=False, groundedness_f1=0.000, answer_precision=0.000


## Honest scope

This is an evaluation study of retrieval and grounded-generation components over 26 Beige Book documents. It is NOT a product, NOT financial advice, and the generation metrics are lexical/structural proxies — hallucination risk remains and no human evaluation panel was run (see README §14). Both QA sets are researcher-authored (single builder, no external annotators).
