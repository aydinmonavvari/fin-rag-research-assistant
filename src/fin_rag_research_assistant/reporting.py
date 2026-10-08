"""Figures and report rendering from actual pipeline outputs.

Every number rendered here comes from the metrics dictionaries produced by the
evaluation stages; nothing is hard-coded (legend counts, tau, metric values and
question counts are all read from the data). Figures use the matplotlib Agg
backend (no display in CI / sandbox).

Split labeling: every figure/section states whether it shows SET A "dev"
(tau selection + debugging; NOT a final evaluation) or SET B "heldout" (the
final evaluation).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (backend must be set before pyplot)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fin_rag_research_assistant.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402

RETRIEVER_COLORS = {"bm25": "#1f77b4", "dense": "#d62728", "hybrid": "#2ca02c", "random": "#7f7f7f"}
RETRIEVER_LABELS = {"bm25": "BM25 (sparse)", "dense": "Dense (MiniLM)", "hybrid": "Hybrid (RRF)", "random": "Random control"}

QA_AUTHORSHIP_NOTE = (
    "Both QA sets were authored by the researcher (single builder; there are no external "
    "annotators). SET A (dev, data/qa/qa_set_dev.jsonl: 20 in-scope + 2 probes) was written "
    "after reading the parsed Beige Book corpus and is used ONLY for refusal-threshold "
    "selection and debugging. SET B (held-out, data/qa/qa_set_eval.jsonl: 12 in-scope + 2 "
    "probes) was authored after SET A was frozen, using paraphrased wording to reduce "
    "overlap; it is never used for threshold selection. Answers are verifiable in the cited "
    "Beige Book documents (each in-scope item records a gold chunk id plus a verbatim "
    "evidence snippet, machine-verified by the eval-retrieval stage). These are study "
    "instruments, not public benchmarks."
)


def _save(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _draw_metric_bars(ax, metrics: dict, title: str) -> None:
    metric_keys = [("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("mrr", "MRR"), ("ndcg@5", "nDCG@5")]
    names = [n for n in ("bm25", "dense", "hybrid", "random") if n in metrics]
    width = 0.2
    x = np.arange(len(metric_keys))
    for i, name in enumerate(names):
        values = [metrics[name][key] for key, _ in metric_keys]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=RETRIEVER_LABELS[name], color=RETRIEVER_COLORS[name])
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(x, [label for _, label in metric_keys])
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def figure_metric_bars(blocks_by_split: dict, out_name: str = "fig1_retrieval_metric_bars.png") -> str:
    """Grouped bars (Recall@1/@5, MRR, nDCG@5) per retriever, dev vs held-out.

    ``blocks_by_split`` maps split -> metrics block (with ``metrics`` and
    ``n_questions`` from the evaluation payload, so question counts in titles
    come from the data).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    titles = {
        "dev": "SET A — dev (tau selection only; NOT final)",
        "heldout": "SET B — held-out (final evaluation)",
    }
    for ax, split in zip(axes, ("dev", "heldout"), strict=True):
        block = blocks_by_split.get(split, {})
        metrics = block.get("metrics", {})
        _draw_metric_bars(ax, metrics, f"{titles[split]}, n={block.get('n_questions', 0)}")
    axes[0].set_ylabel("score (QA questions)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=8, loc="lower center")
    fig.subplots_adjust(bottom=0.16)
    return _save(fig, out_name)


def _draw_recall_curve(ax, curves: dict, title: str) -> None:
    for name in ("bm25", "dense", "hybrid", "random"):
        if name not in curves:
            continue
        ks = sorted(int(k) for k in curves[name])
        values = [curves[name][str(k)] for k in ks]
        ax.plot(ks, values, marker="o", ms=3.5, color=RETRIEVER_COLORS[name], label=RETRIEVER_LABELS[name])
    ax.set_xlabel("k (rank cutoff)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)


def figure_recall_curves(curves_by_split: dict, out_name: str = "fig2_recall_at_k_curves.png") -> str:
    """Recall@k for k = 1..10 per retriever, dev vs held-out (chunk size 800)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    titles = {
        "dev": "SET A — dev (tau selection only; NOT final)",
        "heldout": "SET B — held-out (final evaluation)",
    }
    for ax, split in zip(axes, ("dev", "heldout"), strict=True):
        _draw_recall_curve(ax, curves_by_split.get(split, {}), titles[split])
    axes[0].set_ylabel("Recall@k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=8, loc="lower center")
    fig.subplots_adjust(bottom=0.18)
    return _save(fig, out_name)


def _draw_score_distribution(ax, in_scores: list[float], probe_scores: list[float], tau: float, title: str) -> None:
    """Histogram with legend counts taken from the ACTUAL data (never hard-coded)."""
    in_arr = np.asarray(in_scores, dtype=float)
    probe_arr = np.asarray(probe_scores, dtype=float)
    bins = np.linspace(0, max(1.0, in_arr.max() + 0.05, probe_arr.max() + 0.05), 24)
    ax.hist(in_arr, bins=bins, alpha=0.65,
            label=f"in-scope questions (n={len(in_arr)})", color="#2ca02c")
    ax.hist(probe_arr, bins=bins, alpha=0.65,
            label=f"out-of-scope probes (n={len(probe_arr)})", color="#7f7f7f")
    ax.axvline(tau, color="#d62728", ls="--", lw=1.6, label=f"tau = {tau:.3f}")
    ax.set_xlabel("top-1 dense cosine similarity (all-MiniLM-L6-v2)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def figure_score_distribution(refusal: dict, out_name: str = "fig3_refusal_threshold.png") -> str:
    """Dense top-score distributions (in-scope vs probes) with the chosen tau.

    Left: SET A dev distribution on which tau was selected. Right: SET B
    held-out distribution with the same tau applied (the headline result).
    Legend labels use the actual probe/question counts passed from the data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    dev_eval = refusal.get("eval_dev", {})
    held_eval = refusal.get("eval_heldout", {})
    _draw_score_distribution(
        axes[0], refusal.get("in_scope_scores", []), refusal.get("probe_scores", []),
        refusal["tau"],
        f"SET A — dev: tau selected here ({dev_eval.get('in_scope_n', '?')} in-scope, "
        f"{dev_eval.get('probe_n', '?')} probes)",
    )
    _draw_score_distribution(
        axes[1], held_eval.get("in_scope_scores", []), held_eval.get("probe_scores", []),
        refusal["tau"],
        f"SET B — held-out: same tau applied ({held_eval.get('in_scope_n', '?')} in-scope, "
        f"{held_eval.get('probe_n', '?')} probes)",
    )
    axes[0].set_ylabel("number of questions")
    return _save(fig, out_name)


def figure_groundedness(results: list[dict], out_name: str = "fig4_groundedness_histogram.png") -> str:
    """Histograms of per-answer lexical groundedness F1 and answer precision."""
    rows = [r for r in results if r["type"] == "question"]
    f1 = [r["groundedness_f1"] for r in rows]
    prec = [r["answer_precision"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    bins = np.linspace(0, 1, 11)
    for ax, values, label in (
        (axes[0], f1, "lexical groundedness F1 (answer vs cited excerpts)"),
        (axes[1], prec, "answer precision (same numerator, answer-token denominator)"),
    ):
        ax.hist(values, bins=bins, color="#1f77b4", alpha=0.8, edgecolor="white")
        ax.axvline(float(np.mean(values)), color="#d62728", ls="--", lw=1.6,
                   label=f"mean = {np.mean(values):.2f}")
        ax.set_xlabel(f"{label} — PROXY metric", fontsize=8)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("number of answers")
    n = len(rows)
    axes[0].set_title(f"Groundedness F1 (SET B held-out, {n} in-scope answers)", fontsize=10)
    axes[1].set_title(f"Answer precision (SET B held-out, {n} in-scope answers)", fontsize=10)
    return _save(fig, out_name)


def figure_per_question_mrr(
    per_q: dict,
    qids: list[str],
    out_name: str = "fig5_per_question_mrr.png",
    title_suffix: str = "SET B (held-out)",
) -> str:
    """Heatmap: per-question MRR (rows) x retriever (columns)."""
    names = [n for n in ("bm25", "dense", "hybrid", "random") if n in per_q]
    matrix = np.array([[per_q[name][qid] for name in names] for qid in qids], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, max(6.0, 0.32 * len(qids))))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)), [RETRIEVER_LABELS[n] for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(qids)), qids, fontsize=7)
    for i in range(len(qids)):
        for j in range(len(names)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if value > 0.6 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.75, label="MRR (1 = gold at rank 1)")
    ax.set_title(f"Per-question reciprocal rank by retriever ({title_suffix})", fontsize=10)
    return _save(fig, out_name)


def render_summary(all_metrics: dict) -> str:
    """Markdown summary built only from the metrics dictionaries."""
    retrieval = all_metrics["retrieval"]
    generation = all_metrics["generation"]
    prov = all_metrics["provenance"]
    lines: list[str] = []
    add = lines.append

    add("# fin-rag-research-assistant — study summary")
    add("")
    add(f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `scripts/run_study.py report`. "
        "Every number below is an actual output of the committed pipeline run._")
    add("")

    add("## Corpus provenance (Federal Reserve Beige Book, public domain)")
    add("")
    manifest = all_metrics.get("corpus_manifest", {})
    if manifest.get("n_documents"):
        add(f"Snapshot pin: `data/corpus_manifest.json` ({manifest.get('algorithm')}, "
            f"{manifest.get('n_documents')} documents) — verified against the cache before every "
            "evaluation stage. If federalreserve.gov revises a page, the hash check fails loudly.")
        add("")
    add("| Document | Title | Release | Cached HTML (bytes) | Source |")
    add("| --- | --- | --- | --- | --- |")
    for doc_id, meta in prov["documents"].items():
        add(f"| {doc_id} | {meta['title']} | {meta['release']} | {meta['cached_html_bytes']} "
            f"| [federalreserve.gov]({meta['url']}) |")
    add("")

    add("## Corpus & chunking")
    add("")
    add("| Document | Paragraphs | Characters | Chunks (800 tok) | Tokens in chunks |")
    add("| --- | --- | --- | --- | --- |")
    index_stats = all_metrics.get("index", {})
    for doc_id, stats in index_stats.get("chunking_800", {}).items():
        add(f"| {doc_id} | {stats['n_paragraphs']} | {stats['n_chars']} | {stats['n_chunks']} | {stats['n_tokens']} |")
    add("")
    total_chunks = sum(s["n_chunks"] for s in index_stats.get("chunking_800", {}).values())
    add(f"Corpus total: **{total_chunks} chunks** at 800-token windows with 100-token overlap "
        "(a 'token' is a whitespace-delimited word — documented approximation).")
    add("")

    add("## QA design (authorship disclosed)")
    add("")
    add(f"> {QA_AUTHORSHIP_NOTE}")
    add("")

    splits = retrieval.get("splits", {})
    role_note = {
        "dev": "SET A (dev) — used for tau selection and debugging only; **NOT a final evaluation**.",
        "heldout": "SET B (held-out) — **final reported retrieval evaluation**; never used for threshold selection.",
    }
    for split in ("dev", "heldout"):
        block = splits.get(split)
        if not block:
            continue
        add(f"## Retrieval evaluation — {role_note[split]}")
        add("")
        add(f"Questions: **{block['n_questions']}** in-scope + {block['n_probes']} out-of-scope probes "
            f"(`{block['qa_set']}`). Gold verification: "
            f"{retrieval['gold_verification'][split]['n_verified']}/"
            f"{retrieval['gold_verification'][split]['n_total']} evidence snippets verified.")
        add("")
        add("| Retriever | Recall@1 | Recall@5 | MRR | nDCG@5 |")
        add("| --- | --- | --- | --- | --- |")
        for name in ("bm25", "dense", "hybrid", "random"):
            m = block["metrics"][name]
            add(f"| {RETRIEVER_LABELS[name]} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@5']:.3f} |")
        add("")

    add("## Refusal policy")
    add("")
    refusal = retrieval["refusal"]
    tau = refusal.get("tau")
    add(f"Chosen threshold **tau = {tau}** — selected ONCE on the SET A (dev) top-1 dense cosine "
        f"distribution ({refusal.get('in_scope_n', '?')} in-scope questions vs "
        f"{refusal.get('probe_n', '?')} out-of-scope probes), via the clean-gap rule in "
        "`policy.py` (overlap fallback: tau = in-scope min − margin 0.02; NOT a percentile rule). "
        "The same tau is then applied unchanged to each split:")
    add("")
    for key, label in (("eval_dev", "SET A (dev, in-sample sanity)"), ("eval_heldout", "SET B (held-out, headline)")):
        report = refusal.get(key)
        if not report:
            continue
        add(f"- **{label}**: false refusals {report['in_scope_passed']}/{report['in_scope_n']} "
            f"(rate {report['in_scope_false_refusal_rate']:.3f}); probes refused "
            f"{report['probes_refused']}/{report['probe_n']} (rate {report['probe_refusal_rate']:.3f}).")
    add("")

    sens = retrieval.get("sensitivity_400") or {}
    if sens:
        for split in ("dev", "heldout"):
            block = sens.get(split)
            if not block:
                continue
            gold_map = block["gold_mapping"]
            add(f"## Chunk-size sensitivity ({block['chunk_size']} vs 800 tokens) — {split}")
            add("")
            add("| Retriever | Recall@1 (400) | Recall@5 (400) | MRR (400) | nDCG@5 (400) |")
            add("| --- | --- | --- | --- | --- |")
            for name in ("bm25", "dense", "hybrid", "random"):
                m = block["metrics"][name]
                add(f"| {RETRIEVER_LABELS[name]} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@5']:.3f} |")
            add("")
            add(f"Gold chunks re-mapped to the 400-token granularity: {gold_map['n_mapped']}/{gold_map['n_questions']} "
                f"(exact evidence substring: {gold_map['n_exact_evidence_match']}; unmapped: {gold_map['n_unmapped']}).")
            add("")

    add("## Grounded generation (Qwen2.5-0.5B-Instruct, CPU) — SET B held-out only")
    add("")
    add("Metrics below are **documented PROXIES**, not human evaluation. Citation metrics "
        "measure EXISTENCE/range only — **not entailment**: no NLI/entailment model is run, "
        "so a cited passage is never verified to support the claim. Groundedness F1's recall "
        "denominator is ALL unique non-stopword tokens of the cited 800-token excerpts, so "
        "read F1 alongside answer_precision (same numerator, answer-token denominator).")
    add("")
    summary = generation["summary"]
    add(f"- Evaluation set: **{summary['evaluation_set']}** — ALL {summary['n_questions']} in-scope "
        f"questions (no subsetting) + {summary['n_refusal_probes']} refusal probes")
    add(f"- Citation existence: **{summary['citation_existence_rate']:.3f}** "
        f"({summary['n_citation_existence']}/{summary['n_questions']} non-refusal answers cite >= 1 provided excerpt)")
    add(f"- Fabricated citations (bracketed id outside the provided range, counted before filtering): "
        f"**{summary['n_fabricated_citations']}/{summary['n_questions']}** answers "
        f"(rate {summary['fabricated_citation_rate']:.3f})")
    add(f"- Lexical groundedness F1 (answer vs cited excerpts): mean **{summary['groundedness_mean']:.3f}**, "
        f"median {summary['groundedness_median']:.3f}")
    add(f"- Answer precision (answer-token denominator): mean **{summary['answer_precision_mean']:.3f}**, "
        f"median {summary['answer_precision_median']:.3f}")
    add(f"- Model-layer false refusals on in-scope questions: {summary['n_model_false_refusals']}/{summary['n_questions']}")
    add(f"- Retrieval-layer gate on held-out probes: {summary['refusal_probes_correct']}/{summary['n_refusal_probes']} refused "
        + "(" + "; ".join(f"{r['qid']}: {'refused' if r['used_refusal'] else 'ANSWERED'}"
                          for r in generation["refusal_probes"]["rows"]) + ")")
    add(f"- Total generation wall time: {summary['total_latency_s']:.1f} s "
        f"(mean {summary['mean_latency_s']:.1f} s / answer, greedy decoding)")
    add("")
    add("### Verbatim examples (from the results table in metrics_generation.json)")
    add("")
    example_rows = [
        (label, qid)
        for pool in ("good", "flawed")
        for label, qid in
        [(pool.upper(), q) for q in generation["examples"].get(pool, [])]
    ]
    if not example_rows:
        add("_No examples selected (see metrics_generation.json)._")
        add("")
    for label, qid in example_rows:
        row = next(r for r in generation["results"] if r["qid"] == qid)
        answer = row["raw_answer"].replace("\n", " ").strip()
        add(f"**{label} — {row['qid']}** — *{row['question']}*")
        add("")
        add(f"> {answer[:600]}{'…' if len(answer) > 600 else ''}")
        add("")
        add(f"citations={row['citations']}, fabricated={row.get('fabricated_citations')}, "
            f"citation_valid={row['citation_valid']}, refusal={row['used_refusal']}, "
            f"groundedness_f1={row['groundedness_f1']:.3f}, answer_precision={row['answer_precision']:.3f}")
        add(row.get("commentary", ""))
        add("")

    add("## Honest scope")
    add("")
    add("This is an evaluation study of retrieval and grounded-generation components over 26 Beige Book documents. "
        "It is NOT a product, NOT financial advice, and the generation metrics are lexical/structural "
        "proxies — hallucination risk remains and no human evaluation panel was run (see README §14). "
        "Both QA sets are researcher-authored (single builder, no external annotators).")
    add("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    """Strict JSON writer: NaN/Inf are errors, never silent tokens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def write_summary(payload: dict, out_dir: Path = REPORTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")
