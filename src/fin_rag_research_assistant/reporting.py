"""Figures and report rendering from actual pipeline outputs.

Every number rendered here comes from the metrics dictionaries produced by the
evaluation stages; nothing is hard-coded. Figures use the matplotlib Agg
backend (no display in CI / sandbox).
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

QA_METHODOLOGY_NOTE = (
    "Questions and gold locations were authored during corpus preparation by the "
    "researcher; answers are verifiable in the cited filings (each gold chunk id "
    "plus a verbatim evidence snippet is recorded in data/qa/qa_set.jsonl and "
    "machine-verified by the eval-retrieval stage). This is a study instrument, "
    "not a public benchmark."
)


def _save(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_metric_bars(metrics: dict, out_name: str = "fig1_retrieval_metric_bars.png") -> str:
    """Grouped bars: Recall@1, Recall@5, MRR, nDCG@5 for every retriever."""
    metric_keys = [("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("mrr", "MRR"), ("ndcg@5", "nDCG@5")]
    names = [n for n in ("bm25", "dense", "hybrid", "random") if n in metrics]
    width = 0.2
    x = np.arange(len(metric_keys))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, name in enumerate(names):
        values = [metrics[name][key] for key, _ in metric_keys]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=RETRIEVER_LABELS[name], color=RETRIEVER_COLORS[name])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, [label for _, label in metric_keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (20-question QA set)")
    ax.set_title("Retrieval metrics by retriever (higher is better)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_name)


def figure_recall_curves(curves: dict, out_name: str = "fig2_recall_at_k_curves.png") -> str:
    """Recall@k for k = 1..10 per retriever."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name in ("bm25", "dense", "hybrid", "random"):
        if name not in curves:
            continue
        ks = sorted(int(k) for k in curves[name])
        values = [curves[name][str(k)] for k in ks]
        ax.plot(ks, values, marker="o", ms=3.5, color=RETRIEVER_COLORS[name], label=RETRIEVER_LABELS[name])
    ax.set_xlabel("k (rank cutoff)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k curves (20-question QA set, chunk size 800 tokens)")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_name)


def figure_score_distribution(refusal: dict, out_name: str = "fig3_refusal_threshold.png") -> str:
    """Dense top-score distributions (in-scope vs probes) with the chosen tau."""
    in_scores = np.asarray(refusal["in_scope_scores"], dtype=float)
    probe_scores = np.asarray(refusal["probe_scores"], dtype=float)
    tau = refusal["tau"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bins = np.linspace(0, max(1.0, in_scores.max() + 0.05, probe_scores.max() + 0.05), 24)
    ax.hist(in_scores, bins=bins, alpha=0.65, label="in-scope questions (n=20)", color="#2ca02c")
    ax.hist(probe_scores, bins=bins, alpha=0.65, label="out-of-scope / off-domain probes (n=8)", color="#7f7f7f")
    ax.axvline(tau, color="#d62728", ls="--", lw=1.6, label=f"tau = {tau:.3f}")
    ax.set_xlabel("top-1 dense cosine similarity (all-MiniLM-L6-v2)")
    ax.set_ylabel("number of questions")
    ax.set_title("Refusal threshold: dense score distributions and chosen tau")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_name)


def figure_groundedness(results: list[dict], out_name: str = "fig4_groundedness_histogram.png") -> str:
    """Histogram of per-answer lexical groundedness F1 (generation subset)."""
    values = [r["groundedness_f1"] for r in results if r["type"] == "question"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(0, 1, 11)
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(float(np.mean(values)), color="#d62728", ls="--", lw=1.6, label=f"mean = {np.mean(values):.2f}")
    ax.set_xlabel("lexical groundedness F1 (answer vs cited excerpts) - PROXY metric")
    ax.set_ylabel("number of answers")
    ax.set_title("Groundedness of generated answers (8 in-scope questions)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_name)


def figure_per_question_mrr(per_q: dict, qids: list[str], out_name: str = "fig5_per_question_mrr.png") -> str:
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
    ax.set_title("Per-question reciprocal rank by retriever")
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

    add("## Filings (SEC EDGAR provenance)")
    add("")
    add("| Ticker | Company | Form | Accession | Filed | Period | Primary document |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for ticker, meta in prov["filings"].items():
        add(f"| {ticker} | {meta['company']} | {meta['form']} | {meta['accession']} | {meta['filing_date']} "
            f"| {meta['report_date']} | [html]({meta['document_url']}) |")
    add("")

    add("## Corpus & chunking")
    add("")
    add("| Filing | Paragraphs | Characters | Chunks (800 tok) | Tokens in chunks |")
    add("| --- | --- | --- | --- | --- |")
    index_stats = all_metrics.get("index", {})
    for ticker, stats in index_stats.get("chunking_800", {}).items():
        add(f"| {ticker} | {stats['n_paragraphs']} | {stats['n_chars']} | {stats['n_chunks']} | {stats['n_tokens']} |")
    add("")
    total_chunks = sum(s["n_chunks"] for s in index_stats.get("chunking_800", {}).values())
    add(f"Corpus total: **{total_chunks} chunks** at 800-token windows with 100-token overlap "
        "(a 'token' is a whitespace-delimited word — documented approximation).")
    add("")

    add("## Retrieval evaluation (20 authored questions)")
    add("")
    add(f"> {QA_METHODOLOGY_NOTE}")
    add("")
    add("| Retriever | Recall@1 | Recall@5 | MRR | nDCG@5 |")
    add("| --- | --- | --- | --- | --- |")
    for name in ("bm25", "dense", "hybrid", "random"):
        m = retrieval["metrics"][name]
        add(f"| {RETRIEVER_LABELS[name]} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@5']:.3f} |")
    add("")

    add("## Refusal policy")
    add("")
    refusal = retrieval["refusal"]
    add(f"Chosen threshold **tau = {refusal['tau']}** on the top-1 dense cosine distribution "
        f"({refusal['in_scope_n']} in-scope questions vs {refusal['probe_n']} out-of-scope/off-domain probes). "
        f"In-scope questions passing the gate: {refusal['in_scope_passed']}/{refusal['in_scope_n']} "
        f"(false-refusal rate {refusal['in_scope_false_refusal_rate']:.3f}); probes refused: "
        f"{refusal['probes_refused']}/{refusal['probe_n']} (probe refusal rate {refusal['probe_refusal_rate']:.3f}).")
    add("")

    if retrieval.get("sensitivity_400"):
        sens = retrieval["sensitivity_400"]
        add("## Chunk-size sensitivity (400 vs 800 tokens)")
        add("")
        add("| Retriever | Recall@1 (400) | Recall@5 (400) | MRR (400) | nDCG@5 (400) |")
        add("| --- | --- | --- | --- | --- |")
        for name in ("bm25", "dense", "hybrid", "random"):
            m = sens["metrics"][name]
            add(f"| {RETRIEVER_LABELS[name]} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@5']:.3f} |")
        gold_map = sens["gold_mapping"]
        add("")
        add(f"Gold chunks re-mapped to the 400-token granularity: {gold_map['n_mapped']}/{gold_map['n_questions']} "
            f"(exact evidence substring: {gold_map['n_exact_evidence_match']}; unmapped: {gold_map['n_unmapped']}).")
        add("")

    add("## Grounded generation (Qwen2.5-0.5B-Instruct, CPU)")
    add("")
    add("Metrics below are **documented PROXIES**, not human evaluation.")
    add("")
    summary = generation["summary"]
    add(f"- Questions generated: {summary['n_questions']} (subset of 20) + {summary['n_refusal_probes']} refusal probes")
    add(f"- Citation validity: **{summary['citation_validity_rate']:.3f}** "
        f"({summary['n_citation_valid']}/{summary['n_questions']} answers cite only retrieved, in-range excerpts)")
    add(f"- Lexical groundedness F1 (answer vs cited excerpts): mean **{summary['groundedness_mean']:.3f}**, "
        f"median {summary['groundedness_median']:.3f}")
    add(f"- Refusal probes: {summary['refusal_probes_correct']}/{summary['n_refusal_probes']} correct "
        + "(" + "; ".join(f"{r['qid']}: {'refused' if r['used_refusal'] else 'ANSWERED'}"
                          for r in generation["refusal_probes"]["rows"]) + ")")
    add(f"- Total generation wall time: {summary['total_latency_s']:.1f} s "
        f"(mean {summary['mean_latency_s']:.1f} s / answer, greedy decoding)")
    add("")
    add("### Verbatim examples")
    add("")
    for label, qid in (("GOOD", generation["examples"]["good"][0]),
                       ("GOOD", generation["examples"]["good"][1]),
                       ("FLAWED", generation["examples"]["flawed"][0]),
                       ("FLAWED", generation["examples"]["flawed"][1])):
        row = next(r for r in generation["results"] if r["qid"] == qid)
        answer = row["raw_answer"].replace("\n", " ").strip()
        add(f"**{label} — {row['qid']}** — *{row['question']}*")
        add("")
        add(f"> {answer[:600]}{'…' if len(answer) > 600 else ''}")
        add("")
        add(f"citations={row['citations']}, citation_valid={row['citation_valid']}, "
            f"refusal={row['used_refusal']}, groundedness_f1={row['groundedness_f1']:.3f}")
        add(row.get("commentary", ""))
        add("")

    add("## Honest scope")
    add("")
    add("This is an evaluation study of retrieval and grounded-generation components over two filings. "
        "It is NOT a product, NOT financial advice, and the generation metrics are lexical/structural "
        "proxies — hallucination risk remains and no human evaluation panel was run (see README §14).")
    add("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(payload: dict, out_dir: Path = REPORTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")
