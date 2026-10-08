"""Retrieval evaluation: run every retriever over each QA split, compute metrics.

Split design (anti-circularity): SET A "dev" is used for refusal-threshold
(tau) SELECTION only; SET B "heldout" receives the final evaluation with the
tau chosen on dev. Also assembles the score-distribution data that drives the
refusal-threshold analysis and the chunk-size sensitivity comparison.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from fin_rag_research_assistant.chunk import Chunk
from fin_rag_research_assistant.config import (
    METRIC_KS,
    RANDOM_SEED,
    RECALL_CURVE_KS,
    RRF_K,
    TOP_K,
)
from fin_rag_research_assistant.metrics import evaluate_ranking, recall_at_k
from fin_rag_research_assistant.policy import (
    pick_refusal_threshold,
    refusal_threshold_report,
)
from fin_rag_research_assistant.qa import QAItem
from fin_rag_research_assistant.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RandomRetriever,
    Retrieved,
)


@dataclass
class RetrievalRun:
    retriever_name: str
    ranked_lists: dict[str, list[str]]  # qid -> ranked chunk ids
    score_lists: dict[str, list[float]]  # qid -> scores aligned with ranked ids


def run_retriever(retriever, questions: list[QAItem], top_k: int = TOP_K) -> RetrievalRun:
    """Run one retriever over all in-scope questions."""
    ranked: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for item in questions:
        results: list[Retrieved] = retriever.retrieve(item.question, top_k=top_k)
        ranked[item.qid] = [r.chunk_id for r in results]
        scores[item.qid] = [r.score for r in results]
    return RetrievalRun(retriever.name, ranked, scores)


def metrics_for_run(run: RetrievalRun, questions: list[QAItem]) -> dict[str, float]:
    gold = [item.gold_chunk_id for item in questions]
    ranked_lists = [run.ranked_lists[item.qid] for item in questions]
    return evaluate_ranking(ranked_lists, gold, ks=METRIC_KS)


def per_question_mrr(run: RetrievalRun, questions: list[QAItem]) -> dict[str, float]:
    from fin_rag_research_assistant.metrics import reciprocal_rank

    return {
        item.qid: reciprocal_rank(run.ranked_lists[item.qid], item.gold_chunk_id)
        for item in questions
    }


def build_retrievers(chunks: list[Chunk], with_dense: bool = True) -> dict:
    """Instantiate the full retriever zoo on one chunk set."""
    bm25 = BM25Retriever(chunks)
    out: dict = {"bm25": bm25}
    if with_dense:
        dense = DenseRetriever(chunks)
        out["dense"] = dense
        out["hybrid"] = HybridRetriever(bm25, dense, k=RRF_K)
    out["random"] = RandomRetriever(chunks, seed=RANDOM_SEED)
    return out


def top_dense_scores(dense_run: RetrievalRun) -> dict[str, float]:
    """Best dense cosine score per question (the refusal-policy signal)."""
    return {qid: (scores[0] if scores else float("nan"))
            for qid, scores in dense_run.score_lists.items()}


def refusal_analysis(
    dense_run: RetrievalRun,
    questions: list[QAItem],
    probe_scores: dict[str, float],
) -> dict:
    """Pick tau from the DEV score distribution and report the partition.

    ONLY to be called with the SET A (dev) dense run + SET A probes: this is
    the threshold-SELECTION step. Applying the chosen tau to another split is
    done by :func:`refusal_eval_report`, which never re-picks tau.
    """
    in_scope = [
        top_dense_scores(dense_run)[item.qid] for item in questions
        if item.qid in dense_run.score_lists
    ]
    in_scope = [s for s in in_scope if not np.isnan(s)]
    probes = [s for s in probe_scores.values() if not np.isnan(s)]
    if not probes:
        return {"tau": None, "note": "no out-of-scope probes scored yet"}
    tau = pick_refusal_threshold(in_scope, probes)
    report = refusal_threshold_report(in_scope, probes, tau)
    report["in_scope_scores"] = [round(float(s), 4) for s in in_scope]
    report["probe_scores"] = [round(float(s), 4) for s in probes]
    return report


def select_refusal_threshold(
    dense_run: RetrievalRun,
    dev_questions: list[QAItem],
    dev_probe_scores: dict[str, float],
) -> dict:
    """Explicit dev-only wrapper around :func:`refusal_analysis`.

    Kept as a named seam so the pipeline (and tests) can assert that tau is
    selected from SET A only.
    """
    report = refusal_analysis(dense_run, dev_questions, dev_probe_scores)
    report["selection_set"] = "dev (SET A)"
    return report


def refusal_eval_report(
    dense_run: RetrievalRun,
    questions: list[QAItem],
    probe_scores: dict[str, float],
    tau: float,
    split: str,
) -> dict:
    """Apply an ALREADY-SELECTED tau to one split's dense score distribution.

    Never calls :func:`pick_refusal_threshold` — the tau comes from the dev
    selection and is applied unchanged (held-out evaluation semantics).
    Returns the same partition report as ``refusal_threshold_report`` plus the
    split label and the (rounded) scores behind it.
    """
    in_scope = [
        top_dense_scores(dense_run)[item.qid] for item in questions
        if item.qid in dense_run.score_lists
    ]
    in_scope = [s for s in in_scope if not np.isnan(s)]
    probes = [s for s in probe_scores.values() if not np.isnan(s)]
    report = refusal_threshold_report(in_scope, probes, tau)
    report["split"] = split
    report["in_scope_scores"] = [round(float(s), 4) for s in in_scope]
    report["probe_scores"] = [round(float(s), 4) for s in probes]
    return report


def evaluate_retrieval(
    chunks: list[Chunk],
    questions: list[QAItem],
    with_dense: bool = True,
    top_k: int = TOP_K,
) -> tuple[dict[str, dict[str, float]], dict[str, RetrievalRun], dict]:
    """Full comparison table: retriever -> metrics. Also returns the zoo so the
    caller can reuse the (expensive) dense retriever for probe scoring."""
    retrievers = build_retrievers(chunks, with_dense=with_dense)
    table: dict[str, dict[str, float]] = {}
    runs: dict[str, RetrievalRun] = {}
    for name, retriever in retrievers.items():
        run = run_retriever(retriever, questions, top_k=top_k)
        runs[name] = run
        table[name] = metrics_for_run(run, questions)
    return table, runs, retrievers


# --- gold-evidence verification and chunk-size sensitivity -------------------

def _normalize(text: str) -> str:
    """Whitespace-normalize so evidence matches chunk text exactly."""
    return " ".join(text.split()).lower()


def verify_gold_evidence(chunks: list[Chunk], questions: list[QAItem]) -> dict:
    """Check that every gold chunk verifiably contains its evidence snippet.

    Integrity check for the authored QA set: a question counts as verified when
    the normalized evidence string occurs in the normalized text of its gold
    chunk. Results are reported (never silently dropped).
    """
    by_id = {chunk.chunk_id: _normalize(chunk.text) for chunk in chunks}
    missing = []
    verified = 0
    for item in questions:
        gold_text = by_id.get(item.gold_chunk_id)
        evidence = _normalize(item.evidence)
        if gold_text is not None and evidence in gold_text:
            verified += 1
        else:
            missing.append(item.qid)
    return {"n_total": len(questions), "n_verified": verified, "missing_qids": missing}


def locate_gold_chunk(chunks: list[Chunk], item: QAItem) -> str | None:
    """Find the chunk containing the evidence under a different chunking.

    Exact normalized-substring match first; falls back to the chunk with the
    highest token overlap with the evidence (jaccard), flagged by the caller.
    """
    evidence = _normalize(item.evidence)
    for chunk in chunks:
        if evidence in _normalize(chunk.text):
            return chunk.chunk_id
    evidence_tokens = set(evidence.split())
    best_id, best_score = None, 0.0
    for chunk in chunks:
        tokens = set(_normalize(chunk.text).split())
        union = tokens | evidence_tokens
        score = len(tokens & evidence_tokens) / len(union) if union else 0.0
        if score > best_score:
            best_id, best_score = chunk.chunk_id, score
    return best_id if best_score >= 0.5 else None


def map_questions_to_chunks(
    questions: list[QAItem], chunks: list[Chunk]
) -> tuple[list[QAItem], dict]:
    """Re-map gold chunk ids onto a different chunk granularity (400 vs 800)."""
    mapped: list[QAItem] = []
    n_exact = 0
    unmapped: list[str] = []
    for item in questions:
        gold_id = locate_gold_chunk(chunks, item)
        if gold_id is None:
            unmapped.append(item.qid)
            continue
        exact = evidence_in_chunk(chunks, gold_id, item.evidence)
        n_exact += int(exact)
        mapped.append(dataclasses.replace(item, gold_chunk_id=gold_id))
    info = {
        "n_questions": len(questions),
        "n_mapped": len(mapped),
        "n_exact_evidence_match": n_exact,
        "n_unmapped": len(unmapped),
        "unmapped_qids": unmapped,
    }
    return mapped, info


def evidence_in_chunk(chunks: list[Chunk], chunk_id: str, evidence: str) -> bool:
    by_id = {chunk.chunk_id: _normalize(chunk.text) for chunk in chunks}
    return _normalize(evidence) in by_id.get(chunk_id, "")


def recall_curve(
    run: RetrievalRun, questions: list[QAItem], ks: tuple[int, ...] = RECALL_CURVE_KS
) -> dict[str, float]:
    """Recall@k for k = 1..max depth, from one saved run."""
    gold = {item.qid: item.gold_chunk_id for item in questions}
    out: dict[str, float] = {}
    for k in ks:
        values = [recall_at_k(run.ranked_lists[item.qid], gold[item.qid], k) for item in questions]
        out[str(k)] = sum(values) / len(values)
    return out
