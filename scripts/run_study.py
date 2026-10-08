#!/usr/bin/env python3
"""Study CLI: fetch -> index -> eval-retrieval -> eval-generation -> report.

Every stage is idempotent and caches its outputs (network access only in
`fetch`; heavy model weights only in `eval-*`). Run stages individually or all
at once:

    python scripts/run_study.py fetch
    python scripts/run_study.py index
    python scripts/run_study.py eval-retrieval
    python scripts/run_study.py eval-generation
    python scripts/run_study.py report
    python scripts/run_study.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fin_rag_research_assistant import config  # noqa: E402
from fin_rag_research_assistant.chunk import load_chunks
from fin_rag_research_assistant.corpus import build_and_save_chunks, build_corpus, load_corpus
from fin_rag_research_assistant.evaluation import (
    evaluate_retrieval,
    map_questions_to_chunks,
    per_question_mrr,
    recall_curve,
    refusal_analysis,
    verify_gold_evidence,
)
from fin_rag_research_assistant.generation import (
    evaluate_generation_result,
    generate_answer,
    load_generation_model,
    refusal_correctness,
)
from fin_rag_research_assistant.policy import should_refuse
from fin_rag_research_assistant.qa import load_qa_set
from fin_rag_research_assistant.reporting import (
    QA_METHODOLOGY_NOTE,
    figure_groundedness,
    figure_metric_bars,
    figure_per_question_mrr,
    figure_recall_curves,
    figure_score_distribution,
    write_json,
    write_summary,
)

REPORTS_DIR = config.REPORTS_DIR


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _questions_and_probes():
    items = load_qa_set(config.QA_DIR / "qa_set.jsonl")
    questions = [q for q in items if q.qtype != "out_of_scope"]
    probes = [q for q in items if q.qtype == "out_of_scope"]
    return questions, probes


# --- stages ------------------------------------------------------------------


def stage_fetch(args) -> None:  # noqa: ARG001
    corpus, provenance = build_corpus()
    for ticker, meta in provenance["filings"].items():
        print(f"[fetch] {ticker}: accession {meta['accession']} filed {meta['filing_date']} "
              f"({meta['n_paragraphs']} paragraphs, {meta['n_chars']} chars)")


def stage_index(args) -> None:  # noqa: ARG001
    corpus, _ = load_corpus()
    stats: dict = {}
    for size in (config.CHUNK_SIZE_TOKENS, config.CHUNK_SIZE_SMALL_TOKENS):
        chunks = build_and_save_chunks(corpus, chunk_size=size, overlap=config.CHUNK_OVERLAP_TOKENS)
        per_filing = {
            ticker: {
                "n_paragraphs": len(text.split("\n\n")),
                "n_chars": len(text),
                "n_chunks": sum(1 for c in chunks if c.filing_id == ticker),
                "n_tokens": sum(c.n_tokens for c in chunks if c.filing_id == ticker),
            }
            for ticker, text in sorted(corpus.items())
        }
        stats[f"chunking_{size}"] = per_filing
        print(f"[index] chunk size {size}: {len(chunks)} chunks "
              f"({ {t: per_filing[t]['n_chunks'] for t in per_filing} })")
    write_json(
        REPORTS_DIR / "index_stats.json",
        {"stage": "index", "generated_at": _utc(), **stats},
    )


def stage_eval_retrieval(args) -> None:
    chunks = load_chunks(config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl")
    questions, probes = _questions_and_probes()
    gold_check = verify_gold_evidence(chunks, questions)
    print(f"[eval-retrieval] gold verification: {gold_check['n_verified']}/{gold_check['n_total']} "
          f"(missing: {gold_check['missing_qids']})")

    table, runs, retrievers = evaluate_retrieval(
        chunks, questions, with_dense=not args.no_dense, top_k=config.RETRIEVAL_EVAL_DEPTH
    )
    curves = {name: recall_curve(run, questions) for name, run in runs.items() if run is not None}
    per_q = {name: per_question_mrr(run, questions) for name, run in runs.items()}

    refusal: dict = {"tau": None, "note": "dense retriever unavailable"}
    if "dense" in retrievers:
        probe_scores = {
            probe.qid: float(retrievers["dense"].retrieve(probe.question, top_k=1)[0].score)
            for probe in probes
        }
        refusal = refusal_analysis(runs["dense"], questions, probe_scores)
        print(f"[eval-retrieval] refusal tau = {refusal['tau']} "
              f"(in-scope passed {refusal['in_scope_passed']}/{refusal['in_scope_n']}, "
              f"probes refused {refusal['probes_refused']}/{refusal['probe_n']})")

    sensitivity = None
    if args.sensitivity and not args.no_dense:
        small_path = config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_SMALL_TOKENS}.jsonl"
        chunks_small = load_chunks(small_path)
        mapped, gold_map = map_questions_to_chunks(questions, chunks_small)
        table_small, _runs, _zoo = evaluate_retrieval(
            chunks_small, mapped, with_dense=True, top_k=config.TOP_K
        )
        sensitivity = {"chunk_size": config.CHUNK_SIZE_SMALL_TOKENS, "metrics": table_small,
                       "gold_mapping": gold_map}
        print(f"[eval-retrieval] 400-token sensitivity: "
              f"{gold_map['n_mapped']}/{gold_map['n_questions']} gold chunks re-mapped")

    payload = {
        "stage": "eval-retrieval",
        "generated_at": _utc(),
        "n_questions": len(questions),
        "n_probes": len(probes),
        "qa_methodology_note": QA_METHODOLOGY_NOTE,
        "gold_verification": gold_check,
        "metrics": table,
        "recall_curves": curves,
        "per_question_mrr": per_q,
        "question_ids": [q.qid for q in questions],
        "refusal": refusal,
        "sensitivity_400": sensitivity,
    }
    write_json(REPORTS_DIR / "metrics_retrieval.json", payload)

    figure_metric_bars(table)
    figure_recall_curves(curves)
    if "dense" in retrievers:
        figure_score_distribution(refusal)
    figure_per_question_mrr(per_q, payload["question_ids"])
    print("[eval-retrieval] figures 1, 2, 3, 5 written to figures/")


def stage_eval_generation(args) -> None:
    chunks = load_chunks(config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl")
    questions, probes = _questions_and_probes()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    table, _runs, retrievers = evaluate_retrieval(
        chunks, questions, with_dense=not args.no_dense, top_k=config.TOP_K
    )
    del table  # generation stage re-uses only the retrievers + model
    hybrid = retrievers.get("hybrid") or retrievers["bm25"]
    dense = retrievers.get("dense")

    subset = list(config.GEN_SUBSET_QIDS) or [q.qid for q in questions[: config.GEN_SUBSET_SIZE]]
    refusal_probe_ids = [probe.qid for probe in probes if probe.qid.startswith("OOS-")]
    print(f"[eval-generation] subset: {subset}; refusal probes: {refusal_probe_ids}")

    model, tokenizer = load_generation_model(config.GEN_MODEL_ID)
    results: list[dict] = []
    for qid in subset + refusal_probe_ids:
        item = next((q for q in questions + probes if q.qid == qid), None)
        if item is None:
            raise ValueError(f"qid {qid} not found in the QA set")
        is_question = item.qtype != "out_of_scope"
        retrieved = hybrid.retrieve(item.question, top_k=config.GEN_TOP_K)
        excerpts = [by_id[r.chunk_id].text for r in retrieved]
        raw, latency = generate_answer(item.question, excerpts, model, tokenizer)
        scored = evaluate_generation_result(
            item.qid, item.question, excerpts, [r.chunk_id for r in retrieved], raw, latency
        )
        row = asdict(scored)
        row.pop("prompt_excerpts")  # reproducible from chunk ids; keeps the report small
        row["type"] = "question" if is_question else "refusal_probe"
        row["expected_answer"] = item.answer if is_question else None
        dense_top = float(dense.retrieve(item.question, top_k=1)[0].score) if dense else None
        row["dense_top_score"] = dense_top
        tau_path = REPORTS_DIR / "metrics_retrieval.json"
        if dense_top is not None and tau_path.exists():
            tau = json.loads(tau_path.read_text(encoding="utf-8"))["refusal"].get("tau")
            policy = "refuse" if (tau is not None and should_refuse(dense_top, tau)) else "answer"
            row["policy_decision"] = policy
        results.append(row)
        print(f"[eval-generation] {qid}: refusal={row['used_refusal']} "
              f"citations={row['citations']} f1={row['groundedness_f1']:.3f} "
              f"latency={latency:.1f}s")

    in_scope = [r for r in results if r["type"] == "question"]
    n_valid = sum(1 for r in in_scope if r["citation_valid"])
    f1_values = [r["groundedness_f1"] for r in in_scope]
    refusal_rows = refusal_correctness(
        results, {probe.qid: True for probe in probes if probe.qid.startswith("OOS-")}
    )
    summary = {
        "n_questions": len(in_scope),
        "n_refusal_probes": len(results) - len(in_scope),
        "n_citation_valid": n_valid,
        "citation_validity_rate": n_valid / len(in_scope) if in_scope else 0.0,
        "groundedness_mean": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "groundedness_median": median(f1_values) if f1_values else 0.0,
        "refusal_probes_correct": refusal_rows["n_correct"],
        "total_latency_s": sum(r["latency_s"] for r in results),
        "mean_latency_s": sum(r["latency_s"] for r in results) / len(results),
    }
    good = sorted(
        (r for r in in_scope if r["citation_valid"] and not r["used_refusal"]),
        key=lambda r: -r["groundedness_f1"],
    )
    flawed_pool = sorted(
        (r for r in in_scope if not (r["citation_valid"] and not r["used_refusal"])),
        key=lambda r: (r["groundedness_f1"], -r["latency_s"]),
    )
    fallback = sorted(in_scope, key=lambda r: r["groundedness_f1"])
    flawed: list[dict] = []
    for row in flawed_pool + fallback:
        if row not in flawed:
            flawed.append(row)
        if len(flawed) == 2:
            break
    examples = {
        "good": [r["qid"] for r in good[:2]],
        "flawed": [r["qid"] for r in flawed[:2]],
    }

    payload = {
        "stage": "eval-generation",
        "generated_at": _utc(),
        "model": config.GEN_MODEL_ID,
        "retriever_for_excerpts": hybrid.name,
        "gen_top_k": config.GEN_TOP_K,
        "max_new_tokens": config.GEN_MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False)",
        "metrics_are_proxies": True,
        "results": results,
        "summary": summary,
        "refusal_probes": refusal_rows,
        "examples": examples,
    }
    write_json(REPORTS_DIR / "metrics_generation.json", payload)
    figure_groundedness(results)
    print(f"[eval-generation] citation validity {summary['n_citation_valid']}/"
          f"{summary['n_questions']}, groundedness mean {summary['groundedness_mean']:.3f}, "
          f"refusal probes correct {summary['refusal_probes_correct']}/"
          f"{summary['n_refusal_probes']}")


def stage_report(args) -> None:  # noqa: ARG001
    def _read(name: str) -> dict:
        return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))

    all_metrics = {
        "project": "fin-rag-research-assistant",
        "generated_at": _utc(),
        "config": {
            "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
            "generation_model": config.GEN_MODEL_ID,
            "rrf_k": config.RRF_K,
            "random_seed": config.RANDOM_SEED,
            "gen_subset_qids": list(config.GEN_SUBSET_QIDS),
        },
        "provenance": _read(str(config.PROCESSED_DIR / "provenance.json")),
        "index": _read("index_stats.json"),
        "retrieval": _read("metrics_retrieval.json"),
        "generation": _read("metrics_generation.json"),
    }
    write_json(REPORTS_DIR / "metrics.json", all_metrics)
    write_summary(all_metrics)
    print(f"[report] wrote {REPORTS_DIR / 'metrics.json'} and {REPORTS_DIR / 'summary.md'}")


STAGES = {
    "fetch": stage_fetch,
    "index": stage_index,
    "eval-retrieval": stage_eval_retrieval,
    "eval-generation": stage_eval_generation,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=[*STAGES, "all"], help="pipeline stage to run")
    parser.add_argument("--no-dense", action="store_true",
                        help="skip dense/hybrid retrieval (BM25 + random only)")
    parser.add_argument("--no-sensitivity", dest="sensitivity", action="store_false",
                        help="skip the 400-token chunk-size sensitivity run")
    args = parser.parse_args()

    if args.stage == "all":
        for name, stage in STAGES.items():
            print(f"===== stage: {name} =====")
            stage(args)
    else:
        STAGES[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
