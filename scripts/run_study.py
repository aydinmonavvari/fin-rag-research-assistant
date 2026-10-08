#!/usr/bin/env python3
"""Study CLI: fetch -> index -> eval-retrieval -> eval-generation -> report.

QA design (anti-circularity): two researcher-authored sets —
  SET A "dev"    (data/qa/qa_set_dev.jsonl):    tau selection + debugging only;
  SET B "heldout"(data/qa/qa_set_eval.jsonl):  final evaluation, never used to
                                                 pick tau.
No metric computed on SET A is a final evaluation; artifacts label every block
"dev" or "heldout" explicitly.

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
from fin_rag_research_assistant.beigebook import (  # noqa: E402
    build_corpus_manifest,
    verify_corpus_manifest,
    write_corpus_manifest,
)
from fin_rag_research_assistant.chunk import load_chunks  # noqa: E402
from fin_rag_research_assistant.corpus import (  # noqa: E402
    build_and_save_chunks,
    build_corpus,
    load_corpus,
)
from fin_rag_research_assistant.evaluation import (  # noqa: E402
    build_retrievers,
    map_questions_to_chunks,
    metrics_for_run,
    per_question_mrr,
    recall_curve,
    refusal_eval_report,
    run_retriever,
    select_refusal_threshold,
    verify_gold_evidence,
)
from fin_rag_research_assistant.generation import (  # noqa: E402
    evaluate_generation_result,
    generate_answer,
    load_generation_model,
    refusal_correctness,
)
from fin_rag_research_assistant.policy import should_refuse  # noqa: E402
from fin_rag_research_assistant.qa import (  # noqa: E402
    load_qa_split,
    split_questions_probes,
)
from fin_rag_research_assistant.reporting import (  # noqa: E402
    QA_AUTHORSHIP_NOTE,
    figure_groundedness,
    figure_metric_bars,
    figure_per_question_mrr,
    figure_recall_curves,
    figure_score_distribution,
    write_json,
    write_summary,
)

REPORTS_DIR = config.REPORTS_DIR
SPLITS = ("dev", "heldout")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_splits() -> dict[str, tuple[list, list]]:
    """Load both QA splits; returns {split: (questions, probes)}."""
    out: dict[str, tuple[list, list]] = {}
    for split in SPLITS:
        items = load_qa_split(split)
        out[split] = split_questions_probes(items)
    return out


def _verify_corpus_or_warn(fail_on_mismatch: bool) -> dict:
    """Check cached HTML against the committed sha256 manifest.

    Warns loudly on any mismatch (or missing file); with
    ``--fail-on-corpus-mismatch`` it raises instead, because numbers produced
    from a different snapshot are not comparable to the committed reports.
    """
    report = verify_corpus_manifest()
    if report["ok"]:
        print(f"[corpus] manifest verified: {report['n_checked']}/"
              f"{report['n_documents_in_manifest']} cached files match "
              "data/corpus_manifest.json (sha256)")
        return report
    print("=" * 78)
    print("[corpus] !! CORPUS SNAPSHOT MISMATCH !!")
    for mismatch in report["mismatches"]:
        print(f"[corpus]   {mismatch['doc_id']}: sha256 {mismatch['expected_sha256'][:12]}… "
              f"-> {mismatch['actual_sha256'][:12]}… "
              f"({mismatch['expected_size_bytes']} -> {mismatch['actual_size_bytes']} bytes)")
    for doc_id in report["missing"]:
        print(f"[corpus]   {doc_id}: MISSING from data/raw/")
    print("[corpus] The cached HTML no longer matches the committed snapshot pin.")
    print("[corpus] Results from this cache are NOT comparable with reports that")
    print("[corpus] used the pinned snapshot. Re-run `fetch` + the full study and")
    print("[corpus] treat every number as a new snapshot (see README §15).")
    print("=" * 78)
    if fail_on_mismatch:
        raise SystemExit("[corpus] aborting (--fail-on-corpus-mismatch is set)")
    return report


# --- stages ------------------------------------------------------------------


def stage_fetch(args) -> None:
    corpus, provenance = build_corpus()
    for doc_id, meta in provenance["documents"].items():
        print(f"[fetch] {doc_id}: {meta['n_paragraphs']} paragraphs, "
              f"{meta['n_chars']} chars ({meta['url']})")
    manifest_path = write_corpus_manifest(build_corpus_manifest())
    report = verify_corpus_manifest()
    print(f"[fetch] corpus manifest rebuilt: {manifest_path} "
          f"({report['n_checked']}/{report['n_documents_in_manifest']} files hash-verified)")


def stage_index(args) -> None:
    _verify_corpus_or_warn(args.fail_on_corpus_mismatch)
    corpus, _ = load_corpus()
    stats: dict = {}
    for size in (config.CHUNK_SIZE_TOKENS, config.CHUNK_SIZE_SMALL_TOKENS):
        chunks = build_and_save_chunks(corpus, chunk_size=size, overlap=config.CHUNK_OVERLAP_TOKENS)
        per_doc = {
            doc_id: {
                "n_paragraphs": len(text.split("\n\n")),
                "n_chars": len(text),
                "n_chunks": sum(1 for c in chunks if c.doc_id == doc_id),
                "n_tokens": sum(c.n_tokens for c in chunks if c.doc_id == doc_id),
            }
            for doc_id, text in sorted(corpus.items())
        }
        stats[f"chunking_{size}"] = per_doc
        print(f"[index] chunk size {size}: {len(chunks)} chunks "
              f"({ {t: per_doc[t]['n_chunks'] for t in per_doc} })")
    write_json(
        REPORTS_DIR / "index_stats.json",
        {"stage": "index", "generated_at": _utc(), **stats},
    )


def _eval_split(
    retrievers: dict,
    questions: list,
    probes: list,
    split: str,
    depth: int,
) -> tuple[dict, dict]:
    """Run every retriever on one split's questions; score probes with dense."""
    runs = {name: run_retriever(retriever, questions, top_k=depth)
            for name, retriever in retrievers.items()}
    table = {name: metrics_for_run(run, questions) for name, run in runs.items()}
    curves = {name: recall_curve(run, questions) for name, run in runs.items()}
    per_q = {name: per_question_mrr(run, questions) for name, run in runs.items()}
    dense = retrievers.get("dense")
    probe_scores: dict[str, float] = {}
    if dense is not None:
        probe_scores = {
            probe.qid: float(dense.retrieve(probe.question, top_k=1)[0].score)
            for probe in probes
        }
    block = {
        "qa_set": str(
            (config.QA_DEV_PATH if split == "dev" else config.QA_EVAL_PATH)
            .relative_to(config.PROJECT_ROOT)
        ),
        "n_questions": len(questions),
        "n_probes": len(probes),
        "question_ids": [q.qid for q in questions],
        "probe_ids": [p.qid for p in probes],
        "metrics": table,
        "recall_curves": curves,
        "per_question_mrr": per_q,
        "probe_dense_scores": probe_scores,
    }
    return block, runs


def stage_eval_retrieval(args) -> None:
    _verify_corpus_or_warn(args.fail_on_corpus_mismatch)
    chunks = load_chunks(config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl")
    splits = _load_splits()
    dev_questions, dev_probes = splits["dev"]
    held_questions, held_probes = splits["heldout"]

    gold_check = {
        "dev": verify_gold_evidence(chunks, dev_questions),
        "heldout": verify_gold_evidence(chunks, held_questions),
    }
    for split, check in gold_check.items():
        print(f"[eval-retrieval] gold verification ({split}): "
              f"{check['n_verified']}/{check['n_total']} "
              f"(missing: {check['missing_qids']})")
        if check["missing_qids"]:
            raise ValueError(
                f"unverifiable gold evidence in {split} set: {check['missing_qids']} — "
                "the QA instrument must be fixed before evaluation"
            )

    retrievers = build_retrievers(chunks, with_dense=not args.no_dense)
    depth = config.RETRIEVAL_EVAL_DEPTH
    blocks: dict[str, dict] = {}
    runs_by_split: dict[str, dict] = {}
    for split, (questions, probes) in splits.items():
        block, runs = _eval_split(retrievers, questions, probes, split, depth)
        blocks[split] = block
        runs_by_split[split] = runs
        bm25 = block["metrics"].get("bm25", {})
        print(f"[eval-retrieval] {split}: bm25 recall@1={bm25.get('recall@1', 0.0):.2f} "
              f"recall@5={bm25.get('recall@5', 0.0):.2f} mrr={bm25.get('mrr', 0.0):.3f} "
              f"(n={block['n_questions']})")

    # tau is selected ONCE, on the DEV (SET A) distribution only.
    refusal: dict = {"tau": None, "note": "dense retriever unavailable"}
    if "dense" in retrievers:
        selection = select_refusal_threshold(
            runs_by_split["dev"]["dense"],
            dev_questions,
            blocks["dev"]["probe_dense_scores"],
        )
        tau = selection["tau"]
        refusal = {
            **selection,
            "selection_rule": "clean-gap rule (see policy.pick_refusal_threshold); "
                              "overlap fallback = in_min - margin(0.02)",
            "selection_set": "dev (SET A) — data/qa/qa_set_dev.jsonl",
        }
        print(f"[eval-retrieval] refusal tau = {tau} (selected on dev/SET A only)")
        # Apply the SAME tau to each split's own distribution. dev = in-sample
        # sanity check; heldout = the headline refusal result. These never
        # re-pick tau.
        dev_eval = refusal_eval_report(
            runs_by_split["dev"]["dense"],
            dev_questions, blocks["dev"]["probe_dense_scores"], tau, "dev (in-sample sanity)",
        )
        held_eval = refusal_eval_report(
            runs_by_split["heldout"]["dense"],
            held_questions, blocks["heldout"]["probe_dense_scores"], tau,
            "heldout (headline)",
        )
        for split, report in (("dev", dev_eval), ("heldout", held_eval)):
            blocks[split]["refusal_eval"] = report
            print(f"[eval-retrieval] refusal @ tau ({report['split']}): "
                  f"false refusals {report['in_scope_passed']}/{report['in_scope_n']}, "
                  f"probes refused {report['probes_refused']}/{report['probe_n']}")
        refusal["eval_dev"] = dev_eval
        refusal["eval_heldout"] = held_eval

    # Chunk-size sensitivity, re-mapping gold chunks per split (labeled).
    sensitivity: dict[str, dict] = {}
    if args.sensitivity and not args.no_dense:
        small_path = config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_SMALL_TOKENS}.jsonl"
        chunks_small = load_chunks(small_path)
        for split, (questions, _probes) in splits.items():
            mapped, gold_map = map_questions_to_chunks(questions, chunks_small)
            retrievers_small = build_retrievers(chunks_small, with_dense=True)
            table_small: dict[str, dict] = {}
            for name, retriever in retrievers_small.items():
                run = run_retriever(retriever, mapped, top_k=config.TOP_K)
                table_small[name] = metrics_for_run(run, mapped)
            sensitivity[split] = {
                "chunk_size": config.CHUNK_SIZE_SMALL_TOKENS,
                "metrics": table_small,
                "gold_mapping": gold_map,
            }
            print(f"[eval-retrieval] 400-token sensitivity ({split}): "
                  f"{gold_map['n_mapped']}/{gold_map['n_questions']} gold chunks re-mapped")

    payload = {
        "stage": "eval-retrieval",
        "generated_at": _utc(),
        "split_definitions": {
            "dev": "SET A (data/qa/qa_set_dev.jsonl) — tau selection + debugging only; "
                   "NOT a final evaluation",
            "heldout": "SET B (data/qa/qa_set_eval.jsonl) — held-out final evaluation; "
                       "never used for threshold selection",
        },
        "qa_authorship_note": QA_AUTHORSHIP_NOTE,
        "gold_verification": gold_check,
        "splits": blocks,
        "refusal": refusal,
        "sensitivity_400": sensitivity,
    }
    write_json(REPORTS_DIR / "metrics_retrieval.json", payload)

    figure_metric_bars(blocks)
    figure_recall_curves({s: blocks[s]["recall_curves"] for s in SPLITS})
    if "dense" in retrievers:
        figure_score_distribution(refusal)
    figure_per_question_mrr(
        blocks["heldout"]["per_question_mrr"], blocks["heldout"]["question_ids"],
        title_suffix="SET B (held-out)",
    )
    print("[eval-retrieval] figures 1, 2, 3, 5 written to figures/")


def stage_eval_generation(args) -> None:
    """Grounded generation on ALL held-out (SET B) in-scope questions + probes.

    There is no question subsetting: the earlier first-8-questions subset
    (which silently sampled only the 202510 release) was removed as an
    undisclosed selection bias. SET A is not passed through the generator; its
    role ends at tau selection.
    """
    chunks = load_chunks(config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl")
    held_questions, held_probes = split_questions_probes(load_qa_split("heldout"))
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    retrievers = build_retrievers(chunks, with_dense=not args.no_dense)
    hybrid = retrievers.get("hybrid") or retrievers["bm25"]
    dense = retrievers.get("dense")

    tau_path = REPORTS_DIR / "metrics_retrieval.json"
    tau = None
    if tau_path.exists():
        tau = json.loads(tau_path.read_text(encoding="utf-8"))["refusal"].get("tau")
    print(f"[eval-generation] held-out set: all {len(held_questions)} in-scope questions "
          f"+ {len(held_probes)} probes; tau = {tau} (selected on dev)")

    model, tokenizer = load_generation_model(config.GEN_MODEL_ID)
    results: list[dict] = []
    for item in held_questions + held_probes:
        is_question = item.qtype != "out_of_scope"
        retrieved = hybrid.retrieve(item.question, top_k=config.GEN_TOP_K)
        excerpts = [by_id[r.chunk_id].text for r in retrieved]
        raw, latency = generate_answer(item.question, excerpts, model, tokenizer)
        scored = evaluate_generation_result(
            item.qid, item.question, excerpts, [r.chunk_id for r in retrieved], raw, latency
        )
        row = asdict(scored)
        row.pop("prompt_excerpts")  # reproducible from chunk ids; keeps the report small
        row["qtype"] = item.qtype
        row["type"] = "question" if is_question else "refusal_probe"
        row["expected_answer"] = item.answer if is_question else None
        dense_top = float(dense.retrieve(item.question, top_k=1)[0].score) if dense else None
        row["dense_top_score"] = dense_top
        row["policy_decision"] = (
            "refuse" if (dense_top is not None and tau is not None
                         and should_refuse(dense_top, tau)) else "answer"
        )
        results.append(row)
        fabricated = row["fabricated_citations"]
        print(f"[eval-generation] {item.qid}: refusal={row['used_refusal']} "
              f"citations={row['citations']} fabricated={fabricated} "
              f"f1={row['groundedness_f1']:.3f} precision={row['answer_precision']:.3f} "
              f"latency={latency:.1f}s")

    in_scope = [r for r in results if r["type"] == "question"]
    nonrefusal = [r for r in in_scope if not r["used_refusal"]]
    n_exist = sum(1 for r in nonrefusal if r["citation_existence"])
    n_fabricated = sum(1 for r in nonrefusal if r["fabricated_citations"])
    f1_values = [r["groundedness_f1"] for r in in_scope]
    prec_values = [r["answer_precision"] for r in in_scope]
    refusal_rows = refusal_correctness(results, {p.qid: True for p in held_probes})
    summary = {
        "evaluation_set": "heldout (SET B, data/qa/qa_set_eval.jsonl)",
        "n_questions": len(in_scope),
        "n_refusal_probes": len(results) - len(in_scope),
        # citation_existence_rate: every non-refusal answer cites >= 1 excerpt
        # that was actually provided. NOT entailment — no NLI/entailment model
        # runs anywhere in this study; a cited passage is never verified to
        # support the claim.
        "n_citation_existence": n_exist,
        "citation_existence_rate": n_exist / len(nonrefusal) if nonrefusal else 0.0,
        "n_fabricated_citations": n_fabricated,
        "fabricated_citation_rate": (
            n_fabricated / len(nonrefusal) if nonrefusal else 0.0
        ),
        "n_model_false_refusals": sum(1 for r in in_scope if r["used_refusal"]),
        "groundedness_mean": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "groundedness_median": median(f1_values) if f1_values else 0.0,
        "answer_precision_mean": sum(prec_values) / len(prec_values) if prec_values else 0.0,
        "answer_precision_median": median(prec_values) if prec_values else 0.0,
        "refusal_probes_correct": refusal_rows["n_correct"],
        "total_latency_s": sum(r["latency_s"] for r in results),
        "mean_latency_s": sum(r["latency_s"] for r in results) / len(results),
    }
    good = sorted(
        (r for r in in_scope if r["citation_valid"] and not r["used_refusal"]),
        key=lambda r: -r["answer_precision"],
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
        "citation_metrics_semantics": (
            "citation_existence_rate measures EXISTENCE only: a non-refusal answer "
            "cites at least one excerpt that was provided. fabricated_citation_rate "
            "counts non-refusal answers emitting a bracketed id outside the provided "
            "range (counted before filtering). NEITHER metric verifies that the "
            "cited passage supports the claim — no entailment/NLI model is run."
        ),
        "groundedness_semantics": (
            "groundedness_f1 recall denominator = ALL unique non-stopword tokens of "
            "the cited 800-token excerpts, so recall is structurally small; read F1 "
            "alongside answer_precision (same numerator, answer-token denominator)."
        ),
        "results": results,
        "summary": summary,
        "refusal_probes": refusal_rows,
        "examples": examples,
    }
    write_json(REPORTS_DIR / "metrics_generation.json", payload)
    figure_groundedness(results)
    print(f"[eval-generation] citation existence {summary['n_citation_existence']}/"
          f"{summary['n_questions']}, fabricated citations {summary['n_fabricated_citations']}/"
          f"{summary['n_questions']}, groundedness mean {summary['groundedness_mean']:.3f}, "
          f"answer precision mean {summary['answer_precision_mean']:.3f}, "
          f"refusal probes correct {summary['refusal_probes_correct']}/"
          f"{summary['n_refusal_probes']}")


def stage_report(args) -> None:
    def _read(name: str) -> dict:
        return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))

    manifest_summary: dict = {}
    try:
        manifest = _read(str(config.CORPUS_MANIFEST_PATH))
        manifest_summary = {
            "path": "data/corpus_manifest.json",
            "algorithm": manifest.get("algorithm"),
            "n_documents": manifest.get("n_documents"),
            "note": "sha256 pin of the cached HTML snapshot used for all numbers; "
                    "run_study verifies it before every evaluation stage",
        }
    except FileNotFoundError:
        manifest_summary = {"path": "data/corpus_manifest.json", "present": False}

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
            "qa_dev_path": "data/qa/qa_set_dev.jsonl",
            "qa_eval_path": "data/qa/qa_set_eval.jsonl",
            "tau_selection_set": "dev (SET A) only",
        },
        "corpus_manifest": manifest_summary,
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
    parser.add_argument("--fail-on-corpus-mismatch", action="store_true",
                        help="exit nonzero instead of warning when cached HTML does not "
                             "match data/corpus_manifest.json (sha256)")
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
