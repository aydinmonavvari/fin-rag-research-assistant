"""QA-set schema, loader and validation (two-file dev / held-out scheme).

Both QA sets are authored by the researcher (single builder — there are no
external annotators):

- SET A (``data/qa/qa_set_dev.jsonl``): the original 20 in-scope questions +
  2 out-of-scope probes, written after reading the parsed Beige Book corpus.
  Used ONLY for refusal-threshold (tau) selection and debugging — no metric
  computed on SET A is a final evaluation.
- SET B (``data/qa/qa_set_eval.jsonl``): 12 in-scope questions (2 numeric,
  2 categorical, 2 definitional, 2 mixed, 2 temporal, 2 citation-grounding
  traps) + 2 out-of-scope probes, authored AFTER SET A was frozen, with
  paraphrased question wording to reduce overlap with the corpus and with
  SET A. Used for the final (held-out) retrieval, refusal and generation
  evaluation; it is never used for threshold selection.

Each in-scope item records the id of the chunk that verifiably contains the
answer plus a verbatim evidence snippet (machine-verified by the pipeline).
This is documented as a limitation: the sets are study instruments, not public
benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fin_rag_research_assistant import config

REQUIRED_FIELDS = {
    "qid",
    "source",
    "question",
    "answer",
    "gold_chunk_id",
    "evidence",
    "qtype",
}
VALID_QTYPES = {
    "numeric",
    "categorical",
    "definitional",
    "mixed",
    "temporal",
    "citation_trap",
    "out_of_scope",
}
# qtypes whose items are in-scope (answerable, with a gold chunk + evidence):
IN_SCOPE_QTYPES = VALID_QTYPES - {"out_of_scope"}
VALID_SOURCES = {
    f"{release}-{slug}"
    for release in config.TARGET_RELEASES
    for slug, _name in config.BB_DISTRICTS
} | {f"{release}-summary" for release in config.TARGET_RELEASES}


@dataclass(frozen=True)
class QAItem:
    qid: str
    source: str
    question: str
    answer: str
    gold_chunk_id: str  # chunk that verifiably contains the answer
    evidence: str  # verbatim snippet from the gold chunk
    qtype: str  # numeric | categorical | definitional | out_of_scope

    def to_dict(self) -> dict:
        return asdict(self)


def validate_item(item: dict) -> QAItem:
    """Validate one raw QA record; raises ValueError on schema violations."""
    missing = REQUIRED_FIELDS - set(item)
    if missing:
        raise ValueError(f"QA record missing fields: {sorted(missing)}")
    if not set(item) - REQUIRED_FIELDS == set():
        extra = set(item) - REQUIRED_FIELDS
        raise ValueError(f"QA record has unexpected fields: {sorted(extra)}")
    if item["qtype"] not in VALID_QTYPES:
        raise ValueError(f"qtype '{item['qtype']}' not in {sorted(VALID_QTYPES)}")
    if item["qtype"] != "out_of_scope" and item["source"] not in VALID_SOURCES:
        raise ValueError(f"source '{item['source']}' not in {sorted(VALID_SOURCES)}")
    # Out-of-scope probes deliberately carry no answer/evidence.
    required_nonempty = (
        ["question"] if item["qtype"] == "out_of_scope"
        else ["question", "answer", "evidence"]
    )
    for field in required_nonempty:
        if not str(item[field]).strip():
            raise ValueError(f"field '{field}' must be non-empty")
    if item["qtype"] != "out_of_scope" and not item["gold_chunk_id"]:
        raise ValueError("in-scope questions require a gold_chunk_id")
    return QAItem(
        qid=item["qid"],
        source=item["source"],
        question=item["question"],
        answer=item["answer"],
        gold_chunk_id=item["gold_chunk_id"],
        evidence=item["evidence"],
        qtype=item["qtype"],
    )


def load_qa_set(path: Path) -> list[QAItem]:
    """Load and validate a JSONL QA set; qids must be unique."""
    items: list[QAItem] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON ({exc})") from exc
            item = validate_item(raw)
            if item.qid in seen:
                raise ValueError(f"line {line_no}: duplicate qid '{item.qid}'")
            seen.add(item.qid)
            items.append(item)
    if not items:
        raise ValueError(f"{path}: empty QA set")
    return items


SPLIT_PATHS = {"dev": config.QA_DEV_PATH, "heldout": config.QA_EVAL_PATH}


def load_qa_split(split: str, qa_dir: Path | None = None) -> list[QAItem]:
    """Load one named QA split: ``dev`` (SET A) or ``heldout`` (SET B).

    ``qa_dir`` overrides the directory (used by tests with tmp fixtures); the
    file names stay split-specific.
    """
    path = SPLIT_PATHS.get(split)
    if path is None:
        raise ValueError(f"unknown QA split '{split}' (expected one of {sorted(SPLIT_PATHS)})")
    if qa_dir is not None:
        path = Path(qa_dir) / path.name
    return load_qa_set(path)


def split_questions_probes(items: list[QAItem]) -> tuple[list[QAItem], list[QAItem]]:
    """Partition a QA split into (in-scope questions, out-of-scope probes)."""
    questions = [q for q in items if q.qtype in IN_SCOPE_QTYPES]
    probes = [q for q in items if q.qtype == "out_of_scope"]
    return questions, probes


def save_qa_set(items: list[QAItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
