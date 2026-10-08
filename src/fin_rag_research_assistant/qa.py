"""QA-set schema, loader and validation.

The 20-question QA set was authored during corpus preparation by the
researcher: questions were written AFTER reading the parsed 10-K corpus, each
with the id of the chunk that verifiably contains the answer plus a verbatim
evidence snippet. This is documented as a limitation: the set is a study
instrument, not a public benchmark.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_FIELDS = {
    "qid",
    "ticker",
    "question",
    "answer",
    "gold_chunk_id",
    "evidence",
    "qtype",
}
VALID_QTYPES = {"numeric", "categorical", "definitional", "out_of_scope"}
VALID_TICKERS = {"AAPL", "MSFT"}


@dataclass(frozen=True)
class QAItem:
    qid: str
    ticker: str
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
    if item["qtype"] != "out_of_scope" and item["ticker"] not in VALID_TICKERS:
        raise ValueError(f"ticker '{item['ticker']}' not in {sorted(VALID_TICKERS)}")
    for field in ("question", "answer", "evidence"):
        if not str(item[field]).strip():
            raise ValueError(f"field '{field}' must be non-empty")
    if item["qtype"] != "out_of_scope" and not item["gold_chunk_id"]:
        raise ValueError("in-scope questions require a gold_chunk_id")
    return QAItem(
        qid=item["qid"],
        ticker=item["ticker"],
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


def save_qa_set(items: list[QAItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
