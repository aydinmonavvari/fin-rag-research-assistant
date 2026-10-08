"""QA-set schema, loader and validation (offline, JSONL fixture + repo data)."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant import config
from fin_rag_research_assistant.qa import (
    load_qa_set,
    load_qa_split,
    save_qa_set,
    split_questions_probes,
    validate_item,
)


def _item(**overrides):
    base = {
        "qid": "Q-01",
        "source": "202601-new-york",
        "question": "What was total net sales?",
        "answer": "$391,035 million",
        "gold_chunk_id": "AAPL:c0004:1a2b3c4d",
        "evidence": "Total net sales | $391,035",
        "qtype": "numeric",
    }
    base.update(overrides)
    return base


def test_validate_item_accepts_complete_record():
    item = validate_item(_item())
    assert item.qid == "Q-01"
    assert item.source == "202601-new-york"
    assert item.qtype == "numeric"


def test_validate_item_missing_field_raises():
    broken = _item()
    del broken["answer"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_item(broken)


def test_validate_item_unexpected_field_raises():
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_item(_item(surprise="x"))


def test_validate_item_bad_qtype_raises():
    with pytest.raises(ValueError, match="qtype"):
        validate_item(_item(qtype="vibes"))


def test_validate_item_empty_evidence_raises():
    with pytest.raises(ValueError, match="non-empty"):
        validate_item(_item(evidence="   "))


def test_validate_item_out_of_scope_needs_no_gold_chunk():
    item = validate_item(_item(qid="OOS-01", source="", gold_chunk_id="", qtype="out_of_scope",
                               question="What were U.S. new-home sales in 1998?"))
    assert item.qtype == "out_of_scope"


def test_load_qa_set_roundtrip(tmp_path):
    items = [validate_item(_item(qid=f"Q-{i:02d}")) for i in range(3)]
    path = tmp_path / "qa.jsonl"
    save_qa_set(items, path)
    loaded = load_qa_set(path)
    assert [i.qid for i in loaded] == ["Q-00", "Q-01", "Q-02"]


def test_load_qa_set_duplicate_qid_raises(tmp_path):
    items = [validate_item(_item(qid="Q-00")), validate_item(_item(qid="Q-00"))]
    path = tmp_path / "dup.jsonl"
    save_qa_set(items, path)
    with pytest.raises(ValueError, match="duplicate qid"):
        load_qa_set(path)


def test_load_qa_set_empty_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty QA set"):
        load_qa_set(path)


def test_validate_item_accepts_set_b_qtypes():
    for qtype in ("mixed", "temporal", "citation_trap"):
        item = validate_item(_item(qid=f"B-{qtype}", qtype=qtype))
        assert item.qtype == qtype


def test_load_qa_split_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown QA split"):
        load_qa_split("validation")


def test_load_qa_split_resolves_split_specific_files(tmp_path):
    dev_item = validate_item(_item(qid="Q-01"))
    held_item = validate_item(_item(qid="B-01", qtype="temporal"))
    save_qa_set([dev_item], tmp_path / "qa_set_dev.jsonl")
    save_qa_set([held_item], tmp_path / "qa_set_eval.jsonl")
    assert [i.qid for i in load_qa_split("dev", qa_dir=tmp_path)] == ["Q-01"]
    assert [i.qid for i in load_qa_split("heldout", qa_dir=tmp_path)] == ["B-01"]


def test_split_questions_probes_partitions_by_qtype():
    items = [
        validate_item(_item(qid="Q-01")),
        validate_item(_item(qid="B-11", qtype="citation_trap")),
        validate_item(_item(qid="B-OOS-01", source="", gold_chunk_id="", qtype="out_of_scope",
                            question="What was X?")),
    ]
    questions, probes = split_questions_probes(items)
    assert [q.qid for q in questions] == ["Q-01", "B-11"]
    assert [p.qid for p in probes] == ["B-OOS-01"]


def test_repo_qa_files_follow_two_set_scheme():
    """Integration with the tracked study instruments (both authored by the
    researcher): SET A dev = 20 + 2; SET B held-out = 12 + 2; no qid overlaps
    between sets; every held-out in-scope item is fully specified."""
    dev = load_qa_split("dev")
    held = load_qa_split("heldout")
    dev_questions, dev_probes = split_questions_probes(dev)
    held_questions, held_probes = split_questions_probes(held)
    assert (len(dev_questions), len(dev_probes)) == (20, 2)
    assert (len(held_questions), len(held_probes)) == (12, 2)
    dev_ids = {q.qid for q in dev}
    held_ids = {q.qid for q in held}
    assert not dev_ids & held_ids, "dev and held-out qids must not overlap"
    for q in held_questions:
        assert q.answer.strip() and q.gold_chunk_id and q.evidence.strip()
    held_types = {q.qtype for q in held_questions}
    assert held_types == {"numeric", "categorical", "definitional", "mixed",
                          "temporal", "citation_trap"}


def test_set_b_evidence_verifies_against_chunked_corpus():
    """Every SET B (and SET A) evidence snippet must occur verbatim in its gold
    chunk — the anti-fabrication guard for the researcher-authored sets.
    Skipped when the parsed chunk cache is absent (fresh CI checkout)."""
    from pathlib import Path

    from fin_rag_research_assistant.chunk import load_chunks
    from fin_rag_research_assistant.evaluation import verify_gold_evidence

    chunks_path = config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl"
    if not Path(chunks_path).exists():
        pytest.skip("chunk cache not built yet; run `run_study.py index`")
    chunks = load_chunks(chunks_path)
    for split in ("dev", "heldout"):
        questions, _probes = split_questions_probes(load_qa_split(split))
        check = verify_gold_evidence(chunks, questions)
        assert check["missing_qids"] == [], f"{split}: unverifiable evidence"
        assert check["n_verified"] == check["n_total"]
