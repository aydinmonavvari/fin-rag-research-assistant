"""QA-set schema, loader and validation (offline, JSONL fixture)."""

from __future__ import annotations

import pytest

from fin_rag_research_assistant.qa import load_qa_set, save_qa_set, validate_item


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
