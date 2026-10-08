"""Reporting-writer tests: strict JSON discipline (NaN/Inf must raise)."""

from __future__ import annotations

import json

import pytest

from fin_rag_research_assistant.reporting import write_json


def test_write_json_roundtrips_payload(tmp_path):
    path = tmp_path / "report.json"
    payload = {"a": 1, "b": [1.5, "x"], "c": {"d": None}}
    write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_write_json_rejects_nan_and_inf(tmp_path):
    """allow_nan=False discipline: a NaN sneaking into metrics must crash the
    writer instead of emitting an invalid ``NaN`` token into committed JSON."""
    path = tmp_path / "nan.json"
    with pytest.raises(ValueError, match="Out of range"):
        write_json(path, {"accuracy": float("nan")})
    assert not path.exists()

    path_inf = tmp_path / "inf.json"
    with pytest.raises(ValueError, match="Out of range"):
        write_json(path_inf, {"score": float("inf")})
    assert not path_inf.exists()
