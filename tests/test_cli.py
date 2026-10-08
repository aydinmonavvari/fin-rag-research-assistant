"""CLI smoke test: --help lists every stage; no pipeline side effects."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_study.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    for stage in ("fetch", "index", "eval-retrieval", "eval-generation", "report"):
        assert stage in result.stdout
    assert "--fail-on-corpus-mismatch" in result.stdout


def test_cli_rejects_unknown_stage():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_study.py"), "not-a-stage"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
