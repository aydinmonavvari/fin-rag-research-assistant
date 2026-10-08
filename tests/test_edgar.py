"""EDGAR metadata extraction on the locally cached submissions JSONs (offline)."""

from __future__ import annotations

import json

import pytest

from fin_rag_research_assistant.config import RAW_DIR
from fin_rag_research_assistant.edgar import FilingRef, latest_10k_ref


def _submissions(cik: str) -> dict | None:
    path = RAW_DIR / f"submissions_CIK{cik}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("cik", "prefix"),
    [("0000320193", "0000320193-"), ("0000789019", "0001193125-")],
)
def test_latest_10k_ref_from_cached_submissions(cik: str, prefix: str):
    submissions = _submissions(cik)
    if submissions is None:
        pytest.skip("cached submissions JSON not present (data/raw is git-ignored)")
    ref = latest_10k_ref(submissions, "TEST")
    assert isinstance(ref, FilingRef)
    assert ref.form == "10-K"
    assert ref.accession.startswith(prefix)
    assert ref.primary_doc.endswith(".htm")
    assert ref.filing_date and ref.report_date
    assert ref.accession_nodashes == ref.accession.replace("-", "")
    assert ref.document_url.startswith("https://www.sec.gov/Archives/edgar/data/")


def test_latest_10k_ref_raises_without_10k():
    fake = {"cik": "1", "filings": {"recent": {"form": ["8-K"], "accessionNumber": ["x"],
                                               "primaryDocument": ["y.htm"],
                                               "filingDate": ["2026-01-01"],
                                               "reportDate": ["2026-01-01"]}}}
    with pytest.raises(Exception, match="no 10-K"):
        latest_10k_ref(fake, "TEST")
