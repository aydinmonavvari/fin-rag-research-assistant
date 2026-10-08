"""SEC EDGAR access: submissions metadata, latest 10-K lookup, cached download.

All network access is polite (descriptive User-Agent, sleep between requests)
and cached: nothing is fetched twice in one checkout.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fin_rag_research_assistant import config


@dataclass(frozen=True)
class FilingRef:
    """Pointer to one SEC filing, sufficient to rebuild its archive URL."""

    ticker: str
    cik: str  # 10-digit, zero-padded
    accession: str  # with dashes, e.g. 0000320193-25-000079
    primary_doc: str  # e.g. aapl-20250927.htm
    filing_date: str
    report_date: str
    form: str = "10-K"

    @property
    def accession_nodashes(self) -> str:
        return self.accession.replace("-", "")

    @property
    def document_url(self) -> str:
        return config.SEC_ARCHIVE_URL.format(
            cik=int(self.cik),
            accession_nodashes=self.accession_nodashes,
            primary_doc=self.primary_doc,
        )


class EdgarError(RuntimeError):
    """Raised when EDGAR metadata does not match expectations."""


def _http_get(url: str, *, binary: bool = False) -> bytes | str:
    """GET with the declared UA and polite retry/backoff on throttling.

    SEC's edge intermittently answers 403/429 to datacenter egress IPs even
    with a fully declared User-Agent; the polite response is to back off and
    retry a bounded number of times (never to raise the request rate).
    """
    delays_s = (30.0, 60.0, 120.0)
    payload: bytes
    for attempt, delay in enumerate((0.0, *delays_s)):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": config.SEC_USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {403, 429, 500, 502, 503} or attempt == len(delays_s):
                raise
    if binary:
        return payload
    return payload.decode("utf-8", errors="replace")


def fetch_submissions(cik: str, cache_dir: Path) -> dict:
    """Fetch (or load from cache) the EDGAR submissions JSON for a CIK."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"submissions_CIK{cik}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = config.SEC_SUBMISSIONS_URL.format(cik10=cik)
    payload = _http_get(url)
    cache_path.write_text(payload, encoding="utf-8")
    time.sleep(config.SEC_REQUEST_SLEEP_S)
    return json.loads(payload)


def latest_10k_ref(submissions: dict, ticker: str) -> FilingRef:
    """Find the most recent 10-K accession in an EDGAR submissions document."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    if not (len(forms) == len(accessions) == len(primary_docs) == len(dates)):
        raise EdgarError(f"{ticker}: malformed submissions arrays")
    for i, form in enumerate(forms):
        if form == "10-K":
            return FilingRef(
                ticker=ticker,
                cik=submissions["cik"],
                accession=accessions[i],
                primary_doc=primary_docs[i],
                filing_date=dates[i],
                report_date=report_dates[i] if i < len(report_dates) else "",
            )
    raise EdgarError(f"{ticker}: no 10-K found in recent filings")


def download_filing_document(ref: FilingRef, cache_dir: Path) -> Path:
    """Download (or load from cache) the primary 10-K HTML document."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ref.ticker}_{ref.accession_nodashes}.html"
    if cache_path.exists():
        return cache_path
    payload = _http_get(ref.document_url, binary=True)
    cache_path.write_bytes(payload)
    time.sleep(config.SEC_REQUEST_SLEEP_S)
    return cache_path


def fetch_all_filings(cache_dir: Path | None = None) -> dict[str, tuple[FilingRef, Path]]:
    """Fetch metadata + primary document for every filing in TARGET_FILINGS."""
    cache_dir = Path(cache_dir) if cache_dir else config.RAW_DIR
    out: dict[str, tuple[FilingRef, Path]] = {}
    for ticker, meta in config.TARGET_FILINGS.items():
        subs = fetch_submissions(meta["cik"], cache_dir)
        ref = latest_10k_ref(subs, ticker)
        path = download_filing_document(ref, cache_dir)
        out[ticker] = (ref, path)
    return out
