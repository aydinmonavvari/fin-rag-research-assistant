"""Federal Reserve Beige Book access: release documents, cached download.

Corpus source (documented pivot): the study was originally designed around SEC
10-K filings, but SEC's edge servers return HTTP 403 to this build
environment's datacenter egress IP (a block that persisted across long
backoffs). The corpus therefore uses the Federal Reserve **Beige Book** —
real, public-domain (U.S. government) economic reports released eight times
per year, one national summary plus twelve district reports per release. Every
document's URL and fetch date is recorded in data/processed/provenance.json.

All network access is polite (descriptive User-Agent, sleep between requests)
and cached: nothing is fetched twice in one checkout.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fin_rag_research_assistant import config


@dataclass(frozen=True)
class DocRef:
    """Pointer to one Beige Book document (summary or district report)."""

    release: str  # e.g. "202601"
    doc_id: str  # e.g. "202601-new-york" or "202601-summary"
    title: str  # human-readable label for provenance
    url: str

    @property
    def cache_name(self) -> str:
        return f"beigebook_{self.doc_id}.html"


class BeigeBookError(RuntimeError):
    """Raised when a release index cannot be parsed."""


def _http_get(url: str) -> str:
    """GET with the declared UA and polite retry/backoff on throttling."""
    delays_s = (30.0, 60.0, 120.0)
    payload = b""
    for attempt, delay in enumerate((0.0, *delays_s)):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(
            url, headers={"User-Agent": config.POLITE_USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {403, 429, 500, 502, 503} or attempt == len(delays_s):
                raise
    return payload.decode("utf-8", errors="replace")


def release_doc_refs(release: str) -> list[DocRef]:
    """The 13 documents of one Beige Book release (national summary + 12 districts)."""
    refs = [
        DocRef(
            release=release,
            doc_id=f"{release}-summary",
            title=f"Beige Book {release} — National Summary",
            url=f"{config.BB_BASE_URL}beigebook{release}-summary.htm",
        )
    ]
    for slug, name in config.BB_DISTRICTS:
        refs.append(
            DocRef(
                release=release,
                doc_id=f"{release}-{slug}",
                title=f"Beige Book {release} — {name} (district report)",
                url=f"{config.BB_BASE_URL}beigebook{release}-{slug}.htm",
            )
        )
    return refs


def all_doc_refs() -> list[DocRef]:
    """Documents for every release in TARGET_RELEASES, sorted for determinism."""
    refs: list[DocRef] = []
    for release in config.TARGET_RELEASES:
        refs.extend(release_doc_refs(release))
    return sorted(refs, key=lambda r: r.doc_id)


def fetch_doc(ref: DocRef, cache_dir: Path) -> Path:
    """Download one document (cache-first) and return the cached HTML path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / ref.cache_name
    if cache_path.exists():
        return cache_path
    payload = _http_get(ref.url)
    cache_path.write_text(payload, encoding="utf-8")
    time.sleep(config.REQUEST_SLEEP_S)  # polite rate limit between documents
    return cache_path


def fetch_all_docs(cache_dir: Path | None = None) -> dict[str, tuple[DocRef, Path]]:
    """Fetch every document in TARGET_RELEASES; returns {doc_id: (ref, html_path)}."""
    cache_dir = Path(cache_dir) if cache_dir else config.RAW_DIR
    out: dict[str, tuple[DocRef, Path]] = {}
    for ref in all_doc_refs():
        out[ref.doc_id] = (ref, fetch_doc(ref, cache_dir))
    return out


def load_cached_index(cache_dir: Path) -> dict[str, int]:
    """Diagnostic helper: byte sizes of cached documents (for provenance checks)."""
    cache_dir = Path(cache_dir)
    sizes: dict[str, int] = {}
    for path in sorted(cache_dir.glob("beigebook_*.html")):
        sizes[path.stem.replace("beigebook_", "")] = path.stat().st_size
    return sizes


def summary_fallback_note(cache_dir: Path) -> str:
    """Returns a short note about cached state; used by the fetch stage log."""
    sizes = load_cached_index(cache_dir)
    return json.dumps(sizes, indent=2, sort_keys=True)
