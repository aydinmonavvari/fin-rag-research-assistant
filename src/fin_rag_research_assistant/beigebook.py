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

import hashlib
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


# --- corpus snapshot manifest (sha256 pin) ------------------------------------
# The committed data/corpus_manifest.json pins the EXACT cached Beige Book HTML
# snapshot used for all reported numbers. federalreserve.gov may revise a page
# at any time; the hash check detects that, so a fresh clone that re-downloads
# different bytes is loudly flagged instead of silently producing numbers that
# are not comparable to the committed reports.


def compute_sha256(path: Path) -> str:
    """sha256 hex digest of a file's raw bytes (streamed, constant memory)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_corpus_manifest(
    raw_dir: Path | None = None,
    provenance_path: Path | None = None,
) -> dict:
    """Build the snapshot manifest from cached HTML + provenance metadata.

    Records url, release, fetched_at (from data/processed/provenance.json),
    sha256 of the file bytes and size_bytes for every cached document.
    """
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    provenance_path = (
        Path(provenance_path) if provenance_path else config.PROCESSED_DIR / "provenance.json"
    )
    refs = {ref.doc_id: ref for ref in all_doc_refs()}
    provenance: dict = {}
    fetched_at = ""
    if provenance_path.exists():
        raw_prov = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance = raw_prov.get("documents", {})
        # fetch date is recorded once per build at the provenance top level
        fetched_at = raw_prov.get("fetched_at", "")
    documents: dict[str, dict] = {}
    for path in sorted(raw_dir.glob("beigebook_*.html"), key=lambda p: p.stem):
        key = path.stem.replace("beigebook_", "")
        ref = refs.get(key)
        meta = provenance.get(key, {})
        documents[key] = {
            "url": ref.url if ref else meta.get("url", ""),
            "release": ref.release if ref else meta.get("release", key.split("-")[0]),
            "fetched_at": fetched_at,
            "sha256": compute_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "manifest_version": 1,
        "algorithm": "sha256",
        "corpus_source": "Federal Reserve Beige Book (public domain, U.S. government)",
        "note": (
            "Pins the exact cached Beige Book HTML snapshot used for every reported "
            "number. If federalreserve.gov revises a page, re-downloading produces "
            "different bytes and the hash check fails loudly: re-run `fetch`, then "
            "re-run the full study and treat all numbers as a new snapshot."
        ),
        "n_documents": len(documents),
        "documents": documents,
    }


def write_corpus_manifest(manifest: dict, path: Path | None = None) -> Path:
    """Persist the manifest (strict JSON, no NaN) to data/corpus_manifest.json."""
    path = Path(path) if path else config.CORPUS_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_corpus_manifest(path: Path | None = None) -> dict:
    """Load the committed manifest (raises FileNotFoundError if absent)."""
    path = Path(path) if path else config.CORPUS_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the corpus snapshot pin is missing; run "
            "`python scripts/run_study.py fetch --rebuild-manifest` or see README §15"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_corpus_manifest(
    manifest_path: Path | None = None,
    raw_dir: Path | None = None,
) -> dict:
    """Recompute sha256 + size for every manifest entry against the cache.

    Returns a report dict: ``{"ok", "n_checked", "mismatches", "missing"}``
    where mismatches carry expected/actual sha256 (and sizes). Never raises on
    mismatch — the caller decides whether to warn or fail.
    """
    manifest = load_corpus_manifest(manifest_path)
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    mismatches: list[dict] = []
    missing: list[str] = []
    n_checked = 0
    for doc_id, meta in sorted(manifest.get("documents", {}).items()):
        path = raw_dir / f"beigebook_{doc_id}.html"
        if not path.exists():
            missing.append(doc_id)
            continue
        n_checked += 1
        actual_sha = compute_sha256(path)
        actual_size = path.stat().st_size
        if actual_sha != meta.get("sha256") or actual_size != meta.get("size_bytes"):
            mismatches.append(
                {
                    "doc_id": doc_id,
                    "expected_sha256": meta.get("sha256"),
                    "actual_sha256": actual_sha,
                    "expected_size_bytes": meta.get("size_bytes"),
                    "actual_size_bytes": actual_size,
                }
            )
    return {
        "ok": not mismatches and not missing,
        "n_documents_in_manifest": len(manifest.get("documents", {})),
        "n_checked": n_checked,
        "mismatches": mismatches,
        "missing": missing,
    }
