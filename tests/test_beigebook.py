"""Beige Book document refs and cache-first fetch logic (offline)."""

from __future__ import annotations

from fin_rag_research_assistant.beigebook import DocRef, all_doc_refs, release_doc_refs
from fin_rag_research_assistant.config import BB_BASE_URL, BB_DISTRICTS, TARGET_RELEASES


def test_release_doc_refs_contains_summary_plus_twelve_districts():
    refs = release_doc_refs("202601")
    assert len(refs) == 1 + len(BB_DISTRICTS) == 13
    slugs = {ref.doc_id for ref in refs}
    assert "202601-summary" in slugs
    for slug, _name in BB_DISTRICTS:
        assert f"202601-{slug}" in slugs


def test_doc_ref_urls_follow_federalreserve_scheme():
    refs = all_doc_refs()
    assert refs, "TARGET_RELEASES must be non-empty"
    for ref in refs:
        assert isinstance(ref, DocRef)
        assert ref.url == f"{BB_BASE_URL}beigebook{ref.doc_id}.htm"
        assert ref.url.startswith("https://www.federalreserve.gov/monetarypolicy/")
        assert ref.cache_name == f"beigebook_{ref.doc_id}.html"


def test_all_doc_refs_deterministic_and_sorted():
    refs_a, refs_b = all_doc_refs(), all_doc_refs()
    assert [r.doc_id for r in refs_a] == [r.doc_id for r in refs_b]
    assert [r.doc_id for r in refs_a] == sorted(r.doc_id for r in refs_a)
    # Two releases x 13 documents
    assert len(refs_a) == 13 * len(TARGET_RELEASES)


def test_fetch_doc_cache_first(tmp_path, monkeypatch):
    """Never hits the network when the cache file already exists."""
    from fin_rag_research_assistant import beigebook

    ref = DocRef(
        release="202601",
        doc_id="202601-summary",
        title="Beige Book 202601 — National Summary",
        url=f"{BB_BASE_URL}beigebook202601-summary.htm",
    )
    cached = tmp_path / ref.cache_name
    cached.write_text("<html>cached</html>", encoding="utf-8")

    def _boom(url):  # network access would fail this test
        raise AssertionError(f"network fetch attempted for {url}")

    monkeypatch.setattr(beigebook, "_http_get", _boom)
    path = beigebook.fetch_doc(ref, tmp_path)
    assert path == cached
    assert path.read_text(encoding="utf-8") == "<html>cached</html>"


def test_fetch_doc_downloads_and_writes_cache(tmp_path, monkeypatch):
    from fin_rag_research_assistant import beigebook

    ref = DocRef(
        release="202601",
        doc_id="202601-boston",
        title="Beige Book 202601 — Boston",
        url=f"{BB_BASE_URL}beigebook202601-boston.htm",
    )
    monkeypatch.setattr(beigebook, "_http_get", lambda url: "<html>fetched</html>")
    monkeypatch.setattr(beigebook.time, "sleep", lambda _s: None)  # no waiting in tests
    path = beigebook.fetch_doc(ref, tmp_path)
    assert path.read_text(encoding="utf-8") == "<html>fetched</html>"
    # Second call is served from cache (would raise if it re-fetched with a failing getter)
    monkeypatch.setattr(beigebook, "_http_get", lambda url: (_ for _ in ()).throw(AssertionError))
    assert beigebook.fetch_doc(ref, tmp_path) == path
