"""Beige Book document refs, cache-first fetch logic, and the snapshot manifest."""

from __future__ import annotations

import pytest

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


# --- corpus snapshot manifest (sha256 pin) ------------------------------------


def _make_cache(tmp_path, contents: dict[str, str]) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, html in contents.items():
        (tmp_path / f"beigebook_{name}.html").write_text(html, encoding="utf-8")


def _make_provenance(tmp_path, doc_ids: list[str]) -> None:
    import json

    prov = {
        "fetched_at": "2026-10-08",
        "documents": {
            doc_id: {"url": f"{BB_BASE_URL}beigebook{doc_id}.htm", "release": doc_id.split("-")[0]}
            for doc_id in doc_ids
        },
    }
    (tmp_path / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")


def test_corpus_manifest_roundtrip_and_verify(tmp_path):
    from fin_rag_research_assistant import beigebook

    _make_cache(tmp_path / "raw", {"202601-boston": "<html>a</html>",
                                   "202601-dallas": "<html>b</html>"})
    prov_dir = tmp_path / "prov"
    prov_dir.mkdir()
    _make_provenance(prov_dir, ["202601-boston", "202601-dallas"])

    manifest = beigebook.build_corpus_manifest(
        raw_dir=tmp_path / "raw", provenance_path=prov_dir / "provenance.json"
    )
    assert manifest["n_documents"] == 2
    assert manifest["algorithm"] == "sha256"
    boston = manifest["documents"]["202601-boston"]
    assert boston["url"] == f"{BB_BASE_URL}beigebook202601-boston.htm"
    assert boston["release"] == "202601"
    assert boston["fetched_at"] == "2026-10-08"
    assert boston["size_bytes"] == len("<html>a</html>")

    manifest_path = tmp_path / "corpus_manifest.json"
    beigebook.write_corpus_manifest(manifest, manifest_path)
    report = beigebook.verify_corpus_manifest(manifest_path, raw_dir=tmp_path / "raw")
    assert report["ok"] is True
    assert report["n_checked"] == 2
    assert report["mismatches"] == [] and report["missing"] == []


def test_corpus_manifest_detects_tampered_and_missing_files(tmp_path):
    from fin_rag_research_assistant import beigebook

    _make_cache(tmp_path, {"202601-boston": "<html>a</html>", "202601-dallas": "<html>b</html>"})
    prov_dir = tmp_path / "prov"
    prov_dir.mkdir()
    _make_provenance(prov_dir, ["202601-boston", "202601-dallas"])
    manifest_path = tmp_path / "corpus_manifest.json"
    beigebook.write_corpus_manifest(
        beigebook.build_corpus_manifest(
            raw_dir=tmp_path, provenance_path=prov_dir / "provenance.json"
        ),
        manifest_path,
    )

    # tamper: silently different bytes (the failure mode the pin exists for)
    (tmp_path / "beigebook_202601-boston.html").write_text("<html>REVISED</html>", encoding="utf-8")
    # delete: file absent entirely
    (tmp_path / "beigebook_202601-dallas.html").unlink()

    report = beigebook.verify_corpus_manifest(manifest_path, raw_dir=tmp_path)
    assert report["ok"] is False
    assert report["missing"] == ["202601-dallas"]
    assert len(report["mismatches"]) == 1
    mismatch = report["mismatches"][0]
    assert mismatch["doc_id"] == "202601-boston"
    assert mismatch["expected_sha256"] != mismatch["actual_sha256"]
    assert mismatch["expected_size_bytes"] != mismatch["actual_size_bytes"]


def test_load_corpus_manifest_missing_file_raises(tmp_path):
    from fin_rag_research_assistant import beigebook

    with pytest.raises(FileNotFoundError, match="corpus_manifest"):
        beigebook.load_corpus_manifest(tmp_path / "absent.json")


def test_repo_corpus_manifest_matches_cached_html():
    """The committed data/corpus_manifest.json must match the local cache byte
    for byte. Skipped on fresh checkouts where data/raw/ is not populated
    (CI); locally it guards against silent corpus drift before any rerun."""
    from fin_rag_research_assistant import beigebook, config

    cached = sorted(config.RAW_DIR.glob("beigebook_*.html"))
    if not cached:
        pytest.skip("raw Beige Book cache not present (fresh checkout)")
    manifest_path = config.CORPUS_MANIFEST_PATH
    if not manifest_path.exists():
        pytest.skip("corpus manifest not committed yet")
    assert len(cached) == 26
    report = beigebook.verify_corpus_manifest()
    assert report["ok"] is True, f"corpus drift detected: {report}"
