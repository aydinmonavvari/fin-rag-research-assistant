"""Paths, constants and study configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QA_DIR = DATA_DIR / "qa"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Two-file QA scheme (SET A / SET B). data/qa/qa_set.jsonl (one combined file,
# used for BOTH tau selection and final evaluation) was replaced by:
#   - SET A "dev":  threshold selection + debugging only (never a final result);
#   - SET B "held-out": final evaluation; never used for threshold selection.
# Both files are authored by the researcher (single builder, no external
# annotators) — disclosed in README / research report / generated reports.
QA_DEV_PATH = QA_DIR / "qa_set_dev.jsonl"
QA_EVAL_PATH = QA_DIR / "qa_set_eval.jsonl"

# Committed snapshot pin for the cached raw Beige Book HTML (sha256 per file).
# Built from the cache + provenance; verified automatically by run_study.py.
CORPUS_MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"

# --- Federal Reserve Beige Book access policy --------------------------------
# Documented corpus pivot: the study was originally designed around SEC 10-K
# filings, but SEC's edge servers return HTTP 403 to this build environment's
# datacenter egress IP (a block that persisted across long backoffs). The
# corpus therefore uses the Federal Reserve Beige Book: real, public-domain
# (U.S. government) economic reports. A descriptive User-Agent is still sent
# as good practice, using the GitHub noreply address (privacy-preserving).
# The UA really used for every request is recorded in
# data/processed/provenance.json. The fetch layer backs off and retries
# (bounded) when a server throttles; cache-first logic means documents are
# never fetched twice in one checkout.
BB_BASE_URL = "https://www.federalreserve.gov/monetarypolicy/"
POLITE_USER_AGENT_DEFAULT = "Aydin Monavvari aydinmonavvari@users.noreply.github.com"
POLITE_USER_AGENT = os.environ.get("POLITE_USER_AGENT", POLITE_USER_AGENT_DEFAULT)
REQUEST_SLEEP_S = 0.75  # polite rate limit between document requests

# Twelve Federal Reserve district reports accompany each national summary.
# (slug, human-readable name) — slugs match the federalreserve.gov URL scheme.
BB_DISTRICTS: list[tuple[str, str]] = [
    ("boston", "First District (Boston)"),
    ("new-york", "Second District (New York)"),
    ("philadelphia", "Third District (Philadelphia)"),
    ("cleveland", "Fourth District (Cleveland)"),
    ("richmond", "Fifth District (Richmond)"),
    ("atlanta", "Sixth District (Atlanta)"),
    ("chicago", "Seventh District (Chicago)"),
    ("st-louis", "Eighth District (St. Louis)"),
    ("minneapolis", "Ninth District (Minneapolis)"),
    ("kansas-city", "Tenth District (Kansas City)"),
    ("dallas", "Eleventh District (Dallas)"),
    ("san-francisco", "Twelfth District (San Francisco)"),
]

# Two consecutive releases x (summary + 12 districts) = 26 documents.
TARGET_RELEASES: list[str] = ["202510", "202601"]

# --- Chunking ----------------------------------------------------------------
# "Token" here means a whitespace-delimited word (documented approximation;
# no tokenizer dependency is required for reproducible chunking).
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
CHUNK_SIZE_SMALL_TOKENS = 400  # sensitivity analysis

# --- Retrieval evaluation ----------------------------------------------------
RANDOM_SEED = 42
TOP_K = 5  # candidate depth for retrieval evaluation / generation context
RRF_K = 60  # reciprocal rank fusion constant (Cormack et al. 2009)
METRIC_KS = (1, 5)  # Recall@1, Recall@5

# --- Refusal policy ----------------------------------------------------------
# Refuse when the best available dense cosine score falls below tau. tau is
# selected ONCE on the SET A (dev) score distribution — see policy.py for the
# exact implemented rule — and then applied unchanged to the held-out SET B
# evaluation. REFUSAL_TAU_DEFAULT is only a fallback for when the analysis has
# not been run.
REFUSAL_TAU_DEFAULT = 0.30

# --- Grounded generation -----------------------------------------------------
# Generation is evaluated on ALL held-out (SET B) in-scope questions plus the
# SET B refusal probes — there is no question subsetting (the earlier
# first-8-questions subset was removed as an undisclosed selection bias).
GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GEN_MAX_NEW_TOKENS = 160
GEN_INSTRUCTION = (
    "Answer using ONLY the provided excerpts; cite excerpt numbers; "
    "if the excerpts do not contain the answer, reply exactly INSUFFICIENT_CONTEXT"
)
# Short format reminder repeated in the user turn (small models often need the
# citation format restated; documented prompt-design choice).
GEN_CITATION_REMINDER = (
    "Cite the excerpt numbers you used in square brackets, e.g. [1]."
)

# Human-readable labels for the two national summaries (used in reports).
DOC_LABELS = {
    "202510-summary": "Beige Book October 2025 — National Summary",
    "202601-summary": "Beige Book January 2026 — National Summary",
}

# --- Study-stage parameters (run_study CLI) ----------------------------------
RETRIEVAL_EVAL_DEPTH = 10  # ranked-list depth kept per question (Recall@k curves)
RECALL_CURVE_KS = tuple(range(1, RETRIEVAL_EVAL_DEPTH + 1))
GEN_TOP_K = 3  # excerpts provided to the generator
