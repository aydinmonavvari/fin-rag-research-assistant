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

# --- SEC EDGAR access policy -------------------------------------------------
# Descriptive User-Agent per SEC fair-access policy (https://www.sec.gov/
# privacy-security#automated-access). We deliberately use the GitHub noreply
# address (privacy-preserving) instead of a personal inbox. The UA really used
# for every request is recorded in data/processed/provenance.json. The fetch
# layer backs off and retries (bounded) when SEC's edge throttles datacenter
# egress IPs with 403/429; if a fallback had been necessary, it would be
# recorded there too.
SEC_USER_AGENT_DEFAULT = "Aydin Monavvari aydinmonavvari@users.noreply.github.com"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", SEC_USER_AGENT_DEFAULT)
SEC_REQUEST_SLEEP_S = 0.75  # polite rate limit between EDGAR requests
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{primary_doc}"
)

TARGET_FILINGS: dict[str, dict[str, str]] = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corporation"},
}

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
# Refuse when the best available dense cosine score falls below tau. The value
# is picked from the score distribution of the in-scope questions vs the
# out-of-scope probes (see scripts stage `eval-retrieval`); the default is only
# a fallback if the analysis has not been run.
REFUSAL_TAU_DEFAULT = 0.30

# --- Grounded generation -----------------------------------------------------
GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GEN_MAX_NEW_TOKENS = 160
GEN_N_QUESTIONS = 8  # subset of the 20-question set
GEN_N_REFUSAL_PROBES = 2
GEN_INSTRUCTION = (
    "Answer using ONLY the provided excerpts; cite excerpt numbers; "
    "if the excerpts do not contain the answer, reply exactly INSUFFICIENT_CONTEXT"
)
# Short format reminder repeated in the user turn (small models often need the
# citation format restated; documented prompt-design choice).
GEN_CITATION_REMINDER = (
    "Cite the excerpt numbers you used in square brackets, e.g. [1]."
)

FILING_LABELS = {"AAPL": "Apple Inc. FY2025 10-K", "MSFT": "Microsoft Corp. FY2026 10-K"}

# --- Study-stage parameters (run_study CLI) ----------------------------------
RETRIEVAL_EVAL_DEPTH = 10  # ranked-list depth kept per question (Recall@k curves)
RECALL_CURVE_KS = tuple(range(1, RETRIEVAL_EVAL_DEPTH + 1))
GEN_TOP_K = 3  # excerpts provided to the generator
GEN_SUBSET_SIZE = 8  # in-scope questions sent through the generator
GEN_SUBSET_QIDS: tuple[str, ...] = ()  # explicit subset, fixed after QA authoring
GEN_N_OFF_DOMAIN_PROBES = 6  # auxiliary probes for the tau-selection figure
