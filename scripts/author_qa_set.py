"""Author the 20-question QA set + 2 out-of-scope probes from the parsed corpus.

Questions were written AFTER reading the parsed Beige Book corpus (202510 and
202601 releases). Each in-scope question carries a verbatim evidence snippet;
the script locates the gold chunk containing it and writes data/qa/qa_set.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fin_rag_research_assistant import config  # noqa: E402
from fin_rag_research_assistant.chunk import load_chunks  # noqa: E402

# (qid, source_doc_id, question, answer, evidence_verbatim, qtype)
QUESTIONS = [
    # --- October 2025 release (202510) ---
    ("Q-01", "202510-summary",
     "In the October 2025 Beige Book, how many Federal Reserve Districts reported slight to modest growth in overall economic activity?",
     "Three Districts",
     "with three Districts reporting slight to modest growth in activity, five reporting no change, and four noting a slight softening",
     "numeric"),
    ("Q-02", "202510-summary",
     "Why were auto sales boosted in some Districts ahead of the end of September, according to the October 2025 report?",
     "Strong demand for electric vehicles ahead of the expiration of a federal tax credit at the end of September",
     "auto sales were boosted in some Districts by strong demand for electric vehicles ahead of the expiration of a federal tax credit at the end of September",
     "categorical"),
    ("Q-03", "202510-summary",
     "Which sectors' labor supply was reportedly strained in several Districts due to recent changes to immigration policies?",
     "Hospitality, agriculture, construction, and manufacturing",
     "labor supply in the hospitality, agriculture, construction, and manufacturing sectors was reportedly strained in several Districts due to recent changes to immigration policies",
     "definitional"),
    ("Q-04", "202510-summary",
     "According to the October 2025 National Summary, what intensified labor cost pressures in recent weeks?",
     "Outsized increases in employer-sponsored health insurance expenses",
     "labor cost pressures intensified in recent weeks due to outsized increases in employer-sponsored health insurance expenses",
     "definitional"),
    ("Q-05", "202510-summary",
     "Which materials saw prices pushed down by waning demand, and in which two Districts did this occur?",
     "Steel in the Sixth District and lumber in the Twelfth District",
     "such as steel in the Sixth District and lumber in the Twelfth District",
     "categorical"),
    ("Q-06", "202510-boston",
     "What did Boston District contacts report about commercial real estate activity in the October 2025 Beige Book?",
     "Commercial real estate activity increased slightly, beating expectations",
     "Commercial real estate activity increased slightly, beating expectations",
     "categorical"),
    ("Q-07", "202510-minneapolis",
     "What share of hospitality and tourism firms in the Minneapolis District reported wage increases of three percent or more?",
     "Nearly three-quarters",
     "nearly three-quarters of hospitality and tourism firms reported wage increases of three percent or more",
     "numeric"),
    ("Q-08", "202510-kansas-city",
     "What did bankers note about consumer loan portfolios in the Tenth District in the October 2025 report?",
     "Consumer loan portfolios deteriorated moderately",
     "bankers noted consumer loan portfolios deteriorated moderately",
     "definitional"),
    ("Q-09", "202510-kansas-city",
     "In the Kansas City District survey cited in October 2025, what share of lenders in crop-heavy areas reported declines in farm income and working capital?",
     "Over 80 percent",
     "over 80 percent of lenders in crop-heavy areas reported declines in farm income and working capital, compared to about 40 percent in areas with more cattle production",
     "numeric"),
    ("Q-10", "202510-boston",
     "What happened to the number of Canadian travelers to New England, according to the Boston report in October 2025?",
     "It remained down sharply from the previous year",
     "The number of Canadian travelers to New England remained down sharply from the previous year",
     "categorical"),
    # --- January 2026 release (202601) ---
    ("Q-11", "202601-summary",
     "In the January 2026 Beige Book, how many Federal Reserve Districts reported overall economic activity increasing at a slight to modest pace?",
     "Eight of the twelve Districts",
     "Overall economic activity increased at a slight to modest pace in eight of the twelve Federal Reserve Districts, with three Districts reporting no change and one reporting a modest decline",
     "numeric"),
    ("Q-12", "202601-summary",
     "What factor largely drove the slight to modest growth in consumer spending reported by most banks in January 2026?",
     "The holiday shopping season",
     "largely attributed to the holiday shopping season",
     "categorical"),
    ("Q-13", "202601-summary",
     "In January 2026, how many Districts reported manufacturing growth and how many reported contraction?",
     "Five Districts reported growth and six reported contraction",
     "Manufacturing activity varied with five Districts reporting growth and six reporting contraction",
     "numeric"),
    ("Q-14", "202601-summary",
     "Which District reported a modest decline in agriculture conditions due to weaker demand for exported commodities in January 2026?",
     "Atlanta",
     "Agriculture conditions were largely unchanged with only Atlanta reporting a modest decline due to weaker demand for exported commodities",
     "categorical"),
    ("Q-15", "202601-summary",
     "On what date was information collection completed for the January 2026 Beige Book, and which Reserve Bank prepared the report?",
     "Information collected on or before January 5, 2026; prepared at the Federal Reserve Bank of Richmond",
     "prepared at the Federal Reserve Bank of Richmond based on information collected on or before January 5, 2026",
     "categorical"),
    ("Q-16", "202601-dallas",
     "What oil price range do Eleventh District producers broadly expect for 2026, according to the Dallas report?",
     "The low $60 per barrel range",
     "Producers broadly expect oil prices to remain in the low $60 per barrel range in 2026",
     "numeric"),
    ("Q-17", "202601-new-york",
     "Which upcoming sporting event was cited as boosting New York City tourism bookings in the January 2026 report?",
     "The upcoming World Cup (group travel bookings coming through around it)",
     "tourism contacts were optimistic, especially due to group travel bookings coming through around the upcoming World Cup",
     "categorical"),
    ("Q-18", "202601-chicago",
     "What was contributing to strong competition for industrial-zoned land in the Chicago District, according to developers?",
     "Robust demand for data centers",
     "robust demand for data centers was contributing to strong competition for industrial-zoned land",
     "definitional"),
    ("Q-19", "202601-chicago",
     "How did Chicago District net farm income in 2025 compare with 2024?",
     "About the same as in 2024 and higher than previously expected, after corn and soybean prices rallied in the fourth quarter despite a large harvest",
     "District net farm income for 2025 was about the same as in 2024 and was higher than previously expected, after corn and soybean prices rallied in the fourth quarter despite a large harvest",
     "numeric"),
    ("Q-20", "202601-san-francisco",
     "What weather condition boosted demand for ski resorts and winter sports in the Twelfth District in late December?",
     "Heavy snowfall in the Mountain West region",
     "heavy snowfall in the Mountain West region in late December boosted demand for ski resorts and winter sports",
     "categorical"),
    # --- Out-of-scope refusal probes (answers deliberately NOT in the corpus) ---
    ("OOS-01", "",
     "What target range for the federal funds rate did the FOMC announce at its January 2026 meeting?",
     "",
     "",
     "out_of_scope"),
    ("OOS-02", "",
     "How many employees did the Federal Reserve Board employ in 2026?",
     "",
     "",
     "out_of_scope"),
]


def main() -> int:
    chunks = load_chunks(config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl")
    rows: list[dict] = []
    problems: list[str] = []
    for qid, source, question, answer, evidence, qtype in QUESTIONS:
        if qtype == "out_of_scope":
            rows.append({
                "qid": qid, "source": "", "question": question, "answer": "",
                "gold_chunk_id": "", "evidence": "", "qtype": qtype,
            })
            continue
        # evidence must appear verbatim in the stated source document; overlapping
        # chunk windows can duplicate a sentence, so the FIRST containing chunk
        # (lowest position) is used as the gold chunk (documented choice).
        matches = [c for c in chunks if c.doc_id == source and evidence in c.text]
        if not matches:
            problems.append(f"{qid}: no match in {source} for evidence {evidence[:60]!r}")
            continue
        matches.sort(key=lambda c: c.position)
        rows.append({
            "qid": qid, "source": source, "question": question, "answer": answer,
            "gold_chunk_id": matches[0].chunk_id, "evidence": evidence, "qtype": qtype,
        })
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1
    out = config.QA_DIR / "qa_set.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_in = sum(1 for r in rows if r["qtype"] != "out_of_scope")
    per_type: dict[str, int] = {}
    for r in rows:
        per_type[r["qtype"]] = per_type.get(r["qtype"], 0) + 1
    print(f"wrote {len(rows)} rows to {out} (in-scope: {n_in}) types={per_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
