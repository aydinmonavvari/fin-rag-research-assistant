"""Author the two QA study instruments (SET A dev + SET B held-out).

Both sets are authored by the researcher (single builder); there are no
external annotators. SET A (``data/qa/qa_set_dev.jsonl``) contains the original
20 in-scope questions + 2 out-of-scope probes written after reading the parsed
Beige Book corpus; it is used for refusal-threshold (tau) selection and
debugging only. SET B (``data/qa/qa_set_eval.jsonl``) was authored AFTER SET A
was frozen, using paraphrased wording (questions do not copy corpus sentences),
and is the held-out evaluation set; it is never used for threshold selection.

Each in-scope question carries a verbatim evidence snippet; the script locates
the gold chunk containing it (first containing chunk in document order, since
overlapping windows can duplicate a sentence) and writes the JSONL files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fin_rag_research_assistant import config  # noqa: E402
from fin_rag_research_assistant.chunk import load_chunks  # noqa: E402

# (qid, source_doc_id, question, answer, evidence_verbatim, qtype)
QUESTIONS_DEV = [
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

# SET B (held-out): authored after SET A was frozen. Questions are paraphrased
# (they do not copy corpus sentences, unlike several SET A questions, which was
# a documented weakness). qtypes: numeric, categorical, definitional, mixed,
# temporal (answer depends on release recency), citation_trap (the plausible
# naive answer exists in the corpus but with a subtly different number/scope;
# the gold answer states the correct nuance), out_of_scope probes.
QUESTIONS_HELDOUT = [
    ("B-01", "202601-dallas",
     "What average pay increase did Texas firms surveyed for the January 2026 Dallas Fed report say their workforces received during 2025?",
     "3.5 percent on average, down from 4.3 percent in 2024",
     "Wage growth in 2025 among more than 250 surveyed Texas manufacturing and services firms was 3.5 percent, on average, down from 4.3 percent in 2024",
     "numeric"),
    ("B-02", "202510-cleveland",
     "One tourism business in the Fourth District described a steep annual drop in guests arriving from a neighboring country. How large was the decline it reported?",
     "Visits by Canadians fell by 50 percent (a full half, year over year)",
     "One tourism contact, who reported a year-over-year decline in activity, said that visits by Canadians fell by 50 percent",
     "numeric"),
    ("B-03", "202601-boston",
     "Which disruption outside the housing market did First District contacts in January 2026 partly blame for weaker home purchases?",
     "The federal government shutdown, which led to mortgage delays",
     "Residential home sales showed moderate declines from a year earlier, in part because the federal government shutdown led to mortgage delays",
     "categorical"),
    ("B-04", "202601-minneapolis",
     "How did heavier-than-usual snow affect Ninth District retail and tourism activity around the holidays, according to the January 2026 Minneapolis report?",
     "Spotty holiday sales were typically blamed on poor weather, but the snow and colder temperatures gave winter tourism a solid start in some regions",
     "Spotty sales during the holiday season were typically attributed to poor weather, with much of the District seeing snowier conditions than normal. However, snow and colder temperatures also translated to a solid start for winter tourism activities in some regions",
     "categorical"),
    ("B-05", "202510-richmond",
     "What cause did a North Carolina housing agent quoted in the Richmond District's October 2025 report give for the large share of home listings that had their asking prices cut?",
     "Initial overpricing \u2014 the agent reported that 45 percent of listings experienced price reductions because they had been priced too high at the outset",
     "45 percent of listings experienced price reductions due to initial overpricing",
     "definitional"),
    ("B-06", "202510-philadelphia",
     "What did an October 2025 Third District contact report had happened to the premium on a liability insurance policy over the previous year?",
     "It rose 15 percent from a year earlier (alongside a nearly 10 percent rise in health-care costs)",
     "Another contact reported a 15 percent increase in a liability insurance policy from a year earlier in addition to a nearly 10 percent rise in health-care costs",
     "definitional"),
    ("B-07", "202510-cleveland",
     "In the October 2025 Cleveland report, how far below prior expectations were orders received by suppliers to industrial and agricultural equipment makers, and which buyers were placing higher orders instead?",
     "Orders were roughly 25 percent below previous expectations; firms selling into the fossil fuel industry and electricity generation reported higher orders related to data center construction and operation",
     "orders from these producers were below their previous expectation by roughly 25 percent. By contrast, some firms selling into the fossil fuel industry and electricity generation reported higher orders related to data center construction and operation",
     "mixed"),
    ("B-08", "202601-philadelphia",
     "By January 2026, where did Third District firms' expectations for next year's growth in pay per worker stand relative to pre-2020 norms?",
     "At a trimmed mean of 3.3 percent for the fourth quarter of 2025 \u2014 slightly above the 3.2 percent pre-pandemic (2016 through 2019) average",
     "firms' expectations of the one-year-ahead change in compensation cost per worker held steady at a trimmed mean of 3.3 percent in the fourth quarter of 2025\u2014just a tick higher than the 3.2 percent pre-pandemic average (2016 through 2019)",
     "mixed"),
    ("B-09", "202601-summary",
     "In the most recent Beige Book release included in this corpus, how many of the twelve Federal Reserve Districts described overall activity as expanding at a slight-to-modest pace, and which release is that?",
     "Eight of the twelve Districts \u2014 the January 2026 release (the October 2025 release counted only three)",
     "Overall economic activity increased at a slight to modest pace in eight of the twelve Federal Reserve Districts, with three Districts reporting no change and one reporting a modest decline",
     "temporal"),
    ("B-10", "202601-summary",
     "What does the newest Beige Book report in the corpus say about whether the current pickup in activity breaks from the pattern of the preceding three report cycles?",
     "It does break the pattern: the report calls the increase an improvement over the last three report cycles, in which a majority of Districts reported little change",
     "This marks an improvement over the last three report cycles where a majority of Districts reported little change",
     "temporal"),
    ("B-11", "202601-minneapolis",
     "In the monthly business survey cited in the January 2026 Minneapolis report, what portion of firms said they had raised the prices they charge customers during December?",
     "20 percent raised prices charged to customers in December, while 16 percent lowered them. The 'about a third' figure in the same paragraph refers to firms whose nonlabor INPUT prices rose, and 'more than a third' refers to firms PLANNING increases for January \u2014 neither is the share that raised charged prices in December",
     "Meanwhile, 20 percent of firms increased prices charged to customers, compared with 16 percent that decreased their prices. More than a third of firms anticipated increasing their prices charged to customers in January",
     "citation_trap"),
    ("B-12", "202510-kansas-city",
     "How widespread was reported stress on farm earnings and operating cash across the Tenth District's agricultural lenders in October 2025 \u2014 was it the same everywhere in the district?",
     "No single district-wide share is reported: over 80 percent of lenders in crop-heavy areas reported declines in farm income and working capital, compared with about 40 percent in areas with more cattle production \u2014 the share depends on the area's farm mix",
     "In a recent survey, over 80 percent of lenders in crop-heavy areas reported declines in farm income and working capital, compared to about 40 percent in areas with more cattle production",
     "citation_trap"),
    ("B-OOS-01", "",
     "What was the national unemployment rate for December 2025 according to the Bureau of Labor Statistics?",
     "",
     "",
     "out_of_scope"),
    ("B-OOS-02", "",
     "By how many basis points did the FOMC lower its federal funds target range at the December 2025 meeting?",
     "",
     "",
     "out_of_scope"),
]


def build_rows(chunks, questions) -> tuple[list[dict], list[str]]:
    """Resolve gold chunk ids; returns (rows, problems)."""
    rows: list[dict] = []
    problems: list[str] = []
    for qid, source, question, answer, evidence, qtype in questions:
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
    return rows, problems


def write_set(rows: list[dict], out: Path) -> dict[str, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    per_type: dict[str, int] = {}
    for row in rows:
        per_type[row["qtype"]] = per_type.get(row["qtype"], 0) + 1
    return per_type


def main() -> int:
    chunks = load_chunks(config.PROCESSED_DIR / f"chunks_{config.CHUNK_SIZE_TOKENS}.jsonl")
    status = 0
    for name, questions, out in (
        ("SET A (dev)", QUESTIONS_DEV, config.QA_DEV_PATH),
        ("SET B (held-out)", QUESTIONS_HELDOUT, config.QA_EVAL_PATH),
    ):
        rows, problems = build_rows(chunks, questions)
        if problems:
            print(f"{name} PROBLEMS:")
            for problem in problems:
                print(" -", problem)
            status = 1
            continue
        per_type = write_set(rows, out)
        n_in = sum(1 for r in rows if r["qtype"] != "out_of_scope")
        print(f"{name}: wrote {len(rows)} rows to {out} (in-scope: {n_in}) types={per_type}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
