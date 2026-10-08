"""HTML-to-text parsing on a fixture (stdlib parser, no network)."""

from __future__ import annotations

from fin_rag_research_assistant.parse import extract_paragraphs, full_text, locate_item_headings

FIXTURE_HTML = """<html><head><title>10-K</title><style>.x { color: red; }</style>
<script>var tracking = 1;</script></head>
<body><div><p>Apple's fiscal year ends in late September.</p><p>Short.</p>
<table><tr><td>Total net sales</td><td>$391,035</td></tr>
<tr><td>Net income</td><td>$93,736</td></tr></table>
<p>Revenue was $391,035 million for the period.</p></div></body></html>"""


def test_extracts_paragraph_text_and_joins_table_cells():
    paragraphs = extract_paragraphs(FIXTURE_HTML)
    joined = full_text(paragraphs)
    assert "Apple's fiscal year ends in late September." in joined
    assert "Total net sales | $391,035" in joined  # table cells joined with separator
    assert "Net income | $93,736" in joined
    assert "Revenue was $391,035 million for the period." in joined


def test_drops_script_style_title_content():
    joined = full_text(extract_paragraphs(FIXTURE_HTML))
    assert "var tracking" not in joined
    assert "color: red" not in joined
    assert "10-K" not in joined  # <title> content skipped


def test_short_fragments_are_kept_but_empties_dropped():
    paragraphs = extract_paragraphs(FIXTURE_HTML)
    assert "Short." in paragraphs  # kept: filtering happens at the study layer, not the parser
    assert all(p.strip() for p in paragraphs)


def test_locate_item_headings_best_effort():
    paragraphs = ["Item 1A. Risk Factors", "Some risk text", "Item 7. Management's Discussion"]
    found = locate_item_headings(paragraphs)
    assert found.get("Item 1A") == 0
    assert found.get("Item 7") == 2
