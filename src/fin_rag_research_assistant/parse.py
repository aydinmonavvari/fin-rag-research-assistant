"""HTML-to-text extraction for SEC 10-K filings.

Deliberately simple and dependency-free: a stdlib ``HTMLParser`` subclass that
keeps paragraph-level text, emits one paragraph per table row with cells
joined by a " | " separator, and skips script/style and XBRL-only tags. 10-K
HTML is machine-shaped (XBRL inline tags, nested tables), so the output is a
*study corpus*, not a typeset document — the chunker and provenance records
make that explicit.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

SKIP_TAGS = {
    "script",
    "style",
    "head",
    "title",
    "noscript",
    "ix:header",
    "ix:hidden",
}
BLOCK_TAGS = {
    "p",
    "div",
    "tr",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "br",
    "section",
}
_HEADING_RE = re.compile(r"^h[1-6]$")
_WS_RE = re.compile(r"[ \t\xa0]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._table_depth = 0
        self._cell_parts: list[str] = []
        self._in_cell = False
        self._row_cells: list[str] = []
        self.paragraphs: list[str] = []
        self._buf: list[str] = []

    # -- helpers ----------------------------------------------------------
    def _flush_paragraph(self) -> None:
        text = _WS_RE.sub(" ", "".join(self._buf)).strip()
        self._buf = []
        if text:
            self.paragraphs.append(html.unescape(text))

    # -- parser hooks ------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: ARG002
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._table_depth += 1
            self._flush_paragraph()
        elif tag in {"td", "th"} and self._table_depth:
            self._flush_paragraph()
            self._in_cell = True
            self._cell_parts = []
        elif _HEADING_RE.match(tag):
            self._flush_paragraph()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            self._flush_paragraph()
        elif tag == "tr" and self._table_depth:
            # One paragraph per table row: cells joined with a separator.
            if self._row_cells:
                self._buf.append(" | ".join(self._row_cells))
            self._row_cells = []
            self._flush_paragraph()
        elif tag in {"td", "th"} and self._in_cell:
            cell = _WS_RE.sub(" ", "".join(self._cell_parts)).strip()
            self._cell_parts = []
            self._in_cell = False
            if cell:
                self._row_cells.append(cell)
        elif tag in BLOCK_TAGS or _HEADING_RE.match(tag):
            self._flush_paragraph()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_cell:
            self._cell_parts.append(data)
        else:
            self._buf.append(data)

    def close(self) -> None:  # noqa: D102
        self._flush_paragraph()
        super().close()


def extract_paragraphs(html_text: str) -> list[str]:
    """Parse 10-K HTML into a list of cleaned paragraph strings."""
    parser = _TextExtractor()
    parser.feed(html_text)
    parser.close()
    cleaned = [_MULTI_NL_RE.sub("\n\n", p).strip() for p in parser.paragraphs]
    return [p for p in cleaned if p]


def full_text(paragraphs: list[str]) -> str:
    """Join paragraphs with blank lines (single canonical corpus text)."""
    return "\n\n".join(paragraphs)


def locate_item_headings(paragraphs: list[str]) -> dict[str, int]:
    """Best-effort detection of 10-K item headings ("Item 1A. Risk Factors"...).

    Returns a mapping like {"Item 1": paragraph_index, "Item 1A": ...}.
    Section splitting proved brittle across the two filings (inline XBRL
    markup, mixed encodings of the item titles), so the study chunker does NOT
    depend on it; this is used only for provenance reporting.
    """
    pattern = re.compile(
        r"^item\s+(\d{1,2}[A-B]?)\b[\s.:—-]*(.{0,60})", re.IGNORECASE
    )
    found: dict[str, int] = {}
    for idx, para in enumerate(paragraphs):
        first_line = para.split("\n", 1)[0].strip()
        match = pattern.match(first_line)
        if match:
            key = f"Item {match.group(1).upper()}"
            found.setdefault(key, idx)
    return found
