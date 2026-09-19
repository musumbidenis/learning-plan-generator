"""Read a CDACC document's tables as tables, instead of guessing at columns.

Everything in these documents that matters is in a grid: the OS keeps its
performance criteria in an `ELEMENT | PERFORMANCE CRITERIA` table, and the
curriculum keeps the syllabus in `Learning Outcome | Content | Suggested
Assessment Methods`. The parsers have always rebuilt those columns from word
x-coordinates - `x0 < 150` is a learning outcome, `< 435` is content, the rest
is assessment - and thresholds are guesses. They are why the curriculum's
Suggested Assessment Methods came back carrying half the Content column and
then ran on into the Methods of Delivery section.

`pdfplumber` reads the ruling lines directly. Checked across 47 documents from
the library - 25 Occupational Standards, 22 curricula, spanning Animal
Production to Analytical Chemistry - every OS carried the elements/PC table and
every curriculum carried both the syllabus table and its durations table.

Two things make the cells usable rather than merely present:

* **Continuation rows.** A unit's table runs over a page break, and the row that
  resumes it has an EMPTY first cell (OS p107, curriculum p135 and p137). Those
  rows belong to the row above, so they are stitched onto it.
* **Repeated headers.** Each page of a long table restates the header, which
  would otherwise read as data.

Word documents do not come through here. `pdf_utils.load_word_pages` builds
their columns from real cells already, placing them at fixed pseudo-x positions
(80 / 235 / 440), so the coordinate parsers read exact cells for those formats -
the guesswork this module removes is specific to PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Sequence

import pdfplumber

import runlog
from pdf_utils import clean_text, is_noise_line


@dataclass
class Table:
    """One table, already stitched and cleaned."""
    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    page: int = 0                      # 0-based index of the page it starts on
    end_page: int = 0                  # where it finishes, after stitching
    bottom: float = 0.0                # y of its last rule, on end_page
    page_height: float = 0.0           # of end_page, to see if it hit the edge

    @property
    def width(self) -> int:
        return max((len(r) for r in [self.header] + self.rows), default=0)

    def cell(self, row: Sequence[str], index: int) -> str:
        return row[index] if index < len(row) else ""

    def row_matching(self, pattern: Pattern) -> Optional[List[str]]:
        """The first row whose leading cell matches - e.g. 'Methods of assessment'."""
        for row in self.rows:
            if row and pattern.search(row[0]):
                return row
        return None


# A cell holding nothing but a number: a duration, a quantity, a credit factor.
_RE_BARE_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def _clean_cell(value) -> str:
    """One cell, joined onto a single line and stripped of document furniture.

    `is_noise_line` exists to throw away page furniture from a LINE of a page,
    and one of the things it throws away is a bare number, because that is what
    a page number looks like. In a table a bare number is the data: the whole
    Duration (Hours) column is bare numbers, and filtering it left every
    duration blank. Numbers are kept; the rest of the noise rules still apply.
    """
    text = clean_text(str(value or "").replace("\n", " "))
    if _RE_BARE_NUMBER.match(text):
        return text
    return "" if is_noise_line(text) else text


def _same_header(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether `b` is `a` restated - a header repeated on the next page."""
    norm = lambda cells: [re.sub(r"\W+", "", c.lower())[:24] for c in cells if c]
    left, right = norm(a), norm(b)
    return bool(left) and left == right


def _stitch(rows: List[List[str]]) -> List[List[str]]:
    """Fold continuation rows into the row they continue.

    A row whose first cell is empty is not a new row: it is the tail of the one
    before it, wrapped onto the next page or the next line of the grid. Joining
    them is what keeps a learning outcome's content whole when its table spans
    three pages.
    """
    out: List[List[str]] = []
    for row in rows:
        if not any(row):
            continue
        if out and not row[0].strip():
            previous = out[-1]
            for i, cell in enumerate(row):
                if not cell:
                    continue
                while len(previous) <= i:
                    previous.append("")
                previous[i] = (previous[i] + " " + cell).strip()
            continue
        out.append(list(row))
    return out


# A table carried over a page break begins ABOVE where any fresh table could:
# there is no room for the heading that would otherwise precede one. Measured
# across the benchmark unit, continuations start at y 50.9-73.2 and every table
# that genuinely begins on its page starts at 103.9 or lower down (Variable/
# Range at 280.6, the Evidence Guide at 607.9, the resources grid at 431.3).
_CARRIED_OVER_ABOVE = 90.0


def _continues(previous, page_index: int, top: float, width: int) -> bool:
    """Whether this table is the tail of `previous`, carried over a page break.

    Judged by where it starts, not by its cells. Reading the first cell got it
    wrong twice on one unit: it missed a continuation that resumed with a full
    row ("3. Maintain ICT system security"), and would have taken any row
    beginning with a number for a continuation. Requiring the previous table to
    have reached the page bottom was wrong too - the OS elements table stops
    233pt short of it and still continues overleaf.
    """
    if previous is None or previous.width != width:
        return False
    if page_index != previous.end_page + 1:
        return False
    return top <= _CARRIED_OVER_ABOVE


# The document most recently opened, kept so parsing a whole document's units
# does not reopen it once per unit. An Occupational Standard holds around
# thirty units, and reopening a 100-page PDF for each of them dominated the
# cost of parsing one - enough that a library-wide check never finished. One
# entry is all that is needed: units are parsed a document at a time.
_OPEN_PATH = ""
_OPEN_PDF = None


def close_document() -> None:
    """Release the cached document. Safe when nothing is open."""
    global _OPEN_PATH, _OPEN_PDF
    if _OPEN_PDF is not None:
        try:
            _OPEN_PDF.close()
        except Exception:                   # noqa: BLE001 - closing must not raise
            pass
    _OPEN_PDF = None
    _OPEN_PATH = ""


def _open_document(path: str):
    """The open pdfplumber document for `path`, reusing the last one."""
    global _OPEN_PATH, _OPEN_PDF
    if _OPEN_PATH == path and _OPEN_PDF is not None:
        return _OPEN_PDF
    close_document()
    _OPEN_PDF = pdfplumber.open(path)
    _OPEN_PATH = path
    return _OPEN_PDF


def tables_in_pages(path: str, first: int, last: int) -> List[Table]:
    """Every usable table on pages [first, last], stitched across the range.

    Only the unit's own pages are read, not the whole document: a curriculum
    runs to 147 pages and a unit occupies six of them.
    """
    collected: List[Table] = []
    try:
        pdf = _open_document(path)
        last = min(last, len(pdf.pages) - 1)
        for index in range(max(first, 0), last + 1):
            page = pdf.pages[index]
            for found in page.find_tables():
                raw = found.extract() or []
                rows = [[_clean_cell(c) for c in row] for row in raw]
                rows = [r for r in rows if any(r)]
                if not rows:
                    continue
                top, bottom = found.bbox[1], found.bbox[3]
                width = max(len(r) for r in rows)
                previous = collected[-1] if collected else None

                # Same header restated at the top of a page: the header is
                # furniture, the rest is more of the same table.
                if previous is not None and _same_header(previous.header,
                                                         rows[0]):
                    previous.rows.extend(rows[1:])
                    previous.end_page = index
                    previous.bottom = bottom
                    previous.page_height = float(page.height)
                    continue

                if _continues(previous, index, top, width):
                    previous.rows.extend(rows)
                    previous.end_page = index
                    previous.bottom = bottom
                    previous.page_height = float(page.height)
                    continue

                collected.append(Table(
                    header=rows[0], rows=rows[1:], page=index,
                    end_page=index, bottom=bottom,
                    page_height=float(page.height)))
            # pdfplumber keeps every parsed object of a page it has visited.
            # Over a 150-page document that is a lot of memory for pages whose
            # tables have already been read.
            page.flush_cache()
    except Exception as e:                  # noqa: BLE001 - never lose a parse
        runlog.log(f"Tables: could not read {path} pages {first}-{last} "
                   f"({type(e).__name__}: {e})", level="WARN")
        return []

    for table in collected:
        table.rows = _stitch(table.rows)
    return collected


def find_table(tables: Sequence[Table], *patterns: Pattern) -> Optional[Table]:
    """The first table whose header satisfies every pattern.

    Asking for "the table headed ELEMENT and PERFORMANCE CRITERIA" survives a
    document that puts it on a different page, which a page number does not.
    """
    for table in tables:
        blob = " | ".join(table.header)
        if all(p.search(blob) for p in patterns):
            return table
    return None


def find_all(tables: Sequence[Table], *patterns: Pattern) -> List[Table]:
    """Every table whose header satisfies the patterns, in document order."""
    out = []
    for table in tables:
        blob = " | ".join(table.header)
        if all(p.search(blob) for p in patterns):
            out.append(table)
    return out
