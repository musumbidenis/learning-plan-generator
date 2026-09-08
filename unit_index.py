"""Locate the units inside an Occupational Standard or a Curriculum.

The original detector inferred unit starts from the SHAPE of a page: exactly one
ISCED-shaped code, in a header context, with a title above it. That reads the
CDACC documents it was tuned against, and fails silently on three shapes that
are common in the wild - each verified against real files:

  * a document that carries only TVET CDACC codes (IT/OS/ICTA/...) and no ISCED
    code at all - detection required an ISCED code, so it found nothing;
  * a multi-unit .docx - the loader produced ONE page for the whole file, so at
    most one unit could ever be found;
  * a header that prints the code ABOVE the title - the title search only ever
    walked upwards.

The fix hinges on something these documents nearly always carry: a table of
units in the preliminary pages ("Summary of Units of Learning", "UNIT CATEGORY |
UNIT CODE | UNITS NAME | DURATION"). That roster is authoritative - it names
every unit and its code before the body begins - so instead of guessing where
units start, we read the roster and then go and FIND each named unit.

Four strategies, and the one that finds the most units wins, so this can never
locate fewer units than the original detector did:

  roster      read the front-matter table, then locate each unit it names
  strict      the original shape rule (unchanged, so behaviour is preserved)
  relaxed     the shape rule with EITHER code family, title above or below
  structural  a 'Unit Description' heading near the top of a page

Anything the roster names but we cannot locate is reported in
`IndexResult.missing` rather than silently dropped, so a document that half
parses says so instead of looking complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from models import UnitRef
from pdf_utils import (
    Page,
    RE_CODE_LABEL as _RE_CODE_LABEL,
    RE_ISCED_CODE_SHAPE,
    RE_TVET_CODE_SHAPE,
    clean_text,
    column_lines,
    is_noise_line,
    unit_start_pages,
    unit_title_above_code,
)

def norm(s: str) -> str:
    """Comparison form of a code or title: lowercase, alphanumerics only.

    Defined here rather than imported from `curriculum_parser` so this module
    stays free of the parsers - they may import it, not the other way round.
    """
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# How far into a page's header block a unit title or marker may sit.
_HEADER_LINES = 12

# A roster page lists SEVERAL units; a unit's own page carries one code. Two
# distinct codes in row-shaped lines is the cheapest reliable separator.
_MIN_ROSTER_ROWS = 2

# Words that prefix a code in a roster row ("CORE 0611 351 01A Computer ...").
_RE_ROW_CATEGORY = re.compile(
    r"^\s*(CORE|COMMON|BASIC|ELECTIVE|OPTIONAL|COMPULSORY)\b\s*", re.I)

# Roster rows end in durations and credit factors ("... 150 15"); totals rows
# carry no code and so never parse as a row at all.
_RE_TRAILING_NUMBERS = re.compile(r"[\s\d.,()\-]+$")

# Lines that are table furniture rather than a unit title.
_RE_ROW_NOISE = re.compile(
    r"^(unit|units|code|title|name|category|duration|hours|credit|"
    r"sub\s*total|total|grand\s*total|s/?no)\b", re.I)

# Present at the top of a unit in both document families.
_RE_UNIT_SECTION = re.compile(r"UNIT\s+DESCRIPTION|^\s*Unit\s+Description\b", re.I)

# Corroborating evidence that a page really is a unit body, not a mention.
_RE_BODY_MARKER = re.compile(
    r"ELEMENTS?\s+AND\s+PERFORMANCE|PERFORMANCE\s+CR\w+|"
    r"Learning\s+Outcomes?,?\s+Content|UNIT\s+DESCRIPTION|"
    r"Unit\s+Description|Summary\s+of\s+Learning\s+Outcomes", re.I)


@dataclass
class RosterEntry:
    """One row of the preliminary units table."""
    code: str = ""
    title: str = ""
    page: int = 0

    @property
    def is_isced(self) -> bool:
        return bool(RE_ISCED_CODE_SHAPE.fullmatch((self.code or "").strip()))


@dataclass
class IndexResult:
    """The units found, plus what the roster promised and we couldn't locate."""
    refs: List[UnitRef] = field(default_factory=list)
    roster: List[RosterEntry] = field(default_factory=list)
    missing: List[RosterEntry] = field(default_factory=list)
    strategy: str = "none"


# --------------------------------------------------------------------------- #
# Codes
# --------------------------------------------------------------------------- #
def find_any_code(text: str) -> str:
    """The first code of EITHER family in *text*, or ''.

    Both families identify a unit; requiring the ISCED one is what made
    TVET-only documents unreadable.
    """
    m = RE_ISCED_CODE_SHAPE.search(text or "")
    if m:
        return clean_text(m.group(1))
    m = RE_TVET_CODE_SHAPE.search(text or "")
    return m.group(1) if m else ""


def _distinct_codes(text: str) -> set:
    out = {re.sub(r"\s+", "", c) for c in RE_ISCED_CODE_SHAPE.findall(text or "")}
    out |= {re.sub(r"\s+", "", c) for c in RE_TVET_CODE_SHAPE.findall(text or "")}
    return out


# --------------------------------------------------------------------------- #
# The preliminary units table
# --------------------------------------------------------------------------- #
def _roster_row(text: str) -> Optional[tuple]:
    """Parse '0611 351 01A Computer Essentials 80 8' -> ('0611 351 01A', 'Computer
    Essentials'), or None when the line isn't a units-table row."""
    line = clean_text(text)
    if not line or _RE_ROW_NOISE.match(line):
        return None
    line = _RE_ROW_CATEGORY.sub("", line)

    m = RE_ISCED_CODE_SHAPE.search(line) or RE_TVET_CODE_SHAPE.search(line)
    if not m:
        return None
    code = clean_text(m.group(1))

    # The title normally follows the code; some layouts put it first.
    for candidate in (line[m.end():], line[:m.start()]):
        title = _RE_TRAILING_NUMBERS.sub("", clean_text(candidate)).strip(" -|–")
        title = _RE_ROW_CATEGORY.sub("", title).strip()
        if len(title) >= 4 and not _RE_ROW_NOISE.match(title):
            return code, title
    return None


def read_roster(pages: Sequence[Page]) -> List[RosterEntry]:
    """Every unit named by a preliminary units table, in document order.

    A page qualifies when at least `_MIN_ROSTER_ROWS` of its lines parse as rows
    carrying distinct codes - a unit's own page has a single code and so never
    qualifies. Rows from several qualifying pages are merged and de-duplicated,
    because some documents split the table across a page boundary.
    """
    entries: List[RosterEntry] = []
    seen = set()
    for page in pages:
        rows = []
        for ln in column_lines(page.words, 0, 10_000):
            row = _roster_row(ln.text)
            if row:
                rows.append(row)
        if len({norm(c) for c, _ in rows}) < _MIN_ROSTER_ROWS:
            continue
        for code, title in rows:
            key = norm(code)
            if key and key not in seen:
                seen.add(key)
                entries.append(RosterEntry(code=code, title=title, page=page.index))
    return entries


# --------------------------------------------------------------------------- #
# Locating a unit the roster named
# --------------------------------------------------------------------------- #
def _header_lines(page: Page) -> List[str]:
    return [ln.text for ln in column_lines(page.words, 0, 10_000)[:_HEADER_LINES]]


def _looks_like_a_unit_body(page: Page) -> bool:
    """A real unit page carries one of the section markers both families use."""
    return bool(_RE_BODY_MARKER.search(page.text))


def _locate(pages: Sequence[Page], entry: RosterEntry,
            skip: set) -> Optional[int]:
    """The page where `entry`'s body begins, by title then by code."""
    want_title = norm(entry.title)
    want_code = norm(entry.code)

    # 1. the title standing alone as a heading in the page's header block
    for page in pages:
        if page.index in skip:
            continue
        for text in _header_lines(page):
            if norm(text) == want_title and _looks_like_a_unit_body(page):
                return page.index

    # 2. the code in the header block, with the body markers to back it up
    for page in pages:
        if page.index in skip:
            continue
        for text in _header_lines(page):
            if want_code and norm(text).find(want_code) >= 0 \
                    and _looks_like_a_unit_body(page):
                return page.index
    return None


def _roster_refs(pages: Sequence[Page], roster: List[RosterEntry],
                 source: str) -> tuple:
    """Locate every rostered unit; return (refs, entries we couldn't find)."""
    roster_pages = {e.page for e in roster}
    located: List[tuple] = []          # (start_page, entry)
    missing: List[RosterEntry] = []
    taken: set = set()

    for entry in roster:
        start = _locate(pages, entry, skip=roster_pages | taken)
        if start is None:
            missing.append(entry)
        else:
            taken.add(start)
            located.append((start, entry))

    located.sort(key=lambda t: t[0])
    starts = [s for s, _ in located]
    refs: List[UnitRef] = []
    for i, (start, entry) in enumerate(located):
        end = starts[i + 1] if i + 1 < len(starts) else len(pages)
        refs.append(UnitRef(
            title=entry.title,
            isced_code=entry.code if entry.is_isced else "",
            code="" if entry.is_isced else entry.code,
            source=source, start_page=start, end_page=end))
    return refs, missing


# --------------------------------------------------------------------------- #
# Shape-based fallbacks
# --------------------------------------------------------------------------- #
def _title_near_code(page: Page) -> str:
    """The unit title around the code line, whichever family the code is.

    `unit_title_above_code` searches upward from an ISCED code specifically, so
    it returns nothing at all on a TVET-only document. This repeats the search
    for either family, and falls forward to the lines BELOW the code for headers
    that print the code first.
    """
    lines = column_lines(page.words, 0, 10_000)
    idx = next((i for i, ln in enumerate(lines)
                if RE_ISCED_CODE_SHAPE.search(ln.text)
                or RE_TVET_CODE_SHAPE.search(ln.text)), None)
    if idx is None:
        return ""

    def usable(text: str) -> bool:
        t = (text or "").strip()
        return bool(t) and not is_noise_line(t)             and not RE_ISCED_CODE_SHAPE.search(t)             and not RE_TVET_CODE_SHAPE.search(t)             and not _RE_CODE_LABEL.search(t)             and not _RE_ROW_NOISE.match(t)             and not _RE_BODY_MARKER.search(t)

    for back in range(idx - 1, -1, -1):
        if usable(lines[back].text):
            return clean_text(lines[back].text)
    for ahead in lines[idx + 1:idx + 5]:
        if usable(ahead.text):
            return clean_text(ahead.text)
    return ""


def _relaxed_starts(pages: Sequence[Page]) -> List[int]:
    """The original rule, but accepting EITHER code family and a title on
    either side of the code."""
    out: List[int] = []
    for page in pages:
        codes = _distinct_codes(page.text)
        if len(codes) != 1:
            continue
        if not _looks_like_a_unit_body(page):
            continue
        if _title_near_code(page):
            out.append(page.index)
    return out


def _structural_starts(pages: Sequence[Page]) -> List[int]:
    """Last resort: a 'Unit Description' heading in a page's header block."""
    out: List[int] = []
    for page in pages:
        header = _header_lines(page)
        if any(_RE_UNIT_SECTION.search(t) for t in header):
            out.append(page.index)
    return out


def _refs_from_starts(pages: Sequence[Page], starts: Sequence[int],
                      source: str) -> List[UnitRef]:
    by_index = {p.index: p for p in pages}
    starts = sorted(set(starts))
    refs: List[UnitRef] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(pages)
        page = by_index.get(start)
        if page is None:
            continue
        title = unit_title_above_code(page) or _title_near_code(page)
        if not title:
            continue
        isced = RE_ISCED_CODE_SHAPE.search(page.text)
        tvet = RE_TVET_CODE_SHAPE.search(page.text)
        refs.append(UnitRef(
            title=title,
            isced_code=clean_text(isced.group(1)) if isced else "",
            code=tvet.group(1) if tvet else "",
            source=source, start_page=start, end_page=end))
    return refs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def index_units(pages: Sequence[Page], source: str) -> IndexResult:
    """Find every unit in *pages*, whichever way works best for this document.

    Each strategy is tried and the richest result wins, with the original strict
    detector preferred on a tie - so a document that reads correctly today keeps
    reading exactly the same way.
    """
    pages = list(pages or [])
    if not pages:
        return IndexResult()

    roster = read_roster(pages)
    candidates: List[tuple] = []       # (refs, strategy)

    if roster:
        refs, missing = _roster_refs(pages, roster, source)
        if refs:
            candidates.append((refs, "roster", missing))

    strict = _refs_from_starts(pages, unit_start_pages(pages), source)
    if strict:
        candidates.append((strict, "strict", []))
    if not candidates:
        relaxed = _refs_from_starts(pages, _relaxed_starts(pages), source)
        if relaxed:
            candidates.append((relaxed, "relaxed", []))
    if not candidates:
        structural = _refs_from_starts(pages, _structural_starts(pages), source)
        if structural:
            candidates.append((structural, "structural", []))

    if not candidates:
        return IndexResult(roster=roster, missing=list(roster))

    # Most units wins; 'strict' breaks a tie so today's documents are unchanged.
    order = {"strict": 0, "roster": 1, "relaxed": 2, "structural": 3}
    refs, strategy, missing = max(
        candidates, key=lambda c: (len(c[0]), -order.get(c[1], 9)))

    if strategy != "roster" and roster:
        # Report anything the roster named that the winning strategy didn't find.
        found = {norm(r.title) for r in refs} | {norm(r.isced_code) for r in refs} \
            | {norm(r.code) for r in refs}
        missing = [e for e in roster
                   if norm(e.title) not in found and norm(e.code) not in found]

    return IndexResult(refs=refs, roster=roster, missing=missing,
                       strategy=strategy)
