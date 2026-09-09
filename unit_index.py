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
    norm,
    norm_code_loose,
    unit_start_pages,
    unit_title_above_code,
)


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
        if len(title) < 4 or _RE_ROW_NOISE.match(title):
            continue
        # 'TVET CDACC UNIT CODE: IT/CU/ICTA/CR/01/4/MA' is a unit page's own
        # header, not a table row. Without this every unit page counted as a
        # roster page and contributed a unit named 'TVET CDACC UNIT CODE:'.
        if _RE_CODE_LABEL.search(title):
            continue
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
# 'ISCED UNIT CODE: 0541 541 01A' - a unit states its code under a label; a row
# of a units table never does.
_RE_CODE_FIELD = re.compile(r"\bCODE\s*:", re.I)


def _code_anchor(lines: Sequence) -> Optional[int]:
    """Index of the line bearing this unit's OWN code.

    A curriculum prints each module's units table immediately above the first
    unit of that module, on the same page:

        MODULE 6
        CORE 0612 551 IT/CU/ICTA/CR/02/6/MA ICT Security 150
        ...
        ICT SECURITY                             <- the unit really starts here
        ISCED UNIT CODE: 0612 551 16A

    Anchoring on the first code-shaped line lands inside that table, and the
    title search then reports the unit as 'MODULE 6'. The labelled line is the
    unit's own; documents that use no label are unaffected, since the first
    code-shaped line is then still the anchor.
    """
    for i, ln in enumerate(lines):
        text = ln.text
        if _RE_CODE_FIELD.search(text) and (RE_ISCED_CODE_SHAPE.search(text)
                                            or RE_TVET_CODE_SHAPE.search(text)):
            return i
    for i, ln in enumerate(lines):
        if RE_ISCED_CODE_SHAPE.search(ln.text) or RE_TVET_CODE_SHAPE.search(ln.text):
            return i
    return None


def _title_near_code(page: Page) -> str:
    """The unit title around the code line, whichever family the code is.

    `unit_title_above_code` searches upward from an ISCED code specifically, so
    it returns nothing at all on a TVET-only document. This repeats the search
    for either family, and falls forward to the lines BELOW the code for headers
    that print the code first.
    """
    lines = column_lines(page.words, 0, 10_000)
    idx = _code_anchor(lines)
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


# How far above a code line a unit title may reasonably sit.
_TITLE_LOOKBACK = 6
# Below this a candidate reads as prose, not as a heading.
_TITLE_SCORE_FLOOR = 1.0


def _title_score(text: str) -> float:
    """How much a line reads like a unit title rather than body prose.

    Unit titles in these documents are short and upper-case ("APPLY
    COMMUNICATION SKILLS"). Taking the nearest usable line instead picked up
    trailing evidence-guide prose from the preceding unit - real examples being
    "Oral questioning Context of assessment" and "In a simulated work
    environment Guidance information".
    """
    t = (text or "").strip()
    if not (3 <= len(t) <= 90):
        return -1.0
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return -1.0

    score = 2.0 * (sum(1 for c in letters if c.isupper()) / len(letters))
    words = t.split()
    if len(words) > 12:
        score -= 1.0
    elif len(words) <= 8:
        score += 0.3
    if t.endswith("."):
        score -= 1.0                      # a sentence, not a heading
    return score


# "This unit covers the competencies required to perform computer repair and
# maintenance. It entails ..." -> "Perform computer repair and maintenance"
_RE_DESCRIPTION = re.compile(
    r"UNIT\s+DESCRIPTION\s*:?\s*(.+?)"
    r"(?:ELEMENTS?\s+AND\s+PERFORMANCE|PERFORMANCE\s+CR\w+|"
    r"Summary\s+of\s+Learning|\Z)", re.I | re.S)
_RE_DESCRIPTION_LEAD = re.compile(
    r"^.*?\b(?:competenc\w*\s+(?:required\s+)?to|required\s+to)\s+", re.I | re.S)


def _title_from_description(page: Page) -> str:
    """Fall back to the unit description when no heading is on the page.

    Some documents - legacy .doc conversions especially - carry the unit code
    and description but leave the heading behind on the previous page. The
    description still says exactly what the unit is ("...required to perform
    computer repair and maintenance"), which makes a far better title than the
    stray line of prose that happens to sit above the code.
    """
    m = _RE_DESCRIPTION.search(page.text or "")
    if not m:
        return ""
    body = clean_text(re.sub(r"\s+", " ", m.group(1)))
    lead = _RE_DESCRIPTION_LEAD.search(body)
    if lead:
        body = body[lead.end():]
    # keep the first clause: "...repair and maintenance. It entails ..."
    body = re.split(r"(?<=[a-z])\.\s|\.\s+It\s", body)[0].strip(" .")
    if len(body) < 4 or len(body) > 120:
        return ""
    return body[:1].upper() + body[1:]


def _best_title(page: Page) -> str:
    """The line above the code that best reads as this unit's title."""
    lines = column_lines(page.words, 0, 10_000)
    idx = _code_anchor(lines)
    if idx is None:
        return ""

    def usable(text: str) -> bool:
        t = (text or "").strip()
        return bool(t) and not is_noise_line(t) \
            and not RE_ISCED_CODE_SHAPE.search(t) \
            and not RE_TVET_CODE_SHAPE.search(t) \
            and not _RE_CODE_LABEL.search(t) \
            and not _RE_ROW_NOISE.match(t) \
            and not _RE_BODY_MARKER.search(t)

    window = [ln.text for ln in lines[max(0, idx - _TITLE_LOOKBACK):idx]
              if usable(ln.text)]
    if window:
        # nearest-first on a tie, so an adjacent heading beats a distant one
        best = max(reversed(window), key=_title_score)
        if _title_score(best) >= _TITLE_SCORE_FLOOR:
            return clean_text(best)
        # nothing above reads like a heading - the description knows better
        # than a stray line of prose does
        return _title_from_description(page) or clean_text(window[-1])

    # nothing usable above the code: the header may print the code first
    below = _title_near_code(page)
    if below and _title_score(below) >= _TITLE_SCORE_FLOOR:
        return below
    return _title_from_description(page) or below


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


def _unit_codes(page: Page) -> tuple:
    """(ISCED, TVET) as the unit itself states them, not as a table lists them.

    Read from the top of the page, a module's units table answers first - and
    its cells wrap, so the code comes back clipped ('IT/CU/ICTA/CC/01/6/M').
    Reading from the unit's own code line gets the whole thing.
    """
    lines = column_lines(page.words, 0, 10_000)
    anchor = _code_anchor(lines)
    texts = ["\n".join(ln.text for ln in lines[anchor:])] if anchor is not None else []
    texts.append(page.text)
    isced = tvet = ""
    for text in texts:
        if not isced:
            m = RE_ISCED_CODE_SHAPE.search(text)
            isced = clean_text(m.group(1)) if m else ""
        if not tvet:
            m = RE_TVET_CODE_SHAPE.search(text)
            tvet = m.group(1) if m else ""
    return isced, tvet


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
        title = _best_title(page) or unit_title_above_code(page) \
            or _title_near_code(page)
        if not title:
            continue
        isced_code, tvet_code = _unit_codes(page)
        refs.append(UnitRef(
            title=title,
            isced_code=isced_code, code=tvet_code,
            source=source, start_page=start, end_page=end))
    return refs


# --------------------------------------------------------------------------- #
# Reconciling the roster against what was found
# --------------------------------------------------------------------------- #
# Below this a shared prefix says nothing - 'com' prefixes half the units in an
# ICT curriculum.
_MIN_PREFIX = 6


def _shares_a_start(a: str, b: str) -> bool:
    """Whether two normalised strings agree as far as the shorter one goes.

    Both roster codes and roster titles arrive clipped, because the cell they
    sit in wraps and only its first line is read: 'IT/CU/ICTA/CC/01/6/M' for
    ...MA, 'Network Design and' for 'Network Design and Management'.
    """
    if not a or not b:
        return False
    short, long = sorted((a, b), key=len)
    return len(short) >= _MIN_PREFIX and long.startswith(short)


def _entry_codes(entry: RosterEntry) -> set:
    """Every code the roster row carries, its own and any inside its title.

    A units table often prints the two code families in adjacent columns, which
    the reader joins into one cell: code '0612 451 07A', title
    'IT/CU/ICTA/CR/02/5/MA Network Design and'. Comparing only the first of
    those against a document that quotes the other reported a located unit as
    missing.
    """
    codes = {entry.code}
    text = entry.title or ""
    codes |= set(RE_ISCED_CODE_SHAPE.findall(text))
    codes |= set(RE_TVET_CODE_SHAPE.findall(text))
    return {norm_code_loose(c) for c in codes if c}


def _entry_title(entry: RosterEntry) -> str:
    """The roster row's title with any code taken back out of it."""
    text = RE_ISCED_CODE_SHAPE.sub(" ", entry.title or "")
    text = RE_TVET_CODE_SHAPE.sub(" ", text)
    return norm(text)


def _entry_was_found(entry: RosterEntry, refs: Sequence[UnitRef]) -> bool:
    """Whether a unit the roster names is among the units actually located."""
    codes = _entry_codes(entry)
    title = _entry_title(entry)
    for ref in refs:
        ref_codes = {norm_code_loose(c) for c in (ref.code, ref.isced_code) if c}
        if codes & ref_codes:
            return True
        if any(_shares_a_start(a, b) for a in codes for b in ref_codes):
            return True
        ref_title = norm(ref.title)
        if title and (title == ref_title or _shares_a_start(title, ref_title)):
            return True
    return False


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
        missing = [e for e in roster if not _entry_was_found(e, refs)]

    return IndexResult(refs=refs, roster=roster, missing=missing,
                       strategy=strategy)
