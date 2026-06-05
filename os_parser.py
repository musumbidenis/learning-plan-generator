"""Module A1 - DETERMINISTIC Occupational-Standard parser (NO AI).

Kenya CDACC Occupational Standards are rigidly structured, so we extract by
pattern and word-coordinate, never by AI. A tiny AI fallback exists ONLY for the
degenerate case where zero units parse (a scanned/garbled OS); it lives in
ai_client.py and is invoked by the caller, not here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from models import Element, PerformanceCriterion, Unit, UnitRef
from pdf_utils import (
    Line,
    Page,
    RE_LEVEL,
    RE_OS_CODE,
    clean_text,
    column_lines,
    detect_column_split,
    is_noise_line,
    load_document,
)

# Header / boilerplate lines that appear inside the Element|PC table and must
# never be treated as element titles or PC continuations.
_HEADER_PHRASES = (
    "element", "performance criteria",
    "these describe", "these are assessable", "outcomes which make up",
    "outcomes that make up", "workplace function", "workplace functions",
    "bold and italicized", "(bold and italicized", "level of performance",
    "required level", "which specify the", "that specify the",
    "elaborated in the range",
)

# Section markers that END the elements/PC region. Case-SENSITIVE on purpose:
# these are standalone ALL-CAPS section headers, so we must not match the
# lowercase 'range)' inside the note '(... elaborated in the range)'.
_REGION_END = re.compile(r"^\s*(RANGE|REQUIRED\s+SKILLS|REQUIRED\s+KNOWLEDGE|"
                         r"REQUIRED\s+SKILLS\s+AND\s+KNOWLEDGE|EVIDENCE\s+GUIDE)\b")

# Region-start marker. CDACC OS documents use two layouts:
#   (a) a section heading line "ELEMENTS AND PERFORMANCE CRITERIA" (older units);
#   (b) NO heading - the table starts directly with its column-header row
#       "ELEMENT | PERFORMANCE CRITERIA" (newer units, e.g. the Computer Science
#       OS: Fundamentals of Programming, Mathematics, AI, Computer Organisation,
#       Networks, Algorithms, Information System).
# We detect either: the heading, both column headers on one reconstructed line,
# or a line that begins with the PC column header alone (when the two headers
# land on separate baselines). 'CRITERIA' is misspelled 'CRETIRIA' in places, so
# tolerate CR\w+.
_RE_REGION_START = re.compile(
    r"ELEMENTS?\s+AND\s+PERFORMANCE\s+\w+"               # (a) section heading
    r"|^\s*ELEMENTS?\b.{0,30}?\bPERFORMANCE\s+CR\w+"     # (b) both headers, one line
    r"|^\s*PERFORMANCE\s+CRITERIA\b",                    # (b) PC column header alone
    re.I)

_RE_ELEMENT = re.compile(r"^(\d+)\.(?!\d)\s+(.*\S)")        # '1. Apply ...' (not 1.1)
# PC numbering is usually '1.1 text' but some units write '1.1.text' (trailing
# dot, no space) -> tolerate an optional dot and optional whitespace.
_RE_PC = re.compile(r"^(\d+\.\d+)\.?\s*(.*\S)")             # '1.1 ...' or '1.1.text'
_RE_ACTION_VERB = re.compile(
    r"\b(apply|applying|perform|performing|demonstrate|demonstrating|manage|"
    r"managing|setup|set\s?up|install|installing|configure|configuring|"
    r"develop|developing|implement|implementing|maintain|maintaining|"
    r"identify|identifying|produce|producing|create|creating|conduct|"
    r"conducting|operate|operating)\b", re.I)


def _is_header_line(text: str) -> bool:
    low = text.lower().strip()
    return any(low.startswith(p) or low == p for p in _HEADER_PHRASES)


# --------------------------------------------------------------------------- #
# Unit boundary detection
# --------------------------------------------------------------------------- #
def _unit_start_pages(pages: List[Page]) -> List[int]:
    """Pages that begin a unit: contain ISCED UNIT CODE and UNIT DESCRIPTION."""
    starts = []
    for p in pages:
        up = p.text.upper()
        if "ISCED UNIT CODE" in up and "UNIT DESCRIPTION" in up:
            starts.append(p.index)
    return starts


def _unit_title_from_first_page(page: Page) -> str:
    """The ALL-CAPS line directly above 'ISCED UNIT CODE:'."""
    # Reconstruct full-width lines from words (top-to-bottom).
    lines = column_lines(page.words, x_min=0, x_max=10_000)
    for idx, ln in enumerate(lines):
        if "ISCED UNIT CODE" in ln.text.upper():
            # walk backwards to the nearest non-noise ALL-CAPS title line
            for back in range(idx - 1, -1, -1):
                cand = lines[back].text.strip()
                if not cand or is_noise_line(cand):
                    continue
                letters = [c for c in cand if c.isalpha()]
                if letters and all(c.isupper() for c in letters):
                    return clean_text(cand)
                # first meaningful non-caps line above -> stop
                return clean_text(cand)
    return ""


# --------------------------------------------------------------------------- #
# Element / PC extraction (the anti-bleed core)
# --------------------------------------------------------------------------- #
def _pc_anchored_split(win: List[dict], fallback: float) -> float:
    """Locate the PC column's left edge from where 'x.y' PC-number tokens align.

    `detect_column_split` finds the widest x-gap in a fixed band and is fooled on
    units whose PC column starts far left (x0~217) with deeper-indented wrap lines
    (x0~248): it splits INSIDE the PC column, so every PC lands in the element
    bucket and zero PCs parse (seen in the Computer-Science OS for 'Computer
    Organisation' and 'Artificial Intelligence'). PC numbers ('1.1', '2.3', ...)
    reliably left-align at the column edge, so anchor the split just left of their
    leftmost cluster; fall back to the gap heuristic when no PC token is present.
    """
    pc_xs = [w["x0"] for w in win
             if re.match(r"^\d+\.\d+\.?$", (w["text"] or "").strip())]
    if not pc_xs:
        return fallback
    # leftmost cluster: the smallest rounded x0 that occurs at least twice
    # (a single stray decimal token can't move the boundary).
    counts = Counter(round(x) for x in pc_xs)
    repeated = sorted(x for x, n in counts.items() if n >= 2)
    pc_left = float(repeated[0]) if repeated else min(pc_xs)
    # element-column left edge = leftmost element-number token ('1.', '2.', ...)
    el_xs = [w["x0"] for w in win if re.match(r"^\d+\.$", (w["text"] or "").strip())]
    el_left = min(el_xs) if el_xs else 0.0
    # only trust the anchor when PCs sit clearly right of the element column;
    # otherwise the layout isn't the two-column element|PC table we expect.
    if pc_left - el_left < 25.0:
        return fallback
    return pc_left - 4.0


def _extract_elements(unit_pages: List[Page]) -> List[Element]:
    elements: List[Element] = []
    cur_el: Optional[Element] = None
    cur_pc: Optional[PerformanceCriterion] = None
    started = False

    for page in unit_pages:
        # Full-width line reconstruction is used only to locate the region's
        # vertical bounds on this page (the markers may sit in either column).
        full_lines = column_lines(page.words, 0, 10_000)
        start_y: Optional[float] = None
        end_y: Optional[float] = None
        for ln in full_lines:
            if start_y is None and not started and _RE_REGION_START.search(ln.text):
                start_y = ln.top
            if started or start_y is not None:
                if _REGION_END.search(ln.text):
                    end_y = ln.top
                    break

        if not started and start_y is None:
            continue                          # region hasn't begun yet
        started = True

        # Vertical window for the element/PC table on this page.
        y0 = (start_y + 1) if start_y is not None else -1.0
        y1 = end_y if end_y is not None else 1e9
        win = [w for w in page.words if y0 < w["top"] < y1]
        if not win:
            if end_y is not None:
                break
            continue

        split_x = _pc_anchored_split(win, fallback=detect_column_split(win))
        left = column_lines([w for w in win if w["x0"] < split_x], 0, 10_000)
        right = column_lines([w for w in win if w["x0"] >= split_x], 0, 10_000)

        # --- LEFT column: element titles ---------------------------------- #
        for ln in left:
            if _is_header_line(ln.text):
                continue
            m = _RE_ELEMENT.match(ln.text)
            if m:
                cur_el = Element(number=m.group(1), title=clean_text(m.group(2)))
                elements.append(cur_el)
                cur_pc = None
            elif cur_el is not None and not _RE_PC.match(ln.text):
                cur_el.title = clean_text(cur_el.title + " " + ln.text)

        # --- RIGHT column: performance criteria --------------------------- #
        for ln in right:
            if _is_header_line(ln.text):
                continue
            m = _RE_PC.match(ln.text)
            if m:
                cur_pc = PerformanceCriterion(number=m.group(1),
                                              text=clean_text(m.group(2)))
                major = m.group(1).split(".")[0]
                target = next((e for e in elements if e.number == major), cur_el)
                if target is not None:
                    target.performance_criteria.append(cur_pc)
            elif cur_pc is not None:
                cur_pc.text = clean_text(cur_pc.text + " " + ln.text)

        if end_y is not None:
            break                             # region finished on this page

    return elements


# --------------------------------------------------------------------------- #
# Description, skill task, assessment methods
# --------------------------------------------------------------------------- #
def _extract_description(first_page_text: str) -> str:
    # End the description at the table start, whether that is the heading
    # ("ELEMENTS AND PERFORMANCE ...") or the bare column header ("ELEMENT
    # PERFORMANCE CRITERIA" / "PERFORMANCE CRITERIA") used by heading-less units.
    m = re.search(
        r"UNIT\s+DESCRIPTION\s*:?(.*?)"
        r"(?:ELEMENTS?\s+AND\s+PERFORMANCE"
        r"|ELEMENTS?\s+PERFORMANCE\s+CR\w+"
        r"|PERFORMANCE\s+CRITERIA)",
        first_page_text, re.S | re.I)
    if not m:
        return ""
    return clean_text(re.sub(r"\s+", " ", m.group(1)))


def _skill_task_from_description(description: str) -> str:
    """The description sentence, copied from its first action verb."""
    if not description:
        return ""
    m = _RE_ACTION_VERB.search(description)
    if m:
        return clean_text(description[m.start():])
    return description


def _extract_assessment_methods(unit_pages: List[Page]) -> List[str]:
    """Parse EVIDENCE GUIDE -> 'Methods of Assessment' sub-items (e.g. 3.1 ...)."""
    # Join the text of all unit pages, then isolate the Methods-of-Assessment block.
    full = "\n".join(p.text for p in unit_pages)
    # The label often wraps as 'Methods of\nassessment', so anchor on 'Methods of'
    # and capture every numbered sub-item up to the next section ('Context of ...').
    m = re.search(r"Methods\s+of\b(.*?)(?:\n\s*\d+\.\s*Context|Context\s+of|"
                  r"Guidance\s+information|\Z)", full, re.S | re.I)
    block = m.group(1) if m else ""
    methods: List[str] = []
    for line in block.splitlines():
        line = clean_text(line)
        # sub-items look like '3.1 Observation' (possibly preceded by 'Assessment')
        mm = re.search(r"\b\d+\.\d+\s+(.+)", line)
        if mm:
            cand = clean_text(mm.group(1))
            # drop trailing boilerplate words
            cand = re.sub(r"\b(Assessment|Competency in this unit.*)$", "", cand).strip()
            if cand and not is_noise_line(cand) and len(cand) > 2:
                methods.append(cand)
    # de-duplicate, keep order
    seen = set()
    out = []
    for x in methods:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# Bulleted line in the Required-Skills/Knowledge lists. Bullets render as •/●/▪,
# or as the mangled replacement glyph (U+FFFD) when the font isn't embedded.
_RE_BULLET = re.compile(r"^\s*[•●▪◦·�‣⁌•●▪○·\*]\s*(.+)")


def _extract_required_knowledge(unit_pages: List[Page]) -> List[str]:
    """Parse the 'Required Knowledge' bullet list (underpinning theory).

    Used to enrich the plan - especially the OS-only fallback, where there is no
    curriculum content and these items become the real Learning Key Points.
    """
    full = "\n".join(p.text for p in unit_pages)
    # 'Required Knowledge' subsection runs until the EVIDENCE GUIDE section.
    m = re.search(r"Required\s+Knowledge\b(.*?)(?:EVIDENCE\s+GUIDE|\Z)",
                  full, re.S | re.I)
    block = m.group(1) if m else ""
    out: List[str] = []
    seen = set()
    for raw in block.splitlines():
        mb = _RE_BULLET.match(raw)
        if not mb:
            continue
        item = clean_text(mb.group(1))
        if not item or len(item) < 3 or is_noise_line(item):
            continue
        k = item.lower()
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_os(path: str) -> List[Unit]:
    """Parse every unit of competency out of an Occupational Standard PDF/DOCX."""
    pages = load_document(path)
    return parse_os_pages(pages)


def _cover_level(pages: List[Page]) -> str:
    """The programme KNQF level from the cover/front matter (default for units)."""
    for p in pages[:6]:
        m = re.search(r"KNQF\s+LEVEL\s*[:\-]?\s*(\d+)", p.text, re.I)
        if m:
            return m.group(1)
    return ""


def _isced_of(page_text: str) -> str:
    m = re.search(r"ISCED\s+UNIT\s+CODE\s*:?\s*([0-9][0-9A-Za-z ]+)", page_text, re.I)
    return clean_text(m.group(1)) if m else ""


def _build_unit(unit_pages: List[Page], cover_level: str) -> Unit:
    """Full deterministic extraction for ONE unit (its page span)."""
    first = unit_pages[0]
    unit = Unit()
    unit.unit_title = _unit_title_from_first_page(first)

    mcode = RE_OS_CODE.search(first.text)
    if mcode:
        unit.os_code = mcode.group(1).strip()
    unit.isced_code = _isced_of(first.text)
    mlevel = RE_LEVEL.search(first.text)
    unit.level = mlevel.group(1) if mlevel else cover_level

    unit.description = _extract_description(first.text)
    unit.skill_task = _skill_task_from_description(unit.description)
    unit.elements = _extract_elements(unit_pages)
    unit.assessment_methods = _extract_assessment_methods(unit_pages)
    unit.required_knowledge = _extract_required_knowledge(unit_pages)
    return unit


def index_os_units(pages: List[Page]) -> List[UnitRef]:
    """Cheap pass: list every unit's identity (title + codes) WITHOUT extracting
    elements/PCs. Used to populate the unit-selection list quickly."""
    starts = _unit_start_pages(pages)
    by_index = {p.index: p for p in pages}
    refs: List[UnitRef] = []
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else len(pages)
        first = by_index.get(start)
        if first is None:
            continue
        title = _unit_title_from_first_page(first)
        mcode = RE_OS_CODE.search(first.text)
        code = mcode.group(1).strip() if mcode else ""
        if title:
            refs.append(UnitRef(title=title, isced_code=_isced_of(first.text),
                                code=code, source="OS",
                                start_page=start, end_page=end))
    return refs


def parse_os_unit(pages: List[Page], ref: UnitRef) -> Unit:
    """Deep-parse a single unit identified by its UnitRef page span."""
    cover_level = _cover_level(pages)
    unit_pages = [p for p in pages if ref.start_page <= p.index < ref.end_page]
    return _build_unit(unit_pages, cover_level)


def parse_os_pages(pages: List[Page]) -> List[Unit]:
    cover_level = _cover_level(pages)
    starts = _unit_start_pages(pages)
    units: List[Unit] = []
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else len(pages)
        unit_pages = [p for p in pages if start <= p.index < end]
        unit = _build_unit(unit_pages, cover_level)
        if unit.unit_title and unit.elements:
            units.append(unit)
    return units


def find_unit(units: List[Unit], query: str) -> Optional[Unit]:
    """Match a unit by ISCED code, OS code, or fuzzy title."""
    if not query:
        return None
    q = query.strip().lower()
    qn = re.sub(r"[^a-z0-9]", "", q)
    for u in units:
        if qn and (qn == re.sub(r"[^a-z0-9]", "", u.isced_code.lower())
                   or qn == re.sub(r"[^a-z0-9]", "", u.os_code.lower())):
            return u
    for u in units:
        if q in u.unit_title.lower() or u.unit_title.lower() in q:
            return u
    # token-overlap fallback
    best, best_score = None, 0.0
    qtokens = set(re.findall(r"[a-z]+", q))
    for u in units:
        utokens = set(re.findall(r"[a-z]+", u.unit_title.lower()))
        if not utokens:
            continue
        score = len(qtokens & utokens) / len(qtokens | utokens)
        if score > best_score:
            best, best_score = u, score
    return best if best_score >= 0.4 else None
