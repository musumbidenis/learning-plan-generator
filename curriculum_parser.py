"""Module A2 - DETERMINISTIC Curriculum parser (NO AI).  <-- KEY UPGRADE

The Curriculum's "Learning Outcomes, Content and Suggested Assessment Methods"
table holds the REAL syllabus content. We extract, per Learning Outcome:

    * lo_title          (e.g. 'Apply computer programming skills')
    * sub_topics[]      each 'x.y' content heading -> ONE teaching session
        - title         (e.g. 'Identification of Programming Languages')
        - key_points[]  the 'x.y.z' child lines  (-> Learning Key Points column)
    * suggested_methods the assessment-method hints listed for the unit

The three columns are separated by WORD X-COORDINATES:
    LO column        x0 <  ~150
    Content column   ~150 <= x0 < ~435
    Assessment col   x0 >= ~435
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from models import CurriculumUnit, LearningOutcome, SubTopic, UnitRef
from pdf_utils import (
    Page,
    clean_text,
    column_lines,
    find_isced_code,
    find_tvet_code,
    is_noise_line,
    load_document,
    page_starts_unit,
    unit_title_above_code,
)

# Column boundaries (tuned to CDACC curriculum layout).
LO_MAX_X = 150.0
CONTENT_MAX_X = 436.0

_RE_TABLE_START = re.compile(r"Learning\s+[Oo]utcomes?,\s+Content", re.I)
_RE_TABLE_END = re.compile(r"(Suggested\s+Delivery\s+Methods|Recommended\s+Resources)", re.I)

_RE_LO = re.compile(r"^(\d+)\.(?!\d)\s+(.*\S)")                 # '1. Apply ...'
_RE_SUBTOPIC = re.compile(r"^(\d+\.\d+)(?!\.\d)\s+(.*\S)")      # '1.1 Title' (exactly x.y)
_RE_KEYPOINT = re.compile(r"^(\d+\.\d+(?:\.\d+)+)\s+(.*\S)")    # '1.1.1 ...' (x.y.z+)


def _unit_start_pages(pages: List[Page]) -> List[int]:
    """Pages that begin a unit, detected by the ISCED code SHAPE (not a label)
    via the shared `page_starts_unit`.

    Earlier versions matched a 'UNIT CODE: <code>' label, which broke on
    differently-worded headers and on module-summary tables. `page_starts_unit`
    anchors on the code shape and guards against those tables (exactly one ISCED
    code per page + a non-empty title above it), so it captures each unit's
    first page - including units whose header sits at the bottom of a page with
    the rest of the unit overflowing onto the next.
    """
    return [p.index for p in pages if page_starts_unit(p)]


def _table_window(unit_pages: List[Page]):
    """Yield (page, y_lo, y_hi) for the rows that fall inside the content table."""
    started = False
    for page in unit_pages:
        y_lo, y_hi = 0.0, 1e9
        lines = column_lines(page.words, 0, 10_000)
        if not started:
            for ln in lines:
                if _RE_TABLE_START.search(ln.text):
                    y_lo = ln.top + 1
                    started = True
                    break
            if not started:
                continue
        # find an end marker on this page
        for ln in lines:
            if _RE_TABLE_END.search(ln.text) and ln.top > y_lo:
                y_hi = ln.top
                yield page, y_lo, y_hi
                return
        yield page, y_lo, y_hi


def _window_words(page: Page, y_lo: float, y_hi: float, x_min: float, x_max: float):
    return [w for w in page.words if y_lo <= w["top"] < y_hi]


def parse_curriculum(path: str) -> List[CurriculumUnit]:
    pages = load_document(path)
    return parse_curriculum_pages(pages)


def parse_curriculum_pages(pages: List[Page]) -> List[CurriculumUnit]:
    starts = _unit_start_pages(pages)
    units: List[CurriculumUnit] = []
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else len(pages)
        unit_pages = [p for p in pages if start <= p.index < end]
        unit = _parse_one_unit(unit_pages)
        if unit and unit.learning_outcomes:
            units.append(unit)
    return units


def index_curriculum_units(pages: List[Page]) -> List[UnitRef]:
    """Cheap pass: list every curriculum unit's identity WITHOUT extracting the
    full content table."""
    starts = _unit_start_pages(pages)
    by_index = {p.index: p for p in pages}
    refs: List[UnitRef] = []
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else len(pages)
        first = by_index.get(start)
        if first is None:
            continue
        title = unit_title_above_code(first)
        if title:
            refs.append(UnitRef(title=title, isced_code=find_isced_code(first.text),
                                code=find_tvet_code(first.text),
                                source="CU", start_page=start, end_page=end))
    return refs


def parse_curriculum_unit(pages: List[Page], ref: UnitRef) -> Optional[CurriculumUnit]:
    """Deep-parse a single curriculum unit identified by its UnitRef page span."""
    unit_pages = [p for p in pages if ref.start_page <= p.index < ref.end_page]
    return _parse_one_unit(unit_pages)


def _parse_one_unit(unit_pages: List[Page]) -> Optional[CurriculumUnit]:
    first = unit_pages[0]
    unit = CurriculumUnit()

    # Unit identity by code shape, label-independent (see pdf_utils helpers).
    unit.unit_title = unit_title_above_code(first)
    unit.curriculum_code = find_tvet_code(first.text)
    unit.isced_code = find_isced_code(first.text)
    mdesc = re.search(r"Unit\s+Description\s*:?(.*?)Summary\s+of\s+Learning",
                      first.text, re.S | re.I)
    if mdesc:
        unit.description = clean_text(re.sub(r"\s+", " ", mdesc.group(1)))

    # --- walk the content table --------------------------------------------- #
    lo_titles: Dict[str, str] = {}      # major number -> LO title
    sub_topics: List[SubTopic] = []     # in document order
    methods: List[str] = []
    cur_sub: Optional[SubTopic] = None
    cur_lo_major: Optional[str] = None
    cur_lo_title_parts: List[str] = []

    def flush_lo_title():
        if cur_lo_major and cur_lo_title_parts:
            lo_titles.setdefault(cur_lo_major,
                                 clean_text(" ".join(cur_lo_title_parts)))

    for page, y_lo, y_hi in _table_window(unit_pages):
        words = _window_words(page, y_lo, y_hi, 0, 10_000)

        # The Suggested-Assessment-Methods column is bulleted; detect its left
        # edge per page from the '•' x-positions (it drifts between 429 and 438).
        bullet_xs = [w["x0"] for w in words if w["text"].strip() in ("•", "", "▪")]
        content_max = (min(bullet_xs) - 4.0) if bullet_xs else CONTENT_MAX_X

        lo_lines = column_lines([w for w in words if w["x0"] < LO_MAX_X], 0, 10_000)
        content_lines = column_lines(
            [w for w in words if LO_MAX_X <= w["x0"] < content_max], 0, 10_000)
        method_lines = column_lines(
            [w for w in words if w["x0"] >= content_max], 0, 10_000)

        # LO column -> titles (continuation lines extend the current LO title)
        for ln in lo_lines:
            m = _RE_LO.match(ln.text)
            if m:
                flush_lo_title()
                cur_lo_major = m.group(1)
                cur_lo_title_parts = [m.group(2)]
            elif cur_lo_major and not _RE_SUBTOPIC.match(ln.text):
                cur_lo_title_parts.append(ln.text)
        flush_lo_title()

        # Content column -> sub-topics & key points
        for ln in content_lines:
            text = ln.text
            m_sub = _RE_SUBTOPIC.match(text)
            m_kp = _RE_KEYPOINT.match(text)
            if m_kp:
                if cur_sub is not None:
                    cur_sub.key_points.append(clean_text(m_kp.group(2)))
            elif m_sub:
                cur_sub = SubTopic(number=m_sub.group(1),
                                   title=clean_text(m_sub.group(2)))
                sub_topics.append(cur_sub)
            else:
                # continuation: extend last key point, else the sub-topic title
                if cur_sub is not None and cur_sub.key_points:
                    cur_sub.key_points[-1] = clean_text(
                        cur_sub.key_points[-1] + " " + text)
                elif cur_sub is not None:
                    cur_sub.title = clean_text(cur_sub.title + " " + text)

        # Assessment column -> method hints. Each method starts with a bullet;
        # un-bulleted lines are wrapped continuations of the previous method.
        for ln in method_lines:
            raw = ln.text.strip()
            is_bullet = bool(re.match(r"^[•▪\*\-]", raw))
            t = re.sub(r"^[•▪\*\-]\s*", "", raw).strip()
            if not t or is_noise_line(t):
                continue
            if is_bullet:
                methods.append(clean_text(t))
            elif methods:
                methods[-1] = clean_text(methods[-1] + " " + t)
            # else: a pre-bullet header line (e.g. 'Methods') -> ignore

    # de-dup methods, keep order
    unique_methods = []
    seen = set()
    for mth in methods:
        k = mth.lower()
        if k not in seen and len(mth) > 2:
            seen.add(k)
            unique_methods.append(mth)

    # Group sub-topics into LOs by their major number.
    los: Dict[str, LearningOutcome] = {}
    order: List[str] = []
    for st in sub_topics:
        major = st.number.split(".")[0]
        if major not in los:
            los[major] = LearningOutcome(number=major,
                                         title=lo_titles.get(major, ""),
                                         suggested_methods=unique_methods)
            order.append(major)
        los[major].sub_topics.append(st)

    unit.learning_outcomes = [los[k] for k in order]
    return unit


def find_unit(units: List[CurriculumUnit], isced_code: str = "",
              os_code: str = "", title: str = "") -> Optional[CurriculumUnit]:
    """Match a curriculum unit to an OS unit. ISCED code is the reliable join key.

    OS code and curriculum code differ only in the segment 'OS' vs 'CU'
    (IT/OS/ICTA/CC/02/5/MA  <->  IT/CU/ICTA/CC/02/5/MA), so we also try a
    normalised compare that ignores that segment.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    def norm_code_loose(s: str) -> str:
        # drop the OS/CU discriminator so the two code families compare equal
        return norm(s).replace("os", "", 1) if norm(s).startswith("itos") \
            else norm(s).replace("cu", "", 1) if norm(s).startswith("itcu") else norm(s)

    if isced_code:
        target = norm(isced_code)
        for u in units:
            if norm(u.isced_code) == target:
                return u
    if os_code:
        tgt = norm_code_loose(os_code)
        for u in units:
            if norm_code_loose(u.curriculum_code) == tgt:
                return u
    if title:
        q = title.strip().lower()
        for u in units:
            if q in u.unit_title.lower() or u.unit_title.lower() in q:
                return u
        qt = set(re.findall(r"[a-z]+", q))
        best, best_score = None, 0.0
        for u in units:
            ut = set(re.findall(r"[a-z]+", u.unit_title.lower()))
            if not ut:
                continue
            score = len(qt & ut) / len(qt | ut)
            if score > best_score:
                best, best_score = u, score
        if best_score >= 0.4:
            return best
    return None
