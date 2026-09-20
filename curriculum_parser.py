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

import runlog
import table_reader
from models import CurriculumUnit, LearningOutcome, SubTopic, UnitRef
from pdf_utils import (
    Page,
    clean_text,
    column_lines,
    find_isced_code,
    find_tvet_code,
    is_noise_line,
    load_document,
    norm,
    norm_code_loose,
    unit_start_pages,
    unit_title_above_code,
)

# `norm` / `norm_code_loose` live in pdf_utils so unit_index can share them
# without importing a parser; they are re-exported here because this is where
# they were first defined and `unit_match` imports them from here.

# Column boundaries (tuned to CDACC curriculum layout).
LO_MAX_X = 150.0
CONTENT_MAX_X = 436.0

_RE_TABLE_START = re.compile(r"Learning\s+[Oo]utcomes?,\s+Content", re.I)
# Column headings, which are rows in their own right and must not become
# a learning outcome.
_RE_HEADER_CELL = re.compile(
    r"^\s*(learning\s+outcomes?|content|suggested\s+assessment"
    r"|assessment\s+methods?|s/?no|duration)", re.I)


_RE_TABLE_END = re.compile(r"(Suggested\s+Delivery\s+Methods|Recommended\s+Resources)", re.I)

# Numbering varies between documents: '1.1 Title', '1.1. Title' and '1.1.Title'
# all occur, and a number sometimes sits alone with its text on the next line.
# Requiring whitespace straight after the number - as these did - silently
# dropped EVERY sub-topic and key point in any curriculum using trailing dots,
# losing the whole unit's content. The OS parser has tolerated this for its
# performance criteria all along; these now match it.
#
# The lookahead is what keeps the two apart: '1.1' must not be the start of
# '1.1.1', so a following '.digit' rejects the match.
_RE_LO = re.compile(r"^(\d+)\.(?!\d)\s*(.*)$")                  # '1. Apply ...'
_RE_SUBTOPIC = re.compile(r"^(\d+\.\d+)(?!\.?\d)\.?\s*(.*)$")   # '1.1 Title' / '1.1. Title'
_RE_KEYPOINT = re.compile(r"^(\d+\.\d+(?:\.\d+)+)\.?\s*(.*)$")  # '1.1.1 ...' (x.y.z+)


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
    starts = unit_start_pages(pages)
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
    starts = unit_start_pages(pages)
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


# The three columns, by their headers rather than by where they sit on the page.
_RE_H_LO = re.compile(r"Learning\s+Outcome", re.I)
_RE_H_CONTENT = re.compile(r"Content", re.I)
_RE_H_ASSESS = re.compile(r"Assessment", re.I)
_RE_H_DURATION = re.compile(r"Duration", re.I)

# A numbered item inside a content cell: '1.1', '1.1.1', '1.1.1.1', with or
# without a trailing dot. The whole cell arrives as one run-on string, so the
# numbering is the only thing separating a sub-topic from its key points.
_RE_NUMBER_RUN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)+)\.?\s+")

# Assessment methods arrive bulleted: '- Practical - Projects - Written tests'.
_RE_BULLET_SPLIT = re.compile(r"[\u2022\u25aa\u25cf\u00b7\*]|(?<=\w)\s+-\s+|^-\s*")

_RE_LEADING_NUMBER = re.compile(r"^\s*(\d+)\s*[.)]?\s*(.*)$", re.S)


def _split_methods(cell: str) -> List[str]:
    """A bulleted assessment cell as separate methods."""
    parts = _RE_BULLET_SPLIT.split(cell or "")
    out: List[str] = []
    for part in parts:
        text = clean_text(part or "")
        if text and len(text) > 2 and not is_noise_line(text):
            out.append(text)
    return out


def _split_content(cell: str) -> List[SubTopic]:
    """A content cell as sub-topics, each carrying its own key points.

    The cell is one long string - '1.1. Documentation of ICT security assets
    1.1.1. Introduction to ICT security 1.1.1.1. Definition ...' - so it is cut
    at its numbering. Two dotted parts ('1.1') start a sub-topic; three or more
    ('1.1.1', '1.1.1.1') are key points belonging to the sub-topic above.
    """
    text = (cell or "").strip()
    if not text:
        return []
    marks = list(_RE_NUMBER_RUN.finditer(text))
    if not marks:
        return []

    subs: List[SubTopic] = []
    current: Optional[SubTopic] = None
    for i, mark in enumerate(marks):
        number = mark.group(1)
        body_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = clean_text(text[mark.end():body_end])
        if not body:
            continue
        if number.count(".") == 1:
            current = SubTopic(number=number, title=body)
            subs.append(current)
        elif current is not None:
            current.key_points.append(body)
        else:
            # Key points before any sub-topic: keep them under one made from
            # the outcome itself rather than dropping the syllabus on the floor.
            current = SubTopic(number=number.rsplit(".", 1)[0], title=body)
            subs.append(current)
    return subs


def _column(header: List[str], pattern, default: int) -> int:
    """Index of the column whose heading matches, or `default`."""
    return next((i for i, cell in enumerate(header) if pattern.search(cell)),
                default)


def _durations_by_outcome(tables) -> Dict[str, int]:
    """{'1': 50, '2': 70, ...} from the Summary of Learning Outcomes table.

    Neither column sits where a two-column table puts it once the document
    numbers its own rows: the Refrigeration and Air Conditioning curriculum
    heads that table `S/NO | Learning Outcomes | Duration (Hours)`, so reading
    column 1 as the hours gives the outcome's TITLE and every duration came
    back zero. Both columns are found by their headings, and the outcome's
    number is taken from its own cell, falling back to the S/NO column that
    carries it when the title cell does not.
    """
    table = table_reader.find_table(tables, _RE_H_LO, _RE_H_DURATION)
    if table is None:
        return {}
    lo_col = _column(table.header, _RE_H_LO, 0)
    hour_col = _column(table.header, _RE_H_DURATION, 1)
    hours: Dict[str, int] = {}
    for row in table.rows:
        value = table.cell(row, hour_col).strip()
        if not value.isdigit():
            continue                       # skips the 'Total Hours' row
        match = _RE_LEADING_NUMBER.match(table.cell(row, lo_col)) \
            or _RE_LEADING_NUMBER.match(row[0] if row else "")
        if match:
            hours[match.group(1)] = int(value)
    return hours


def _outcomes_from_tables(unit_pages: List[Page]) -> List[LearningOutcome]:
    """Learning outcomes read from this unit's ruled tables, or [].

    Reading the cells beats splitting the page by x-coordinate - it is what
    stops the Suggested Assessment Methods column swallowing half of Content
    and running on into Methods of Delivery. But it is not always better:
    some curricula have grids whose cells are merged and split so irregularly
    that pdfplumber fragments one table into 9-, 6-, 4- and 2-column pieces,
    and the result is worse than the coordinate walk. So this returns a
    CANDIDATE, and `_choose_outcomes` decides - see there.
    """
    if not unit_pages:
        return []
    source = getattr(unit_pages[0], "source", "")
    if not source:
        return []
    tables = table_reader.tables_in_pages(
        source, unit_pages[0].index, unit_pages[-1].index)
    if not tables:
        return []

    syllabus = table_reader.find_table(tables, _RE_H_LO, _RE_H_CONTENT,
                                       _RE_H_ASSESS)
    if syllabus is None:
        return []

    hours = _durations_by_outcome(tables)
    outcomes: List[LearningOutcome] = []
    for row in syllabus.rows:
        if len(row) < 2:
            continue
        match = _RE_LEADING_NUMBER.match(row[0])
        if not match:
            continue
        number, title = match.group(1), clean_text(match.group(2))
        sub_topics = _split_content(row[1] if len(row) > 1 else "")
        if not sub_topics:
            continue
        outcomes.append(LearningOutcome(
            number=number,
            title=title,
            sub_topics=sub_topics,
            suggested_methods=_split_methods(row[2] if len(row) > 2 else ""),
            duration_hours=hours.get(number, 0)))

    return outcomes


def _richness(outcomes: List[LearningOutcome]) -> tuple:
    """(sub-topics, key points) recovered - the two things a parse can lose.

    One sub-topic becomes one session, so losing sub-topics loses whole rows of
    the plan; losing key points thins the rows that remain. They are reported
    separately because neither substitutes for the other.
    """
    return (sum(len(o.sub_topics) for o in outcomes),
            sum(len(st.key_points) for o in outcomes for st in o.sub_topics))


def _choose_outcomes(from_tables: List[LearningOutcome],
                     from_layout: List[LearningOutcome]
                     ) -> List[LearningOutcome]:
    """Keep the tables only when they cost nothing, but always take their columns.

    The two parses divide a unit's content differently, so ranking them on any
    single score trades one axis for the other and some unit always loses.
    Measured over 625 units: ranking key points first cost FARM IRRIGATION AND
    DRAINAGE SYSTEMS 24 sub-topics - 24 sessions - and ranking sub-topics first
    instead cost AGRICULTURAL REFRIGERATION 7 key points. There is no ordering
    that wins everywhere.

    So there is no ordering. The tables are used only when they are at least as
    good on BOTH counts, which makes losing content impossible by construction;
    otherwise the coordinate walk stands. It means forgoing the occasional
    trade (a unit where the tables would add key points at the cost of a
    sub-topic), and that is the price of the guarantee.

    Assessment methods and durations come from the tables either way, matched
    by outcome number: those are the two columns the coordinate walk cannot
    read - one is where the Content bleed came from, the other lives in a table
    it never looks at.
    """
    table_subs, table_points = _richness(from_tables)
    layout_subs, layout_points = _richness(from_layout)

    use_tables = bool(from_tables) and (table_subs >= layout_subs
                                        and table_points >= layout_points)
    chosen = from_tables if use_tables else (from_layout or from_tables)

    if not use_tables:
        extras = {o.number: (o.suggested_methods, o.duration_hours)
                  for o in from_tables}
        for outcome in chosen:
            methods, hours = extras.get(outcome.number, (None, 0))
            if methods:
                outcome.suggested_methods = list(methods)
            outcome.duration_hours = hours

    if chosen:
        subs, points = _richness(chosen)
        runlog.log(f"Curriculum: {len(chosen)} learning outcomes, {subs} "
                   f"sub-topics, {points} key points, "
                   f"{sum(o.duration_hours for o in chosen)} hours "
                   f"(from {'tables' if use_tables else 'layout'})")
    return chosen


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

    # The unit is parsed BOTH ways and the better result kept - see
    # `_choose_outcomes` for why neither wins outright.
    table_outcomes = _outcomes_from_tables(unit_pages)

    # --- walk the content table --------------------------------------------- #
    lo_titles: Dict[str, str] = {}      # major number -> LO title
    sub_topics: List[SubTopic] = []     # in document order
    methods: List[str] = []
    cur_sub: Optional[SubTopic] = None
    cur_lo_major: Optional[str] = None
    cur_lo_title_parts: List[str] = []
    # kept so an unnumbered table can be rebuilt from its layout below
    raw_lo: List[tuple] = []
    raw_content: List[tuple] = []

    def flush_lo_title():
        """Keep the fullest version of a learning outcome's title.

        The same outcome is named twice - once in the unit's summary table and
        again in the content table - and the summary copy is often clipped by
        the column, so first-wins left titles truncated to a single word
        ("Manage" for "Manage computer devices").
        """
        if cur_lo_major and cur_lo_title_parts:
            candidate = clean_text(" ".join(cur_lo_title_parts))
            if len(candidate) > len(lo_titles.get(cur_lo_major, "")):
                lo_titles[cur_lo_major] = candidate

    for page, y_lo, y_hi in _table_window(unit_pages):
        words = _window_words(page, y_lo, y_hi, 0, 10_000)

        # The Suggested-Assessment-Methods column is bulleted; detect its left
        # edge per page from the '•' x-positions (it drifts between 429 and 438).
        bullet_xs = [w["x0"] for w in words if w["text"].strip() in ("•", "", "▪")]
        content_max = (min(bullet_xs) - 4.0) if bullet_xs else CONTENT_MAX_X

        lo_lines = column_lines([w for w in words if w["x0"] < LO_MAX_X], 0, 10_000)
        content_lines = column_lines(
            [w for w in words if LO_MAX_X <= w["x0"] < content_max], 0, 10_000)
        raw_lo += [(page.index, ln) for ln in lo_lines]
        raw_content += [(page.index, ln) for ln in content_lines]
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

    if not sub_topics:
        # Word carries the '1.1' / '1.1.1' numbering as automatic list
        # formatting, which never reaches the text, so nothing matches and the
        # whole content table is lost. Its shape still holds the structure.
        layout_outcomes = _outcomes_from_layout(
            raw_lo, raw_content, unique_methods)
    else:
        layout_outcomes = [los[k] for k in order]

    unit.learning_outcomes = _choose_outcomes(table_outcomes, layout_outcomes)
    return unit


def _outcomes_from_layout(lo_lines, content_lines,
                          methods: List[str]) -> List[LearningOutcome]:
    """Rebuild learning outcomes from the table's shape when nothing is numbered.

    Each row is one learning outcome: the left cell names it, and the content
    cell beside it holds its syllabus lines. Without the numbering there is no
    way to tell a sub-topic from its key points, so the outcome becomes a single
    sub-topic - one session - carrying all of its content as key points. That
    keeps every line of the syllabus rather than discarding the lot.
    """
    outcomes: List[LearningOutcome] = []
    anchors: List[tuple] = []          # (page_index, top, SubTopic)
    for page_index, ln in lo_lines:
        title = clean_text(ln.text)
        if not title or is_noise_line(title) or _RE_HEADER_CELL.match(title):
            continue
        number = str(len(outcomes) + 1)
        sub = SubTopic(number=f"{number}.1", title=title)
        outcomes.append(LearningOutcome(number=number, title=title,
                                        sub_topics=[sub],
                                        suggested_methods=methods))
        anchors.append((page_index, ln.top, sub))

    if not anchors:
        return []

    for page_index, ln in content_lines:
        text = clean_text(ln.text)
        if not text or is_noise_line(text) or _RE_HEADER_CELL.match(text):
            continue
        target = anchors[0][2]
        for a_page, a_top, sub in anchors:
            if (a_page, a_top) <= (page_index, ln.top + 2.0):
                target = sub
        target.key_points.append(text)
    return outcomes


def find_unit(units: List[CurriculumUnit], isced_code: str = "",
              os_code: str = "", title: str = "") -> Optional[CurriculumUnit]:
    """Match a curriculum unit to an OS unit. ISCED code is the reliable join key.

    OS code and curriculum code differ only in the segment 'OS' vs 'CU', so we
    also try a normalised compare that ignores that segment.
    """
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
