"""Module D3 - assessment_docs: render one AssessmentTool as four .docx files.

The four documents a CDACC assessment tool is submitted as:

    build_pc_distribution        the unit's PC weighting table + THIS CAT's marks
    build_table_of_specifications  elements x Bloom levels x item format
    build_candidates_tool        what the candidate is handed
    build_assessors_tool         what the assessor marks from

Each returns bytes, the way `doc_builder.document_to_bytes` does, so the app can
hand them straight to a download button.

NOTHING HERE COMPUTES A MARK. Every number reaching this module is already final
(`assessment_allocation` did the arithmetic before the model was ever called);
this module only lays those numbers out. The sums that do appear - a row total, a
column total - are sums of rendered cells, printed so the reader can see the
table reconcile, never a re-allocation.

House style follows doc_builder: Times New Roman, "Table Grid", 100%-wide
tables, one paragraph per line (never iterate a raw string), and a header block
of paired bold "Label: value" cells. Its helpers are imported rather than copied;
session_plan_builder already duplicates several of them and this module does not
make that worse.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence, Tuple

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

import assessment_config as config
from assessment_models import (BLOOM_LEVELS, PRACTICAL, THEORY, AssessmentTool,
                               UnitWeighting)
# Reused as-is from the Learning Plan builder - the reference implementation for
# fonts, cell writing, cell margins and table width. FONT_NAME/FONT_SIZE are
# re-exported so this module never names a second font.
from doc_builder import (FONT_NAME, FONT_SIZE, _set_cell_text,
                         _set_label_value_cell, _set_table_cell_margins,
                         _set_table_width_pct, _style_run)
from doc_builder import _setup_page as _setup_landscape_page
# Only these four live in session_plan_builder; they are generic, so they are
# imported rather than duplicated a third time.
from session_plan_builder import (_merge, _set_cell_shade, _set_grid_cols,
                                  _set_repeat_header)

TITLE_SIZE = 14.0
HEADING_SIZE = 12.0
TOS_FONT_SIZE = 8.0          # the ToS carries 20 columns; it needs the room
HEADER_FILL = "D9D9D9"       # column-header and total rows, as the templates shade them

# A4, matching doc_builder's landscape page and session_plan_builder's margins.
LANDSCAPE_USABLE_CM = 27.7
PORTRAIT_USABLE_CM = 18.46

# What a PC that this CAT does not assess shows in the CAT's mark column.
NOT_ASSESSED = "N/A"

ASSESSMENT_TYPE_LABELS = {THEORY: "Written (theory)", PRACTICAL: "Practical"}

PC_DISTRIBUTION_TITLE = "PERFORMANCE CRITERIA DISTRIBUTION"
TOS_TITLE = "TABLE OF SPECIFICATIONS"
CANDIDATE_TITLE = "CANDIDATE'S ASSESSMENT TOOL"
ASSESSOR_TITLE = "ASSESSOR'S ASSESSMENT TOOL"

# Element | PC | PC text | Theory weight | Practical weight (+ the CAT's own
# mark column, whose heading carries the CAT label and so is built per document).
PC_DISTRIBUTION_HEADERS = ("Element", "PC", "Performance Criterion",
                           "Theory Weight", "Practical Weight")
PC_DISTRIBUTION_WIDTHS = [2.2, 1.4, 13.5, 3.2, 3.4, 4.0]

# The item-format sub-columns the CDACC ToS template prints under every Bloom
# level. The key is `Item.item_format`; 'mcq' has no Item counterpart on purpose
# (see `_mcq_allowed`).
MCQ = "mcq"
SHORT_RESPONSE = "short_response"
EXTENDED_RESPONSE = "extended_response"
ITEM_FORMAT_COLUMNS: Tuple[Tuple[str, str], ...] = (
    (MCQ, "MCQ"),
    (SHORT_RESPONSE, "SR"),
    (EXTENDED_RESPONSE, "ER"),
)
# `Item.item_format` defaults to this, so an allocation with no item yet lands in
# the same column its item would.
DEFAULT_ITEM_FORMAT = SHORT_RESPONSE


# --------------------------------------------------------------------------- #
# Page, title and table plumbing
# --------------------------------------------------------------------------- #
def _setup_page(doc: Document, landscape: bool = True) -> float:
    """Set the page up and return the usable width in cm.

    doc_builder's `_setup_page` does the whole job for the wide documents (A4
    landscape, 1 cm margins, Normal styled to Times New Roman); the two reading
    documents only flip it back to portrait afterwards.
    """
    _setup_landscape_page(doc)
    if landscape:
        return LANDSCAPE_USABLE_CM
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    for margin in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, margin, Cm(1.27))
    return PORTRAIT_USABLE_CM


def _scaled(widths: Sequence[float], usable_cm: float) -> List[float]:
    """Scale relative column widths so they fill the usable page width."""
    total = float(sum(widths)) or 1.0
    return [w / total * usable_cm for w in widths]


def _new_table(doc: Document, widths_cm: Sequence[float]):
    """A fresh full-width "Table Grid" whose column grid is already seeded."""
    table = doc.add_table(rows=0, cols=len(widths_cm))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_grid_cols(table, list(widths_cm))
    _set_table_width_pct(table, 100)
    _set_table_cell_margins(table, top=0.05, bottom=0.05, left=0.12, right=0.12)
    return table


def _add_row(table, values: Sequence, *, header: bool = False,
             size: float = FONT_SIZE, shade: str = ""):
    """One row; each value is a string (one paragraph) or a list of lines."""
    cells = table.add_row().cells
    for cell, value in zip(cells, values):
        lines = value if isinstance(value, list) else [str(value)]
        _set_cell_text(cell, lines, size=size, header=header)
        if shade:
            _set_cell_shade(cell, shade)
    return cells


def _add_title(doc: Document, text: str) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _style_run(par.add_run(text.upper()), size=TITLE_SIZE, bold=True)


def _add_heading(doc: Document, text: str) -> None:
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(8)
    par.paragraph_format.space_after = Pt(2)
    _style_run(par.add_run(text.upper()), size=HEADING_SIZE, bold=True)


def _add_lines(doc: Document, lines: Sequence[str], *, bullet: bool = False,
               size: float = FONT_SIZE) -> None:
    """One paragraph per line - never iterate a raw string."""
    if isinstance(lines, str):
        lines = [lines]
    for line in [str(l) for l in (lines or []) if str(l).strip()]:
        par = doc.add_paragraph()
        par.paragraph_format.space_before = Pt(0)
        par.paragraph_format.space_after = Pt(0)
        _style_run(par.add_run(f"•  {line}" if bullet else line), size=size)


def _bullets(lines: Sequence[str]) -> List[str]:
    return [f"•  {str(l).strip()}" for l in (lines or []) if str(l).strip()]


def _to_bytes(doc: Document) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# The header block every one of the four documents carries
# --------------------------------------------------------------------------- #
def _header_pairs(tool: AssessmentTool) -> List[Tuple[str, str]]:
    """The six identity fields, in the order the header block prints them."""
    cat = tool.cat
    return [
        # "Competency", not "Competence": every published CDACC assessment
        # tool spells it that way. The Learning Plan and Session Plan keep
        # "Competence" because that is what the RVNP/KTTC template they copy
        # actually says, and `learning_plan_parser` reads that label back.
        ("Unit of Competency", tool.unit_title),
        ("ISCED Unit Code", tool.isced_code),
        ("TVET CDACC Unit Code", tool.cdacc_code),
        ("Assessment", cat.label),
        ("Assessment Type",
         ASSESSMENT_TYPE_LABELS.get(cat.assessment_type, cat.assessment_type)),
        ("Total Marks", str(cat.total_marks)),
    ]


def _add_header_block(doc: Document, tool: AssessmentTool,
                      usable_cm: float):
    """Paired bold 'Label: value' cells, laid out as doc_builder's header rows."""
    table = _new_table(doc, [usable_cm / 2.0, usable_cm / 2.0])
    pairs = _header_pairs(tool)
    for idx in range(0, len(pairs), 2):
        cells = table.add_row().cells
        for cell, (label, value) in zip(cells, pairs[idx:idx + 2]):
            _set_label_value_cell(cell, label, value)
    return table


def _gap(doc: Document) -> None:
    """A small paragraph so consecutive tables do not fuse into one."""
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(0)
    par.paragraph_format.space_after = Pt(0)
    _style_run(par.add_run(""), size=4)


# --------------------------------------------------------------------------- #
# 1. PC distribution
# --------------------------------------------------------------------------- #
def build_pc_distribution(weighting: UnitWeighting,
                          tool: AssessmentTool) -> bytes:
    """The unit's weighting table with THIS CAT's mark column appended.

    Every PC of the unit is listed, whether this CAT assesses it or not: the
    table is the unit's, and its purpose is to show what this CAT took out of
    it. A PC this CAT does not assess shows 'N/A' rather than 0, because 0 would
    read as "assessed and worth nothing".
    """
    doc = Document()
    usable = _setup_page(doc, landscape=True)
    _add_title(doc, PC_DISTRIBUTION_TITLE)
    _add_header_block(doc, tool, usable)
    _gap(doc)

    widths = _scaled(PC_DISTRIBUTION_WIDTHS, usable)
    table = _new_table(doc, widths)
    cat_column = f"{tool.cat.label} Marks"
    _add_row(table, list(PC_DISTRIBUTION_HEADERS) + [cat_column],
             header=True, shade=HEADER_FILL)
    _set_repeat_header(table.rows[-1])

    marks = tool.marks_by_pc()          # already final, straight from the model
    for element in weighting.elements:
        for idx, pc in enumerate(element.pcs):
            element_cell = ([element.number, element.title]
                            if idx == 0 and element.title else element.number)
            _add_row(table, [
                element_cell,
                pc.number,
                pc.text,
                pc.theory_weight,
                pc.practical_weight,
                str(marks[pc.number]) if pc.number in marks else NOT_ASSESSED,
            ])
        _add_row(table, _element_subtotal_row(element, marks),
                 header=True, shade=HEADER_FILL)

    _add_row(table, [
        "", "", "GRAND TOTAL",
        _stated_or_sum(weighting.stated_grand_theory,
                       weighting.grand_total(THEORY)),
        _stated_or_sum(weighting.stated_grand_practical,
                       weighting.grand_total(PRACTICAL)),
        tool.cat.total_marks,
    ], header=True, shade=HEADER_FILL)
    return _to_bytes(doc)


def _stated_or_sum(stated: Optional[int], fallback: int) -> int:
    """The total the pasted table stated, or the PC rows' own sum if it stated none.

    The stated figure wins: it is the document's number, and the parser has
    already verified it against the rows.
    """
    return fallback if stated is None else stated


def _element_subtotal_row(element, marks: Dict[str, int]) -> List:
    """An element's sub-total row, populated in all three mark columns."""
    assessed = [marks[pc.number] for pc in element.pcs if pc.number in marks]
    title = f" - {element.title}" if element.title else ""
    return [
        element.number,
        "",
        f"Sub-total: Element {element.number}{title}",
        _stated_or_sum(element.stated_theory_total, element.total(THEORY)),
        _stated_or_sum(element.stated_practical_total, element.total(PRACTICAL)),
        str(sum(assessed)) if assessed else NOT_ASSESSED,
    ]


# --------------------------------------------------------------------------- #
# 2. Table of specifications
# --------------------------------------------------------------------------- #
def _mcq_allowed(knqf_level: str) -> bool:
    """Whether MCQ items may appear at this KNQF level.

    NOT A BUG, AND NOT DEAD CODE. The CDACC ToS template prints an MCQ
    sub-column under every Bloom level, so the column is rendered at every
    level; but at the levels in `config.CONSTRUCTED_RESPONSE_ONLY_LEVELS` (5 and
    6) the candidate must supply the response, so no MCQ item exists and the
    column is rendered EMPTY. Template and format rule disagree; both are
    obeyed - the column is drawn, the cells stay blank. Do not "fix" the blanks
    by filling them, and do not drop the column to hide them.
    """
    return str(knqf_level or "").strip() not in config.CONSTRUCTED_RESPONSE_ONLY_LEVELS


def _tos_grid(tool: AssessmentTool) -> Dict[Tuple[str, str, str], int]:
    """{(element, bloom, item format): marks}, from the items or the allocations.

    A written tool has items and they carry the format. A practical tool has
    none, so its allocations - which carry the same element, Bloom level and
    final marks - fill the grid in the default format column.
    """
    grid: Dict[Tuple[str, str, str], int] = {}
    if tool.items:
        for item in tool.items:
            key = (item.element_number, item.bloom,
                   item.item_format or DEFAULT_ITEM_FORMAT)
            grid[key] = grid.get(key, 0) + item.marks
        return grid
    for alloc in tool.allocations:
        key = (alloc.element_number, alloc.bloom, DEFAULT_ITEM_FORMAT)
        grid[key] = grid.get(key, 0) + alloc.marks
    return grid


def _tos_elements(tool: AssessmentTool) -> List[Tuple[str, str]]:
    """(number, title) for every element this CAT touches, in allocation order."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for alloc in tool.allocations:
        if alloc.element_number not in seen:
            seen.add(alloc.element_number)
            out.append((alloc.element_number, alloc.element_title))
    for item in tool.items:
        if item.element_number not in seen:
            seen.add(item.element_number)
            out.append((item.element_number, ""))
    return out


def build_table_of_specifications(tool: AssessmentTool) -> bytes:
    """Elements down, the six Bloom levels across, item formats beneath each.

    Both totals are printed - a total per element row and a total per format
    column - and both are sums of the cells actually rendered, so the table
    visibly reconciles to the CAT total in its bottom-right corner.
    """
    doc = Document()
    usable = _setup_page(doc, landscape=True)
    _add_title(doc, TOS_TITLE)
    _add_header_block(doc, tool, usable)
    _gap(doc)

    formats = ITEM_FORMAT_COLUMNS
    n_formats = len(formats)
    # Element | (6 levels x formats) | Total
    widths = _scaled([4.0] + [1.0] * (len(BLOOM_LEVELS) * n_formats) + [1.8],
                     usable)
    table = _new_table(doc, widths)

    # --- two header rows: the levels, then their format sub-columns ---------- #
    # Both rows are created before anything is written, because Element and
    # Total span them vertically and a merge must happen before the text.
    top = table.add_row().cells
    sub = table.add_row().cells
    for cell, heading in ((top[0].merge(sub[0]), "Element"),
                          (top[-1].merge(sub[-1]), "Total")):
        _set_cell_text(cell, [heading], size=TOS_FONT_SIZE, header=True)
        _set_cell_shade(cell, HEADER_FILL)
    for position, level in enumerate(BLOOM_LEVELS):
        first = 1 + position * n_formats
        merged = _merge(top, first, first + n_formats - 1)
        _set_cell_text(merged, [level.title()], size=TOS_FONT_SIZE, header=True)
        _set_cell_shade(merged, HEADER_FILL)
        for offset, (_key, label) in enumerate(formats):
            cell = sub[first + offset]
            _set_cell_text(cell, [label], size=TOS_FONT_SIZE, header=True)
            _set_cell_shade(cell, HEADER_FILL)
    for row in table.rows:          # only the two header rows exist yet
        _set_repeat_header(row)

    # --- one row per element, then the column totals ------------------------- #
    grid = _tos_grid(tool)
    mcq_ok = _mcq_allowed(tool.knqf_level)
    column_totals = [0] * (len(BLOOM_LEVELS) * n_formats)
    grand = 0

    for number, title in _tos_elements(tool):
        values: List[int] = []
        for level in BLOOM_LEVELS:
            for key, _label in formats:
                # The MCQ column is drawn at every level but stays empty at the
                # constructed-response-only levels - see `_mcq_allowed`.
                cell = 0 if (key == MCQ and not mcq_ok) else grid.get(
                    (number, level, key), 0)
                values.append(cell)
        row_total = sum(values)
        grand += row_total
        column_totals = [a + b for a, b in zip(column_totals, values)]
        label = f"{number}. {title}" if title else str(number)
        _add_row(table, [label] + [_num(v) for v in values] + [_num(row_total)],
                 size=TOS_FONT_SIZE)

    _add_row(table,
             ["TOTAL"] + [_num(v) for v in column_totals] + [_num(grand)],
             header=True, size=TOS_FONT_SIZE, shade=HEADER_FILL)
    return _to_bytes(doc)


def _num(value: int) -> str:
    """A mark cell: blank when nothing was allocated, so the grid stays readable."""
    return str(value) if value else ""


# --------------------------------------------------------------------------- #
# 3. The candidate's tool
# --------------------------------------------------------------------------- #
def _time_allowed(tool: AssessmentTool) -> str:
    if tool.task_brief is not None and tool.task_brief.time_allowed.strip():
        return tool.task_brief.time_allowed
    return f"{tool.cat.duration_minutes} minutes"


def _candidate_instructions(tool: AssessmentTool) -> List[str]:
    """Standing instructions, worded from the CAT's own facts.

    Nothing in the AssessmentTool carries paper instructions, so they are built
    here from the numbers that are already in it - question count, duration,
    total marks - and nowhere else.
    """
    return [
        "1. Write your name and registration number in the spaces provided.",
        f"2. This paper consists of {len(tool.items)} question(s).",
        ("3. Read the scenario in Section A, then answer ALL the questions in "
         "the spaces provided." if len(tool.scenarios) == 1
         else "3. Answer ALL the questions in the spaces provided."),
        f"4. Time allowed: {_time_allowed(tool)}.",
        f"5. The paper is marked out of {tool.cat.total_marks} marks.",
        "6. Do not write on this paper anything other than your answers.",
    ]


def _scenario_numbers(tool: AssessmentTool) -> Dict[str, int]:
    return {s.id: n for n, s in enumerate(tool.scenarios, start=1)}


def _add_written_paper(doc: Document, tool: AssessmentTool) -> None:
    _add_heading(doc, "Instructions to the candidate")
    _add_lines(doc, _candidate_instructions(tool))

    numbers = _scenario_numbers(tool)
    if tool.scenarios:
        # Set once, here, and referred to by number from every item that uses
        # it - a scenario is never reprinted under its questions.
        _add_heading(doc, "Section A: Scenario"
                     if len(tool.scenarios) == 1 else "Section A: Scenarios")
        for scenario in tool.scenarios:
            par = doc.add_paragraph()
            par.paragraph_format.space_after = Pt(0)
            _style_run(par.add_run(
                f"Scenario {numbers[scenario.id]}: {scenario.title}"), bold=True)
            _add_lines(doc, [scenario.text])

    _add_heading(doc, "Section B: Questions" if tool.scenarios else "Questions")
    for item in tool.items:
        par = doc.add_paragraph()
        par.paragraph_format.space_before = Pt(6)
        par.paragraph_format.space_after = Pt(0)
        _style_run(par.add_run(f"{item.number}. "), bold=True)
        # With a single scenario the instructions have already said the whole
        # paper refers to it, so "(Refer to Scenario 1)" on all twelve
        # questions is noise on the page. It is printed only where the
        # candidate has a choice of situations to keep straight.
        if (len(tool.scenarios) > 1
                and item.scenario_id and item.scenario_id in numbers):
            _style_run(par.add_run(
                f"(Refer to Scenario {numbers[item.scenario_id]}) "), bold=False)
        _style_run(par.add_run(item.stem), bold=False)
        _style_run(par.add_run(f"  ({item.marks} marks)"), bold=True)


def _add_task_brief(doc: Document, tool: AssessmentTool, usable_cm: float) -> None:
    """The practical brief: what to do and under what conditions - no more.

    Deliberately silent about the items of evaluation and how the marks split
    between them: the candidate is assessed on the work, not on a checklist they
    were shown beforehand. The observation and product checklists live only in
    the assessor's tool.
    """
    brief = tool.task_brief
    _add_heading(doc, "Practical task")
    table = _new_table(doc, _scaled([4.5, 14.0], usable_cm))
    rows = [
        ("Task", [brief.task] if brief else []),
        ("Conditions", [brief.conditions] if brief else []),
        ("Tools, Equipment and Materials",
         _bullets(brief.tools_equipment_materials) if brief else []),
        ("Safety Requirements",
         _bullets(brief.safety_requirements) if brief else []),
        ("Time Allowed", [_time_allowed(tool)]),
    ]
    for label, lines in rows:
        cells = table.add_row().cells
        _set_cell_text(cells[0], [label], header=True)
        _set_cell_text(cells[1], list(lines) or [""])


def build_candidates_tool(tool: AssessmentTool) -> bytes:
    """What the candidate is handed: the question paper, or the task brief."""
    doc = Document()
    usable = _setup_page(doc, landscape=False)
    _add_title(doc, CANDIDATE_TITLE)
    _add_header_block(doc, tool, usable)
    if tool.is_practical:
        _add_task_brief(doc, tool, usable)
    else:
        _add_written_paper(doc, tool)
    return _to_bytes(doc)


# --------------------------------------------------------------------------- #
# 4. The assessor's tool
# --------------------------------------------------------------------------- #
MARKING_SCHEME_HEADERS = ("Item", "Question", "Expected Response", "Marks")
MARKING_SCHEME_WIDTHS = [1.4, 5.5, 9.0, 1.6]

CHECKLIST_HEADERS = ("No.", "Item of Evaluation", "PC(s)", "Marks")
CHECKLIST_WIDTHS = [1.2, 11.0, 3.0, 1.6]

ORAL_HEADERS = ("No.", "Question", "PC(s)", "Response Indicators")
ORAL_WIDTHS = [1.2, 6.5, 2.2, 7.0]


def _add_marking_scheme(doc: Document, tool: AssessmentTool,
                        usable_cm: float) -> None:
    """One row per marking point, grouped under the item it belongs to."""
    _add_heading(doc, "Marking scheme")
    table = _new_table(doc, _scaled(MARKING_SCHEME_WIDTHS, usable_cm))
    _add_row(table, MARKING_SCHEME_HEADERS, header=True, shade=HEADER_FILL)
    _set_repeat_header(table.rows[-1])

    for item in tool.items:
        points = item.marking_scheme or []
        if not points:
            _add_row(table, [item.number, item.stem, "", item.marks])
            continue
        for idx, point in enumerate(points):
            first = idx == 0
            _add_row(table, [
                item.number if first else "",
                [item.stem, f"({item.marks} marks)"] if first else "",
                point.text,
                point.marks,
            ])
    _add_row(table, ["", "TOTAL", "", tool.cat.total_marks],
             header=True, shade=HEADER_FILL)


def _add_checklist(doc: Document, heading: str, items, usable_cm: float) -> None:
    _add_heading(doc, heading)
    table = _new_table(doc, _scaled(CHECKLIST_WIDTHS, usable_cm))
    _add_row(table, CHECKLIST_HEADERS, header=True, shade=HEADER_FILL)
    _set_repeat_header(table.rows[-1])
    total = 0
    for entry in items:
        total += entry.marks
        _add_row(table, [
            entry.number,
            [entry.text] + _bullets(entry.sub_parts),
            ", ".join(entry.pc_numbers),
            entry.marks,
        ])
    _add_row(table, ["", "Sub-total", "", total], header=True, shade=HEADER_FILL)


def _add_oral_questions(doc: Document, tool: AssessmentTool,
                        usable_cm: float) -> None:
    _add_heading(doc, "Oral questions")
    table = _new_table(doc, _scaled(ORAL_WIDTHS, usable_cm))
    _add_row(table, ORAL_HEADERS, header=True, shade=HEADER_FILL)
    _set_repeat_header(table.rows[-1])
    for question in tool.oral_questions:
        _add_row(table, [
            question.number,
            question.question,
            ", ".join(question.pc_numbers),
            _bullets(question.response_indicators) or [""],
        ])


def build_assessors_tool(tool: AssessmentTool) -> bytes:
    """What the assessor marks from: the scheme, or the checklists."""
    doc = Document()
    # Landscape: both the marking scheme and the checklists carry a wide text
    # column beside their mark column.
    usable = _setup_page(doc, landscape=True)
    _add_title(doc, ASSESSOR_TITLE)
    _add_header_block(doc, tool, usable)
    if tool.is_practical:
        # The two checklists stay separate tables: they are inspected at
        # different moments - one while the work happens, one on the product.
        _add_checklist(doc, "Observation checklist",
                       tool.observation_checklist, usable)
        _gap(doc)
        _add_checklist(doc, "Product checklist", tool.product_checklist, usable)
        if tool.oral_questions:
            _gap(doc)
            _add_oral_questions(doc, tool, usable)
    else:
        _add_marking_scheme(doc, tool, usable)
    return _to_bytes(doc)
