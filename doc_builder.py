"""Module D - doc_builder: render the Learning Plan to a .docx (RVNP format).

A4 portrait, Times New Roman, "Table Grid". Two tables:
  1) a header-info table (paired label/value cells + merged Skill/Benchmark rows)
  2) the 9-column session table (REF KTTC/TP/LP/F07).

Every list is rendered LINE-BY-LINE (each string -> its own paragraph). We never
iterate a raw string, which is what causes the 'G / r / o / u / p' char-splitting
bug when a string is mistakenly treated as a list.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

from models import PlanInputs, Session, Unit

FONT_NAME = "Times New Roman"

SESSION_HEADERS = [
    "Week", "Session No", "Session Title", "Specific Learning Outcomes",
    "Learning Key Points", "Trainee Activities", "Resources & References",
    "Learning Checks / Assessments", "Reflections & Date",
]
# Relative column widths (sum scaled to the usable page width).
SESSION_WIDTHS = [0.55, 0.7, 1.5, 2.1, 2.3, 2.4, 1.8, 1.9, 1.2]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _set_cell_text(cell, lines: List[str], bold_predicate: Optional[Callable] = None,
                   size: float = 8.0, header: bool = False) -> None:
    """Write `lines` into a table cell, one paragraph per line.

    `bold_predicate(line) -> bool` decides which whole lines render bold.
    A bare string is wrapped to a single-element list to defeat char-splitting.
    """
    if isinstance(lines, str):
        lines = [lines]
    lines = [l for l in (lines or []) if str(l).strip() != ""] or [""]

    # reuse the default first paragraph, then add the rest
    cell.text = ""
    first_par = cell.paragraphs[0]
    for idx, line in enumerate(lines):
        par = first_par if idx == 0 else cell.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.space_before = Pt(0)
        run = par.add_run(str(line))
        run.font.name = FONT_NAME
        run.font.size = Pt(size)
        # ensure East-Asian font mapping too (avoids fallback fonts)
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), FONT_NAME)
        rfonts.set(qn("w:hAnsi"), FONT_NAME)
        if header:
            run.bold = True
        elif bold_predicate is not None and bold_predicate(str(line)):
            run.bold = True


def _is_caps_heading(line: str) -> bool:
    """A 'mostly uppercase' content sub-heading (e.g. OVERVIEW OF ...)."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.85 and not line.strip().startswith("-")


def _keypoint_bold(line: str) -> bool:
    return _is_caps_heading(line)


def _activity_bold(line: str) -> bool:
    s = line.strip().lower()
    return s.startswith("follow up activity") or s.startswith("due date")


def _assessment_bold(line: str) -> bool:
    s = line.strip().lower()
    return s.startswith(("knowledge checks", "skills:", "skills :", "attitudes"))


def _outcome_bold(line: str) -> bool:
    return line.strip().lower().startswith("by the end")


def _set_col_widths(table, widths_cm: List[float]) -> None:
    table.autofit = False
    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)


# --------------------------------------------------------------------------- #
# Header info table
# --------------------------------------------------------------------------- #
def _add_label_value_row(table, l1, v1, l2, v2) -> None:
    row = table.add_row().cells
    for cell, text, is_label in ((row[0], l1, True), (row[1], v1, False),
                                 (row[2], l2, True), (row[3], v2, False)):
        _set_cell_text(cell, [text], size=10)
        if is_label:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.bold = True


def _skill_task_lines(unit: Unit) -> List[str]:
    """'Skill or Job Task' = the numbered element titles (matches the gold)."""
    return [f"{el.number}. {el.title}" for el in unit.elements] or [unit.skill_task]


def _benchmark_lines(unit: Unit) -> List[str]:
    """Elements (1,2,3) each followed by their PCs keeping x.y numbering."""
    out: List[str] = []
    for el in unit.elements:
        out.append(f"{el.number}. {el.title}")
        for pc in el.performance_criteria:
            out.append(f"{pc.number} {pc.text}")
    return out


def build_header_table(doc: Document, unit: Unit, inputs: PlanInputs,
                       display_code: str) -> None:
    table = doc.add_table(rows=0, cols=4)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    _add_label_value_row(table, "Unit of Competence", unit.unit_title,
                         "Unit Code", display_code)
    _add_label_value_row(table, "Name of Trainer", inputs.trainer_name,
                         "Admission Number", "")
    _add_label_value_row(table, "Institution", inputs.institution,
                         "Level", inputs.level or unit.level)
    _add_label_value_row(table, "Date of Preparation", inputs.date_of_preparation,
                         "Date of Revision", "")
    _add_label_value_row(table, "Number of Trainees", str(inputs.num_trainees),
                         "Class", inputs.class_code)

    # Merged 'Skill or Job Task' row
    srow = table.add_row().cells
    smerged = srow[0].merge(srow[1]).merge(srow[2]).merge(srow[3])
    _set_cell_text(smerged, ["Skill or Job Task:"] + _skill_task_lines(unit), size=10,
                   bold_predicate=lambda l: l.strip().lower().startswith("skill or job"))

    # Merged 'Benchmark or Criteria' row
    brow = table.add_row().cells
    bmerged = brow[0].merge(brow[1]).merge(brow[2]).merge(brow[3])
    _set_cell_text(
        bmerged, ["Benchmark or Criteria to be used:"] + _benchmark_lines(unit),
        size=10,
        bold_predicate=lambda l: l.strip().lower().startswith("benchmark or criteria")
        or bool(re.match(r"^\d+\.\s+\D", l.strip())))


# --------------------------------------------------------------------------- #
# Session table
# --------------------------------------------------------------------------- #
def build_session_table(doc: Document, sessions: List[Session]) -> None:
    table = doc.add_table(rows=1, cols=9)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # header row
    for cell, head in zip(table.rows[0].cells, SESSION_HEADERS):
        _set_cell_text(cell, [head], size=8, header=True)

    for s in sessions:
        cells = table.add_row().cells
        _set_cell_text(cells[0], [str(s.week)], size=8)
        _set_cell_text(cells[1], [s.session_no], size=8)
        _set_cell_text(cells[2], [s.session_title], size=8)
        _set_cell_text(cells[3], s.learning_outcomes, size=8,
                       bold_predicate=_outcome_bold)
        _set_cell_text(cells[4], s.key_points, size=8, bold_predicate=_keypoint_bold)
        _set_cell_text(cells[5], s.trainee_activities, size=8,
                       bold_predicate=_activity_bold)
        _set_cell_text(cells[6], s.resources, size=8)
        _set_cell_text(cells[7], s.assessments, size=8,
                       bold_predicate=_assessment_bold)
        _set_cell_text(cells[8], [""], size=8)        # Reflections & Date - blank

    # widths
    usable = 19.0   # cm (A4 portrait, ~1cm margins)
    total = sum(SESSION_WIDTHS)
    widths = [w / total * usable for w in SESSION_WIDTHS]
    _set_col_widths(table, widths)


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def _setup_page(doc: Document) -> None:
    sec = doc.sections[0]
    # A4 portrait
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, m, Cm(1.0))
    # default style font
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = Pt(10)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), FONT_NAME)
    rfonts.set(qn("w:hAnsi"), FONT_NAME)


def build_document(unit: Unit, sessions: List[Session], inputs: PlanInputs,
                   display_code: str = "") -> Document:
    doc = Document()
    _setup_page(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    trun = title.add_run("LEARNING PLAN")
    trun.bold = True
    trun.font.name = FONT_NAME
    trun.font.size = Pt(14)

    build_header_table(doc, unit, inputs, display_code or unit.os_code)
    doc.add_paragraph()
    build_session_table(doc, sessions)
    return doc


def save_document(unit: Unit, sessions: List[Session], inputs: PlanInputs,
                  path: str, display_code: str = "") -> str:
    doc = build_document(unit, sessions, inputs, display_code)
    doc.save(path)
    return path


def document_to_bytes(unit: Unit, sessions: List[Session], inputs: PlanInputs,
                      display_code: str = "") -> bytes:
    import io
    doc = build_document(unit, sessions, inputs, display_code)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
