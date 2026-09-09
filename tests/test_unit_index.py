"""Unit detection across document shapes the original detector could not read."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_index
from models import UnitRef
from pdf_utils import Page


def page(index, lines):
    """Build a Page from (text, x0, top) triples, mimicking extracted words."""
    words = []
    for text, x0, top in lines:
        x = x0
        for tok in text.split():
            words.append({"text": tok, "x0": x, "x1": x + 5 * len(tok),
                          "top": float(top), "bottom": float(top) + 10})
            x += 5 * len(tok) + 5
    txt = "\n".join(l[0] for l in sorted(lines, key=lambda l: l[2]))
    return Page(index=index, text=txt, words=words)


def unit_page(index, title, code, code_first=False):
    header = [("UNIT CODE: " + code, 80, 50), (title, 80, 70)] if code_first \
        else [(title, 80, 50), ("UNIT CODE: " + code, 80, 70)]
    return page(index, header + [
        ("UNIT DESCRIPTION: This unit specifies the competences required to "
         "do the thing.", 80, 90),
        ("ELEMENT", 80, 130), ("PERFORMANCE CRITERIA", 240, 130),
        ("1. Do the thing", 80, 150), ("1.1 The thing is done", 240, 150)])


def roster_page(index, rows, header="Summary of Units of Learning"):
    lines = [(header, 80, 30),
             ("Unit Code Units Title Unit Duration Credit Factor", 80, 45)]
    for i, (code, title, dur) in enumerate(rows):
        lines.append(("%s %s %d 8" % (code, title, dur), 80, 60 + i * 14))
    lines.append(("Sub Total 480 48", 80, 60 + len(rows) * 14))
    return page(index, lines)


# --------------------------------------------------------------------------- #
# Reading the preliminary units table
# --------------------------------------------------------------------------- #
def test_reads_the_units_table_from_the_front_matter():
    pages = [roster_page(0, [("0611 351 01A", "Computer Essentials", 80),
                             ("0611 351 02A", "Computer Operations", 100)])]
    roster = unit_index.read_roster(pages)
    assert [(e.code, e.title) for e in roster] == [
        ("0611 351 01A", "Computer Essentials"),
        ("0611 351 02A", "Computer Operations")]


def test_reads_a_units_table_that_leads_with_a_category_column():
    pages = [page(0, [
        ("UNIT CATEGORY UNIT CODE UNITS NAME DURATION (HOURS)", 80, 30),
        ("CORE 0611 351 01A Computer Essentials 80", 80, 50),
        ("CORE 0611 351 02A Computer Operations 100", 80, 64),
        ("Sub Total Hours 480", 80, 78)])]
    roster = unit_index.read_roster(pages)
    assert [e.title for e in roster] == ["Computer Essentials",
                                         "Computer Operations"]


def test_reads_a_units_table_of_tvet_codes():
    pages = [page(0, [
        ("Summary of Units of Competency", 80, 30),
        ("IT/OS/ICTA/CC/02/5/MA Apply Communication Skills 60", 80, 50),
        ("IT/OS/ICTA/CC/03/5/MA Apply Digital Literacy 40", 80, 64)])]
    assert [e.title for e in unit_index.read_roster(pages)] == [
        "Apply Communication Skills", "Apply Digital Literacy"]


def test_totals_rows_are_not_mistaken_for_units():
    pages = [roster_page(0, [("0611 351 01A", "Computer Essentials", 80),
                             ("0611 351 02A", "Computer Operations", 100)])]
    titles = [e.title.lower() for e in unit_index.read_roster(pages)]
    assert not any("total" in t for t in titles)


def test_a_units_own_page_is_never_read_as_a_roster():
    """One code on the page - a roster lists several."""
    assert unit_index.read_roster(
        [unit_page(0, "COMPUTER ESSENTIALS", "0611 351 01A")]) == []


def test_a_roster_split_across_two_pages_is_merged():
    pages = [roster_page(0, [("0611 351 01A", "Computer Essentials", 80),
                             ("0611 351 02A", "Computer Operations", 100)]),
             roster_page(1, [("0612 351 03A", "Computer Network Setup", 150),
                             ("0714 351 04A", "Computer Repair", 150)])]
    assert len(unit_index.read_roster(pages)) == 4


# --------------------------------------------------------------------------- #
# The shapes that used to yield nothing
# --------------------------------------------------------------------------- #
def test_finds_units_in_a_document_with_only_tvet_codes():
    """The old detector required an ISCED code and found zero here."""
    pages = [unit_page(0, "APPLY COMMUNICATION SKILLS", "IT/OS/ICTA/CC/02/5/MA"),
             unit_page(1, "APPLY DIGITAL LITERACY", "IT/OS/ICTA/CC/03/5/MA")]
    res = unit_index.index_units(pages, "OS")
    assert [r.title for r in res.refs] == ["APPLY COMMUNICATION SKILLS",
                                           "APPLY DIGITAL LITERACY"]
    assert [r.code for r in res.refs] == ["IT/OS/ICTA/CC/02/5/MA",
                                          "IT/OS/ICTA/CC/03/5/MA"]


def test_finds_a_unit_whose_code_is_printed_above_its_title():
    pages = [unit_page(0, "COMPUTER ESSENTIALS", "0611 351 01A", code_first=True)]
    res = unit_index.index_units(pages, "CU")
    assert [r.title for r in res.refs] == ["COMPUTER ESSENTIALS"]


def test_units_get_contiguous_non_overlapping_page_spans():
    pages = [unit_page(i, "UNIT %d" % i, "061%d 351 0%dA" % (i, i))
             for i in range(3)]
    refs = unit_index.index_units(pages, "OS").refs
    assert [(r.start_page, r.end_page) for r in refs] == [(0, 1), (1, 2), (2, 3)]


def test_an_empty_document_yields_nothing_rather_than_raising():
    res = unit_index.index_units([], "OS")
    assert res.refs == [] and res.roster == [] and res.strategy == "none"


# --------------------------------------------------------------------------- #
# Reporting the shortfall
# --------------------------------------------------------------------------- #
def test_a_rostered_unit_missing_from_the_body_is_reported():
    pages = [roster_page(0, [("0611 351 01A", "Computer Essentials", 80),
                             ("0611 351 02A", "Computer Operations", 100)]),
             unit_page(1, "COMPUTER ESSENTIALS", "0611 351 01A")]
    res = unit_index.index_units(pages, "CU")
    assert [r.title.upper() for r in res.refs] == ["COMPUTER ESSENTIALS"]
    assert [m.title for m in res.missing] == ["Computer Operations"]


def test_nothing_is_reported_missing_when_every_rostered_unit_is_found():
    pages = [roster_page(0, [("0611 351 01A", "Computer Essentials", 80),
                             ("0611 351 02A", "Computer Operations", 100)]),
             unit_page(1, "COMPUTER ESSENTIALS", "0611 351 01A"),
             unit_page(2, "COMPUTER OPERATIONS", "0611 351 02A")]
    res = unit_index.index_units(pages, "CU")
    assert len(res.refs) == 2
    assert res.missing == []


def test_the_roster_page_itself_is_never_returned_as_a_unit():
    pages = [roster_page(0, [("0611 351 01A", "Computer Essentials", 80),
                             ("0611 351 02A", "Computer Operations", 100)]),
             unit_page(1, "COMPUTER ESSENTIALS", "0611 351 01A"),
             unit_page(2, "COMPUTER OPERATIONS", "0611 351 02A")]
    assert all(r.start_page != 0
               for r in unit_index.index_units(pages, "CU").refs)


# --------------------------------------------------------------------------- #
# Multi-unit DOCX
# --------------------------------------------------------------------------- #
def test_a_multi_unit_docx_yields_one_page_per_unit(tmp_path):
    """load_docx_pages used to return ONE page, capping the file at one unit."""
    from docx import Document
    from pdf_utils import load_document

    doc = Document()
    titles = ["APPLY COMMUNICATION SKILLS", "APPLY DIGITAL LITERACY",
              "PERFORM COMPUTER OPERATIONS"]
    for i, title in enumerate(titles, start=1):
        doc.add_paragraph(title)
        doc.add_paragraph("ISCED UNIT CODE: 0611 351 0%dA" % i)
        doc.add_paragraph("UNIT DESCRIPTION: This unit specifies the "
                          "competences required to do the thing.")
        t = doc.add_table(rows=2, cols=2)
        t.cell(0, 0).text = "ELEMENT"
        t.cell(0, 1).text = "PERFORMANCE CRITERIA"
        t.cell(1, 0).text = "1. Do a thing"
        t.cell(1, 1).text = "1.1 The thing is done"
        doc.add_page_break()
    path = str(tmp_path / "three_units.docx")
    doc.save(path)

    pages = load_document(path)
    assert len(pages) >= 3
    assert [r.title for r in unit_index.index_units(pages, "OS").refs] == titles


# --------------------------------------------------------------------------- #
# A unit that starts under its module's units table, on the same page
# --------------------------------------------------------------------------- #
def module_table_then_unit_page(index):
    """The real shape of a CDACC curriculum: 'MODULE 6', the table of the two
    units in it, then the first of those units starting below."""
    return page(index, [
        ("MODULE 6", 80, 20),
        ("UNIT UNIT CATEGORY ISCED UNIT CODE TVET CDACC UNIT CODE "
         "UNIT NAME DURATION (HOURS)", 80, 34),
        ("CORE 0612 551 IT/CU/ICTA/CR/02/6/MA ICT Security 150", 80, 48),
        ("CORE 0613 551 IT/CU/ICTA/CR/03/6/MA Desktop Application 280", 80, 62),
        ("Total Hours 430", 80, 76),
        ("ICT SECURITY", 80, 100),
        ("ISCED UNIT CODE: 0612 551 16A", 80, 114),
        ("TVET CDACC UNIT CODE: IT/CU/ICTA/CR/02/6/MA", 80, 128),
        ("Unit Description", 80, 142),
        ("This unit covers the competencies required to manage ICT security.",
         80, 156),
        ("Learning Outcomes", 80, 170), ("Content", 300, 170),
        ("1. Assess security needs", 80, 184),
        ("1.1 Security threats", 300, 184)])


def test_a_unit_under_its_module_table_is_named_after_itself():
    """This shipped a unit called 'MODULE 6'.

    The title search anchored on the first code-shaped line, which is a row of
    the table above; the unit's own code line is the one carrying a label.
    """
    refs = unit_index.index_units([module_table_then_unit_page(0)], "CU").refs
    assert [r.title for r in refs] == ["ICT SECURITY"]


def test_the_unit_takes_its_own_code_not_the_tables_clipped_one():
    refs = unit_index.index_units([module_table_then_unit_page(0)], "CU").refs
    assert refs[0].code == "IT/CU/ICTA/CR/02/6/MA"
    assert refs[0].isced_code == "0612 551 16A"


def test_a_units_own_code_header_is_not_read_as_a_table_row():
    """A unit page carries both code families, which made it 'a roster page'
    listing a unit named 'TVET CDACC UNIT CODE:'."""
    roster = unit_index.read_roster([module_table_then_unit_page(0)])
    assert all("CODE" not in e.title.upper() for e in roster)
    assert [e.title for e in roster] == ["ICT Security", "Desktop Application"]


# --------------------------------------------------------------------------- #
# Reconciling the roster against what was found
# --------------------------------------------------------------------------- #
def test_a_roster_row_carrying_the_other_code_family_is_not_reported_missing():
    """The two code columns land in one cell: code '0612 451 07A', title
    'IT/CU/ICTA/CR/02/5/MA Network Design and'. The document quotes the TVET
    code, so comparing only the ISCED one cried wolf on a located unit."""
    entry = unit_index.RosterEntry(
        code="0612 451 07A",
        title="IT/CU/ICTA/CR/02/5/MA Network Design and")
    refs = [UnitRef(title="NETWORK DESIGN AND MANAGEMENT",
                    code="IT/CU/ICTA/CR/02/5/MA")]
    assert unit_index._entry_was_found(entry, refs)


def test_a_clipped_roster_code_still_matches_the_unit():
    """Table cells wrap: 'IT/CU/ICTA/CC/01/6/M' is '...MA' with the last
    character on the next line."""
    entry = unit_index.RosterEntry(code="IT/CU/ICTA/CC/01/6/M", title="Discrete")
    refs = [UnitRef(title="DISCRETE MATHEMATICAL CONCEPTS",
                    code="IT/CU/ICTA/CC/01/6/MA")]
    assert unit_index._entry_was_found(entry, refs)


def test_a_clipped_roster_title_still_matches_the_unit():
    entry = unit_index.RosterEntry(code="0714 351 04A",
                                   title="Computer Repair and")
    refs = [UnitRef(title="COMPUTER REPAIR AND MAINTENANCE",
                    isced_code="0714 351 09A")]
    assert unit_index._entry_was_found(entry, refs)


def test_a_unit_the_document_really_lacks_is_still_reported():
    entry = unit_index.RosterEntry(code="0322 551 11A",
                                   title="Manage Organization Records")
    refs = [UnitRef(title="CONDUCT INDEXING AND ABSTRACTING",
                    isced_code="0322 551 10A")]
    assert not unit_index._entry_was_found(entry, refs)


def test_a_short_shared_prefix_is_not_taken_as_a_match():
    """'Com' opens half the units in an ICT curriculum."""
    entry = unit_index.RosterEntry(code="0611 351 09A", title="Com")
    refs = [UnitRef(title="COMPUTER ESSENTIALS", isced_code="0611 351 01A")]
    assert not unit_index._entry_was_found(entry, refs)
