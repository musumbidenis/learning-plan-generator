"""Table reading, against synthetic pdfplumber pages - no real PDFs.

The behaviour worth pinning down is the stitching: these documents put one
table across three pages, and every earlier attempt at deciding what continues
what got it wrong in a different way.
"""

import re

import pytest

import table_reader
from table_reader import Table


class FakeFound:
    """Stands in for a pdfplumber Table object."""

    def __init__(self, rows, top, bottom):
        self._rows = rows
        self.bbox = (0.0, top, 500.0, bottom)

    def extract(self):
        return self._rows


class FakePage:
    def __init__(self, tables, height=842.0):
        self._tables = tables
        self.height = height

    def find_tables(self):
        return self._tables

    def flush_cache(self):
        """Real pages release their parsed objects; these have none."""


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def plumber(monkeypatch):
    """Let a test describe pages, and have tables_in_pages read them."""
    # The module keeps the last document open so a whole file's units do not
    # reopen it; every test here uses the same path, so it must be cleared or
    # one test's pages answer the next one's.
    table_reader.close_document()

    def install(pages):
        monkeypatch.setattr(table_reader.pdfplumber, "open",
                            lambda path: FakePdf(pages))
    return install


HEADER = ["ELEMENT", "PERFORMANCE CRITERIA"]


def test_a_table_carried_over_a_page_break_is_one_table(plumber):
    """The OS elements table ends 233pt above the page bottom and still
    continues overleaf, so 'did it reach the bottom' is the wrong question.
    What marks a continuation is starting ABOVE where any heading could sit."""
    plumber([
        FakePage([FakeFound([HEADER, ["1. Assess needs", "1.1 Assets are documented."]],
                            top=207.8, bottom=608.9)]),
        FakePage([FakeFound([["3. Maintain security", "3.1 Monitoring is carried out."]],
                            top=50.9, bottom=179.3)]),
    ])

    tables = table_reader.tables_in_pages("x.pdf", 0, 1)

    assert len(tables) == 1
    assert [r[0] for r in tables[0].rows] == ["1. Assess needs", "3. Maintain security"]
    assert tables[0].page == 0 and tables[0].end_page == 1


def test_a_new_table_lower_down_the_page_is_its_own_table(plumber):
    """Range and the Evidence Guide follow the elements table on the same and
    the next page; merging them would fold three sections into one."""
    plumber([
        FakePage([FakeFound([HEADER, ["1. Assess needs", "1.1 x"]],
                            top=207.8, bottom=608.9)]),
        FakePage([FakeFound([["Variable", "Range"], ["1. Assets", "Software"]],
                            top=280.6, bottom=716.6)]),
    ])

    tables = table_reader.tables_in_pages("x.pdf", 0, 1)

    assert len(tables) == 2
    assert tables[1].header == ["Variable", "Range"]


def test_a_header_repeated_on_the_next_page_is_not_data(plumber):
    plumber([
        FakePage([FakeFound([HEADER, ["1. Assess needs", "1.1 x"]],
                            top=207.8, bottom=760.0)]),
        FakePage([FakeFound([HEADER, ["2. Install controls", "2.1 y"]],
                            top=73.0, bottom=400.0)]),
    ])

    tables = table_reader.tables_in_pages("x.pdf", 0, 1)

    assert len(tables) == 1
    assert [r[0] for r in tables[0].rows] == ["1. Assess needs", "2. Install controls"]


def test_a_row_with_an_empty_first_cell_extends_the_row_above(plumber):
    """How the curriculum wraps one learning outcome's content across pages."""
    plumber([FakePage([FakeFound([
        ["Learning Outcome", "Content", "Suggested Assessment Methods"],
        ["1. Assess the needs", "1.1 Documentation", "- Practical"],
        ["", "1.1.4 Importance of assessing", "- Written tests"],
    ], top=120.0, bottom=742.0)])])

    tables = table_reader.tables_in_pages("x.pdf", 0, 0)

    assert len(tables[0].rows) == 1
    assert tables[0].rows[0][1] == "1.1 Documentation 1.1.4 Importance of assessing"
    assert tables[0].rows[0][2] == "- Practical - Written tests"


def test_a_table_of_a_different_width_never_continues_the_last_one(plumber):
    plumber([
        FakePage([FakeFound([HEADER, ["1. Assess", "1.1 x"]], top=200.0, bottom=760.0)]),
        FakePage([FakeFound([["S/No.", "Item", "Qty"], ["1.", "Textbooks", "5"]],
                            top=60.0, bottom=300.0)]),
    ])

    tables = table_reader.tables_in_pages("x.pdf", 0, 1)

    assert len(tables) == 2


def test_a_bare_number_survives_cleaning(plumber):
    """The whole Duration (Hours) column is bare numbers, and the page-furniture
    filter treats a bare number as a page number. Filtering it blanked every
    duration in the curriculum."""
    plumber([FakePage([FakeFound([
        ["Learning Outcomes", "Duration (Hours)"],
        ["1. Assess security needs", "50"],
        ["Total Hours", "150"],
    ], top=600.0, bottom=738.0)])])

    tables = table_reader.tables_in_pages("x.pdf", 0, 0)

    assert tables[0].rows[0] == ["1. Assess security needs", "50"]
    assert tables[0].rows[1] == ["Total Hours", "150"]


def test_page_furniture_is_still_dropped(plumber):
    plumber([FakePage([FakeFound([
        ["Learning Outcomes", "Duration (Hours)"],
        ["©TVET CDACC 2025", "50"],
    ], top=600.0, bottom=738.0)])])

    tables = table_reader.tables_in_pages("x.pdf", 0, 0)

    assert tables[0].rows[0][0] == ""


def test_an_entirely_empty_table_is_ignored(plumber):
    """Footers are detected as 1x3 tables of blanks on every page."""
    plumber([FakePage([FakeFound([["", "", ""]], top=759.1, bottom=764.9)])])

    assert table_reader.tables_in_pages("x.pdf", 0, 0) == []


def test_a_document_that_cannot_be_opened_yields_nothing(monkeypatch):
    """A parse must degrade to the coordinate path, never raise."""
    def boom(path):
        raise OSError("not a pdf")

    monkeypatch.setattr(table_reader.pdfplumber, "open", boom)

    assert table_reader.tables_in_pages("x.pdf", 0, 3) == []


# --------------------------------------------------------------------------- #
# Finding the table you want
# --------------------------------------------------------------------------- #
def _tables():
    return [
        Table(header=["Variable", "Range"], rows=[["1. Assets", "Software"]]),
        Table(header=["ELEMENT", "PERFORMANCE CRITERIA"],
              rows=[["1. Assess", "1.1 x"]]),
        Table(header=["Learning Outcomes", "Duration (Hours)"],
              rows=[["1. Assess", "50"]]),
    ]


def test_a_table_is_found_by_its_header_not_its_page():
    found = table_reader.find_table(_tables(), re.compile("ELEMENT", re.I),
                                    re.compile("PERFORMANCE", re.I))

    assert found.rows == [["1. Assess", "1.1 x"]]


def test_every_pattern_must_match():
    assert table_reader.find_table(
        _tables(), re.compile("ELEMENT", re.I),
        re.compile("Duration", re.I)) is None


def test_a_row_is_found_by_its_leading_cell():
    table = Table(header=["", ""],
                  rows=[["1. Critical aspects", "evidence that..."],
                        ["3. Methods of assessment", "5.1 Practical 5.2 Projects"]])

    row = table.row_matching(re.compile(r"Methods? of assessment", re.I))

    assert row[1].startswith("5.1 Practical")


def test_find_all_returns_them_in_order():
    tables = _tables() + [Table(header=["Learning Outcomes", "Duration (Hours)"],
                                rows=[["1. Other unit", "40"]])]

    found = table_reader.find_all(tables, re.compile("Duration", re.I))

    assert len(found) == 2 and found[0].rows[0][1] == "50"
