"""Choosing between the two ways of reading a Word document.

No Word, no LibreOffice, no real documents: what is under test is the choice
and what it remembers, so the two readings are stubbed and the scores are
handed to it.
"""

import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import document_reader
import word_to_pdf


def _docx(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", b"<w:document/>")
    return str(path)


@pytest.fixture
def docx(tmp_path):
    return _docx(tmp_path / "Curriculum.docx")


@pytest.fixture
def both_readings(tmp_path, monkeypatch):
    """Stand in for the two readings, and say what each recovered."""
    converted = tmp_path / "converted.pdf"
    converted.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(word_to_pdf, "convert", lambda p: str(converted))

    state = {"read": []}

    def install(word_score, pdf_score):
        def load_word_pages(path):
            state["read"].append("word")
            return ["word pages"]

        def load_pdf_pages(path):
            state["read"].append("pdf")
            return ["pdf pages"]

        monkeypatch.setattr(document_reader, "load_word_pages", load_word_pages)
        monkeypatch.setattr(document_reader, "load_pdf_pages", load_pdf_pages)
        monkeypatch.setattr(
            document_reader, "_score",
            lambda pages, role: word_score if pages == ["word pages"]
            else pdf_score)
        return state

    return install


# --------------------------------------------------------------------------- #
# The choice
# --------------------------------------------------------------------------- #
def test_the_conversion_is_kept_when_it_recovered_more(docx, both_readings):
    """Refrigeration and Air Conditioning: nothing at all read as Word,
    seven outcomes a unit once converted."""
    both_readings(word_score=0, pdf_score=340)

    assert document_reader.read(docx, document_reader.CU) == ["pdf pages"]


def test_the_word_reading_is_kept_when_the_conversion_lost_content(docx,
                                                                   both_readings):
    """COMMUNICATION SKILLS on Agricultural Engineering: 176 recovered
    reading the file, 98 once converted."""
    both_readings(word_score=176, pdf_score=98)

    assert document_reader.read(docx, document_reader.CU) == ["word pages"]


def test_a_tie_goes_to_the_conversion(docx, both_readings):
    """It is the only reading that carries the durations, so on equal content
    it has recovered more."""
    both_readings(word_score=120, pdf_score=120)

    assert document_reader.read(docx, document_reader.CU) == ["pdf pages"]


# --------------------------------------------------------------------------- #
# Remembering it
# --------------------------------------------------------------------------- #
def test_the_verdict_is_not_worked_out_twice(docx, both_readings):
    state = both_readings(word_score=10, pdf_score=99)
    document_reader.read(docx, document_reader.CU)
    state["read"].clear()

    document_reader.read(docx, document_reader.CU)

    assert state["read"] == ["pdf"]         # not both readings again


def test_each_side_of_a_programme_decides_for_itself(docx, both_readings):
    """The same file can be opened as either, and one verdict must not erase
    the other."""
    both_readings(word_score=10, pdf_score=99)
    document_reader.read(docx, document_reader.CU)
    both_readings(word_score=99, pdf_score=10)
    document_reader.read(docx, document_reader.OS)

    assert document_reader.read(docx, document_reader.CU) == ["pdf pages"]
    assert document_reader.read(docx, document_reader.OS) == ["word pages"]


# --------------------------------------------------------------------------- #
# When there is no choice to make
# --------------------------------------------------------------------------- #
def test_a_pdf_is_read_once_and_never_scored(tmp_path, monkeypatch):
    pdf = tmp_path / "Curriculum.pdf"
    pdf.write_bytes(b"%PDF-1.7 a real one")
    monkeypatch.setattr(document_reader, "load_pdf_pages", lambda p: ["pages"])
    monkeypatch.setattr(document_reader, "_score",
                        lambda *a: pytest.fail("scored a PDF"))
    monkeypatch.setattr(word_to_pdf, "convert",
                        lambda p: pytest.fail("converted a PDF"))

    assert document_reader.read(str(pdf), document_reader.CU) == ["pages"]


def test_with_no_converter_the_word_reading_is_used_unscored(docx, monkeypatch):
    monkeypatch.setattr(word_to_pdf, "convert", lambda p: None)
    monkeypatch.setattr(document_reader, "load_word_pages", lambda p: ["word"])
    monkeypatch.setattr(document_reader, "_score",
                        lambda *a: pytest.fail("scored with nothing to compare"))

    assert document_reader.read(docx, document_reader.CU) == ["word"]
