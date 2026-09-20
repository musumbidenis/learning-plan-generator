"""Converting a Word document before reading it.

Nothing here starts Word or LibreOffice: the converters are stubbed, because
what matters is the decisions around them - what is cached, what happens when
there is no converter, and that a failure still leaves the document readable.
"""

import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import word_to_pdf


def _write_docx(path, body=b"<w:document/>"):
    """The least that sniffs as a Word document - it is never opened here."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", body)
    return str(path)


@pytest.fixture
def docx(tmp_path):
    return _write_docx(tmp_path / "Curriculum.docx")


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(word_to_pdf, "CACHE_DIR", str(tmp_path / "converted"))


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def test_the_cache_name_changes_when_the_document_does(docx):
    before = word_to_pdf.cache_path(docx)

    with open(docx, "ab") as fh:
        fh.write(b"one more revision")

    assert word_to_pdf.cache_path(docx) != before


def test_two_documents_of_the_same_name_do_not_share_a_conversion(tmp_path):
    """Every programme folder calls its file Curriculum.docx."""
    a, b = tmp_path / "a", tmp_path / "b"
    for folder in (a, b):
        folder.mkdir()
        _write_docx(folder / "Curriculum.docx", folder.name.encode())

    assert word_to_pdf.cache_path(str(a / "Curriculum.docx")) != \
        word_to_pdf.cache_path(str(b / "Curriculum.docx"))


def test_an_already_converted_document_is_not_converted_again(docx, monkeypatch):
    dst = word_to_pdf.cache_path(docx)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(b"%PDF-1.7 pretend")

    def never(*a, **k):
        raise AssertionError("converted a document that was already converted")

    monkeypatch.setattr(word_to_pdf, "_convert_with_word", never)
    monkeypatch.setattr(word_to_pdf, "_convert_with_libreoffice", never)

    assert word_to_pdf.convert(docx) == dst


def test_an_empty_conversion_is_not_left_behind_to_be_reused(docx, monkeypatch):
    """A converter that fails half way leaves a 0-byte file, and caching that
    would make the failure permanent."""
    def touch_and_fail(src, dst):
        open(dst, "wb").close()
        return "Word gave up"

    monkeypatch.setattr(word_to_pdf, "converter_name", lambda: word_to_pdf.WORD)
    monkeypatch.setattr(word_to_pdf, "_convert_with_word", touch_and_fail)
    monkeypatch.setattr(word_to_pdf, "_convert_with_libreoffice",
                        lambda s, d: "not installed")

    assert word_to_pdf.convert(docx) is None
    assert not os.path.exists(word_to_pdf.cache_path(docx))


# --------------------------------------------------------------------------- #
# Choosing a converter
# --------------------------------------------------------------------------- #
def test_with_no_converter_installed_nothing_is_converted(docx, monkeypatch):
    monkeypatch.setattr(word_to_pdf, "converter_name", lambda: "")

    assert word_to_pdf.convert(docx) is None


def test_libreoffice_is_tried_when_word_fails(docx, monkeypatch):
    def word_fails(src, dst):
        return "Word is not responding"

    def libreoffice_works(src, dst):
        with open(dst, "wb") as fh:
            fh.write(b"%PDF-1.7 converted")
        return ""

    monkeypatch.setattr(word_to_pdf, "converter_name", lambda: word_to_pdf.WORD)
    monkeypatch.setattr(word_to_pdf, "_convert_with_word", word_fails)
    monkeypatch.setattr(word_to_pdf, "_convert_with_libreoffice",
                        libreoffice_works)

    assert word_to_pdf.convert(docx) == word_to_pdf.cache_path(docx)


def test_a_converter_that_raises_does_not_take_the_load_down(docx, monkeypatch):
    def boom(src, dst):
        raise OSError("the COM server disappeared")

    monkeypatch.setattr(word_to_pdf, "converter_name", lambda: word_to_pdf.WORD)
    monkeypatch.setattr(word_to_pdf, "_convert_with_word", boom)
    monkeypatch.setattr(word_to_pdf, "_convert_with_libreoffice", boom)

    assert word_to_pdf.convert(docx) is None
