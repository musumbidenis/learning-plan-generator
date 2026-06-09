"""Label-independent unit identity (code-shape anchored). ZERO API calls.

Units are located by the SHAPE of the ISCED / TVET-CDACC codes, not by the
wording of the surrounding label - so headers that differ across documents
(or carry CDACC's real-world typos) all resolve the same unit name. These
synthetic pages exercise that directly, plus the summary-table guard.
"""

from pdf_utils import (
    Page,
    _group_words_into_lines,
    find_isced_code,
    find_tvet_code,
    page_starts_unit,
    unit_title_above_code,
)

X = 70.0


def _w(text, x, y):
    return {"text": text, "x0": x, "x1": x + 5 * len(text), "top": y, "bottom": y + 9}


def _row(text, y, x=X):
    out, cx = [], x
    for tok in text.split():
        out.append(_w(tok, cx, y))
        cx += 6 * len(tok) + 4
    return out


def _page(rows):
    words = []
    for y, text in rows:
        words += _row(text, y)
    text = "\n".join(" ".join(w["text"] for w in grp)
                     for grp in _group_words_into_lines(words))
    return Page(index=0, text=text, words=words)


# Same unit, five differently-worded code labels -> same title every time.
LABEL_VARIANTS = [
    "ISCED UNIT CODE: 0613 451 05A",      # canonical OS label
    "UNIT CODE: 0613 451 05A",            # missing 'ISCED' prefix
    "ISCDE UNIT CODE: 0613 451 05A",      # transposed-prefix typo
    "ISCED Unit Code : 0613 451 05A",     # mixed case + spaced colon
    "0613 451 05A",                       # bare code, no label at all
]


def test_title_resolves_regardless_of_label():
    for label in LABEL_VARIANTS:
        page = _page([(10, "APPLY COMPUTER PROGRAMMING PRINCIPLES"),
                      (28, label),
                      (42, "UNIT CODE: IT/OS/ICTA/CC/02/5/MA"),
                      (60, "UNIT DESCRIPTION")])
        assert unit_title_above_code(page) == "APPLY COMPUTER PROGRAMMING PRINCIPLES", label
        assert page_starts_unit(page), label


def test_title_resolves_when_tvet_code_sits_above_isced():
    """Some documents order the header title -> TVET code -> ISCED code, so the
    line directly above the ISCED anchor is the TVET CDACC code, NOT the title
    (seen in the 'Information Manager' OS). Title resolution must walk past the
    code line to the real name - otherwise the dropdown shows the TVET code."""
    page = _page([(10, "APPLY DIGITAL LITERACY"),
                  (28, "TVET CDACC Unit Code: LIS/OS/LIA/BC/01/5/MA"),
                  (42, "ISCED Unit Code: 0611 551 01A"),
                  (60, "UNIT DESCRIPTION:")])
    assert unit_title_above_code(page) == "APPLY DIGITAL LITERACY"
    assert find_tvet_code(page.text) == "LIS/OS/LIA/BC/01/5/MA"
    assert find_isced_code(page.text) == "0611 551 01A"
    assert page_starts_unit(page)


def test_code_extraction_by_shape():
    assert find_isced_code("UNIT CODE: 0613 451 05A trailing") == "0613 451 05A"
    assert find_isced_code("no code here") == ""
    assert find_tvet_code("TVET CDACC UNIT CODE: IT/CU/ICTA/CC/02/5/MA") == "IT/CU/ICTA/CC/02/5/MA"
    assert find_tvet_code("ICT/OS/CS/CR/01/6/MA") == "ICT/OS/CS/CR/01/6/MA"
    assert find_tvet_code("plain words only") == ""


def test_summary_table_is_not_a_unit_start():
    """A page listing several unit codes (a summary/MODULE/CATEGORY table) must
    NOT register as a unit start - the distinct-ISCED-count guard rejects it."""
    summary = _page([(10, "BASIC UNITS OF COMPETENCY"),
                     (30, "UNIT CODE: 0031 441 01A"),
                     (44, "UNIT CODE: 0417 441 02A"),
                     (58, "UNIT CODE: 0413 441 03A")])
    assert not page_starts_unit(summary)


def test_code_without_title_is_not_a_unit_start():
    """One code but nothing meaningful above it -> not a unit start."""
    page = _page([(30, "ISCED UNIT CODE: 0031 441 01A"),
                  (44, "UNIT DESCRIPTION")])
    assert not page_starts_unit(page)
