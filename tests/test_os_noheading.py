"""Heading-less OS units must still yield elements + PCs. ZERO API calls.

Newer CDACC Occupational Standards (e.g. the Computer Science Technician OS:
Fundamentals of Programming, Mathematics, AI, Computer Organisation, etc.) omit
the "ELEMENTS AND PERFORMANCE CRITERIA" section heading - the element/PC table
starts directly with its "ELEMENT | PERFORMANCE CRITERIA" column-header row.
These synthetic pages reproduce that layout (no real PDF needed) and guard the
region-start detection that previously required the heading.
"""

from pdf_utils import Page, _group_words_into_lines
import os_parser

EL_X = 70.0    # element (left) column
PC_X = 240.0   # performance-criteria (right) column


def _w(text, x, y):
    return {"text": text, "x0": x, "x1": x + 5 * len(text), "top": y, "bottom": y + 9}


def _row(cells, y):
    """cells = [(x, 'multi word text'), ...] laid out on one baseline `y`."""
    out = []
    for x, text in cells:
        cx = x
        for tok in text.split():
            out.append(_w(tok, cx, y))
            cx += 6 * len(tok) + 4
    return out


def _page(words):
    text = "\n".join(" ".join(w["text"] for w in grp)
                     for grp in _group_words_into_lines(words))
    return Page(index=0, text=text, words=words)


def test_headingless_unit_same_baseline_headers():
    """ELEMENT and PERFORMANCE CRITERIA share one baseline (no heading line)."""
    words = []
    words += _row([(EL_X, "APPLY FUNDAMENTALS OF PROGRAMMING")], 10)
    words += _row([(EL_X, "ISCED UNIT CODE: 0613 554 08A")], 28)
    words += _row([(EL_X, "UNIT CODE: ICT/OS/CS/CR/04/6/MA")], 42)
    words += _row([(EL_X, "UNIT DESCRIPTION")], 60)
    words += _row([(EL_X, "This unit covers fundamentals of programming.")], 74)
    words += _row([(EL_X, "ELEMENT"), (PC_X, "PERFORMANCE CRITERIA")], 110)
    words += _row([(EL_X, "These describe the key outcomes"),
                   (PC_X, "These are assessable statements")], 126)
    words += _row([(EL_X, "1. Identify Programming Concepts"),
                   (PC_X, "1.1 Programming concepts are applied.")], 150)
    words += _row([(PC_X, "1.2 Phases of program development are implemented.")], 166)
    words += _row([(EL_X, "2. Configure the Java environment"),
                   (PC_X, "2.1 Java is installed")], 200)
    words += _row([(EL_X, "RANGE")], 240)

    els = os_parser._extract_elements([_page(words)])
    assert [e.number for e in els] == ["1", "2"]
    assert els[0].title == "Identify Programming Concepts"
    assert [pc.number for pc in els[0].performance_criteria] == ["1.1", "1.2"]
    assert [pc.number for pc in els[1].performance_criteria] == ["2.1"]


def test_headingless_unit_separate_baseline_headers():
    """ELEMENT and PERFORMANCE CRITERIA fall on separate baselines."""
    words = []
    words += _row([(EL_X, "CONFIGURE OPERATING SYSTEMS")], 10)
    words += _row([(EL_X, "ISCED UNIT CODE: 0613 554 02A")], 28)
    words += _row([(EL_X, "UNIT DESCRIPTION")], 60)
    words += _row([(EL_X, "This unit covers operating systems.")], 74)
    words += _row([(EL_X, "ELEMENT")], 108)
    words += _row([(PC_X, "PERFORMANCE CRITERIA")], 120)
    words += _row([(EL_X, "1. Identify fundamentals"),
                   (PC_X, "1.1 Principles of computer software are applied")], 150)
    words += _row([(EL_X, "RANGE")], 190)

    els = os_parser._extract_elements([_page(words)])
    assert [e.number for e in els] == ["1"]
    assert [pc.number for pc in els[0].performance_criteria] == ["1.1"]


def test_description_stops_at_headingless_table():
    words = []
    words += _row([(EL_X, "APPLY FUNDAMENTALS OF PROGRAMMING")], 10)
    words += _row([(EL_X, "ISCED UNIT CODE: 0613 554 08A")], 28)
    words += _row([(EL_X, "UNIT DESCRIPTION")], 60)
    words += _row([(EL_X, "This unit covers fundamentals of programming.")], 74)
    words += _row([(EL_X, "ELEMENT"), (PC_X, "PERFORMANCE CRITERIA")], 110)
    words += _row([(EL_X, "1. Identify Programming Concepts"),
                   (PC_X, "1.1 Programming concepts are applied.")], 150)
    page = _page(words)
    desc = os_parser._extract_description(page.text)
    assert desc == "This unit covers fundamentals of programming."
