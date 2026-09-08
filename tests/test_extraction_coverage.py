"""Content that used to be silently dropped from real documents.

Each test here stands for a way a real Occupational Standard or Curriculum was
losing its performance criteria or learning key points entirely.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import curriculum_parser as cp
import os_parser
from pdf_utils import Page


def page(index, lines):
    words = []
    for text, x0, top in lines:
        for tok in text.split():
            words.append({"text": tok, "x0": x0, "x1": x0 + 5 * len(tok),
                          "top": float(top), "bottom": float(top) + 10})
    txt = "\n".join(l[0] for l in sorted(lines, key=lambda l: (l[2], l[1])))
    return Page(index=index, text=txt, words=words)


# --------------------------------------------------------------------------- #
# Curriculum numbering with a trailing dot
# --------------------------------------------------------------------------- #
def test_numbering_forms_are_told_apart():
    """'1.1.' is a sub-topic; '1.1.1.' is one of its key points."""
    assert cp._RE_SUBTOPIC.match("1.1. Selection of Hardware").groups() \
        == ("1.1", "Selection of Hardware")
    assert cp._RE_SUBTOPIC.match("1.1 Selection").groups() == ("1.1", "Selection")
    assert cp._RE_KEYPOINT.match("1.1.1. Introduction").groups() \
        == ("1.1.1", "Introduction")
    assert cp._RE_KEYPOINT.match("1.1.1.1. Meaning").groups() \
        == ("1.1.1.1", "Meaning")
    # a sub-topic pattern must never swallow a key point
    assert cp._RE_SUBTOPIC.match("1.1.1. Introduction") is None
    assert cp._RE_KEYPOINT.match("1.1. Selection") is None


def test_a_number_alone_on_its_line_still_opens_a_sub_topic():
    assert cp._RE_SUBTOPIC.match("1.2.").groups() == ("1.2", "")


def test_trailing_dot_numbering_still_yields_content():
    """This numbering dropped EVERY sub-topic and key point of a unit."""
    pages = [page(0, [
        ("COMPUTER ESSENTIALS", 80, 20),
        ("UNIT CODE: 0611 351 01A", 80, 34),
        ("Learning outcomes, Content and Suggested Assessment Methods", 80, 48),
        ("1. Manage computer devices", 80, 62),
        ("1.1. Selection of Computer Hardware devices", 200, 62),
        ("1.1.1. Introduction to computer devices", 210, 76),
        ("1.1.2. Computer case and monitor", 210, 90),
        ("1.2. Connection of devices", 200, 104),
        ("1.2.1. Cabling", 210, 118),
    ])]
    unit = cp._parse_one_unit(pages)
    assert [lo.number for lo in unit.learning_outcomes] == ["1"]
    lo = unit.learning_outcomes[0]
    assert [s.number for s in lo.sub_topics] == ["1.1", "1.2"]
    assert lo.sub_topics[0].key_points == ["Introduction to computer devices",
                                           "Computer case and monitor"]
    assert lo.sub_topics[1].key_points == ["Cabling"]


def test_the_fullest_learning_outcome_title_wins():
    """The summary table clips the title; the content table spells it out."""
    pages = [page(0, [
        ("UNIT CODE: 0611 351 01A", 80, 10),
        ("Learning outcomes, Content and Suggested Assessment Methods", 80, 24),
        ("1. Manage", 80, 38),
        ("1.1. Selection of devices", 200, 38),
        ("2. Manage desktop settings", 80, 52),
        ("2.1. Desktop", 200, 52),
        ("1. Manage computer devices", 80, 66),
        ("1.2. Connection", 200, 66),
    ])]
    unit = cp._parse_one_unit(pages)
    titles = {lo.number: lo.title for lo in unit.learning_outcomes}
    assert titles["1"] == "Manage computer devices"


def test_a_copyright_footer_is_not_read_as_part_of_a_title():
    from pdf_utils import is_noise_line
    assert is_noise_line("©QAI 2025")
    assert is_noise_line("�QAI 2025")
    assert not is_noise_line("Perform Computer Network Maintenance")


# --------------------------------------------------------------------------- #
# Word auto-numbering: the numbers never reach the text
# --------------------------------------------------------------------------- #
def _unnumbered_os_page():
    """An element/PC table exactly as a .doc yields it - no numbers at all."""
    return page(0, [
        ("APPLY COMMUNICATION SKILLS", 80, 10),
        ("UNIT CODE: 0031 441 01B", 80, 24),
        ("UNIT DESCRIPTION", 80, 38),
        ("This unit covers the competencies required to communicate.", 80, 52),
        ("ELEMENTS AND PERFORMANCE CRITERIA", 80, 66),
        ("ELEMENT", 80, 80), ("PERFORMANCE CRITERIA", 235, 80),
        ("Apply communication channels", 80, 94),
        ("Specific channels are identified.", 235, 94),
        ("Challenges are addressed.", 235, 108),
        ("Channels are evaluated.", 235, 122),
        ("Apply written communication skills", 80, 136),
        ("Types of written communication are identified.", 235, 136),
        ("Guidelines are analyzed.", 235, 150),
        ("RANGE", 80, 164),
    ])


def test_an_unnumbered_element_table_still_yields_elements_and_criteria():
    """Word applies the numbering as list formatting, so none of it is text.

    Nothing matched '1.' or '1.1' and the entire table was discarded - every
    element and every performance criterion of a .doc source.
    """
    elements = os_parser._extract_elements([_unnumbered_os_page()])
    assert [e.title for e in elements] == ["Apply communication channels",
                                           "Apply written communication skills"]
    assert [e.number for e in elements] == ["1", "2"]
    assert [len(e.performance_criteria) for e in elements] == [3, 2]


def test_generated_criteria_are_numbered_under_their_element():
    elements = os_parser._extract_elements([_unnumbered_os_page()])
    assert [pc.number for pc in elements[0].performance_criteria] == \
        ["1.1", "1.2", "1.3"]
    assert elements[0].performance_criteria[1].text == "Challenges are addressed."


def test_a_numbered_table_is_not_second_guessed():
    """The layout fallback must never override real numbering."""
    pages = [page(0, [
        ("APPLY SKILLS", 80, 10),
        ("UNIT CODE: 0031 441 01B", 80, 24),
        ("ELEMENTS AND PERFORMANCE CRITERIA", 80, 38),
        ("ELEMENT", 80, 52), ("PERFORMANCE CRITERIA", 235, 52),
        ("1. Do the first thing", 80, 66),
        ("1.1 The first thing is done", 235, 66),
        ("1.2 It is checked", 235, 80),
        ("2. Do the second thing", 80, 94),
        ("2.1 The second thing is done", 235, 94),
        ("RANGE", 80, 108),
    ])]
    elements = os_parser._extract_elements(pages)
    assert [e.number for e in elements] == ["1", "2"]
    assert [e.title for e in elements] == ["Do the first thing",
                                           "Do the second thing"]
    assert [pc.number for pc in elements[0].performance_criteria] == ["1.1", "1.2"]


def test_an_unnumbered_curriculum_table_keeps_its_content():
    """Same cause on the curriculum side: the whole syllabus was being lost."""
    pages = [page(0, [
        ("COMPUTER OPERATIONS", 80, 10),
        ("UNIT CODE: 0611 451 06A", 80, 24),
        ("Learning outcomes, Content and Suggested Assessment Methods", 80, 38),
        ("Learning Outcome", 80, 52), ("Content", 235, 52),
        ("Process word document", 80, 66),
        ("Ergonomics risk factors", 235, 66),
        ("Creation of computerized word document", 235, 80),
        ("Types of word processors", 235, 94),
        ("Manipulate computerized spreadsheet", 80, 108),
        ("Introduction to spreadsheets", 235, 108),
    ])]
    unit = cp._parse_one_unit(pages)
    assert [lo.title for lo in unit.learning_outcomes] == [
        "Process word document", "Manipulate computerized spreadsheet"]
    first = unit.learning_outcomes[0].sub_topics[0]
    assert first.key_points == ["Ergonomics risk factors",
                                "Creation of computerized word document",
                                "Types of word processors"]
    assert unit.learning_outcomes[1].sub_topics[0].key_points == [
        "Introduction to spreadsheets"]


def test_the_column_headings_do_not_become_a_learning_outcome():
    pages = [page(0, [
        ("UNIT CODE: 0611 451 06A", 80, 10),
        ("Learning outcomes, Content and Suggested Assessment Methods", 80, 24),
        ("Learning Outcome", 80, 38), ("Content", 235, 38),
        ("Process word document", 80, 52),
        ("Ergonomics risk factors", 235, 52),
    ])]
    unit = cp._parse_one_unit(pages)
    assert [lo.title for lo in unit.learning_outcomes] == ["Process word document"]
