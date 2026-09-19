"""Parsing the OS and Curriculum from cells - no PDFs, no network.

These cover the pieces that read a cell's contents, and the decision about
which parse of a unit to keep.
"""

import re

import curriculum_parser as cp
import drive_client
import os_parser
from models import LearningOutcome, SubTopic
from table_reader import Table


# --------------------------------------------------------------------------- #
# A content cell -> sub-topics and their key points
# --------------------------------------------------------------------------- #
CONTENT = ("1.1. Documentation of ICT security assets "
           "1.1.1. Introduction to ICT security "
           "1.1.1.1. Definition of terms "
           "1.2 ICT security threats "
           "1.2.1 Types of threats")


def test_a_content_cell_becomes_sub_topics_carrying_their_key_points():
    """The whole cell arrives as one run-on string, so the numbering is the
    only thing separating a sub-topic from the points beneath it."""
    subs = cp._split_content(CONTENT)

    assert [s.number for s in subs] == ["1.1", "1.2"]
    assert subs[0].title == "Documentation of ICT security assets"
    assert subs[0].key_points == ["Introduction to ICT security",
                                  "Definition of terms"]
    assert subs[1].key_points == ["Types of threats"]


def test_four_level_numbering_is_a_key_point_not_a_sub_topic():
    """'1.1.1.1' is as deep as these documents go and belongs to 1.1."""
    subs = cp._split_content("1.1 Assets 1.1.1.1 Definition of terms")

    assert len(subs) == 1 and subs[0].key_points == ["Definition of terms"]


def test_key_points_before_any_sub_topic_are_not_thrown_away():
    """A cell that continues from the previous page can open mid-list."""
    subs = cp._split_content("3.2.1 Importance of monitoring 3.2.2 Preparation")

    assert len(subs) == 1 and subs[0].number == "3.2"
    assert len(subs[0].key_points) == 1


def test_a_cell_with_no_numbering_yields_nothing():
    assert cp._split_content("Some prose with no numbers at all") == []


# --------------------------------------------------------------------------- #
# The assessment cell - the column that used to bleed
# --------------------------------------------------------------------------- #
def test_a_bulleted_assessment_cell_becomes_separate_methods():
    cell = "• Practical • Projects • Third Party Reports"

    assert cp._split_methods(cell) == ["Practical", "Projects",
                                       "Third Party Reports"]


def test_dash_bulleted_methods_are_split_too():
    assert cp._split_methods("- Practical - Written tests") == [
        "Practical", "Written tests"]


# --------------------------------------------------------------------------- #
# Durations
# --------------------------------------------------------------------------- #
def _duration_table():
    return Table(header=["Learning Outcomes", "Duration (Hours)"],
                 rows=[["1. Assess security needs", "50"],
                       ["2. Install security control measures", "70"],
                       ["3. Maintain ICT system security", "30"],
                       ["Total Hours", "150"]])


def test_durations_are_read_per_outcome_and_the_total_ignored():
    hours = cp._durations_by_outcome([_duration_table()])

    assert hours == {"1": 50, "2": 70, "3": 30}
    assert sum(hours.values()) == 150


def test_no_duration_table_is_not_an_error():
    assert cp._durations_by_outcome([]) == {}


# --------------------------------------------------------------------------- #
# Which parse of a unit to keep
# --------------------------------------------------------------------------- #
def _outcome(number, subs, points, methods=(), hours=0):
    return LearningOutcome(
        number=number, title=f"Outcome {number}",
        sub_topics=[SubTopic(number=f"{number}.{i}", title=f"t{i}",
                             key_points=[f"k{j}" for j in range(points)])
                    for i in range(subs)],
        suggested_methods=list(methods), duration_hours=hours)


def test_the_tables_win_when_they_recovered_more():
    """On the benchmark unit reading cells finds two sub-topics the coordinate
    walk misses entirely."""
    chosen = cp._choose_outcomes([_outcome("1", 4, 3)], [_outcome("1", 2, 3)])

    assert len(chosen[0].sub_topics) == 4


def test_the_layout_wins_when_the_grid_was_too_irregular_to_read():
    """Some curricula have cells merged and split so badly that pdfplumber
    fragments one table into 9-, 6-, 4- and 2-column pieces; there the
    coordinate walk recovers more, and must be kept."""
    chosen = cp._choose_outcomes([_outcome("1", 1, 1)], [_outcome("1", 9, 5)])

    assert len(chosen[0].sub_topics) == 9


def test_methods_and_hours_come_from_the_tables_even_when_the_layout_wins():
    """Those two columns are exactly what the coordinate walk cannot read: the
    bleed came from one, and it never looks at the table holding the other."""
    chosen = cp._choose_outcomes(
        [_outcome("1", 1, 1, methods=["Practical", "Projects"], hours=50)],
        [_outcome("1", 9, 5, methods=["Practical 3.2.1 Importance of ICT"])])

    assert len(chosen[0].sub_topics) == 9            # layout content kept
    assert chosen[0].suggested_methods == ["Practical", "Projects"]
    assert chosen[0].duration_hours == 50


def test_an_outcome_the_tables_never_saw_keeps_its_own_methods():
    chosen = cp._choose_outcomes([], [_outcome("1", 3, 2, methods=["Oral"])])

    assert chosen[0].suggested_methods == ["Oral"]


# --------------------------------------------------------------------------- #
# The OS grid
# --------------------------------------------------------------------------- #
def test_an_element_cell_yields_every_criterion_it_runs_together():
    """Column 1 holds all of an element's PCs in one cell."""
    pairs = os_parser._split_numbered_cell(
        "1.1 Assets are documented. 1.2 Threats are identified. "
        "1.3 Risk is assessed.")

    assert [n for n, _ in pairs] == ["1.1", "1.2", "1.3"]
    assert pairs[0][1] == "Assets are documented."


def test_elements_and_criteria_are_read_from_the_grid():
    table = Table(header=["ELEMENT", "PERFORMANCE CRITERIA"],
                  rows=[["1. Assess security needs",
                         "1.1 Assets are documented. 1.2 Threats are identified."],
                        ["2. Install control measures",
                         "2.1 Physical measures are implemented."]])

    elements = os_parser._elements_from_table([table])

    assert [e.number for e in elements] == ["1", "2"]
    assert elements[0].title == "Assess security needs"
    assert len(elements[0].performance_criteria) == 2
    assert elements[1].performance_criteria[0].number == "2.1"


def test_the_evidence_guide_row_yields_the_assessment_methods():
    table = Table(header=["", ""],
                  rows=[["1. Critical aspects of competency", "evidence that..."],
                        ["3. Methods of assessment",
                         "Competency may be assessed through: 5.1 Practical "
                         "5.2 Projects 5.3 Written tests"]])

    assert os_parser._methods_from_table([table]) == [
        "Practical", "Projects", "Written tests"]


def test_no_elements_grid_means_fall_back_rather_than_invent():
    assert os_parser._elements_from_table([]) == []
    assert os_parser._methods_from_table([]) == []


# --------------------------------------------------------------------------- #
# Drive throttling reads as throttling
# --------------------------------------------------------------------------- #
def test_googles_sorry_page_is_reported_as_rate_limiting():
    """A 403 carrying Google's HTML interstitial means too many requests. The
    old message sent the reader to check API restrictions, which are fine."""
    message = drive_client._explain(
        403, "<html><head><title>Sorry...</title></head><body>")

    assert "rate limiting" in message.lower()
    assert "restrictions" not in message.lower()


def test_a_genuine_restriction_403_still_says_so():
    message = drive_client._explain(
        403, "Requests to this API drive.googleapis.com method "
             "google.apps.drive.v3.DriveFiles.Get are blocked.")

    assert "restrict" in message.lower()
