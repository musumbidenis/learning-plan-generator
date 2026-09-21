"""Matching the curriculum's taught content onto the elements being assessed.

The risk these tests exist for is not a crash. It is an element quietly
carrying the wrong learning outcome's content, which produces a paper full of
well-written questions about the wrong half of the unit - and nothing
downstream can tell, because the questions look fine.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assessment_content as content_builder
from models import CurriculumUnit, Element, LearningOutcome, SubTopic


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _element(number, title):
    return Element(number=number, title=title)


def _outcome(number, title, topics=(), hours=0, methods=()):
    return LearningOutcome(
        number=number, title=title, duration_hours=hours,
        suggested_methods=list(methods),
        sub_topics=[SubTopic(number=n, title=t, key_points=list(points))
                    for n, t, points in topics])


def _curriculum(*outcomes):
    return CurriculumUnit(unit_title="Apply Computer Programming Principles",
                          learning_outcomes=list(outcomes))


SECURITY = ("1.1", "Identification of ICT security threats",
            ["Definition of terms", "Types of malware",
             "Social engineering attacks"])
CONTROLS = ("1.2", "Application of security controls",
            ["Access control models", "Firewall configuration"])


# --------------------------------------------------------------------------- #
# The match
# --------------------------------------------------------------------------- #
def test_an_element_takes_the_outcome_with_its_own_title():
    curriculum = _curriculum(
        _outcome("1", "Apply programming logic"),
        _outcome("2", "Identify ICT security threats", [SECURITY]))

    content = content_builder.content_for(
        curriculum, [_element("2", "Identify ICT security threats")])

    assert [b.outcome_number for b in content] == ["2"]


def test_the_titles_win_when_the_numbering_disagrees():
    """The two documents are written by different committees and their
    numbering drifts. Trusting the number here would assess element 1 out of
    element 3's content."""
    curriculum = _curriculum(
        _outcome("1", "Apply programming logic", [CONTROLS]),
        _outcome("3", "Identify ICT security threats", [SECURITY]))

    content = content_builder.content_for(
        curriculum, [_element("1", "Identify ICT security threats")])

    assert content[0].outcome_number == "3"
    assert content[0].topics[0].number == "1.1"


def test_a_strong_pair_settles_before_a_weak_one():
    """Matching element by element in order lets the first element take an
    outcome that belongs to a later one, and everything after it shifts by
    one. The best pairs across the whole unit settle first."""
    curriculum = _curriculum(
        _outcome("1", "Identify ICT security threats and vulnerabilities"),
        _outcome("2", "Identify ICT security threats"))

    matched = content_builder.match_outcomes(
        [_element("1", "Identify ICT security threats"),
         _element("2", "Identify ICT security threats and vulnerabilities")],
        curriculum.learning_outcomes)

    assert matched["1"].number == "2"
    assert matched["2"].number == "1"


def test_two_elements_never_share_one_outcome():
    curriculum = _curriculum(_outcome("1", "Identify ICT security threats",
                                      [SECURITY]))

    matched = content_builder.match_outcomes(
        [_element("1", "Identify ICT security threats"),
         _element("2", "Identify ICT security threats")],
        curriculum.learning_outcomes)

    assert len(set(id(o) for o in matched.values())) == len(matched)


def test_a_drifted_title_falls_back_to_the_same_number():
    """Titles do get rewritten between the two documents. The number is a
    weaker signal than the title, not no signal at all."""
    curriculum = _curriculum(
        _outcome("1", "Undertake threat and vulnerability profiling",
                 [SECURITY]))

    content = content_builder.content_for(
        curriculum, [_element("1", "Identify ICT security threats")])

    assert content[0].topics[0].title == SECURITY[1]


def test_an_element_that_matches_nothing_carries_no_content():
    """No content is better than another element's content: the paper is then
    written from the PC alone, which is shallow but not wrong."""
    curriculum = _curriculum(_outcome("1", "Apply programming logic",
                                      [CONTROLS]))

    content = content_builder.content_for(
        curriculum, [_element("7", "Maintain poultry housing")])

    assert content == []


def test_a_curriculum_that_was_never_read_is_not_an_error():
    assert content_builder.content_for(None, [_element("1", "Anything")]) == []
    assert content_builder.content_for(_curriculum(), [_element("1", "X")]) == []


def test_an_outcome_with_no_sub_topics_adds_nothing():
    """An outcome the parser found by title but could not read content from
    would otherwise print an empty element heading and tell the model it had
    been given content it has not."""
    curriculum = _curriculum(_outcome("1", "Identify ICT security threats"))

    assert content_builder.content_for(
        curriculum, [_element("1", "Identify ICT security threats")]) == []


# --------------------------------------------------------------------------- #
# Only the elements this CAT covers
# --------------------------------------------------------------------------- #
def test_only_the_wanted_elements_come_back():
    curriculum = _curriculum(
        _outcome("1", "Identify ICT security threats", [SECURITY]),
        _outcome("2", "Apply security controls", [CONTROLS]))
    elements = [_element("1", "Identify ICT security threats"),
                _element("2", "Apply security controls")]

    content = content_builder.content_for(curriculum, elements, {"2"})

    assert [b.element_number for b in content] == ["2"]


def test_the_match_is_made_across_the_whole_unit_before_filtering():
    """Handing in only the assessed subset is how an element ends up holding
    content it has nothing to do with. Element 2's title is a fair match for
    outcome 1 and would claim it if element 1 were not in the room; with the
    whole unit present, element 1 takes outcome 1 outright and element 2 falls
    to its own."""
    curriculum = _curriculum(
        _outcome("1", "Identify ICT security threats", [SECURITY]),
        _outcome("2", "Apply ICT security controls", [CONTROLS]))
    whole_unit = [
        _element("1", "Identify ICT security threats"),
        _element("2", "Identify ICT security threats and control measures")]

    passing_the_subset = content_builder.content_for(
        curriculum, whole_unit[1:], {"2"})
    passing_the_unit = content_builder.content_for(curriculum, whole_unit, {"2"})

    assert passing_the_subset[0].topics[0].number == "1.1"   # the wrong one
    assert passing_the_unit[0].topics[0].number == "1.2"


# --------------------------------------------------------------------------- #
# What the model is shown
# --------------------------------------------------------------------------- #
def test_the_block_carries_the_numbers_a_trainer_can_point_at():
    curriculum = _curriculum(_outcome("1", "Identify ICT security threats",
                                      [SECURITY], hours=50))
    content = content_builder.content_for(
        curriculum, [_element("1", "Identify ICT security threats")])

    rendered = content_builder.render(content)

    assert "ELEMENT 1 - Identify ICT security threats" in rendered
    assert "1.1 Identification of ICT security threats" in rendered
    assert "- Types of malware" in rendered
    assert "50 hours" in rendered


def test_nothing_matched_renders_to_nothing_at_all():
    """An empty string is what tells the prompt builder to leave the section
    out entirely, rather than printing a heading over nothing."""
    assert content_builder.render([]) == ""


def test_the_suggested_methods_are_gathered_once_each():
    curriculum = _curriculum(
        _outcome("1", "Identify ICT security threats", [SECURITY],
                 methods=["Written tests", "Practical"]),
        _outcome("2", "Apply security controls", [CONTROLS],
                 methods=["Practical", "Portfolio of evidence"]))
    elements = [_element("1", "Identify ICT security threats"),
                _element("2", "Apply security controls")]

    content = content_builder.content_for(curriculum, elements)

    assert content_builder.methods(content) == [
        "Written tests", "Practical", "Portfolio of evidence"]


def test_the_summary_counts_what_the_paper_has_to_work_with():
    curriculum = _curriculum(_outcome("1", "Identify ICT security threats",
                                      [SECURITY, CONTROLS]))
    content = content_builder.content_for(
        curriculum, [_element("1", "Identify ICT security threats")])

    line = content_builder.summarise(content)

    assert "2 sub-topic(s)" in line
    assert "5 key point(s)" in line


def test_the_summary_says_plainly_when_there_is_nothing():
    assert "PCs alone" in content_builder.summarise([])
