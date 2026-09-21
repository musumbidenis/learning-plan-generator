"""Reading a pasted PC weighting table - no documents, no network.

The primary fixture is the real CDACC weighting table for PERFORM TOUR GUIDE
OPERATIONS (TO/OS/TTM/CR/01/6/MA, KNQF 6, theory:practical 2:3): two elements,
twelve performance criteria, sub-totals 18/27 each and a grand total of 36/54.
It is written once in tab-separated form and converted to the other two paste
shapes, because the three shapes are the same table and must parse to the same
object - the separator is an accident of where the user copied from.
"""

import assessment_weighting as aw
from assessment_models import PRACTICAL, THEORY, UnitWeighting, WeightedElement, WeightedPC


# --------------------------------------------------------------------------- #
# The real table, as pasted from Word
# --------------------------------------------------------------------------- #
TABBED = "\n".join([
    "PERFORM TOUR GUIDE OPERATIONS",
    "UNIT CODE: TO/OS/TTM/CR/01/6/MA",
    "THEORY : PRACTICAL RATIO 2:3",
    "Element\tPerformance Criteria\tTheory\tPractical",
    "1. Prepare for tour guiding operations",
    "1.1\tTour itinerary is obtained as per the tour programme\t4\t4",
    "1.2\tTour guiding equipment is assembled as per the itinerary\t2\t5",
    "1.3\tClient profile is established as per the booking details\t3\t6",
    "1.4\tTour route is confirmed as per the itinerary\t3\t4",
    "1.5\tSafety requirements are observed as per the SOP\t3\t4",
    "1.6\tTransport arrangements are confirmed as per the tour programme\t3\t4",
    "\tSub Total\t18\t27",
    "2. Conduct tour guiding operations",
    "2.1\tClients are received as per the organisation procedure\t3\t4",
    "2.2\tTour commentary is delivered as per the itinerary\t3\t4",
    "2.3\tAttractions are interpreted as per the tour programme\t4\t5",
    "2.4\tClient queries are handled as per customer care standards\t3\t5",
    "2.5\tEmergencies are managed as per the SOP\t3\t5",
    "2.6\tTour report is compiled as per the organisation procedure\t2\t4",
    "\tSub Total\t18\t27",
    "\tGRAND TOTAL\t36\t54",
])


def _as_pipes(tabbed: str) -> str:
    """The same table as a user who tidied the paste into a Markdown grid."""
    return "\n".join("| " + " | ".join(line.split("\t")) + " |"
                     for line in tabbed.splitlines())


def _as_spaces(tabbed: str) -> str:
    """The same table as it arrives from a PDF viewer: columns become runs of
    spaces and nothing else survives."""
    return "\n".join("    ".join(line.split("\t")).rstrip()
                     for line in tabbed.splitlines())


def _mini(*rows: str) -> str:
    return "\n".join(rows)


def _blocking(problems):
    return [p for p in problems if p.blocking]


def _warnings(problems):
    return [p for p in problems if not p.blocking]


# --------------------------------------------------------------------------- #
# The primary fixture
# --------------------------------------------------------------------------- #
def test_the_real_table_parses_with_nothing_wrong_with_it():
    """The benchmark: a clean paste of a real CDACC table must raise no
    problem at all, or every later warning is noise the user learns to skip."""
    weighting, problems = aw.parse(TABBED)

    assert problems == []
    assert [el.number for el in weighting.elements] == ["1", "2"]
    assert len(weighting.pcs) == 12


def test_the_elements_keep_their_titles_and_their_own_pcs():
    weighting, _ = aw.parse(TABBED)

    assert weighting.elements[0].title == "Prepare for tour guiding operations"
    assert [pc.number for pc in weighting.elements[0].pcs] == [
        "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
    assert [pc.number for pc in weighting.elements[1].pcs] == [
        "2.1", "2.2", "2.3", "2.4", "2.5", "2.6"]
    assert all(pc.element_number == "2" for pc in weighting.elements[1].pcs)


def test_every_pc_keeps_its_two_weights_and_its_text():
    """The weights are the whole point of the table; the text is what the
    model later writes an item from, so a PC with weights but no text is as
    useless as no PC at all."""
    weighting, _ = aw.parse(TABBED)

    assert [pc.theory_weight for pc in weighting.pcs] == [
        4, 2, 3, 3, 3, 3, 3, 3, 4, 3, 3, 2]
    assert [pc.practical_weight for pc in weighting.pcs] == [
        4, 5, 6, 4, 4, 4, 4, 4, 5, 5, 5, 4]
    assert weighting.pc("1.1").text == (
        "Tour itinerary is obtained as per the tour programme")
    assert all(pc.text for pc in weighting.pcs)


def test_the_sub_total_rows_are_checksums_and_not_performance_criteria():
    """'Sub Total 18 27' ends with two integers exactly as a PC row does, so
    reading by shape alone would file two more PCs numbered nothing."""
    weighting, _ = aw.parse(TABBED)

    assert len(weighting.pcs) == 12
    assert [el.stated_theory_total for el in weighting.elements] == [18, 18]
    assert [el.stated_practical_total for el in weighting.elements] == [27, 27]
    assert weighting.elements[0].total(THEORY) == 18
    assert weighting.elements[0].total(PRACTICAL) == 27


def test_the_grand_total_row_goes_to_the_unit_not_to_an_element():
    weighting, _ = aw.parse(TABBED)

    assert weighting.stated_grand_theory == 36
    assert weighting.stated_grand_practical == 54
    assert weighting.grand_total(THEORY) == 36
    assert weighting.grand_total(PRACTICAL) == 54


def test_the_ratio_is_derived_from_the_grand_totals_in_lowest_terms():
    weighting, _ = aw.parse(TABBED)

    assert weighting.ratio == (2, 3)


def test_a_forty_sixty_unit_reduces_to_the_same_two_three():
    """Units state the same ratio at different scales - 40:60 and 36:54 are
    both 2:3 - and `assessment_allocation` compares ratios, not totals."""
    weighting, _ = aw.parse(_mini(
        "1. Prepare",
        "1.1\tTour itinerary is obtained\t40\t60",
        "\tGRAND TOTAL\t40\t60"))

    assert weighting.ratio == (2, 3)


def test_the_unit_is_identified_from_the_lines_above_the_table():
    weighting, _ = aw.parse(TABBED)

    assert weighting.unit_title == "PERFORM TOUR GUIDE OPERATIONS"
    assert weighting.cdacc_code == "TO/OS/TTM/CR/01/6/MA"
    assert weighting.knqf_level == "6"


# --------------------------------------------------------------------------- #
# The three paste shapes
# --------------------------------------------------------------------------- #
def test_tabs_pipes_and_spaces_all_parse_to_the_same_table():
    """Which separator arrives depends on whether the user copied from Word,
    from a Markdown-tidied paste or from a PDF viewer - never on the document.
    All three must give the same weighting or the same unit is assessed
    differently depending on where it was copied from."""
    from_tabs, tab_problems = aw.parse(TABBED)
    from_pipes, pipe_problems = aw.parse(_as_pipes(TABBED))
    from_spaces, space_problems = aw.parse(_as_spaces(TABBED))

    assert from_pipes == from_tabs
    assert from_spaces == from_tabs
    assert tab_problems == pipe_problems == space_problems == []


def test_a_pdf_paste_that_lost_its_column_runs_is_still_read_by_shape():
    """The worst case: a PDF viewer collapses every column run to one space,
    so nothing is left but the row's shape - an id in front, two weights
    behind."""
    weighting, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1 Tour itinerary is obtained as per the tour programme 4 4",
        "Sub Total 4 4"))

    assert problems == []
    assert weighting.pc("1.1").text == (
        "Tour itinerary is obtained as per the tour programme")
    assert weighting.pc("1.1").theory_weight == 4


def test_a_pc_whose_text_ends_in_a_number_keeps_the_right_weights():
    """'... within 24 hours 3 5' - the trailing pair is the weights and the 24
    is text, which is why both integers are required rather than one."""
    weighting, _ = aw.parse(_mini(
        "1. Conduct tour guiding operations",
        "1.1 Tour report is submitted within 24 hours 3 5"))

    pc = weighting.pc("1.1")
    assert (pc.theory_weight, pc.practical_weight) == (3, 5)
    assert pc.text == "Tour report is submitted within 24 hours"


# --------------------------------------------------------------------------- #
# Rows that wrapped
# --------------------------------------------------------------------------- #
def test_a_pc_whose_text_wrapped_is_rejoined():
    """Long PC text wraps inside its Word cell, so one row arrives as two
    lines. Left unjoined the tail line is a PC with no number and the PC keeps
    half its text."""
    weighting, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.2\tTour guiding equipment and materials are\t2\t5",
        "\tassembled as per the tour itinerary"))

    assert problems == []
    assert len(weighting.pcs) == 1
    assert weighting.pc("1.2").text == (
        "Tour guiding equipment and materials are assembled as per the "
        "tour itinerary")


def test_a_wrap_that_carried_the_weights_down_with_it_is_still_read():
    """When the cell wraps before its last line, the two weights sit beside
    the TAIL of the text and the numbered line has none at all."""
    weighting, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.2\tTour guiding equipment and materials are",
        "\tassembled as per the tour itinerary\t2\t5"))

    assert problems == []
    pc = weighting.pc("1.2")
    assert (pc.theory_weight, pc.practical_weight) == (2, 5)
    assert pc.text.endswith("assembled as per the tour itinerary")


def test_a_page_number_between_two_wrapped_lines_is_not_taken_as_a_weight():
    """A PDF paste carries the page furniture with it, and a bare '37' landing
    inside a wrapped row would otherwise be appended to the PC's text."""
    weighting, _ = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.2\tTour guiding equipment and materials are\t2\t5",
        "37",
        "\tassembled as per the tour itinerary"))

    assert "37" not in weighting.pc("1.2").text


# --------------------------------------------------------------------------- #
# What validate refuses to let through
# --------------------------------------------------------------------------- #
def test_a_sub_total_that_does_not_add_up_is_blocking():
    """The checksum's whole purpose. This is element 1 of the real table with
    PC 1.3 lost to a page break: the five rows that arrived add up to 15 and 21
    where the document's own sub-total row says 18 and 27."""
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained\t4\t4",
        "1.2\tEquipment is assembled\t2\t5",
        "1.4\tTour route is confirmed\t3\t4",
        "1.5\tSafety requirements are observed\t3\t4",
        "1.6\tTransport arrangements are confirmed\t3\t4",
        "\tSub Total\t18\t27"))

    theory = [p for p in _blocking(problems) if "theory sub-total" in p.message]
    assert theory and theory[0].where == "1"
    assert "18" in theory[0].message and "15" in theory[0].message
    assert any("practical sub-total" in p.message for p in _blocking(problems))


def test_sub_totals_that_do_not_reach_the_grand_total_are_blocking():
    """A whole element can vanish between two page breaks; each surviving
    element then adds up perfectly and only the grand total notices."""
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained\t4\t4",
        "\tSub Total\t4\t4",
        "\tGRAND TOTAL\t36\t54"))

    assert any("grand total says 36" in p.message for p in _blocking(problems))


def test_a_heading_that_disagrees_with_the_totals_is_blocking():
    """The heading is retyped by hand and the totals are not, so the totals
    win - but one of the two is wrong and only the user knows which."""
    weighting, problems = aw.parse(_mini(
        "THEORY : PRACTICAL RATIO 1:1",
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained\t36\t54",
        "\tGRAND TOTAL\t36\t54"))

    assert weighting.ratio == (2, 3)
    assert any("1:1" in p.message and "2:3" in p.message
               for p in _blocking(problems))


def test_a_heading_that_agrees_with_the_totals_says_nothing():
    _, problems = aw.parse(_mini(
        "THEORY : PRACTICAL RATIO 2:3",
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained\t36\t54",
        "\tGRAND TOTAL\t36\t54"))

    assert problems == []


def test_a_pc_before_any_element_is_kept_but_blocks():
    """A paste that starts mid-table has PCs with no element heading above
    them. Dropping them hides the damage; keeping them shows the user the rows
    to go back for."""
    weighting, problems = aw.parse(_mini(
        "1.1\tTour itinerary is obtained\t4\t4"))

    assert weighting.pc("1.1") is not None
    assert any("before any element" in p.message and p.where == "1.1"
               for p in _blocking(problems))


def test_a_pc_filed_under_the_wrong_element_is_blocking():
    """Checked on the object rather than on a paste: `assessment_allocation`
    reads a PC through its element, so a mis-filed PC corrupts that element's
    share of the marks however it got there."""
    weighting = UnitWeighting(elements=[WeightedElement(
        number="1", title="Prepare",
        pcs=[WeightedPC(number="2.1", element_number="2", text="Clients are "
                        "received", theory_weight=3, practical_weight=4)])])

    problems = aw.validate(weighting)

    assert any("2.1" in p.message and p.blocking for p in problems)


def test_a_pc_with_no_text_is_blocking():
    """A PDF paste can lose a whole cell and leave the weights behind. Nothing
    can be written from a PC that says nothing."""
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\t\t4\t4"))

    assert any("no text" in p.message and p.where == "1.1"
               for p in _blocking(problems))


def test_a_pc_row_that_lost_its_weights_is_blocking():
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained\t4\t4",
        "1.2\tEquipment is assembled"))

    assert any("no theory and practical weights" in p.message
               and p.where == "1.2" for p in _blocking(problems))


def test_the_same_pc_twice_is_blocking():
    """Pastes get made twice when the first one looked short, and the second
    lands under the first."""
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained\t4\t4",
        "1.1\tTour itinerary is obtained\t4\t4"))

    assert any("appears twice" in p.message for p in _blocking(problems))


def test_a_pc_weighted_zero_warns_without_blocking():
    """The document may really weight a PC zero for theory - a purely manual
    criterion - so this is a warning. It still has to be said: that PC can
    never appear on a written paper."""
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\tEquipment is assembled as per the itinerary\t0\t5",
        "\tSub Total\t0\t5"))

    assert _blocking(problems) == []
    warned = _warnings(problems)
    assert len(warned) == 1
    assert warned[0].where == "1.1" and "theory" in warned[0].message


def test_an_element_with_no_performance_criteria_is_blocking():
    _, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "2. Conduct tour guiding operations",
        "2.1\tClients are received\t3\t4"))

    assert any("no performance criteria" in p.message and p.where == "1"
               for p in _blocking(problems))


# --------------------------------------------------------------------------- #
# Nothing raises
# --------------------------------------------------------------------------- #
def test_prose_with_no_table_in_it_reports_rather_than_raises():
    weighting, problems = aw.parse(
        "The learner shall be assessed through written and practical tests.")

    assert weighting.elements == []
    assert any("No elements" in p.message for p in _blocking(problems))


def test_an_empty_paste_reports_rather_than_raises():
    weighting, problems = aw.parse("")

    assert isinstance(weighting, UnitWeighting)
    assert _blocking(problems)


def test_a_table_cut_off_mid_row_returns_what_was_read():
    """Users paste in two halves. The first half has to come back usable
    enough to show, with the damage named."""
    weighting, problems = aw.parse(_mini(
        "1. Prepare for tour guiding operations",
        "1.1\tTour itinerary is obtained as per the tour programme\t4\t4",
        "1.2\tTour guiding equipment and mater"))

    assert len(weighting.pcs) == 2
    assert weighting.pc("1.1").theory_weight == 4
    assert _blocking(problems)


# --------------------------------------------------------------------------- #
# The element row as a Word table actually pastes it
# --------------------------------------------------------------------------- #
def test_an_element_row_without_a_trailing_dot_is_still_an_element():
    """In a Word table the element number sits in its own CELL, so the glued
    line reads '1<TAB>Manage tourist arrival and departures' with no dot.
    That is the commonest paste there is, and it matched nothing: both PCs
    were filed under a synthetic element and the table came back broken."""
    text = ("1\tManage tourist arrival and departures\n"
            "1.1\tTour transfer resources are assembled\t4\t4\n"
            "1.2\tTourists are received per procedure\t2\t5\n"
            "Sub Total\t6\t9\n"
            "GRAND TOTAL\t6\t9")

    weighting, problems = aw.parse(text)

    assert [(e.number, e.title) for e in weighting.elements] == [
        ("1", "Manage tourist arrival and departures")]
    assert len(weighting.elements[0].pcs) == 2
    assert problems == []


def test_the_dotted_element_row_still_works():
    text = ("1. Manage tourist arrival and departures\n"
            "1.1\tTour transfer resources are assembled\t4\t4\n"
            "Sub Total\t4\t4\n"
            "GRAND TOTAL\t4\t4")

    weighting, problems = aw.parse(text)

    assert weighting.elements[0].title == "Manage tourist arrival and departures"
    assert problems == []


def test_a_wrapped_line_opening_with_a_figure_is_not_an_element():
    """'24 hours after arrival' would otherwise read as element 24 and take
    the rest of the table with it."""
    text = ("1\tManage tourist arrival and departures\n"
            "1.1\tTour transfer resources are assembled\n"
            "24 hours after the arrival is confirmed\t4\t4\n"
            "Sub Total\t4\t4\n"
            "GRAND TOTAL\t4\t4")

    weighting, problems = aw.parse(text)

    assert [e.number for e in weighting.elements] == ["1"]
    assert "24 hours" in weighting.elements[0].pcs[0].text
    assert problems == []
