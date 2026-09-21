"""Module C - allocating a CAT's marks. Pure arithmetic, ZERO API calls.

The regression fixture is PERFORM TOUR GUIDE OPERATIONS (theory:practical
2:3), whose PC weighting table was typed out of the real CDACC document. Its
theory column is reproduced cell for cell; where the practical column differs,
the test says why.
"""

import random

import assessment_allocation as alloc
from assessment_config import (MAX_TOTAL_MARKS, MIN_MARKS_PER_PC,
                               SPLIT_ITEM_THRESHOLD, VIABLE_MARKS_PER_PC,
                               marks_per_response)
from assessment_models import (BLOOM_LEVELS, PRACTICAL, THEORY, CatDefinition,
                               UnitWeighting, WeightedElement, WeightedPC)


# --------------------------------------------------------------------------- #
# The real unit
# --------------------------------------------------------------------------- #
THEORY_WEIGHTS = {"1.1": 4, "1.2": 2, "1.3": 3, "1.4": 3, "1.5": 3, "1.6": 3,
                  "2.1": 3, "2.2": 3, "2.3": 4, "2.4": 3, "2.5": 3, "2.6": 2}
PRACTICAL_WEIGHTS = {"1.1": 4, "1.2": 5, "1.3": 6, "1.4": 4, "1.5": 4,
                     "1.6": 4, "2.1": 4, "2.2": 4, "2.3": 5, "2.4": 5,
                     "2.5": 5, "2.6": 4}
PC_NUMBERS = list(THEORY_WEIGHTS)


def tour_guide() -> UnitWeighting:
    elements = []
    for number, title in (("1", "Prepare for tour guiding"),
                          ("2", "Conduct tour guiding")):
        pcs = [WeightedPC(number=n, element_number=number,
                          text=f"PC {n}",
                          theory_weight=THEORY_WEIGHTS[n],
                          practical_weight=PRACTICAL_WEIGHTS[n])
               for n in PC_NUMBERS if n.startswith(number + ".")]
        elements.append(WeightedElement(number=number, title=title, pcs=pcs))
    return UnitWeighting(unit_title="PERFORM TOUR GUIDE OPERATIONS",
                         knqf_level="5", elements=elements, ratio=(2, 3))


def whole_unit(assessment_type: str, total: int) -> CatDefinition:
    return CatDefinition(assessment_type=assessment_type, total_marks=total,
                         selected_pcs=list(PC_NUMBERS))


# --------------------------------------------------------------------------- #
# allocate
# --------------------------------------------------------------------------- #
def test_theory_column_matches_the_source_document():
    """The CDACC document's own theory column: weights summing to 36 spread
    over 50 marks. Every cell has to come back the way the document prints it,
    or the generated paper contradicts the curriculum it came from."""
    marks = [a.marks for a in alloc.allocate(tour_guide(),
                                             whole_unit(THEORY, 50))]

    assert marks == [6, 3, 4, 4, 4, 4, 4, 4, 6, 4, 4, 3]
    assert sum(marks) == 50


def test_practical_column_is_the_document_corrected():
    """Weights summing to 54 over 40 marks. The source document prints 2.6 as
    1 mark and its column as 39 - a manual slip, since the same weight of 4
    earns 3 marks everywhere else on that column. The engine is EXPECTED to
    differ there: it prints 3 and totals 40."""
    marks = [a.marks for a in alloc.allocate(tour_guide(),
                                             whole_unit(PRACTICAL, 40))]

    assert marks == [3, 4, 4, 3, 3, 3, 3, 3, 4, 4, 3, 3]
    assert sum(marks) == 40
    assert marks[-1] == 3        # the document says 1
    assert sum(marks) != 39      # the document says 39


def test_naive_rounding_would_have_overshot_the_practical_paper():
    """Why largest remainder is not a matter of taste: rounding each practical
    share on its own gives a paper out of 41 when the user asked for 40."""
    naive = sum(round(w * 40 / 54) for w in PRACTICAL_WEIGHTS.values())

    assert naive == 41
    assert sum(a.marks for a in alloc.allocate(
        tour_guide(), whole_unit(PRACTICAL, 40))) == 40


def test_a_partial_cat_still_awards_its_whole_total():
    """CAT 1 covers element 1 only. Its 30 marks are shared over those six PCs
    alone - dividing by the unit's grand total would hand out roughly half the
    paper and silently lose the rest."""
    cat = CatDefinition(assessment_type=THEORY, total_marks=30,
                        selected_pcs=["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"])
    marks = alloc.allocate(tour_guide(), cat)

    assert [a.pc_number for a in marks] == ["1.1", "1.2", "1.3", "1.4",
                                            "1.5", "1.6"]
    assert sum(a.marks for a in marks) == 30


def test_pcs_come_back_in_the_table_order_not_the_click_order():
    """The user ticks 2.3 then 1.1; the PC distribution table prints 1.1 then
    2.3, in every document."""
    cat = CatDefinition(assessment_type=THEORY, total_marks=10,
                        selected_pcs=["2.3", "1.1", "2.1"])

    assert [a.pc_number for a in alloc.allocate(tour_guide(), cat)] == [
        "1.1", "2.1", "2.3"]


def _two_pcs(first_weight, second_weight, total):
    unit = UnitWeighting(elements=[WeightedElement(number="1", pcs=[
        WeightedPC(number="1.1", element_number="1",
                   theory_weight=first_weight),
        WeightedPC(number="1.2", element_number="1",
                   theory_weight=second_weight)])])
    cat = CatDefinition(assessment_type=THEORY, total_marks=total,
                        selected_pcs=["1.1", "1.2"])
    return [a.marks for a in alloc.allocate(unit, cat)]


def test_an_equal_remainder_goes_to_the_heavier_pc():
    """Weights 1 and 3 over 6 marks leave 1.5 and 4.5 - the same remainder, one
    mark to give away. The heavier PC takes it, so the spare mark lands on the
    criterion the unit says matters more, and lands there every time."""
    assert _two_pcs(1, 3, 6) == [1, 5]


def test_an_equal_remainder_on_equal_weights_goes_to_the_lower_pc_number():
    """Nothing tells these two apart, so the order of the document does. Left
    to set iteration or click order this mark would wander between
    regenerations of the same CAT."""
    assert _two_pcs(1, 1, 5) == [3, 2]


def test_an_unweighted_selection_falls_back_to_equal_shares():
    """A table pasted with an empty practical column must not divide by zero -
    check_total blocks it, but allocate still has to return something."""
    unit = tour_guide()
    for pc in unit.pcs:
        pc.practical_weight = 0
    marks = [a.marks for a in alloc.allocate(unit, whole_unit(PRACTICAL, 24))]

    assert marks == [2] * 12


# --------------------------------------------------------------------------- #
# check_total
# --------------------------------------------------------------------------- #
def _blocking(problems):
    return [p for p in problems if p.blocking]


def test_a_total_below_one_mark_is_blocking():
    problems = _blocking(alloc.check_total(tour_guide(),
                                           whole_unit(THEORY, 0)))

    assert problems and any("at least 1 mark" in p.message for p in problems)


def test_fewer_marks_than_pcs_is_blocking_and_offers_both_fixes():
    """8 marks over 12 PCs leaves four PCs on 0, which cannot be assessed. The
    user cannot tell from the form whether to raise the total or drop PCs, so
    the message has to name both numbers."""
    problems = _blocking(alloc.check_total(tour_guide(),
                                           whole_unit(THEORY, 8)))
    message = " ".join(p.message for p in problems)

    assert problems
    assert "12" in message and "0 marks" in message
    assert "raise the total to 12" in message.lower()
    assert "deselect 4" in message.lower()


def test_a_viable_total_raises_nothing_blocking():
    assert _blocking(alloc.check_total(tour_guide(),
                                       whole_unit(THEORY, 50))) == []


def test_a_thin_total_warns_and_names_the_capped_pcs():
    """18 marks over 12 PCs is under the viable figure: the lightest PCs come
    out on a single mark, which can only be asked for as recall. The warning
    names them so the user can see which parts of the unit go thin."""
    problems = alloc.check_total(tour_guide(), whole_unit(THEORY, 18))
    warnings = [p for p in problems if not p.blocking]

    assert _blocking(problems) == []
    assert len(warnings) == 1
    assert str(VIABLE_MARKS_PER_PC) in warnings[0].message
    capped = [a.pc_number for a in alloc.allocate(tour_guide(),
                                                  whole_unit(THEORY, 18))
              if a.marks <= MIN_MARKS_PER_PC]
    assert capped and all(n in warnings[0].message for n in capped)


def test_a_total_over_the_ceiling_only_warns():
    problems = alloc.check_total(tour_guide(),
                                 whole_unit(THEORY, MAX_TOTAL_MARKS + 1))

    assert _blocking(problems) == []
    assert any(str(MAX_TOTAL_MARKS) in p.message for p in problems)


def test_an_empty_selection_is_blocking_and_does_not_divide_by_zero():
    cat = CatDefinition(assessment_type=THEORY, total_marks=50,
                        selected_pcs=[])

    assert _blocking(alloc.check_total(tour_guide(), cat))


def test_a_pc_that_is_not_in_the_table_is_reported_against_itself():
    """A stale selection kept from a re-pasted table: it can never be given
    marks, so `where` has to point at the PC the user must untick."""
    cat = CatDefinition(assessment_type=THEORY, total_marks=50,
                        selected_pcs=["1.1", "9.9"])
    problems = _blocking(alloc.check_total(tour_guide(), cat))

    assert [p.where for p in problems] == ["9.9"]


def test_an_unweighted_column_is_blocking():
    unit = tour_guide()
    for pc in unit.pcs:
        pc.practical_weight = 0
    problems = _blocking(alloc.check_total(unit, whole_unit(PRACTICAL, 40)))

    assert any("practical" in p.message for p in problems)


# --------------------------------------------------------------------------- #
# assign_bloom
# --------------------------------------------------------------------------- #
def _bloomed(assessment_type, total):
    unit = tour_guide()
    cat = whole_unit(assessment_type, total)
    return alloc.assign_bloom(alloc.allocate(unit, cat), unit.knqf_level,
                              total)


def test_every_allocated_mark_keeps_its_pc_and_gets_a_level():
    """The grid may cut a PC in two but may never change what it is worth: the
    marking scheme and the table of specifications are printed from the same
    list and have to agree PC by PC."""
    before = {a.pc_number: a.marks for a in alloc.allocate(
        tour_guide(), whole_unit(THEORY, 50))}
    after = {}
    for a in _bloomed(THEORY, 50):
        assert a.bloom in BLOOM_LEVELS
        after[a.pc_number] = after.get(a.pc_number, 0) + a.marks

    assert after == before
    assert sum(after.values()) == 50


def test_the_theory_paper_reaches_all_six_levels():
    """50 marks over twelve PCs leaves a 2-mark EVALUATING cell and a 2-mark
    CREATING cell in each element, and the lightest PCs are what fill them. A
    paper that stops at APPLYING is the failure this guards against."""
    entries = _bloomed(THEORY, 50)

    assert {a.bloom for a in entries} == set(BLOOM_LEVELS)


def test_a_column_too_small_to_hold_a_question_borrows_until_it_can():
    """The practical's CREATING cell came out at one mark per element, and the
    lightest PC on it is worth three. That cell used to be left empty, costing
    the paper a whole Bloom level; now it borrows from the richest column that
    can spare a mark until it can hold a real question, and the 3-mark PC goes
    in. The element's total is untouched either way - that is what keeps the
    table of specifications adding up along its rows."""
    entries = _bloomed(PRACTICAL, 40)

    assert {a.bloom for a in entries} == set(BLOOM_LEVELS)
    assert sum(a.marks for a in entries) == 40


def test_no_item_is_worth_less_than_one_answer_at_its_own_level():
    """"Design a maintenance routine. (1 mark)" is not a hard question, it is
    an unanswerable one. A one-mark item is always recall."""
    for assessment_type, total in ((THEORY, 50), (PRACTICAL, 40),
                                   (THEORY, 30), (THEORY, 20)):
        for a in _bloomed(assessment_type, total):
            assert a.marks >= marks_per_response(a.bloom), (
                f"{a.pc_number} at {a.bloom} for {a.marks}")


# --------------------------------------------------------------------------- #
# Keeping every column answerable
# --------------------------------------------------------------------------- #
def test_a_deficient_column_borrows_from_the_richest_one():
    #      know und app ana eva cre
    assert alloc._respect_minimums([3, 8, 4, 4, 1, 0]) == [3, 7, 4, 4, 2, 0]


def test_borrowing_never_leaves_the_lender_unable_to_hold_a_question():
    out = alloc._respect_minimums([0, 2, 2, 2, 1, 0])
    assert all(v == 0 or v >= 2 for v in out[1:])
    assert sum(out) == 7


def test_a_column_nobody_can_lend_to_gives_its_marks_up():
    """Three columns of exactly two have nothing spare, so the odd mark at
    CREATING cannot be topped up and goes to the highest column that is
    already viable rather than sliding the paper down to recall."""
    out = alloc._respect_minimums([0, 2, 2, 2, 0, 1])
    assert sum(out) == 7
    assert out[5] == 0
    assert all(v == 0 or v >= 2 for v in out[1:])


def test_knowledge_is_never_deficient():
    assert alloc._respect_minimums([1, 4, 4, 0, 0, 0]) == [1, 4, 4, 0, 0, 0]


def test_respecting_the_minimums_never_changes_the_total():
    import random
    random.seed(11)
    for _ in range(300):
        row = [random.randint(0, 6) for _ in range(6)]
        assert sum(alloc._respect_minimums(list(row))) == sum(row)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def test_a_spare_mark_goes_to_knowledge_rather_than_to_creating():
    """The fault this replaced: a 6-mark PC cut into 5 and 1, with the 1
    landing on CREATING because that column happened to have room - a paper
    asking the candidate to design something for one mark. One mark is a
    recall mark, so the tail goes to KNOWLEDGE."""
    #                       know und app ana eva cre
    cut = alloc._split_at([0, 5, 0, 0, 0, 4], 6)

    assert cut is not None
    head_column, head, tail_column, tail = cut
    assert (head_column, head) == (1, 5)
    assert (tail_column, tail) == (0, 1)


def test_a_cut_with_no_answerable_shape_is_not_made_at_all():
    """Nothing to cut against: the PC goes somewhere whole instead."""
    assert alloc._split_at([0, 0, 0, 0, 0, 0], 6) is None


def test_neither_half_of_a_cut_is_ever_too_small_for_where_it_lands():
    import random
    random.seed(5)
    for _ in range(500):
        demand = [random.randint(0, 8) for _ in range(6)]
        marks = random.randint(6, 12)
        cut = alloc._split_at(list(demand), marks)
        if cut is None:
            continue
        head_column, head, tail_column, tail = cut
        assert head + tail == marks
        assert head >= marks_per_response(BLOOM_LEVELS[head_column])
        assert tail >= marks_per_response(BLOOM_LEVELS[tail_column])


def _split_paper():
    """A CAT on three PCs out of 30: 13, 10 and 7 marks, every one of them over
    the split threshold and none matching a column."""
    unit = tour_guide()
    cat = CatDefinition(assessment_type=THEORY, total_marks=30,
                        selected_pcs=["1.1", "1.2", "1.3"])
    return alloc.assign_bloom(alloc.allocate(unit, cat), "5", 30)


def test_a_heavy_pc_is_split_across_two_levels():
    """Three questions cannot cover six columns. Splitting is what lets a short
    CAT still spread across the taxonomy, and each half keeps the PC it came
    from so the totals never move."""
    entries = _split_paper()
    by_pc = {}
    for a in entries:
        by_pc.setdefault(a.pc_number, []).append(a)

    assert {n: sum(e.marks for e in v) for n, v in by_pc.items()} == {
        "1.1": 13, "1.2": 7, "1.3": 10}
    assert all(len(v) == 2 for v in by_pc.values())
    assert len({a.bloom for a in entries}) >= 4


def test_a_split_is_two_entries_of_at_least_a_mark_at_different_levels():
    """A PC cut into 2 and 2 would print as two half-questions on one
    criterion, and a cut into three would print as three. Two entries, distinct
    levels, nothing below a mark."""
    for entries in (_split_paper(), _bloomed(THEORY, 50),
                    _bloomed(PRACTICAL, 40)):
        by_pc = {}
        for a in entries:
            by_pc.setdefault(a.pc_number, []).append(a)
        for parts in by_pc.values():
            assert len(parts) <= 2
            if len(parts) == 2:
                assert sum(p.marks for p in parts) > SPLIT_ITEM_THRESHOLD
                assert parts[0].bloom != parts[1].bloom
                assert all(p.marks >= 1 for p in parts)


def test_element_rows_still_hold_their_own_marks():
    """The table of specifications is read across as well as down: element 1's
    row has to total what element 1 was allocated."""
    rows = {}
    for a in alloc.allocate(tour_guide(), whole_unit(PRACTICAL, 40)):
        rows[a.element_number] = rows.get(a.element_number, 0) + a.marks
    after = {}
    for a in _bloomed(PRACTICAL, 40):
        after[a.element_number] = after.get(a.element_number, 0) + a.marks

    assert after == rows == {"1": 20, "2": 20}


def test_the_column_targets_total_the_paper():
    """0.20 + 0.20 + 0.25 + 0.20 + 0.08 + 0.07 of 50 is 10, 10, 12.5, 10, 4,
    3.5 - two cells that cannot be marks. Integerised they still have to come
    to 50, or the specification asks for a paper of a different size."""
    columns = alloc._column_targets("5", 50)

    assert sum(columns) == 50
    assert len(columns) == len(BLOOM_LEVELS)
    assert all(c >= 0 for c in columns)


def test_an_empty_allocation_bloomes_to_nothing():
    assert alloc.assign_bloom([], "5", 0) == []


# --------------------------------------------------------------------------- #
# Properties - any table, any total
# --------------------------------------------------------------------------- #
def _random_unit(rng, count):
    pcs, elements, per = [], [], max(1, count // 2)
    for index in range(count):
        element = str(index // per + 1)
        pcs.append(WeightedPC(number=f"{element}.{index % per + 1}",
                              element_number=element,
                              theory_weight=rng.randint(1, 9),
                              practical_weight=rng.randint(1, 9)))
    for element in sorted({pc.element_number for pc in pcs},
                          key=lambda n: int(n)):
        elements.append(WeightedElement(
            number=element,
            pcs=[pc for pc in pcs if pc.element_number == element]))
    return UnitWeighting(knqf_level=rng.choice(["3", "4", "5", "6", ""]),
                         elements=elements)


def test_allocation_always_totals_exactly_over_random_tables():
    """hypothesis is not installed here, so a seeded loop stands in for it:
    250 weight vectors x totals. The column adds up, nothing goes negative and
    no selected PC drops out - the three things every document downstream
    assumes without checking."""
    rng = random.Random(20240917)
    for _ in range(250):
        unit = _random_unit(rng, rng.randint(1, 20))
        selected = rng.sample([pc.number for pc in unit.pcs],
                              rng.randint(1, len(unit.pcs)))
        cat = CatDefinition(assessment_type=rng.choice([THEORY, PRACTICAL]),
                            total_marks=rng.randint(1, 300),
                            selected_pcs=selected)
        out = alloc.allocate(unit, cat)

        assert sum(a.marks for a in out) == cat.total_marks
        assert all(a.marks >= 0 for a in out)
        assert {a.pc_number for a in out} == set(selected)


def test_bloom_assignment_never_loses_a_mark_over_random_papers():
    """The same loop carried through the grid: whatever the profile and the
    element sizes, every entry carries a level and every PC keeps its total."""
    rng = random.Random(1976)
    for _ in range(250):
        unit = _random_unit(rng, rng.randint(1, 20))
        selected = rng.sample([pc.number for pc in unit.pcs],
                              rng.randint(1, len(unit.pcs)))
        total = rng.randint(len(selected), 300)
        cat = CatDefinition(assessment_type=rng.choice([THEORY, PRACTICAL]),
                            total_marks=total, selected_pcs=selected)
        before = alloc.allocate(unit, cat)
        after = alloc.assign_bloom(before, unit.knqf_level, total)

        assert all(a.bloom in BLOOM_LEVELS for a in after)
        assert sum(a.marks for a in after) == total
        per_pc = {}
        for a in after:
            per_pc.setdefault(a.pc_number, []).append(a.marks)
        assert {n: sum(v) for n, v in per_pc.items()} == {
            a.pc_number: a.marks for a in before}
        for number, parts in per_pc.items():
            assert len(parts) <= 2
            assert len(parts) == 1 or (sum(parts) > SPLIT_ITEM_THRESHOLD
                                       and min(parts) >= 1)


def test_an_unwritable_item_is_reported_before_anything_is_generated():
    """The trainer fixes this by changing the selection, so they are told at
    the distribution table rather than after paying for a generation."""
    from assessment_models import Allocation
    problems = alloc.check_items([
        Allocation(element_number="1", element_title="E1", pc_number="1.1",
                   pc_text="c", weight=4, marks=1, bloom="analysing"),
        Allocation(element_number="1", element_title="E1", pc_number="1.2",
                   pc_text="c", weight=4, marks=1, bloom="knowledge"),
    ])

    assert len(problems) == 1
    assert problems[0].where == "PC 1.1"
    assert problems[0].blocking


def test_a_sound_distribution_reports_nothing():
    assert alloc.check_items(_bloomed(THEORY, 50)) == []
