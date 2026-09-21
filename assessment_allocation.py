"""Module C - DETERMINISTIC mark allocation (NO AI).

`assessment_models` hands this part the two things a mark can depend on: the
unit's own weighting table and what the user asked for. Everything here is
arithmetic on those two - no I/O, no model, no Streamlit - because the same
numbers are printed in three different documents (the marking scheme, the
table of specifications and the PC distribution table) and the only way they
can agree is for each mark to be computed exactly once, here, before the model
is ever called.

WHY LARGEST REMAINDER, NEVER round()
    PERFORM TOUR GUIDE OPERATIONS, practical, twelve PCs weighted
    4 5 6 4 4 4 4 4 5 5 5 4 (base 54) out of 40 marks. Rounding each share on
    its own gives 3 4 4 3 3 3 3 3 4 4 4 3 - which totals 41, on a paper that is
    out of 40. That drift is not a matter of taste in rounding: it is a
    document that contradicts itself. Largest remainder hands out the floors
    first and then the marks that are left over, one each, so the column always
    totals exactly what the user asked for.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from assessment_config import (MAX_TOTAL_MARKS, MIN_MARKS_PER_PC,
                               SPLIT_ITEM_THRESHOLD, VIABLE_MARKS_PER_PC,
                               profile_for)
from assessment_models import (BLOOM_LEVELS, Allocation, CatDefinition,
                               Problem, UnitWeighting, WeightedPC)


# --------------------------------------------------------------------------- #
# Apportionment
# --------------------------------------------------------------------------- #
def _pc_sort_key(number: str) -> Tuple[Tuple[int, ...], str]:
    """'1.10' after '1.2', which sorting the strings would get backwards."""
    parts = tuple(int(p) for p in re.findall(r"\d+", number or ""))
    return (parts or (0,), number or "")


def _largest_remainder(raw: Sequence, total: int,
                       tiebreak: Sequence) -> List[int]:
    """Whole values summing to `total` exactly, nearest to the raw shares.

    Every entry keeps its floor; the marks left over go one each to the largest
    fractional remainders. `tiebreak[i]` settles equal remainders and is what
    makes the result reproducible rather than dependent on the order the PCs
    happened to arrive in.
    """
    floors = [math.floor(x) for x in raw]
    short = total - sum(floors)
    if short <= 0:
        return floors
    ranked = sorted(range(len(floors)),
                    key=lambda i: (-(raw[i] - floors[i]), tiebreak[i]))
    for i in ranked[:short]:
        floors[i] += 1
    return floors


def _selected(weighting: UnitWeighting,
              cat: CatDefinition) -> List[WeightedPC]:
    """The selected PCs in the table's own order, not the order they were
    clicked: every document prints them 1.1, 1.2, 1.3 down the page."""
    wanted = set(cat.selected_pcs or [])
    return [pc for pc in weighting.pcs if pc.number in wanted]


def allocate(weighting: UnitWeighting,
             cat: CatDefinition) -> List[Allocation]:
    """Share this CAT's total out over its selected PCs, by weight.

    The base is the SELECTED PCs' weights only. A CAT that covers half a unit
    still awards the whole total the user entered, so the denominator has to
    shrink with the selection - dividing by the unit's grand total would leave
    the paper short of its own total.

    The shares are held as exact fractions and integerised by largest
    remainder, so the column totals `cat.total_marks` to the mark. Ties go to
    the heavier PC first, then to the lower PC number - arbitrary but fixed, so
    that regenerating a CAT never silently moves a mark from 1.2 to 2.6.

    A selection carrying no weight at all for this assessment type falls back
    to equal shares rather than dividing by zero; `check_total` reports that as
    blocking, because it means the pasted table is wrong.
    """
    pcs = _selected(weighting, cat)
    if not pcs:
        return []

    weights = [pc.weight_for(cat.assessment_type) for pc in pcs]
    base = sum(weights)
    if base <= 0:
        weights = [1] * len(pcs)
        base = len(pcs)

    total = max(0, int(cat.total_marks))
    raw = [Fraction(w * total, base) for w in weights]
    tiebreak = [(-w, _pc_sort_key(pc.number)) for w, pc in zip(weights, pcs)]
    marks = _largest_remainder(raw, total, tiebreak)

    out: List[Allocation] = []
    for pc, weight, mark in zip(pcs, weights, marks):
        element = weighting.element_of(pc.number)
        out.append(Allocation(
            element_number=pc.element_number or (element.number if element else ""),
            element_title=element.title if element else "",
            pc_number=pc.number,
            pc_text=pc.text,
            weight=weight,
            marks=mark,
        ))
    return out


# --------------------------------------------------------------------------- #
# Is the total the user typed a total this CAT can carry?
# --------------------------------------------------------------------------- #
def check_total(weighting: UnitWeighting,
                cat: CatDefinition) -> List[Problem]:
    """What is wrong with this total, said in terms the user can act on.

    A blocking problem names the constraint that broke and both ways out of it
    - raise the total, or deselect PCs - because from the form alone the user
    cannot tell which of the two numbers is the one they meant to change.
    """
    problems: List[Problem] = []
    known = _selected(weighting, cat)
    seen = {pc.number for pc in known}
    for number in cat.selected_pcs or []:
        if number not in seen:
            problems.append(Problem(
                f"PC {number} is selected but is not in the weighting table, "
                f"so it cannot be allocated any marks. Re-paste the table or "
                f"deselect {number}.", blocking=True, where=number))

    if not known:
        problems.append(Problem(
            "No performance criterion is selected, so there is nothing to "
            "allocate marks to. Select at least one PC.", blocking=True))
        return problems

    total = int(cat.total_marks)
    count = len(known)

    if total < 1:
        problems.append(Problem(
            f"The total is {total} marks. A CAT has to be marked out of at "
            f"least 1 mark.", blocking=True))

    needed = count * MIN_MARKS_PER_PC
    if total < needed:
        keepable = max(0, total) // MIN_MARKS_PER_PC
        problems.append(Problem(
            f"{total} marks cannot cover {count} selected PCs: at "
            f"{MIN_MARKS_PER_PC} mark each they need at least {needed}. Below "
            f"that at least one PC allocates to 0 marks and cannot be assessed "
            f"at all. Either raise the total to {needed} or more, or deselect "
            f"{count - keepable} PCs to leave {keepable}.", blocking=True))

    if sum(pc.weight_for(cat.assessment_type) for pc in known) <= 0:
        problems.append(Problem(
            f"None of the {count} selected PCs carries a "
            f"{cat.assessment_type} weight, so there is nothing to apportion "
            f"by and every PC would fall back to an equal share. Check the "
            f"{cat.assessment_type} column of the pasted table.",
            blocking=True))

    # Only worth saying once the paper can actually be allocated: below the
    # floor the marks this names are 0, not 1, and the blocking problem above
    # already says so.
    if needed <= total < count * VIABLE_MARKS_PER_PC:
        capped = [a.pc_number for a in allocate(weighting, cat)
                  if a.marks <= MIN_MARKS_PER_PC]
        named = (f" {', '.join(capped)} would be capped at "
                 f"{MIN_MARKS_PER_PC} mark.") if capped else ""
        problems.append(Problem(
            f"{total} marks over {count} PCs is under {VIABLE_MARKS_PER_PC} "
            f"marks each, which forces those PCs down to single-mark recall "
            f"items.{named}", blocking=False))

    if total > MAX_TOTAL_MARKS:
        problems.append(Problem(
            f"{total} marks is above the {MAX_TOTAL_MARKS}-mark ceiling for a "
            f"single CAT. Mark the paper out of less and scale it afterwards.",
            blocking=False))

    return problems


# --------------------------------------------------------------------------- #
# The element x Bloom grid
# --------------------------------------------------------------------------- #
def _column_targets(knqf_level: str, total: int) -> List[int]:
    """The profile's percentages as whole marks that still total the paper."""
    profile = profile_for(knqf_level)
    raw = [profile.get(level, 0.0) * total for level in BLOOM_LEVELS]
    return _largest_remainder(raw, total, list(range(len(BLOOM_LEVELS))))


def _fit_grid(row_totals: Sequence[int], col_totals: Sequence[int],
              tolerance: float = 1e-9,
              max_passes: int = 50) -> List[List[float]]:
    """Iterative proportional fitting: the grid that honours both margins.

    Rows are scaled to their element's marks, then columns to the profile's
    marks, until neither moves. From a flat start it settles on the first pass,
    which is the point - the cells are right because both margins were fitted,
    not because of a formula that only happens to hold for a flat start.
    """
    rows, cols = len(row_totals), len(col_totals)
    grid = [[1.0] * cols for _ in range(rows)]
    for _ in range(max_passes):
        drift = 0.0
        for r in range(rows):
            line = sum(grid[r])
            if line > 0:
                factor = row_totals[r] / line
                drift = max(drift, abs(factor - 1.0))
                grid[r] = [v * factor for v in grid[r]]
        for c in range(cols):
            column = sum(grid[r][c] for r in range(rows))
            if column > 0:
                factor = col_totals[c] / column
                drift = max(drift, abs(factor - 1.0))
                for r in range(rows):
                    grid[r][c] *= factor
        if drift <= tolerance:
            break
    return grid


def _best_whole(demand: List[int], marks: int) -> int:
    """Where a whole PC goes: the tightest cell that still holds it, or failing
    that the emptiest one, so an overflow lands where there is most room."""
    fits = [i for i, d in enumerate(demand) if d >= marks]
    if fits:
        return min(fits, key=lambda i: (demand[i], i))
    return min(range(len(demand)), key=lambda i: (-demand[i], i))


def _place(allocations: List[Allocation],
           demand: List[int]) -> List[Allocation]:
    """Hand one element's PCs to Bloom columns, biggest PC first.

    Biggest first because the small PCs are what the gaps get filled with:
    placing 3 marks before 6 leaves the 6 nowhere to go but over a margin.

    A PC over SPLIT_ITEM_THRESHOLD is cut in two when some column needs less
    than that PC carries - one entry finishes that column exactly, the other
    goes where it fits. A CAT on three PCs out of 30 marks allocates 13, 10 and
    7 while no column is worth more than 8: without splitting, three questions
    would have to cover six columns and half the taxonomy would go empty.

    A PC at or below the threshold is never cut, because two 2-mark halves of
    one criterion print as half a question each. A column smaller than every PC
    that could go in it therefore stays empty - on the tour-guide practical,
    CREATING asks for one mark per element and the smallest PC there is worth
    three.
    """
    out: Dict[str, List[Allocation]] = {}
    for a in sorted(allocations,
                    key=lambda x: (-x.marks, _pc_sort_key(x.pc_number))):
        marks = a.marks
        exact = next((i for i, d in enumerate(demand) if d == marks and d > 0),
                     None)
        fillable = [i for i, d in enumerate(demand) if 1 <= d <= marks - 1]
        if exact is not None:
            demand[exact] -= marks
            pieces = [(exact, marks)]
        elif marks > SPLIT_ITEM_THRESHOLD and fillable:
            first = min(fillable, key=lambda i: (-demand[i], i))
            head = demand[first]
            demand[first] = 0
            rest = marks - head
            second = _best_whole(demand, rest)
            demand[second] -= rest
            pieces = sorted([(first, head), (second, rest)])
        else:
            column = _best_whole(demand, marks)
            demand[column] -= marks
            pieces = [(column, marks)]
        out[a.pc_number] = [
            Allocation(element_number=a.element_number,
                       element_title=a.element_title,
                       pc_number=a.pc_number,
                       pc_text=a.pc_text,
                       weight=a.weight,
                       marks=piece,
                       bloom=BLOOM_LEVELS[column])
            for column, piece in pieces]
    return [entry for a in allocations for entry in out[a.pc_number]]


def assign_bloom(allocations: List[Allocation], knqf_level: str,
                 total_marks: int) -> List[Allocation]:
    """Give every allocated mark a Bloom level, without moving any of them.

    Both margins of the grid are fixed before a single PC is placed: the rows
    are the marks `allocate` already gave each element, the columns are the
    KNQF level's profile integerised by largest remainder so that the six of
    them still total the paper. Only then are the PCs fitted into the cells -
    which is why the table of specifications adds up along its rows and down
    its columns, and why no PC's total changes to make it do so.

    `total_marks` sizes the profile. When it disagrees with what the
    allocations actually carry, the allocations win - a mark that exists has to
    appear somewhere in the grid - and the profile is sized to them instead.

    Returned in the order given, a split PC's two entries adjacent and in
    taxonomy order. Every entry carries a level, and the marks still total what
    came in.
    """
    if not allocations:
        return []

    carried = sum(a.marks for a in allocations)
    asked = int(total_marks)
    columns = _column_targets(knqf_level, asked if asked == carried else carried)

    order: List[str] = []
    rows: Dict[str, int] = {}
    for a in allocations:
        if a.element_number not in rows:
            order.append(a.element_number)
            rows[a.element_number] = 0
        rows[a.element_number] += a.marks

    grid = _fit_grid([rows[e] for e in order], columns)

    out: List[Allocation] = []
    for index, element in enumerate(order):
        demand = _largest_remainder(grid[index], rows[element],
                                    list(range(len(BLOOM_LEVELS))))
        out.extend(_place([a for a in allocations
                           if a.element_number == element], demand))
    return out
