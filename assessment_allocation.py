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
                               MAX_RESPONSES_PER_ITEM, marks_per_response,
                               natural_level, profile_for)
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


def _need(column: int) -> int:
    """The smallest item the Bloom column at `column` can carry."""
    return marks_per_response(BLOOM_LEVELS[column])


def _best_whole(demand: List[int], marks: int) -> int:
    """Where a whole PC goes: the tightest cell that still holds it, or failing
    that the emptiest one, so an overflow lands where there is most room.

    Cells that cannot carry an item this small are passed over first - one
    mark at ANALYSING is a question asking the candidate to analyse something
    for a single mark, which is not a hard question but an unanswerable one.
    Only if no column can carry it does the usual choice apply, because the
    mark exists and has to be somewhere.
    """
    allowed = [i for i in range(len(demand)) if marks >= _need(i)]
    fits = [i for i in allowed if demand[i] >= marks]
    if fits:
        return min(fits, key=lambda i: (demand[i], i))
    if allowed:
        return min(allowed, key=lambda i: (-demand[i], i))
    return min(range(len(demand)), key=lambda i: (-demand[i], i))


def _split_at(demand: List[int], marks: int):
    """Where to cut a PC in two so BOTH halves are writable, or None.

    The head finishes some column exactly and the tail goes where it fits.
    Both have to clear the minimum of the column they land in: splitting a
    6-mark PC into 5 and 1 is fine when the 1 goes to KNOWLEDGE and wrong when
    it goes to CREATING, which is how papers were ending up asking a candidate
    to design something for one mark.

    Returned as (head_column, head_marks, tail_column, tail_marks), preferring
    the largest head - the same preference as before, now among the cuts that
    leave two answerable questions rather than among all of them.
    """
    options = []
    for column, head in enumerate(demand):
        if not _need(column) <= head <= marks - 1:
            continue
        tail = marks - head
        remaining = list(demand)
        remaining[column] = 0
        landing = _best_whole(remaining, tail)
        if tail < _need(landing):
            continue
        options.append((column, head, landing, tail))
    if not options:
        return None
    return max(options, key=lambda o: (o[1], -o[0]))


def check_items(allocations: Sequence[Allocation]) -> List[Problem]:
    """Items too small for the verb their Bloom level allows.

    `_respect_minimums` and `_split_at` between them stop these being made,
    and on realistic weightings none appear at all. They can still survive a
    CAT stretched over more performance criteria than it has marks for, where
    a PC allocates to one or two marks whatever anyone does with it - and the
    trainer needs to be told that here, looking at the distribution table,
    rather than after paying for a generation that could not have been written.
    """
    out: List[Problem] = []
    for a in allocations:
        need = marks_per_response(a.bloom)
        if a.bloom and a.marks < need:
            out.append(Problem(
                message=(f"PC {a.pc_number} is set at {a.bloom} for "
                         f"{a.marks} mark(s), but a {a.bloom} question asks "
                         f"for a developed answer and cannot be marked out of "
                         f"less than {need}. Select fewer performance criteria "
                         f"or raise the total marks."),
                blocking=True, where=f"PC {a.pc_number}"))
    return out


def _respect_minimums(demand: List[int]) -> List[int]:
    """Make every column able to carry a whole question, keeping the total.

    A column asking for one mark at ANALYSING is a question nobody can write:
    the bank gives the item "analyse", and one mark buys one point, so the
    paper ends up asking a candidate to analyse something for a single mark.
    Published CDACC papers never do this - a one-mark item is always recall.

    So a deficient column BORROWS rather than closing. A mark is taken from
    the richest column that can still carry a question without it, until the
    deficient one is viable. Borrowing is tried first because closing the
    column costs the paper a whole Bloom level, and a CAT is expected to reach
    all six; only when nobody can lend does the column give its marks up, to
    the highest column that is already viable so the paper does not slide down
    the taxonomy.

    Runs from CREATING down to KNOWLEDGE, so a column topped up by a borrow is
    never revisited and the sweep terminates. KNOWLEDGE needs one mark and so
    is never deficient. The element's total is unchanged either way, which is
    what keeps the table of specifications adding up along its rows.
    """
    need = [marks_per_response(level) for level in BLOOM_LEVELS]
    spill = 0
    for i in reversed(range(len(BLOOM_LEVELS))):
        while 0 < demand[i] < need[i]:
            lenders = [j for j in range(len(BLOOM_LEVELS))
                       if j != i and demand[j] - 1 >= need[j]]
            if not lenders:
                break
            richest = max(lenders, key=lambda j: (demand[j], -j))
            demand[richest] -= 1
            demand[i] += 1
        if 0 < demand[i] < need[i]:
            spill += demand[i]
            demand[i] = 0
    if spill:
        home = next((i for i in reversed(range(len(BLOOM_LEVELS)))
                     if demand[i] >= need[i]), 0)
        demand[home] += spill
    return demand


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

    A cut that would leave a piece too small for the column it lands in is not
    made at all; the PC goes somewhere whole instead. See `_split_at`.
    """
    out: Dict[str, List[Allocation]] = {}
    for a in sorted(allocations,
                    key=lambda x: (-x.marks, _pc_sort_key(x.pc_number))):
        marks = a.marks
        exact = next((i for i, d in enumerate(demand)
                      if d == marks and d > 0 and marks >= _need(i)), None)
        cut = (_split_at(demand, marks)
               if exact is None and marks > SPLIT_ITEM_THRESHOLD else None)
        if exact is not None:
            demand[exact] -= marks
            pieces = [(exact, marks)]
        elif cut is not None:
            first, head, second, rest = cut
            demand[first] -= head
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

    THE CRITERION DECIDES. A performance criterion states its own cognitive
    demand in its own verb - "Tools and equipment are identified", "Faults are
    diagnosed", "Maintenance schedule is developed" - and that is the level
    the item is written at. This used to be settled the other way round: a
    target profile was integerised first and each PC was fitted into whatever
    cell was left, which is how a criterion about identifying tools ended up
    demanding that a candidate design something. Forcing a level onto a
    criterion that does not ask for it does not make a paper harder, it makes
    it unanswerable.

    A criterion whose verb is in no bank says nothing about its own level, and
    only those are placed by the profile - across the marks they carry, not
    across the paper, so a level the profile wants is reached only if a free
    criterion can honestly carry it.

    Nothing here is expected to span the six levels. A CAT assesses the
    criteria it assesses, and if all of them are identification criteria then
    it is an identification paper. `total_marks` is kept in the signature
    because callers pass it, and is used only to size the profile for the free
    criteria.

    Returned in the order given, and the marks still total what came in.
    """
    if not allocations:
        return []

    spoken: List[Allocation] = []
    silent: List[Allocation] = []
    for a in allocations:
        (spoken if natural_level(a.pc_text) else silent).append(a)

    placed: Dict[int, List[Allocation]] = {}
    for a in spoken:
        placed[id(a)] = _sized(a, natural_level(a.pc_text))
    if silent:
        for a, entries in _by_profile(silent, knqf_level).items():
            placed[a] = entries
    return [entry for a in allocations for entry in placed[id(a)]]


def _sized(a: Allocation, bloom: str) -> List[Allocation]:
    """One criterion as however many questions its marks can honestly fill.

    A criterion's level is settled by its own verb, so splitting it can no
    longer be about reaching a different level - both halves sit at the same
    one. It is about size. A KNOWLEDGE criterion holding twelve marks buys one
    mark a response, and "List TWELVE ICT security threats" is a list-writing
    exercise rather than an assessment item; published papers ask for three to
    five. So it becomes two questions of six marks, or three of four, until no
    question asks for more than MAX_RESPONSES_PER_ITEM.

    The marks are split by the same largest-remainder rule as everything else,
    so the criterion's total is untouched and the paper still adds up.
    """
    per_response = marks_per_response(bloom)
    responses = a.marks / per_response if per_response else a.marks
    pieces = max(1, math.ceil(responses / MAX_RESPONSES_PER_ITEM))
    if pieces == 1:
        return [_with_bloom(a, bloom, a.marks)]
    even = [Fraction(a.marks, pieces)] * pieces
    shares = _largest_remainder(even, a.marks, list(range(pieces)))
    return [_with_bloom(a, bloom, share) for share in shares if share > 0]


def _with_bloom(a: Allocation, bloom: str, marks: int) -> Allocation:
    return Allocation(element_number=a.element_number,
                      element_title=a.element_title, pc_number=a.pc_number,
                      pc_text=a.pc_text, weight=a.weight, marks=marks,
                      bloom=bloom)


def _by_profile(allocations: List[Allocation],
                knqf_level: str) -> Dict[int, List[Allocation]]:
    """Place the criteria that do not name a level, using the KNQF profile.

    The old whole-paper machinery, now working only on the marks these
    criteria carry. Both margins of the grid are still fixed before a PC is
    placed - rows are each element's marks, columns the profile integerised by
    largest remainder - so the table of specifications still adds up along its
    rows and down its columns.
    """
    carried = sum(a.marks for a in allocations)
    columns = _column_targets(knqf_level, carried)

    order: List[str] = []
    rows: Dict[str, int] = {}
    for a in allocations:
        if a.element_number not in rows:
            order.append(a.element_number)
            rows[a.element_number] = 0
        rows[a.element_number] += a.marks

    grid = _fit_grid([rows[e] for e in order], columns)

    out: Dict[int, List[Allocation]] = {}
    for index, element in enumerate(order):
        demand = _respect_minimums(
            _largest_remainder(grid[index], rows[element],
                               list(range(len(BLOOM_LEVELS)))))
        mine = [a for a in allocations if a.element_number == element]
        for a, entries in zip(mine, _place_grouped(mine, demand)):
            out[id(a)] = entries
    return out


def _place_grouped(allocations: List[Allocation],
                   demand: List[int]) -> List[List[Allocation]]:
    """`_place`'s result, kept grouped by the PC each entry came from."""
    flat = _place(allocations, demand)
    grouped: Dict[str, List[Allocation]] = {}
    for entry in flat:
        grouped.setdefault(entry.pc_number, []).append(entry)
    return [grouped.get(a.pc_number, []) for a in allocations]
