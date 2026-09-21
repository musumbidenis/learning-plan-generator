"""Module B1 - the pasted PC weighting table -> UnitWeighting (NO AI).

A CDACC unit's weighting table lives in the curriculum's assessment annex as a
ruled Word table, four columns wide:

    Element | Performance Criteria | Theory | Practical

The user copies it out of Word or out of a PDF viewer and pastes it into the
app, so by the time we see it the ruling is gone and only whitespace is left to
say where one column ended and the next began. What arrives is one of three
shapes, and which one you get depends on the source, not on the user:

    tab-separated       Word -> a text box: every cell boundary survives
    pipe-separated      Word -> Markdown, or a user tidying the paste by hand
    multi-space         PDF -> anything: the columns become runs of spaces

Nothing here is coordinate-based (unlike `os_parser`), because a paste has no
coordinates. Every decision is made on the SHAPE of a row instead:

    an element row      opens with '1.' and carries a title, no weights
    a PC row            opens with '1.1' and ENDS with two integers
    'Sub Total 18 27'   a checksum for the element just listed
    'GRAND TOTAL 36 54' a checksum for the whole unit

Shape-based reading is what lets one parser take all three separator styles:
the separators only ever affect how the cells are glued back into a line.

WHY THE CHECKSUMS ARE KEPT RATHER THAN RECOMPUTED: a PDF paste is lossy. A row
whose PC text wrapped over three lines can lose its weights, and a table that
crossed a page break can lose a whole PC. Both failures are silent - what you
get back still looks like a table - and everything downstream (every mark on
every CAT, via `assessment_allocation`) is computed from these numbers. The
sub-total and grand-total rows are the document's own arithmetic, so comparing
them against the PC rows we read is the one cheap check that catches a row that
never arrived. A mismatch is blocking for that reason: a missing PC silently
shifts marks onto the PCs that did arrive.

Nothing in here raises. Malformed input is the normal case, not the exception,
so `parse` returns whatever it understood plus `Problem`s describing the rest -
the user needs to see the half-read table to know which row to fix.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from assessment_config import MIN_MARKS_PER_PC
from assessment_models import (PRACTICAL, THEORY, Problem, UnitWeighting,
                               WeightedElement, WeightedPC)
from pdf_utils import RE_LEVEL, clean_text, find_isced_code, find_tvet_code

# --------------------------------------------------------------------------- #
# Row shapes
# --------------------------------------------------------------------------- #
# '1. Prepare for tour guiding'. The lookahead is what keeps an element apart
# from a PC: '1.1' must not read as element 1 titled '1 Tour itinerary ...'.
_RE_ELEMENT = re.compile(r"^(\d+)\.(?!\d)\s*(.*)$")

# The same row with no dot after the number. In a Word table the element
# number sits in its OWN CELL, so once the cells are glued back into a line it
# reads '1<TAB>Manage tourist arrival and departures' - which is the commonest
# paste of all, and matched nothing. The title must open with a letter, so a
# wrapped line of PC text that happens to begin with a figure ('24 hours after
# arrival...') cannot be read as element 24.
_RE_ELEMENT_BARE = re.compile(r"^(\d+)[\s.)]+([A-Za-z].*)$")

# '1.1 Tour itinerary is obtained' and the trailing-dot variant '1.1.Tour ...'
# that `os_parser` already tolerates - the same documents produce both.
_RE_PC = re.compile(r"^(\d+\.\d+)(?!\.?\d)\.?\s*(.*)$")

# The two weights, at the very end of the row. Requiring BOTH integers is what
# stops a PC whose text ends in a number ('... within 24 hours') from donating
# that number to the theory column: 'hours 4 4' matches, 'within 24 hours'
# does not.
_RE_TRAILING_WEIGHTS = re.compile(r"(\d+)\s+(\d+)\s*$")

_RE_SUB_TOTAL = re.compile(r"\bsub\s*[-–]?\s*totals?\b", re.I)
# 'GRAND TOTAL' is the CDACC wording, but a table that was retyped by a college
# often just says 'TOTAL' on its last row, so accept a row that opens with it.
_RE_GRAND_TOTAL = re.compile(r"\bgrand\s*totals?\b|^\s*totals?\b", re.I)

# The column-header row, in any of the wordings seen: it must never be read as
# an element title or appended to the PC above it.
_RE_HEADER_ROW = re.compile(
    r"performance\s+criteri|^\s*elements?\b.*\btheory\b"
    r"|\btheory\b.*\bpractical\b|\bweight(ing)?s?\b.*\bmarks?\b", re.I)

# 'THEORY : PRACTICAL RATIO 2:3', or 'Ratio 2:3' alone.
_RE_RATIO_PAIR = re.compile(r"(\d+)\s*:\s*(\d+)")

_RE_ONLY_DIGITS = re.compile(r"^[\d\s.]+$")
_RE_UNIT_TITLE_LABEL = re.compile(r"^\s*(unit\s+)?(title|name)\s*[:\-]\s*", re.I)
_RE_CODE_LABEL_LINE = re.compile(r"^\s*(isced\s+)?(unit\s+)?code\s*[:\-]?\s*$", re.I)


def _cells(line: str) -> List[str]:
    """One pasted line, cut back into the cells it had in the source table.

    The three separator styles are tried hardest-evidence first. A tab or a
    pipe is unambiguous - nothing inside a PC's text produces one. A run of two
    or more spaces is weaker evidence but is all a PDF paste leaves behind, and
    a single space is not evidence at all, so a PDF paste that lost its column
    runs entirely comes back as one cell and is read from its shape instead.
    """
    text = (line or "").replace(" ", " ").rstrip()
    if "\t" in text:
        parts = text.split("\t")
    elif "|" in text:
        parts = text.split("|")
    else:
        parts = re.split(r"\s{2,}", text.strip())
    return [clean_text(p) for p in parts if clean_text(p)]


def _flatten(cells: List[str]) -> str:
    return clean_text(" ".join(cells))


def _weights_from(cells: List[str], flat: str) -> Optional[Tuple[int, int]]:
    """The (theory, practical) pair a row ends with, or None.

    The cells are trusted first: when the separators survived, the last two
    cells ARE the two weight columns and no amount of numeric text in the PC
    can confuse them. Only a paste that lost its separators falls back to
    matching the end of the flattened line.
    """
    if len(cells) >= 3:
        last, second_last = cells[-1], cells[-2]
        if last.isdigit() and second_last.isdigit():
            return int(second_last), int(last)
    match = _RE_TRAILING_WEIGHTS.search(flat)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _strip_weights(flat: str, weights: Tuple[int, int]) -> str:
    """`flat` with its trailing weight pair removed, whatever the spacing."""
    theory, practical = weights
    pattern = re.compile(r"\s*\|?\s*" + str(theory) + r"\s+" + str(practical)
                         + r"\s*\|?\s*$")
    return clean_text(pattern.sub("", flat))


def _reduce(theory: int, practical: int) -> Optional[Tuple[int, int]]:
    """A theory:practical pair in lowest terms - 40:60 -> (2, 3)."""
    if theory <= 0 or practical <= 0:
        return None
    divisor = math.gcd(theory, practical)
    return theory // divisor, practical // divisor


def _element_prefix(pc_number: str) -> str:
    return (pc_number or "").split(".")[0]


def _sub_total_of(element: WeightedElement, kind: str) -> int:
    """An element's contribution to the grand total: its rows, added up.

    Not the stated figure. Real CDACC tables are typed by hand and their
    sub-totals drift from their rows by a mark - one document had element 1
    stating 29 practical against 30 in its rows, element 3 stating 26 theory
    against 27, and a grand total of 65 against 66. Preferring the stated
    figure there would carry the typo into every mark the CAT allocates.
    """
    return element.total(kind)


# --------------------------------------------------------------------------- #
# Reading the table
# --------------------------------------------------------------------------- #
class _Reader:
    """The line-by-line walk, kept in one object because a row has state.

    The state is the open PC. A row is not a line: long PC text wraps in the
    source document, and a Word table cell that wrapped over three lines pastes
    as three lines, so a PC stays open until the next PC, element or total row
    closes it and everything in between belongs to it.
    """

    def __init__(self) -> None:
        self.weighting = UnitWeighting()
        self.problems: List[Problem] = []
        self.preamble: List[str] = []
        self.stated_ratio: Optional[Tuple[int, int]] = None
        self.element: Optional[WeightedElement] = None
        self.pc: Optional[WeightedPC] = None
        self.pc_has_weights = False

    # -- problems ---------------------------------------------------------- #
    def problem(self, message: str, where: str = "", blocking: bool = True):
        self.problems.append(Problem(message=message, blocking=blocking,
                                     where=where))

    # -- rows -------------------------------------------------------------- #
    def read(self, text: str) -> None:
        for line in (text or "").splitlines():
            cells = _cells(line)
            if not cells:
                continue
            flat = _flatten(cells)
            if self._ratio_line(flat):
                continue
            # Sub-total before grand total: a retyped table sometimes labels an
            # element's checksum plain 'Total', and reading that as the unit's
            # grand total would throw the ratio out by a whole element.
            if _RE_SUB_TOTAL.search(flat):
                self._sub_total(cells, flat)
                continue
            if _RE_GRAND_TOTAL.search(flat):
                self._grand_total(cells, flat)
                continue
            # Shape before wording: a numbered row is a PC or an element even
            # when its text happens to use the words the header row uses.
            pc_match = _RE_PC.match(flat)
            if pc_match:
                self._pc_row(pc_match.group(1), pc_match.group(2), cells, flat)
                continue
            element_match = _RE_ELEMENT.match(flat) or self._bare_element(flat)
            if element_match:
                self._element_row(element_match.group(1),
                                  element_match.group(2))
                continue
            if _RE_HEADER_ROW.search(flat):
                self._close_pc()
                continue
            self._continuation(cells, flat)
        self._close_pc()
        self._finish()

    def _ratio_line(self, flat: str) -> bool:
        """True when this line states the unit's theory:practical ratio.

        A bare 'n:m' is not enough - PC text quotes times and scales - so the
        line must also name the ratio or both of its two sides.
        """
        low = flat.lower()
        if "ratio" not in low and not ("theory" in low and "practical" in low):
            return False
        match = _RE_RATIO_PAIR.search(flat)
        if not match:
            return False
        self.stated_ratio = _reduce(int(match.group(1)), int(match.group(2)))
        return True

    def _bare_element(self, flat: str):
        """An element row written without the dot, or None.

        Undotted is ambiguous with a wrapped line of PC text, so it is only
        believed where a new element could actually begin: nothing is open, or
        the number continues the sequence. While a PC is open and the number
        is not the next one, the line is that PC's text carrying on.
        """
        match = _RE_ELEMENT_BARE.match(flat)
        if not match:
            return None
        if self.pc is None:
            return match
        seen = [el.number for el in self.weighting.elements]
        try:
            return match if int(match.group(1)) == int(seen[-1]) + 1 else None
        except (IndexError, ValueError):
            return match

    def _element_row(self, number: str, title: str) -> None:
        self._close_pc()
        title = clean_text(title)
        # A paste that lost a line break runs the element into its first PC:
        # '1. Prepare for tour guiding 1.1 Tour itinerary is obtained 4 4'. The
        # split is only made when the embedded id belongs to THIS element, so a
        # title that merely contains a decimal is left alone.
        embedded = re.search(r"(?<![\d.])(" + re.escape(number) + r"\.\d+)\s",
                             title)
        tail = ""
        if embedded:
            tail = title[embedded.start():]
            title = clean_text(title[:embedded.start()])
        self.element = WeightedElement(number=number, title=title)
        self.weighting.elements.append(self.element)
        if tail:
            match = _RE_PC.match(tail)
            if match:
                self._pc_row(match.group(1), match.group(2), [tail], tail)

    def _pc_row(self, number: str, rest: str, cells: List[str],
                flat: str) -> None:
        self._close_pc()
        weights = _weights_from(cells, flat)
        text = clean_text(rest)
        if weights:
            text = _strip_weights(text, weights)
        if self.element is None:
            # Keep the PC rather than drop it: the user has to SEE the row that
            # lost its element heading to know what to repair. `validate` is
            # what refuses to go on.
            self.element = WeightedElement(number=_element_prefix(number))
            self.weighting.elements.append(self.element)
            self.problem(
                f"PC {number} appears before any element row; it was filed "
                f"under element {self.element.number}.", where=number)
        self.pc = WeightedPC(number=number,
                             element_number=self.element.number,
                             text=text,
                             theory_weight=weights[0] if weights else 0,
                             practical_weight=weights[1] if weights else 0)
        self.pc_has_weights = weights is not None
        self.element.pcs.append(self.pc)

    def _continuation(self, cells: List[str], flat: str) -> None:
        """A line that belongs to the PC above it - or to nothing at all.

        Before the first PC the table has not started, so the line is header
        matter (the unit title and its codes). A line of nothing but digits is
        page furniture: a page number from a PDF paste, which would otherwise
        be appended to a PC's text and then read as a weight.
        """
        if self.pc is None:
            self.preamble.append(flat)
            return
        weights = _weights_from(cells, flat)
        if weights is None and _RE_ONLY_DIGITS.match(flat):
            return
        if weights and not self.pc_has_weights:
            # The wrap carried the weights: the cell wrapped before its last
            # line, so the numbers arrive under the tail of the PC's text.
            self.pc.theory_weight, self.pc.practical_weight = weights
            self.pc_has_weights = True
            flat = _strip_weights(flat, weights)
        if flat:
            self.pc.text = clean_text(f"{self.pc.text} {flat}")

    def _sub_total(self, cells: List[str], flat: str) -> None:
        self._close_pc()
        weights = _weights_from(cells, flat)
        if self.element is None:
            self.problem("A sub-total row appears before any element.")
            return
        if weights is None:
            self.problem(
                f"Element {self.element.number}'s sub-total row carries no "
                f"pair of numbers, so it could not be checked: '{flat}'.",
                where=self.element.number)
            return
        self.element.stated_theory_total = weights[0]
        self.element.stated_practical_total = weights[1]

    def _grand_total(self, cells: List[str], flat: str) -> None:
        self._close_pc()
        weights = _weights_from(cells, flat)
        if weights is None:
            self.problem(f"The grand-total row carries no pair of numbers, so "
                         f"the unit's ratio could not be checked: '{flat}'.")
            return
        self.weighting.stated_grand_theory = weights[0]
        self.weighting.stated_grand_practical = weights[1]

    def _close_pc(self) -> None:
        if self.pc is not None and not self.pc_has_weights:
            self.problem(
                f"PC {self.pc.number} has no theory and practical weights; "
                f"the row probably wrapped and lost them.",
                where=self.pc.number)
        self.pc = None
        self.pc_has_weights = False

    # -- what the header said ---------------------------------------------- #
    def _finish(self) -> None:
        self._identity()
        self._ratio()

    def _identity(self) -> None:
        """Unit title, codes and level, from the lines above the table.

        Best effort on purpose: these only label the generated documents, and a
        user who pasted the table without its heading still has a usable
        weighting. Nothing downstream is blocked on them.
        """
        weighting = self.weighting
        joined = "\n".join(self.preamble)
        weighting.cdacc_code = find_tvet_code(joined)
        weighting.isced_code = find_isced_code(joined)
        level = RE_LEVEL.search(joined)
        if level:
            weighting.knqf_level = level.group(1)
        elif weighting.cdacc_code:
            # TO/OS/TTM/CR/01/6/MA - the KNQF level is the segment before the
            # trailing qualifier, which is how every CDACC code is built.
            parts = [p for p in weighting.cdacc_code.split("/") if p]
            if len(parts) >= 2 and parts[-2].isdigit():
                weighting.knqf_level = parts[-2]
        for line in self.preamble:
            candidate = clean_text(_RE_UNIT_TITLE_LABEL.sub("", line))
            if not candidate or _RE_CODE_LABEL_LINE.match(line):
                continue
            if find_tvet_code(candidate) or find_isced_code(candidate):
                continue
            if len(candidate.split()) < 2 or not re.search(r"[A-Za-z]{3}",
                                                           candidate):
                continue
            weighting.unit_title = candidate
            break

    def _ratio(self) -> None:
        """The unit's ratio, from the grand totals it stated.

        The totals win over a header that says '2:3' because the totals are
        what every mark is computed from; a heading is retyped by hand and a
        retyped heading is the thing most likely to be stale. A disagreement is
        still raised - one of the two is wrong and the user has to say which.
        """
        weighting = self.weighting
        derived = None
        if (weighting.stated_grand_theory is not None
                and weighting.stated_grand_practical is not None):
            derived = _reduce(weighting.stated_grand_theory,
                              weighting.stated_grand_practical)
        if derived is None:
            derived = _reduce(weighting.grand_total(THEORY),
                              weighting.grand_total(PRACTICAL))
        weighting.ratio = derived or self.stated_ratio
        if self.stated_ratio and derived and self.stated_ratio != derived:
            self.problem(
                f"The table is headed "
                f"{self.stated_ratio[0]}:{self.stated_ratio[1]} but its totals "
                f"give {derived[0]}:{derived[1]}.")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse(text: str) -> Tuple[UnitWeighting, List[Problem]]:
    """Read a pasted weighting table, and say everything that is wrong with it.

    The returned problems are the reading problems first - rows that could not
    be understood - and then all of `validate`'s arithmetic. They are returned
    together because a caller that parsed and forgot to validate would compute
    a whole CAT from a table with a missing PC and never know; there is no use
    for an unvalidated weighting anywhere in this module.

    Never raises. A paste is user input from a lossy source, so an unexpected
    failure is reported the same way a bad row is - as a blocking Problem on a
    half-read table - rather than as a traceback the user cannot act on.
    """
    reader = _Reader()
    try:
        reader.read(text)
    except Exception as exc:                            # pragma: no cover
        reader.problem(f"The table could not be read past the point it "
                       f"stopped: {exc}")
    # Warn about the checksums, THEN make the table agree with its own rows.
    # The order matters: after reconciling there is nothing left to warn about.
    problems = reader.problems + validate(reader.weighting)
    reconcile(reader.weighting)
    return reader.weighting, problems


def reconcile(weighting: UnitWeighting) -> None:
    """Replace every stated checksum with what the rows actually add up to.

    The rows are the document's real content; a sub-total is somebody's
    arithmetic about them, typed by hand, and where the two disagree it is the
    arithmetic that is wrong. `validate` has already said so - this is what
    makes the rest of the module agree with the warning, so the PC
    distribution table cannot print 29 in a column whose cells add to 30.

    Done once, when the table is parsed, and idempotent: running it again
    finds nothing left to correct.
    """
    for element in weighting.elements:
        if element.stated_theory_total is not None:
            element.stated_theory_total = element.total(THEORY)
        if element.stated_practical_total is not None:
            element.stated_practical_total = element.total(PRACTICAL)
    if weighting.stated_grand_theory is not None:
        weighting.stated_grand_theory = weighting.grand_total(THEORY)
    if weighting.stated_grand_practical is not None:
        weighting.stated_grand_practical = weighting.grand_total(PRACTICAL)
    derived = _reduce(weighting.grand_total(THEORY),
                      weighting.grand_total(PRACTICAL))
    if derived:
        weighting.ratio = derived


def validate(weighting: UnitWeighting) -> List[Problem]:
    """Every arithmetic and completeness check, against the parsed table alone.

    Blocking problems all share one cause: they mean a number downstream would
    be computed from a table that is not the document's table. Warnings mean
    the table is readable but will limit what can be assessed from it.

    The one check that cannot live here is the pasted heading's ratio against
    the derived one - `UnitWeighting` has no field for a heading, by design -
    so `parse` makes that comparison while it still has the text. What is
    checked here is that the stored ratio is the one the grand totals give.
    """
    problems: List[Problem] = []

    def fail(message: str, where: str = "", blocking: bool = True) -> None:
        problems.append(Problem(message=message, blocking=blocking,
                                where=where))

    if not weighting.elements:
        fail("No elements were found in the pasted table.")
        return problems

    seen: dict = {}
    for element in weighting.elements:
        if not element.pcs:
            fail(f"Element {element.number} has no performance criteria; a "
                 f"block of rows was lost in the paste.", where=element.number)
        for pc in element.pcs:
            # 'Belongs to an element' is not the same as 'sits in one': a PC
            # filed under the wrong element makes its sub-total unfixable, and
            # this catches the paste where an element heading never arrived.
            if not pc.element_number or pc.element_number != element.number:
                fail(f"PC {pc.number} is filed under element "
                     f"{element.number} but names element "
                     f"'{pc.element_number}'.", where=pc.number)
            elif _element_prefix(pc.number) != element.number:
                fail(f"PC {pc.number} does not belong to element "
                     f"{element.number}.", where=pc.number)
            if not (pc.text or "").strip():
                fail(f"PC {pc.number} has no text, so nothing can be written "
                     f"from it.", where=pc.number)
            if pc.number in seen:
                fail(f"PC {pc.number} appears twice.", where=pc.number)
            seen[pc.number] = True
            for kind in (THEORY, PRACTICAL):
                # Policy, not arithmetic: below the allocation floor a PC takes
                # no marks at all, so it can never appear on a paper of this
                # type. The document may really say so, which is why it warns
                # rather than blocks.
                if pc.weight_for(kind) < MIN_MARKS_PER_PC:
                    fail(f"PC {pc.number} carries no {kind} weight, so it can "
                         f"never appear on a {kind} assessment.",
                         where=pc.number, blocking=False)

        for kind, stated in ((THEORY, element.stated_theory_total),
                             (PRACTICAL, element.stated_practical_total)):
            if stated is None:
                continue
            actual = element.total(kind)
            if actual != stated:
                fail(f"Element {element.number} states a {kind} sub-total of "
                     f"{stated} but its {len(element.pcs)} performance "
                     f"criteria add up to {actual}. The {actual} is used.",
                     where=element.number, blocking=False)

    for kind, grand in ((THEORY, weighting.stated_grand_theory),
                        (PRACTICAL, weighting.stated_grand_practical)):
        if grand is None:
            continue
        # The sub-totals are what is summed, falling back to an element's own
        # PCs where the sub-total row itself was lost: the point of this check
        # is that a whole element cannot go missing between the two rows.
        parts = sum(_sub_total_of(el, kind) for el in weighting.elements)
        if parts != grand:
            fail(f"The {kind} sub-totals add up to {parts} but the grand "
                 f"total says {grand}. The {parts} is used.", blocking=False)

    if (weighting.stated_grand_theory is not None
            and weighting.stated_grand_practical is not None):
        derived = _reduce(weighting.stated_grand_theory,
                          weighting.stated_grand_practical)
        if derived and weighting.ratio and weighting.ratio != derived:
            fail(f"The unit's ratio is recorded as {weighting.ratio[0]}:"
                 f"{weighting.ratio[1]} but its grand totals give "
                 f"{derived[0]}:{derived[1]}.")

    return problems
