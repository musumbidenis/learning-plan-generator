"""The ten checks a generated assessment tool has to pass, and one repair pass.

The model writes prose under tight constraints and mostly holds to them; these
checks are for the times it does not. They fall into two kinds, and the
difference is the whole design of this file:

    NOT REPAIRABLE  the paper is wrong in a way no rewording can fix - a PC
                    that was never assessed, an item written to the wrong
                    marks, a total that does not reconcile, a level the
                    allocation could never reach, a checklist of the wrong
                    length. The fix is upstream (allocate again) or by hand.

    REPAIRABLE      the wording is wrong - the lead verb sits in another
                    level's bank, a stem gives away another item's key, a stem
                    telegraphs its own answer, an item of evaluation is its PC
                    restated, a selected-response format crept in. `repair`
                    sends those items back, alone, with what is wrong with
                    each.

`blocking` on the returned Problem follows that split: a repairable problem
does not block, because the repair pass clears it without the user; anything
not repairable does, because only a person can resolve it. What survives
MAX_REPAIR_PASSES is handed back for hand editing.

Similarity is plain token overlap - no new dependency, and nothing here is
subtle enough to need one. Two texts are compared on their content words with
the vocabulary they were both handed (the PC text, the allocation) subtracted,
so that sharing the topic of the question does not read as sharing its answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import runlog
from ai_client import (AIError, _chat_json, _emit_progress, _strict,
                       load_api_key, load_model_name, resolve_model)
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               ITEM_INDEPENDENCE_OVERLAP,
                               ITEM_VS_PC_SIMILARITY, MAX_CHECKLIST_ITEMS,
                               MAX_REPAIR_PASSES, MIN_CHECKLIST_ITEMS,
                               TEMPERATURE, VERB_BANK, level_of_verb)
from assessment_models import (BLOOM_LEVELS, AssessmentTool, ChecklistItem,
                               Item, MarkingPoint, Problem)

# The checks, by the name they report under.
PC_COVERAGE = "pc_coverage"
MARK_FIDELITY = "mark_fidelity"
TOTAL_RECONCILIATION = "total_reconciliation"
BLOOM_CONFORMANCE = "bloom_conformance"
BLOOM_COMPLETENESS = "bloom_completeness"
ITEM_INDEPENDENCE = "item_independence"
STEM_CLUE = "stem_clue"
PRACTICAL_ITEM_COUNT = "practical_item_count"
ITEM_NOT_PC = "item_not_pc"
FORMAT_COMPLIANCE = "format_compliance"

# The five a model can be asked to fix by rewriting the offending item. The
# other five are arithmetic or coverage: rewording cannot change them.
REPAIRABLE_CHECKS = frozenset({BLOOM_CONFORMANCE, ITEM_INDEPENDENCE, STEM_CLUE,
                               ITEM_NOT_PC, FORMAT_COMPLIANCE})


@dataclass(eq=False)
class AssessmentProblem(Problem):
    """A `Problem` that also says which check found it and whether a model can
    fix it.

    `assessment_models.Problem` carries message / blocking / where and is
    shared with the weighting parser, which has no notion of a repair pass.
    Rather than widen the shared contract for one consumer, this subclass adds
    the two fields the repair loop needs. It IS a Problem - anything reading
    `.message` or `.blocking` (the UI, the doc builder) is unaffected.
    """
    # Put back by hand: `Problem` is a plain dataclass, so it already set
    # __hash__ to None, and eq=False on this subclass only declines to
    # override it. The UI puts problems in sets to split the repairable ones
    # from the rest, and two findings are never "the same problem" anyway -
    # each names its own item.
    __hash__ = object.__hash__
    check: str = ""
    repairable: bool = False
    item_number: int = 0


def _problem(check: str, message: str, where: str = "",
             item_number: int = 0) -> AssessmentProblem:
    repairable = check in REPAIRABLE_CHECKS
    return AssessmentProblem(message=message, blocking=not repairable,
                             where=where, check=check, repairable=repairable,
                             item_number=item_number)


def repairable(problems: Iterable[Problem]) -> List[AssessmentProblem]:
    """Only the problems a rewrite can clear."""
    return [p for p in problems
            if isinstance(p, AssessmentProblem) and p.repairable]


def blocking(problems: Iterable[Problem]) -> List[Problem]:
    """Only the problems that must be settled before the documents are built."""
    return [p for p in problems if p.blocking]


# --------------------------------------------------------------------------- #
# Plain token overlap
# --------------------------------------------------------------------------- #
_RE_WORD = re.compile(r"[a-z0-9]+")

# Words that carry no subject matter. A stem and its marking key share these
# whatever they are about, so counting them would make every item look like a
# giveaway.
_STOPWORDS = frozenset("""
a an and are as at be been being by can could do does for from had has have
her his how i if in into is it its may might must not of on or our shall she
should so than that the their them then there these they this those to up upon
was were what when where which while who whom why will with would you your
candidate trainee assessor item question answer state list give name explain
describe identify outline define using use used work working task marks mark
one two three four five six seven eight nine ten
""".split())


def tokens(text: str) -> Set[str]:
    """The content words of a piece of text, lowercased and de-duplicated."""
    return {w for w in _RE_WORD.findall((text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def overlap(needle: Set[str], haystack: Set[str]) -> float:
    """How much of `needle` appears in `haystack`, 0.0 to 1.0.

    Asymmetric on purpose. The questions asked here are "how much of this
    marking key turns up in that stem" and "how much of this PC is restated in
    this item" - both about how much of the FIRST text the second contains, not
    about how alike the two are in general.
    """
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


# --------------------------------------------------------------------------- #
# The lead verb
# --------------------------------------------------------------------------- #
_RE_LEAD = re.compile(r"^\s*(?:\(?[a-z0-9]{1,3}[.)]\s*)*(.*)$", re.I | re.S)


def lead_verb(stem: str) -> str:
    """The verb a stem opens with, two words first ('carry out', 'break down').

    Any item numbering the model put in front of its own stem is stepped over,
    so '1. State FOUR...' is read the same as 'State FOUR...'.
    """
    body = _RE_LEAD.match(stem or "").group(1)
    words = _RE_WORD.findall(body.lower())
    if not words:
        return ""
    if len(words) > 1 and level_of_verb(f"{words[0]} {words[1]}"):
        return f"{words[0]} {words[1]}"
    return words[0]


# --------------------------------------------------------------------------- #
# Selected-response formats
# --------------------------------------------------------------------------- #
CONSTRUCTED_FORMATS = ("short_response", "extended_response")

_SELECTED_RESPONSE_SIGNS = (
    (re.compile(r"\b(?:which|choose|select|tick|circle)\b[^.?]{0,40}"
                r"\b(?:of\s+the\s+following|the\s+correct|one)\b", re.I),
     "offers the candidate options to choose between"),
    (re.compile(r"\btrue\s+or\s+false\b|\bstate\s+whether\b[^.?]{0,30}"
                r"\btrue\b", re.I),
     "is a true/false item"),
    (re.compile(r"\bmatch(?:ing)?\b[^.?]{0,30}\b(?:following|column|list|"
                r"items?|pairs?)\b", re.I),
     "is a matching item"),
    (re.compile(r"(?:^|\n)\s*[A-D][.)]\s+\S+[\s\S]*?\n\s*[B-D][.)]\s+\S+"),
     "lists lettered alternatives to pick from"),
    (re.compile(r"_{4,}"), "is a fill-in-the-blank item"),
)


def selected_response_fault(item: Item) -> str:
    """Why this item is a selected-response item, or '' when it is not."""
    if item.item_format and item.item_format not in CONSTRUCTED_FORMATS:
        return f"is formatted as '{item.item_format}'"
    for pattern, why in _SELECTED_RESPONSE_SIGNS:
        if pattern.search(item.stem or ""):
            return why
    return ""


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
def _key_tokens(item: Item) -> Set[str]:
    return tokens(" ".join(p.text for p in item.marking_scheme))


def _shared_vocabulary(tool: AssessmentTool) -> Set[str]:
    """The words every item was handed: the PC texts and element titles.

    Subtracted from both sides of an independence or clue comparison. Two items
    on the same unit inevitably share its vocabulary; that is the syllabus
    talking, not one item leaking into another.
    """
    return tokens(" ".join(f"{a.pc_text} {a.element_title}"
                           for a in tool.allocations))


def _check_pc_coverage(tool: AssessmentTool) -> List[AssessmentProblem]:
    """1. Every allocated PC is actually assessed somewhere."""
    wanted = []
    for a in tool.allocations:
        if a.pc_number not in wanted:
            wanted.append(a.pc_number)

    if tool.is_practical:
        covered = {n for c in (tool.observation_checklist
                               + tool.product_checklist) for n in c.pc_numbers}
        covered |= {n for q in tool.oral_questions for n in q.pc_numbers}
        where = "item of evaluation"
    else:
        covered = {i.pc_number for i in tool.items if i.pc_number}
        where = "item"

    return [_problem(PC_COVERAGE,
                     f"PC {n} is allocated marks but no {where} assesses it",
                     where=n)
            for n in wanted if n not in covered]


def _check_mark_fidelity(tool: AssessmentTool) -> List[AssessmentProblem]:
    """2. Per-item marks equal the allocation, exactly.

    Written: one item per allocation row, in order, at that row's marks and
    under that row's PC and level. Practical: the marks of the items tracing
    to a PC sum to that PC's allocated marks.
    """
    out: List[AssessmentProblem] = []
    if tool.is_practical:
        allocated = tool.marks_by_pc()
        given: Dict[str, int] = {}
        for c in tool.observation_checklist + tool.product_checklist:
            for n in c.pc_numbers:
                given[n] = given.get(n, 0) + c.marks
        for pc, marks in allocated.items():
            if given.get(pc, 0) != marks:
                out.append(_problem(
                    MARK_FIDELITY,
                    f"PC {pc} is allocated {marks} mark(s) but its items of "
                    f"evaluation carry {given.get(pc, 0)}", where=pc))
        return out

    allocs = tool.allocations
    if len(tool.items) != len(allocs):
        out.append(_problem(
            MARK_FIDELITY,
            f"the allocation asks for {len(allocs)} item(s) and "
            f"{len(tool.items)} came back; the items cannot be matched to "
            f"their marks"))
        return out

    for item, a in zip(tool.items, allocs):
        where = f"item {item.number}"
        if item.marks != a.marks:
            out.append(_problem(
                MARK_FIDELITY,
                f"item {item.number} is allocated {a.marks} mark(s) but "
                f"carries {item.marks}", where=where,
                item_number=item.number))
        if item.pc_number != a.pc_number:
            out.append(_problem(
                MARK_FIDELITY,
                f"item {item.number} should assess PC {a.pc_number} but is "
                f"tagged PC {item.pc_number or '(none)'}", where=where,
                item_number=item.number))
        if item.element_number != a.element_number:
            out.append(_problem(
                MARK_FIDELITY,
                f"item {item.number} should sit under element "
                f"{a.element_number} but is tagged "
                f"{item.element_number or '(none)'}", where=where,
                item_number=item.number))
        if a.bloom and item.bloom != a.bloom:
            out.append(_problem(
                MARK_FIDELITY,
                f"item {item.number} should be at {a.bloom} but is tagged "
                f"{item.bloom or '(none)'}", where=where,
                item_number=item.number))
    return out


def _check_totals(tool: AssessmentTool) -> List[AssessmentProblem]:
    """3. The paper totals to the CAT, and every marking scheme to its item."""
    out: List[AssessmentProblem] = []
    total = tool.cat.total_marks

    if tool.is_practical:
        given = sum(c.marks for c in (tool.observation_checklist
                                      + tool.product_checklist))
        if total and given != total:
            out.append(_problem(
                TOTAL_RECONCILIATION,
                f"the items of evaluation come to {given} mark(s); "
                f"{tool.cat.label} is out of {total}"))
        return out

    given = sum(i.marks for i in tool.items)
    if total and given != total:
        out.append(_problem(
            TOTAL_RECONCILIATION,
            f"the items come to {given} mark(s); {tool.cat.label} is out of "
            f"{total}"))
    for item in tool.items:
        points = sum(p.marks for p in item.marking_scheme)
        if points != item.marks:
            out.append(_problem(
                TOTAL_RECONCILIATION,
                f"item {item.number} is worth {item.marks} mark(s) but its "
                f"marking scheme awards {points}",
                where=f"item {item.number}", item_number=item.number))
    return out


def _check_bloom_conformance(tool: AssessmentTool) -> List[AssessmentProblem]:
    """4. The lead verb comes from the bank for the item's own level."""
    out: List[AssessmentProblem] = []
    for item in tool.items:
        bank = VERB_BANK.get(item.bloom)
        if bank is None:
            out.append(_problem(
                BLOOM_CONFORMANCE,
                f"item {item.number} is tagged '{item.bloom or '(none)'}', "
                f"which is not a Bloom level",
                where=f"item {item.number}", item_number=item.number))
            continue
        verb = lead_verb(item.stem)
        if verb in bank:
            continue
        found = level_of_verb(verb)
        sits = (f"'{verb}' is a {found} verb" if found
                else f"'{verb or '(none)'}' is in no verb bank")
        out.append(_problem(
            BLOOM_CONFORMANCE,
            f"item {item.number} is at {item.bloom} but opens with "
            f"{sits}; open it with one of: {', '.join(bank)}",
            where=f"item {item.number}", item_number=item.number))
    return out


def _check_bloom_completeness(tool: AssessmentTool) -> List[AssessmentProblem]:
    """5. All six levels appear on the paper.

    Not repairable, and not the model's fault when it fires: the levels are
    fixed in the allocation before a word is written, so a paper that misses
    one was allocated that way. The fix is to allocate again (splitting a
    well-funded PC across two levels is what `assessment_allocation` does for
    exactly this), not to reword an item.
    """
    if tool.is_practical:
        return []                      # a checklist item has no Bloom level
    present = {i.bloom for i in tool.items}
    missing = [lv for lv in BLOOM_LEVELS if lv not in present]
    if not missing:
        return []
    return [_problem(
        BLOOM_COMPLETENESS,
        f"the paper reaches no item at: {', '.join(missing)}; a CAT is "
        f"expected to span all six levels. Re-allocate rather than reword.")]


def _check_independence(tool: AssessmentTool) -> List[AssessmentProblem]:
    """6. No item's stem carries another item's marking key."""
    out: List[AssessmentProblem] = []
    shared = _shared_vocabulary(tool)
    keys = {i.number: _key_tokens(i) - shared for i in tool.items}
    stems = {i.number: tokens(i.stem) - shared for i in tool.items}
    for item in tool.items:
        for other in tool.items:
            if other.number == item.number:
                continue
            key = keys.get(other.number, set())
            if not key:
                continue
            score = overlap(key, stems.get(item.number, set()))
            if score >= ITEM_INDEPENDENCE_OVERLAP:
                out.append(_problem(
                    ITEM_INDEPENDENCE,
                    f"item {item.number} carries {score:.0%} of item "
                    f"{other.number}'s marking key in its stem, so a "
                    f"candidate can answer item {other.number} from it; "
                    f"the items must be answerable independently",
                    where=f"item {item.number}", item_number=item.number))
    return out


def _check_stem_clues(tool: AssessmentTool) -> List[AssessmentProblem]:
    """7. No stem telegraphs its own answer."""
    out: List[AssessmentProblem] = []
    shared = _shared_vocabulary(tool)
    for item in tool.items:
        key = _key_tokens(item) - shared
        if not key:
            continue
        score = overlap(key, tokens(item.stem) - shared)
        if score >= ITEM_INDEPENDENCE_OVERLAP:
            out.append(_problem(
                STEM_CLUE,
                f"item {item.number}'s stem already contains {score:.0%} of "
                f"its own marking key, so it gives away its answer; ask for "
                f"what it currently states",
                where=f"item {item.number}", item_number=item.number))
    return out


def _parent_items(tool: AssessmentTool) -> List[ChecklistItem]:
    """Parent items only - a sub-part is part of its parent, not an item."""
    return list(tool.observation_checklist) + list(tool.product_checklist)


def _check_practical_count(tool: AssessmentTool) -> List[AssessmentProblem]:
    """8. The two checklists together hold a workable number of items."""
    if not tool.is_practical:
        return []
    count = len(_parent_items(tool))
    if MIN_CHECKLIST_ITEMS <= count <= MAX_CHECKLIST_ITEMS:
        return []
    direction = "only " if count < MIN_CHECKLIST_ITEMS else ""
    return [_problem(
        PRACTICAL_ITEM_COUNT,
        f"the observation and product checklists hold {direction}{count} "
        f"parent item(s) of evaluation (sub-parts do not count); "
        f"{MIN_CHECKLIST_ITEMS} to {MAX_CHECKLIST_ITEMS} are expected")]


def _check_item_not_pc(tool: AssessmentTool) -> List[AssessmentProblem]:
    """9. An item of evaluation is not its performance criterion restated."""
    if not tool.is_practical:
        return []
    pc_text = {a.pc_number: a.pc_text for a in tool.allocations}
    out: List[AssessmentProblem] = []
    for c in _parent_items(tool):
        item_tokens = tokens(c.text)
        for pc in c.pc_numbers:
            source = tokens(pc_text.get(pc, ""))
            if not source:
                continue
            score = overlap(source, item_tokens)
            if score >= ITEM_VS_PC_SIMILARITY:
                out.append(_problem(
                    ITEM_NOT_PC,
                    f"item of evaluation {c.number} repeats {score:.0%} of "
                    f"PC {pc} ('{pc_text.get(pc, '')}'); an item of "
                    f"evaluation says what the assessor observes or inspects "
                    f"in THIS task, not what the standard requires in general",
                    where=f"item of evaluation {c.number}",
                    item_number=c.number))
    return out


def _check_format(tool: AssessmentTool) -> List[AssessmentProblem]:
    """10. No selected-response item where only constructed response is used."""
    if str(tool.knqf_level).strip() not in CONSTRUCTED_RESPONSE_ONLY_LEVELS:
        return []
    out: List[AssessmentProblem] = []
    for item in tool.items:
        fault = selected_response_fault(item)
        if fault:
            out.append(_problem(
                FORMAT_COMPLIANCE,
                f"item {item.number} {fault}; at KNQF level "
                f"{tool.knqf_level} every item is constructed response "
                f"({', '.join(CONSTRUCTED_FORMATS)})",
                where=f"item {item.number}", item_number=item.number))
    return out


_CHECKS = (_check_pc_coverage, _check_mark_fidelity, _check_totals,
           _check_bloom_conformance, _check_bloom_completeness,
           _check_independence, _check_stem_clues, _check_practical_count,
           _check_item_not_pc, _check_format)


def validate(tool: AssessmentTool) -> List[Problem]:
    """Everything wrong with a generated tool, in check order.

    Each Problem is an `AssessmentProblem`, which names the check and says
    whether a rewrite can clear it; `blocking` is True for exactly those that
    cannot be repaired. An empty list means the tool is ready to print.
    """
    out: List[Problem] = []
    for check in _CHECKS:
        try:
            out.extend(check(tool))
        except Exception as e:      # noqa: BLE001 - a broken check must not eat the rest
            runlog.error(f"Assessment: the {check.__name__} check failed: {e}")
    return out


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #
REPAIR_SYSTEM = """You are a senior TVET assessor in Kenya, correcting individual items on a Continuous Assessment Test that has already been written and approved in every other respect.

You are given the items that failed review, each with exactly what is wrong with it, and the stems of the items that PASSED. The passing items are frozen: they are not yours to change, and they are shown to you only so your corrections do not collide with them.

RULES
- Return one corrected item per failing item, under the same item number. Return nothing else.
- Change only what the fault names. The item's element, PC, Bloom level and marks are fixed and are echoed back unchanged.
- The marking scheme's points still sum to the item's stated marks. Never restate, adjust or total a mark.
- The corrected item must not repeat, hint at or give away the answer to any frozen item or to any other corrected item.
- The corrected stem must not give away its own answer.
- Every item is constructed response. Never a multiple-choice, true/false, matching or fill-in-the-blank item.
- Keep the item on its own performance criterion and its own scenario.

Return ONE JSON object and nothing else."""


PRACTICAL_REPAIR_SYSTEM = """You are a senior TVET assessor in Kenya, correcting individual items of evaluation on a practical assessment checklist that has already been written and approved in every other respect.

An item of evaluation is NOT a performance criterion. A performance criterion is general and comes from the occupational standard. An item of evaluation is specific to the task the candidate was set, and says what the assessor will actually watch them do or inspect in the finished work.

Rewrite ONLY the items you are given. For each one, keep the same observable behaviour or outcome, the same marks, and the same performance criteria - change the WORDING so it describes what is seen during THIS task rather than repeating the standard's general statement. Return every item you were given, by its number, and nothing else."""


def _practical_repair_schema() -> dict:
    return _strict({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "item": {"type": "string"},
                    },
                    "required": ["number", "item"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    })


def _checklist_items(tool: AssessmentTool) -> List[ChecklistItem]:
    return list(tool.observation_checklist) + list(tool.product_checklist)


def build_practical_repair_prompt(tool: AssessmentTool,
                                  faults: Dict[int, List[str]]) -> str:
    """The failing items of evaluation, with every other one frozen."""
    by_number = {c.number: c for c in _checklist_items(tool)}
    failing = []
    for number in sorted(faults):
        item = by_number.get(number)
        if item is None:
            continue
        pcs = ", ".join(
            f"{n} ({_pc_text(tool, n)})" for n in item.pc_numbers) or "(none)"
        failing.append(
            f"ITEM {item.number} - {item.marks} mark(s)\n"
            f"   traces to: {pcs}\n"
            f"   wording: {item.text}\n"
            f"   WHAT IS WRONG:\n"
            + "\n".join(f"     - {why}" for why in faults[number]))
    frozen = "\n".join(f"   {c.number}. {c.text}"
                        for c in _checklist_items(tool)
                        if c.number not in faults)
    return (f"TASK: {tool.task_brief.task if tool.task_brief else ''}\n\n"
            f"REWRITE THESE ITEMS OF EVALUATION:\n\n"
            + "\n\n".join(failing)
            + f"\n\nEVERY OTHER ITEM IS FIXED AND MUST NOT BE REPEATED OR "
              f"CONTRADICTED:\n{frozen or '   (none)'}\n")


def _pc_text(tool: AssessmentTool, number: str) -> str:
    return next((a.pc_text for a in tool.allocations
                 if a.pc_number == number), "")


def _apply_practical_repair(tool: AssessmentTool, payload,
                            faults: Dict[int, List[str]]) -> int:
    """Put corrected wordings back. Marks and PC tracing are never touched."""
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        runlog.warn("Assessment: the repair answer had no items in it")
        return 0
    by_number = {c.number: c for c in _checklist_items(tool)}
    landed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            number = int(row.get("number"))
        except (TypeError, ValueError):
            continue
        text = str(row.get("item") or "").strip()
        item = by_number.get(number)
        if not text or item is None or number not in faults:
            continue
        item.text = text
        landed += 1
    return landed


def _repair_practical(tool: AssessmentTool, faults: Dict[int, List[str]],
                      api_key: str, model: str, progress_cb) -> AssessmentTool:
    """The written path's loop, on items of evaluation."""
    if not _checklist_items(tool):
        return tool
    for attempt in range(1, MAX_REPAIR_PASSES + 1):
        _emit_progress(progress_cb,
                       f"Assessment: repairing {len(faults)} item(s) of "
                       f"evaluation, pass {attempt} of {MAX_REPAIR_PASSES}")
        try:
            payload = _chat_json(build_practical_repair_prompt(tool, faults),
                                 api_key, model, _practical_repair_schema(),
                                 "assessment_practical_repair",
                                 progress_cb=progress_cb,
                                 temperature=TEMPERATURE,
                                 system=PRACTICAL_REPAIR_SYSTEM)
        except AIError as e:
            runlog.error(f"Assessment: repair pass {attempt} failed, keeping "
                         f"the items as they are: {e}")
            return tool
        if not _apply_practical_repair(tool, payload, faults):
            return tool
        faults = _faults_by_item(validate(tool))
        if not faults:
            return tool
    return tool


def _repair_schema() -> dict:
    return _strict({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "stem": {"type": "string"},
                        "item_format": {"type": "string"},
                        "marking_scheme": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "marks": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
        },
    })


def _faults_by_item(problems: Sequence[Problem]) -> Dict[int, List[str]]:
    """{item number: [what is wrong]} for the repairable problems only."""
    out: Dict[int, List[str]] = {}
    for p in problems:
        if isinstance(p, AssessmentProblem) and p.repairable and p.item_number:
            out.setdefault(p.item_number, []).append(p.message)
    return out


def _item_by_number(tool: AssessmentTool, number: int) -> Optional[Item]:
    return next((i for i in tool.items if i.number == number), None)


def build_repair_prompt(tool: AssessmentTool,
                        faults: Dict[int, List[str]]) -> str:
    """The failing items and their faults, with every other item frozen."""
    failing = []
    for number in sorted(faults):
        item = _item_by_number(tool, number)
        if item is None:
            continue
        verbs = ", ".join(VERB_BANK.get(item.bloom, [])) or "(any)"
        scheme = "\n".join(f"     * {p.text} [{p.marks}]"
                           for p in item.marking_scheme)
        failing.append(
            f"ITEM {item.number} - element {item.element_number}, PC "
            f"{item.pc_number}, {item.bloom}, {item.marks} mark(s), "
            f"scenario '{item.scenario_id or 'none'}'\n"
            f"   allowed lead verbs: {verbs}\n"
            f"   stem: {item.stem}\n"
            f"   marking scheme:\n{scheme or '     * (none)'}\n"
            f"   WHAT IS WRONG:\n"
            + "\n".join(f"     - {f}" for f in faults[number]))

    frozen = "\n".join(
        f"ITEM {i.number} (frozen, PC {i.pc_number}): {i.stem}"
        for i in tool.items if i.number not in faults)

    scenarios = "\n".join(f"[{s.id}] {s.title}: {s.text}"
                          for s in tool.scenarios)

    return f"""UNIT: {tool.unit_title}
ASSESSMENT: {tool.cat.label} ({tool.cat.assessment_type})
KNQF LEVEL: {tool.knqf_level}

SCENARIOS (unchanged):
{scenarios or "(none)"}

ITEMS THAT PASSED - FROZEN, do not rewrite, do not duplicate, do not answer:
{frozen or "(none)"}

ITEMS TO CORRECT:
{chr(10).join(failing)}

Return the {len(failing)} corrected item(s) now, by number."""


def _apply_repair(tool: AssessmentTool, payload,
                  faults: Dict[int, List[str]]) -> int:
    """Put corrected stems and schemes back, in place. Returns how many landed.

    Only the items that failed are touched, and only their wording: an item
    the model returns that was not asked for is ignored, so a model that
    decides to improve a frozen item cannot.
    """
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        runlog.warn("Assessment: the repair answer had no items in it")
        return 0
    landed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            number = int(row.get("number"))
        except (TypeError, ValueError):
            continue
        if number not in faults:
            continue                       # frozen, or never asked for
        item = _item_by_number(tool, number)
        stem = row.get("stem")
        if item is None or not isinstance(stem, str) or not stem.strip():
            continue
        item.stem = stem.strip()
        fmt = row.get("item_format")
        if isinstance(fmt, str) and fmt.strip():
            item.item_format = fmt.strip()
        scheme = _marking_points(row.get("marking_scheme"))
        if scheme:
            item.marking_scheme = scheme
        landed += 1
    return landed


def _marking_points(raw) -> List[MarkingPoint]:
    out: List[MarkingPoint] = []
    for row in raw if isinstance(raw, list) else []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        marks = row.get("marks")
        out.append(MarkingPoint(text=text,
                                marks=marks if isinstance(marks, int) else 1))
    return out


def repair(tool: AssessmentTool, problems: Sequence[Problem], api_key: str = "",
           model: str = "", progress_cb=None) -> AssessmentTool:
    """Send back ONLY the failing items, with the rest of the paper frozen.

    Regenerating the whole paper is the obvious move and the wrong one: the
    model rewrites the items that were fine along with the ones that were not,
    and the new draft fails a different set of checks. Nothing converges - the
    failures just move around, and the user watches a paper they had almost
    accepted turn into a paper they have not read. So the failing items go back
    alone, each with the exact violation text, and every other item is shown
    as frozen context so the corrections do not collide with it.

    At most MAX_REPAIR_PASSES passes. Whatever is still wrong after that is
    left in place for the user to edit by hand - call `validate` again on the
    returned tool to see it.

    A practical tool is repaired the same way, on its items of evaluation. The
    one repairable fault it produces is an item that restates its performance
    criterion, which is the commonest thing wrong with a CDACC checklist and
    the whole reason the check exists - reporting it for hand editing would
    leave the tool failing the rule it was written to enforce. What the rewrite
    may not change is what the assessor is told to inspect, or the marks, so
    both are frozen and re-checked after the pass.
    """
    faults = _faults_by_item(problems)
    if not faults:
        return tool
    # The caller hands us what the UI had, which is usually nothing: the model
    # was settled inside `generate` and never came back out. Settling it again
    # here costs one lookup and is the difference between a repair pass and a
    # request to a model named ''.
    if not model:
        api_key = api_key or load_api_key()
        model = resolve_model(api_key, load_model_name(), progress_cb)
    if tool.is_practical:
        return _repair_practical(tool, faults, api_key, model, progress_cb)
    if not tool.items:
        return tool

    for attempt in range(1, MAX_REPAIR_PASSES + 1):
        _emit_progress(progress_cb,
                       f"Assessment: repairing {len(faults)} item(s), pass "
                       f"{attempt} of {MAX_REPAIR_PASSES}")
        try:
            payload = _chat_json(build_repair_prompt(tool, faults), api_key,
                                 model, _repair_schema(), "assessment_repair",
                                 progress_cb=progress_cb,
                                 temperature=TEMPERATURE, system=REPAIR_SYSTEM)
        except AIError as e:
            runlog.error(f"Assessment: repair pass {attempt} failed, keeping "
                         f"the items as they are: {e}")
            return tool

        landed = _apply_repair(tool, payload, faults)
        runlog.log(f"Assessment: repair pass {attempt} corrected {landed} "
                   f"item(s)")

        faults = _faults_by_item(validate(tool))
        if not faults:
            _emit_progress(progress_cb,
                           f"Assessment: every repairable fault cleared after "
                           f"pass {attempt}")
            return tool

    runlog.warn(f"Assessment: {len(faults)} item(s) still fail after "
                f"{MAX_REPAIR_PASSES} repair pass(es); they are left for hand "
                f"editing: " + ", ".join(str(n) for n in sorted(faults)))
    _emit_progress(progress_cb,
                   f"Assessment: {len(faults)} item(s) need editing by hand")
    return tool
