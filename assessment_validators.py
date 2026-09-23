"""The fifteen checks a generated tool has to pass, and one repair pass.

The model writes prose under tight constraints and mostly holds to them; these
checks are for the times it does not. They fall into two kinds, and the
difference is the whole design of this file:

    NOT REPAIRABLE  the paper is wrong in a way no rewording can fix - a PC
                    that was never assessed, an item written to the wrong
                    marks, a total that does not reconcile, an item
                    worth less than one answer at its own level, a checklist
                    of the wrong length.
                    The fix is upstream (allocate again) or by hand.

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

import assessment_content
import runlog
from ai_client import (AIError, _chat_json, _emit_progress, _strict,
                       load_api_key, load_model_name, resolve_model)
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               ITEM_INDEPENDENCE_OVERLAP,
                               ITEM_VS_PC_SIMILARITY, MAX_CHECKLIST_ITEMS,
                               MAX_REPAIR_PASSES, MIN_CHECKLIST_ITEMS,
                               REPAIR_TEMPERATURE, VERB_BANK,
                               MAX_VERB_WORDS, allowed_verbs,
                               level_of_verb, marks_per_response,
                               response_type, sector_for)
from assessment_models import (BLOOM_LEVELS, AssessmentTool, ChecklistItem,
                               Item, MarkingPoint, Problem)

# The checks, by the name they report under.
PC_COVERAGE = "pc_coverage"
MARK_FIDELITY = "mark_fidelity"
TOTAL_RECONCILIATION = "total_reconciliation"
BLOOM_CONFORMANCE = "bloom_conformance"
ITEM_INDEPENDENCE = "item_independence"
STEM_CLUE = "stem_clue"
PRACTICAL_ITEM_COUNT = "practical_item_count"
ITEM_NOT_PC = "item_not_pc"
FORMAT_COMPLIANCE = "format_compliance"
MARKS_FIT_VERB = "marks_fit_verb"
PLACEHOLDER_KEY = "placeholder_key"
UNFUNDED_ITEM = "unfunded_item"
COURSE_REFERENCE = "course_reference"
SYLLABUS_RECITATION = "syllabus_recitation"
DUPLICATE_ITEM = "duplicate_item"

# The ones a model can be asked to fix by rewriting the offending item. The
# rest are arithmetic, coverage or allocation: rewording cannot change them -
# an item too small for its own Bloom level needs more marks, not better
# words.
REPAIRABLE_CHECKS = frozenset({BLOOM_CONFORMANCE, ITEM_INDEPENDENCE, STEM_CLUE,
                               ITEM_NOT_PC, FORMAT_COMPLIANCE,
                               PLACEHOLDER_KEY, COURSE_REFERENCE,
                               SYLLABUS_RECITATION, DUPLICATE_ITEM})


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
             item_number: int = 0, blocks: Optional[bool] = None
             ) -> AssessmentProblem:
    """A finding. It blocks the documents unless a rewrite could clear it.

    `blocks` overrides that pairing for a finding where neither half holds:
    something no rewrite can clear but that is not worth withholding the
    documents over either, because the trainer can see it at a glance and
    generate again. Such a finding is reported, and the paper stays
    downloadable.
    """
    fixable = check in REPAIRABLE_CHECKS
    return AssessmentProblem(message=message,
                             blocking=(not fixable) if blocks is None
                             else blocks,
                             where=where, check=check, repairable=fixable,
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
# Sentence end, not decimal point: "1.1" and "Mr." must not split a stem.
_RE_SENTENCE = re.compile(r"(?<![A-Z0-9])[.!?]+\s+")


def _verb_at(text: str) -> str:
    """The verb `text` opens with, longest phrase first.

    'carry out' and 'break down' are two words; the sector house style adds
    longer ones - 'illustrate and label', 'describe with the aid of a sketch',
    and the Health Sciences opening 'what do you understand by', which is five
    and whose first word is not a verb at all. Taking the longest prefix that
    resolves is what lets those be recognised without a special case each.
    """
    words = _RE_WORD.findall((text or "").lower())
    if not words:
        return ""
    for size in range(min(MAX_VERB_WORDS, len(words)), 1, -1):
        phrase = " ".join(words[:size])
        if level_of_verb(phrase):
            return phrase
    return words[0]


def _openings(stem: str) -> List[str]:
    """Where the ask could begin, best candidate first.

    A CDACC question often sets a one-clause situation before it asks
    anything - "Mr. M has experienced conflict among workmates. Identify FOUR
    ways..." - and the verb that carries the Bloom level is the one opening
    the ASK, not the one opening the stem. Published papers also put a role
    before it inside the same sentence: "As the safety coordinator in your
    organisation, outline FOUR steps...".

    So three places are looked at and no more: the start of the stem, the
    start of its last sentence, and the point after that sentence's first
    comma. Hunting for a bank verb anywhere in the stem would pass any item
    that happened to contain one and gut the check.
    """
    body = _RE_LEAD.match(stem or "").group(1).strip()
    if not body:
        return []
    out = [body]
    sentences = [p for p in _RE_SENTENCE.split(body) if p.strip()]
    if len(sentences) > 1:
        out.append(sentences[-1].strip())
    head, comma, rest = out[-1].partition(",")
    if comma and rest.strip() and len(_RE_WORD.findall(head)) <= 12:
        out.append(rest.strip())
    return out


def lead_verb(stem: str) -> str:
    """The verb the stem opens with.

    Kept for the message a failing item is given, which names what the item
    actually starts with. `bloom_verb` is what decides whether it passes.
    """
    openings = _openings(stem)
    return _verb_at(openings[0]) if openings else ""


def bloom_verb(stem: str, bank: Sequence[str]) -> str:
    """The verb this stem offers for `bank`, or its opening verb if none fits.

    An item is conformant when the ask opens with a verb from its own level's
    bank, wherever the ask begins.
    """
    openings = _openings(stem)
    for opening in openings:
        verb = _verb_at(opening)
        if verb in bank:
            return verb
    # Nothing fits, so report the verb where the ask most likely begins rather
    # than the first word of the stem. On "The firm keeps records. Write FOUR
    # notes about them." the useful thing to tell the repair pass is that the
    # question opens with 'write', not with 'the'.
    return _verb_at(openings[-1]) if openings else ""


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
    """4. The lead verb comes from the bank for the item's own level.

    The bank is the generic one PLUS whatever this sector's published papers
    actually open with at that level - so an Electrical paper may say
    "Calculate the current" and a Mechanical one "Inscribe a circle" without
    being reported for it. See `assessment_config.allowed_verbs`.
    """
    sector = sector_for(tool.programme, tool.unit_title)
    out: List[AssessmentProblem] = []
    for item in tool.items:
        bank = allowed_verbs(item.bloom, sector) if item.bloom in VERB_BANK \
            else None
        if bank is None:
            out.append(_problem(
                BLOOM_CONFORMANCE,
                f"item {item.number} is tagged '{item.bloom or '(none)'}', "
                f"which is not a Bloom level",
                where=f"item {item.number}", item_number=item.number))
            continue
        verb = bloom_verb(item.stem, bank)
        if verb in bank:
            continue
        found = level_of_verb(verb)
        sits = (f"'{verb}' is a {found} verb" if found
                else f"'{verb or '(none)'}' is in no verb bank")
        out.append(_problem(
            BLOOM_CONFORMANCE,
            f"item {item.number} is at {item.bloom} but its ask opens with "
            f"{sits}; open the question with one of: {', '.join(bank)}",
            where=f"item {item.number}", item_number=item.number))
    return out


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


def _check_marks_fit_verb(tool: AssessmentTool) -> List[AssessmentProblem]:
    """11. An item is worth at least one answer at its own Bloom level.

    "Explain the fire triangle. (1 mark)" is not a hard question, it is an
    unanswerable one: the verb asks for a developed answer and the marks buy a
    single named thing. Published CDACC papers never do it - a one-mark item
    is always recall.

    Rewording cannot clear this, so it blocks. `assessment_allocation` is
    where it is prevented, and `check_items` says so at the distribution table
    before anything is generated; this catches an item the model wrote to a
    mark its allocation did not give it.
    """
    out: List[AssessmentProblem] = []
    for item in tool.items:
        need = marks_per_response(item.bloom)
        if item.bloom and 0 < item.marks < need:
            out.append(_problem(
                MARKS_FIT_VERB,
                f"item {item.number} is at {item.bloom} for {item.marks} "
                f"mark(s); a {item.bloom} question cannot be marked out of "
                f"less than {need}",
                where=f"item {item.number}", item_number=item.number))
    return out


# A marking point that is a slot rather than an answer. Seen live: "Threat
# classified as ___", "Way 1 ___", "Measure 3". Three or more underscores, or
# a bare noun-and-number with nothing else in it.
_RE_BLANK = re.compile(r"_{3,}|\.{5,}")
# The NUMBER is what makes it a slot. Without it these are ordinary one-word
# answers - "Threat" is a correct response to "list the terms defined in this
# unit", and flagging it told a trainer to rewrite a marking point that was
# right. Only "Threat 1", "Way 2", "Step 3" name a position rather than a
# thing.
_SLOT_NOUN = (r"point|step|way|item|measure|reason|factor|type|indicator|"
              r"answer|method|example|stage|principle|control|threat|"
              r"response|element|option|part|component|finding|scenario|"
              r"feature|benefit|cause|effect|advantage|function|"
              r"characteristic|tool")

_RE_SLOT = re.compile(
    r"^\s*(?:the\s+)?(?:" + _SLOT_NOUN + r")\s*\d+\s*[:\-.]?\s*\W*$", re.I)

# The ordinal form, which the first pattern missed entirely: "First tool",
# "Second malware type with propagation method and detection technique". A
# real answer does not begin by numbering itself. One or two words are allowed
# between the ordinal and the slot noun so "First malware type" is caught
# while "First aid kit is checked before the shift" is not - 'kit' is not a
# slot noun.
_RE_ORDINAL_SLOT = re.compile(
    r"^\s*(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth)\s+(?:\w+\s+){0,2}(?:" + _SLOT_NOUN + r")\b",
    re.I)

# The lettered form, and the one that survives having real words after it:
# "Finding A - Likelihood: Medium, Impact: High". The rating is genuine and
# the finding is not there at all, so the earlier patterns - which both need
# the slot to be the WHOLE point - let it through. An assessor holding this
# cannot tell whether a candidate's finding is the one that was wanted,
# because no finding was ever named.
#
# Only a bare letter or digit counts as the label. "Finding AB1234 on the
# payroll server" names something; "Finding A" names a position.
# The noun is matched without regard to case; the LABEL is not. A lower-case
# letter there is a word - "component c of the mixture" - and an upper-case
# one is a placeholder.
_RE_LABELLED_SLOT = re.compile(
    r"^\s*(?i:(?:the\s+)?(?:" + _SLOT_NOUN + r"))"
    r"\s+[A-Z0-9]\s*(?:[:\-–—]|$)")


def is_placeholder(text: str) -> bool:
    """A marking point that names a slot instead of stating the answer."""
    body = (text or "").strip()
    if not body:
        return False
    return bool(_RE_BLANK.search(body) or _RE_SLOT.match(body)
                or _RE_ORDINAL_SLOT.match(body)
                or _RE_LABELLED_SLOT.match(body))


def _check_placeholder_key(tool: AssessmentTool) -> List[AssessmentProblem]:
    """12. Every marking point states an answer, not a blank to fill in.

    The marking scheme is what an assessor holds while marking a script. A
    scheme of "Way 1 ___", "Way 2 ___" is not one, and a paper carrying it is
    unusable however good its questions read. Caught rather than trusted
    because the standing instructions forbid it and a live generation did it
    anyway, on five items out of six.
    """
    out: List[AssessmentProblem] = []
    for item in tool.items:
        blanks = [p.text for p in item.marking_scheme if is_placeholder(p.text)]
        if not blanks:
            continue
        out.append(_problem(
            PLACEHOLDER_KEY,
            f"item {item.number} has {len(blanks)} marking point(s) that name "
            f"a blank instead of stating the answer (e.g. \"{blanks[0]}\"); "
            f"write what the assessor should look for",
            where=f"item {item.number}", item_number=item.number))
    return out


# A stem that points at the course instead of asking about the trade:
# "covered in the unit", "as described in this module", "listed in the
# curriculum".
_RE_COURSE_REFERENCE = re.compile(
    r"\b(?:covered|taught|discussed|described|studied|learnt|learned|listed|"
    r"outlined|given|mentioned|stated|presented|introduced|defined|named|"
    r"explained|identified|shown|provided|supplied|included|encountered|"
    r"examined|addressed|highlighted|set out)\s+"
    r"(?:in|during)\s+(?:the|this|your)\s+"
    r"(?:unit|course|module|class|lesson|curriculum|content|syllabus|"
    r"training|programme|program|topic|teaching notes|notes|handout|"
    r"lecture|reference notes|learning guide|manual)"
    r"|\b(?:as\s+per|according\s+to)\s+(?:the|this|your)\s+"
    r"(?:unit|course|module|lesson|curriculum|syllabus|teaching notes|notes)\b"
    # The notes exist so the model has something to ask about. Naming them in
    # the question tells the candidate the answer is in a document, and it is
    # a document they were never given.
    r"|\bin\s+the\s+(?:teaching\s+)?notes\b", re.I)


def _check_course_reference(tool: AssessmentTool) -> List[AssessmentProblem]:
    """14. A question asks about the trade, never about the course.

    "List FOUR ICT security threats covered in the unit" is a different and
    much easier question than "List FOUR ICT security threats": it tells the
    candidate the answer is a list they were given, and it tells them the
    marking scheme is that list. A competent tradesperson who never sat this
    particular course could not answer it, which is the test of whether an
    item assesses competence or attendance.

    The standing instructions forbid it in as many words and the model does it
    anyway - four of six items in one live run, and in runs both with and
    without reference notes, so it is not something the notes introduced. It
    is a wording fault, which makes it repairable: deleting the phrase is
    almost always the whole fix.
    """
    out: List[AssessmentProblem] = []
    for item in tool.items:
        found = _RE_COURSE_REFERENCE.search(item.stem or "")
        if not found:
            continue
        out.append(_problem(
            COURSE_REFERENCE,
            f"item {item.number} asks about the course rather than the trade "
            f"(\"{found.group(0)}\"); a candidate who knows the work should "
            f"be able to answer without having sat this unit - ask the "
            f"question directly",
            where=f"item {item.number}", item_number=item.number))
    return out


KEY_POINT_ECHO_SHARE = 0.8
MIN_ECHOED_POINTS = 3


def _key_point_texts(tool: AssessmentTool) -> List[Set[str]]:
    """Every taught key point, as a set of its content words."""
    out: List[Set[str]] = []
    for block in tool.content:
        for topic in block.topics:
            for point in topic.key_points:
                words = tokens(point)
                if words:
                    out.append(words)
    return out


def _check_syllabus_recitation(tool: AssessmentTool) -> List[AssessmentProblem]:
    """15. The answer is not already printed in the key point it came from.

    This is the shallowness fault, stated precisely enough to catch.

    A curriculum key point often lists its own examples - "Types of malware:
    virus, worm, trojan, ransomware". Ask "List FOUR types of malware" and the
    marking scheme is that line, word for word. The candidate is being tested
    on whether they can read a heading, and the question could have been
    written by the syllabus itself. It is the single commonest way a generated
    paper comes out thin, and no other check sees it: the marks add up, the
    verb is right for the level, the item sits on its PC and nothing is
    placeholder.

    Caught by asking whether one key point contains nearly the whole marking
    scheme. Knowledge items are not the target - recall is a legitimate thing
    to assess, and "Name FOUR indicators of an insider threat" is recall the
    syllabus does not spell out. What is caught is recall of the SYLLABUS
    rather than of the trade.

    Repairable, because it is a question-writing fault: the same key point,
    asked about rather than read out, gives a better question at the same
    level and the same marks.
    """
    key_points = _key_point_texts(tool)
    if not key_points:
        return []                         # no curriculum read; nothing to echo
    out: List[AssessmentProblem] = []
    for item in tool.items:
        answers = [tokens(p.text) for p in item.marking_scheme]
        answers = [a for a in answers if a]
        if len(answers) < MIN_ECHOED_POINTS:
            continue
        for point in key_points:
            echoed = sum(1 for a in answers if overlap(a, point) >= 0.7)
            if echoed / len(answers) < KEY_POINT_ECHO_SHARE:
                continue
            listed = ", ".join(sorted(point)[:6])
            out.append(_problem(
                SYLLABUS_RECITATION,
                f"item {item.number} is answered by one line of the "
                f"curriculum: {echoed} of its {len(answers)} marking points "
                f"are already printed in the key point it came from "
                f"({listed}), so the question tests reading the syllabus "
                f"rather than knowing the work. Change WHAT IS ASKED, not "
                f"the wording - the new marking scheme must contain "
                f"different answers from those words. Ask what a worker has "
                f"to know ABOUT those things: how one is told from another, "
                f"how each arrives or is used, what is done about it",
                where=f"item {item.number}", item_number=item.number))
            break
    return out


DUPLICATE_ITEM_OVERLAP = 0.7


def _check_duplicate_items(tool: AssessmentTool) -> List[AssessmentProblem]:
    """16. No two items on the paper are the same question twice.

    A criterion too big for one question is split into several items at the
    same level, and each is written from the same material. Left to itself the
    model writes the same question each time: one live paper came back with

        1. Identify FOUR ICT security threats...
        2. List FOUR types of ICT security threats...
        3. Name FOUR common ICT security threats...

    and all three marking schemes were malware, social engineering,
    vulnerabilities, risk. A candidate answers once and is paid three times,
    and a paper claiming to cover twelve marks of the criterion covers four.

    Nothing else sees it. The marks add up, each item sits on its own row, the
    verbs are in the right bank, and `_check_independence` is asking the
    opposite question - whether a stem leaks ANOTHER item's answer, which is
    about items being too different to share, not too alike.

    Compared on the marking schemes rather than the stems, because the stems
    are reworded and the answers are not. Repairable: the second and later
    items need a different question on the same topic, which is what the
    notes are there to supply.
    """
    out: List[AssessmentProblem] = []
    keys = [(item, tokens(" ".join(p.text for p in item.marking_scheme)))
            for item in tool.items]
    for index, (item, mine) in enumerate(keys):
        if not mine:
            continue
        for earlier, theirs in keys[:index]:
            if not theirs:
                continue
            if (overlap(mine, theirs) >= DUPLICATE_ITEM_OVERLAP
                    and overlap(theirs, mine) >= DUPLICATE_ITEM_OVERLAP):
                out.append(_problem(
                    DUPLICATE_ITEM,
                    f"item {item.number} asks the same question as item "
                    f"{earlier.number} - the two marking schemes are the same "
                    f"answer in different words, so the candidate is paid "
                    f"twice for one piece of knowledge. Ask something "
                    f"different on the same topic",
                    where=f"item {item.number}", item_number=item.number))
                break
    return out


def _check_unfunded_items(tool: AssessmentTool) -> List[AssessmentProblem]:
    """13. Every item of evaluation carries at least one mark.

    Seen live, and invisible to every other check: a practical tool came back
    with ten items where a product-checklist row - "Submitted vulnerability
    assessment report containing the scanned findings" - carried nought. The
    totals still reconciled, because that PC's nine marks were already spread
    across two other rows, so nothing else noticed. What prints is an
    assessor's checklist with a row worth no marks: either the assessor scores
    it and the paper does not add up, or they skip it and the candidate did
    that work for nothing.

    Not repairable - the repair pass freezes marks, and it is not a wording
    fault - but it does not block either. It is one visible row in a document
    the trainer is about to read, and they can give it marks or drop it.
    """
    if not tool.is_practical:
        return []
    out: List[AssessmentProblem] = []
    for item in _parent_items(tool):
        if item.marks <= 0:
            out.append(_problem(
                UNFUNDED_ITEM,
                f"item of evaluation {item.number} carries no marks "
                f"(\"{item.text[:60]}\"); give it marks out of the "
                f"{', '.join(item.pc_numbers) or 'PC'} it traces to, or drop "
                f"it - an assessor cannot score a row worth nothing",
                where=f"item of evaluation {item.number}",
                item_number=item.number, blocks=False))
    return out


_CHECKS = (_check_pc_coverage, _check_mark_fidelity, _check_totals,
           _check_bloom_conformance, _check_independence, _check_stem_clues, _check_practical_count,
           _check_item_not_pc, _check_format, _check_marks_fit_verb,
           _check_placeholder_key, _check_unfunded_items,
           _check_course_reference, _check_syllabus_recitation,
           _check_duplicate_items)


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
- Every item is constructed response. Never a multiple-choice, true/false, matching or fill-in-the-blank item. Give response_type as exactly "short_response" or "extended_response".
- Keep the item on its own performance criterion. Build the question and its marking scheme out of the TEACHING NOTES, which are what the trainer teaches this unit from, and stay on a topic the notes cover. A PC is the competency link and the source of the marks; it is never content.
- Keep the stem concise - normally one sentence - and state exactly how many responses are wanted.
- Ask about the trade, never about the course. No "covered in the unit", "as described in this module", "listed in the curriculum". Delete the phrase and ask the question directly.
- Where a fault says two items ask the same question, the later one needs a DIFFERENT question on the same topic, with a marking scheme of different answers. Both items stay on their own rows, levels and marks.
- Never set the question a curriculum key point already answers. Where a fault says the marking scheme is printed in the key point, rewriting the stem is NOT the fix: the corrected item must have a DIFFERENT marking scheme, holding answers that do not appear in that line. Stay on the same key point, the same Bloom level and the same marks, and ask what a competent worker has to know ABOUT those things - how one is told from another, how each arrives or is used, what is done about it.
  Wrong fix: "Identify FOUR types of malware covered in the unit" becomes "Identify FOUR common types of malware". The answer is still virus, worm, trojan, ransomware.
  Right fix: "State FOUR ways malware reaches a workstation" - infected removable media, an attachment from a phishing message, a drive-by download, software from an unofficial source.
- Every marking point states the answer the assessor looks for. Never a blank, a numbered slot ("Way 1", "Step 2"), or "1 mark for each correct answer".
- Do not add a scenario to connect the item to any other item.

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
    return (f"TASK: {tool.task_brief.task if tool.task_brief else ''}"
            f"{_content_section(tool)}\n\n"
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
                                 temperature=REPAIR_TEMPERATURE,
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
                        "response_type": {"type": "string"},
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


def _content_section(tool: AssessmentTool) -> str:
    """The taught content, for a prompt that must not stray outside it."""
    rendered = assessment_content.render(tool.content)
    if not rendered:
        return ""
    return ("\n\nTEACHING NOTES - the corrected item is set from this and "
            "nothing else:\n" + rendered)


def build_repair_prompt(tool: AssessmentTool,
                        faults: Dict[int, List[str]]) -> str:
    """The failing items and their faults, with every other item frozen."""
    sector = sector_for(tool.programme, tool.unit_title)
    failing = []
    for number in sorted(faults):
        item = _item_by_number(tool, number)
        if item is None:
            continue
        verbs = ", ".join(allowed_verbs(item.bloom, sector)) or "(any)"
        scheme = "\n".join(f"     * {p.text} [{p.marks}]"
                           for p in item.marking_scheme)
        failing.append(
            f"ITEM {item.number} - element {item.element_number}, PC "
            f"{item.pc_number}, {item.bloom}, {item.marks} mark(s)\n"
            f"   allowed lead verbs: {verbs}\n"
            f"   stem: {item.stem}\n"
            f"   marking scheme:\n{scheme or '     * (none)'}\n"
            f"   WHAT IS WRONG:\n"
            + "\n".join(f"     - {f}" for f in faults[number]))

    frozen = "\n".join(
        f"ITEM {i.number} (frozen, PC {i.pc_number}): {i.stem}"
        for i in tool.items if i.number not in faults)

    # The taught content goes back with the item. Without it a repair is a
    # rewrite made from the model's own knowledge of the trade - which is the
    # exact fault the content was introduced to prevent, reintroduced at the
    # one point where nobody is looking at the wording again afterwards.
    content = _content_section(tool)

    return f"""UNIT: {tool.unit_title}
ASSESSMENT: {tool.cat.label} ({tool.cat.assessment_type})
KNQF LEVEL: {tool.knqf_level}{content}

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
        fmt = row.get("response_type")
        if isinstance(fmt, str) and fmt.strip():
            item.item_format = response_type(fmt)
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
                                 temperature=REPAIR_TEMPERATURE, system=REPAIR_SYSTEM)
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
