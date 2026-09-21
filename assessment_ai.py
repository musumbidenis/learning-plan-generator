"""The assessment module's single AI call: allocations in, written words out.

One request per assessment tool. NEVER a batch of several CATs in one call:
item independence - no item's stem may give away another item's marking key -
can only be checked, and only be written to, within one paper. A batched
response makes the model juggle several papers at once and makes the check
meaningless across the batch boundary, which is precisely where the leaks
appear.

The division of labour this file sits in is set out in `assessment_models`:
every mark is computed in plain arithmetic before the model is called. Here the
model receives fixed numbers and writes words around them. It never computes,
totals, merges or adjusts a mark.

What comes back is NOT silently corrected. If the model writes an item to a
different mark than its allocation, that item was written to a different budget
- its marking scheme will be wrong too - so the divergence is kept as returned
and left for `assessment_validators` to catch and for the repair pass to fix.
Quietly stamping the right number over it would hide a wrong question behind a
right total.

Standing instructions live in AS_WRITTEN_SYSTEM / AS_PRACTICAL_SYSTEM and the
per-call data in the user message, the same split as `ai_client.LP_SYSTEM` -
instructions and data are different things and separating them keeps both
readable (and lets a provider cache the fixed half).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, List, Optional

import runlog
from ai_client import (AIError, _chat_json, _emit_progress, _strict,
                       load_api_key, load_model_name, resolve_model)
import assessment_content
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               MAX_CHECKLIST_ITEMS, MIN_CHECKLIST_ITEMS,
                               TEMPERATURE, VERB_BANK)
from assessment_models import (AssessmentTool, ChecklistItem, Item,
                               MarkingPoint, OralQuestion, Scenario, TaskBrief)

# Where a raw generation is kept so a bad paper can be read back afterwards.
# The directory ignores itself (a '*' .gitignore written on creation), so no
# generation ever reaches a commit and the project's own .gitignore does not
# have to know this module exists.
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       ".assessment_raw")

# How many generations to keep. Enough to look back over a session's work,
# bounded so the directory cannot grow without end.
MAX_RAW_FILES = 60


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #
def written_schema() -> dict:
    """Scenarios and items, wrapped in an object.

    A strict schema's root must be an object, so both arrays travel under
    named keys. `scenario_id` is required-but-may-be-empty rather than
    optional: strict decoding requires every declared property, and an empty
    string is how an item says it stands on its own.
    """
    return _strict({
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "text": {"type": "string"},
                    },
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "element_number": {"type": "string"},
                        "pc_number": {"type": "string"},
                        "bloom": {"type": "string"},
                        "item_format": {"type": "string"},
                        "scenario_id": {"type": "string"},
                        "stem": {"type": "string"},
                        "marks": {"type": "integer"},
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


def _checklist_schema() -> dict:
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "pc_numbers": str_array,
                "marks": {"type": "integer"},
                "sub_parts": str_array,
            },
        },
    }


def practical_schema() -> dict:
    """The candidate's brief, the two checklists and any oral questions."""
    str_array = {"type": "array", "items": {"type": "string"}}
    return _strict({
        "type": "object",
        "properties": {
            "task_brief": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "conditions": {"type": "string"},
                    "tools_equipment_materials": str_array,
                    "safety_requirements": str_array,
                    "time_allowed": {"type": "string"},
                },
            },
            "observation_checklist": _checklist_schema(),
            "product_checklist": _checklist_schema(),
            "oral_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pc_numbers": str_array,
                        "question": {"type": "string"},
                        "response_indicators": str_array,
                    },
                },
            },
        },
    })


# --------------------------------------------------------------------------- #
# Standing instructions
# --------------------------------------------------------------------------- #
AS_WRITTEN_SYSTEM = """You are a senior TVET assessor in Kenya, writing the items of a Continuous Assessment Test (CAT) for one unit of competency under the TVET CDACC framework.

You are GIVEN the unit, the CAT, and a MARK ALLOCATION TABLE with one row per item. You do not choose what is assessed, at which Bloom level, or for how many marks. You write the words.

THE MARKS ARE NOT YOURS
Each allocation row states the marks for that item. Write the item to exactly that figure. Never invent, merge, split, round or adjust a mark, never move marks between items, and never state or compute a total anywhere - not for a section, not for the paper. The paper's totals are printed from the table you were given.

ONE ITEM PER ROW
Return exactly one item per allocation row, in the order the rows are given. Echo that row's element number, PC number, Bloom level and marks onto the item verbatim. Do not add an item, drop an item or reorder them.

WHAT YOU MAY ASSESS
You are given the CONTENT TAUGHT for this unit - the sub-topics and key points the curriculum sets out under each element, which is what the trainees actually sat through. Set every item from that content. Do not assess a topic that is not there. Do not invent equipment, standards, legislation, formulae, software, suppliers or terminology the content does not mention, and do not fall back on your own knowledge of the trade to fill a gap - where the content is thin on a performance criterion, ask a narrower question rather than a better-informed one.
Spread the paper across that content. Where an element lists many key points, draw the items from different ones rather than circling the same idea twice, and let the scenario touch enough of the taught ground for that to be possible.
Where no taught content is given for an element, and only then, work from the performance criterion alone and keep the item general.

FORMAT - CONSTRUCTED RESPONSE ONLY
Every item is answered in the candidate's own words. Use only:
- short_response: a few lines to a short paragraph.
- extended_response: a longer structured answer, or an answer to a case study.
Never write a multiple-choice item, a true/false item, a matching item, a fill-in-the-blank, or any item with options to choose between. Do not offer lettered or numbered alternatives (A, B, C / i, ii, iii) as candidate answers to pick from.

THE LEAD VERB
Each item opens with a verb from the ALLOWED VERBS list given for that row's Bloom level, and with no other verb. The verb is the first word of the stem. An item at KNOWLEDGE does not ask for analysis, and an item at ANALYSING does not ask for recall.

TELL THE CANDIDATE WHAT IS WANTED
Quantify the response expected. "State FOUR safety measures observed when working on a live circuit" - not "State the safety measures". The number asked for, the count of the marks and the number of marking points agree.

EVERY ITEM STANDS ALONE
The items may be answered in any order, independently.
- No stem may contain, define, hint at or paraphrase the answer to any other item.
- No stem may telegraph its own answer. If the stem names the thing it asks the candidate to name, rewrite it.
- Do not write "as described in question 3", "using your answer above", or any cross-reference.

MARKING SCHEME
Each item carries a marking scheme of discrete, mark-bearing points whose marks SUM to that item's stated marks. One point per mark is the norm; a point worth more says so in its own marks field. Each point is the substance an assessor looks for, written as a short sentence or phrase - not "1 mark for each correct answer".

THE PAPER OPENS WITH A SCENARIO
Write the scenario first. Every item then hangs off it: the candidate reads the situation once and answers every question about that situation. No item is asked directly, out of the air, as a bare context-free question.
A scenario is a realistic Kenyan workplace situation - a named workshop, garage, salon, farm, clinic, hotel, site, office or SME - and it gives the candidate the facts to work from: where they are, who they are in it, what has happened, and what they have been asked to do. Use plausible Kenyan places, roles, equipment and quantities, in Kenya Shillings where money appears. Make it rich enough that every item has something concrete to bite on, and draw its detail from the taught content so that the situation is one the trainees are equipped to reason about. It never contains the answer to any item.
Write ONE scenario. Write a second, or at most a third, only where the performance criteria genuinely belong to separate workplace situations that cannot honestly be folded into one. Never one scenario per item.
EVERY item carries the id of the scenario it belongs to. A scenario_id is never empty.

KEEP THE STEMS SHORT
The scenario carries the situation, so the stem does not restate it. A stem is normally one sentence: the lead verb, what is wanted, and how many. An item may add a small detail of its own - a reading just taken, a figure quoted, a further thing the supervisor now asks for - where that detail is what turns a general question into a question about this scenario. What a stem may not do is set the workshop, the customer and the job out all over again before it gets round to asking anything.
Weak, because it asks nothing of the scenario: "Explain FOUR causes of overheating in a petrol engine."
Strong: "Explain FOUR likely causes of the overheating Mutiso reported on the Probox." 

TERMINOLOGY (Kenya CBET) - MANDATORY
Use: trainee, candidate, assessor, unit of competency, performance criteria, competency, Continuous Assessment Test (CAT).
Never use: student, pupil, learner, teacher, lecturer, instructor, exam, quiz, or test as a noun.
Spelling: British / Kenyan English - organise, practise (as a verb), programme, labelled, capitalised, centred.

Return ONE JSON object and nothing else."""


AS_PRACTICAL_SYSTEM = """You are a senior TVET assessor in Kenya, writing the practical assessment for one unit of competency under the TVET CDACC framework.

You are GIVEN the unit, the CAT, and a MARK ALLOCATION TABLE stating the marks fixed for each performance criterion. You do not choose what is assessed or for how many marks. You write the task the candidate is set and the items of evaluation the assessor works from.

THE MARKS ARE NOT YOURS
Distribute each PC's stated marks across the items of evaluation that trace to it. The marks of the items tracing to a PC sum to exactly that PC's stated marks. Never alter a PC's total, never move marks between PCs, and never state a grand total anywhere.

WHAT THE TASK MAY REQUIRE
You are given the CONTENT TAUGHT for this unit - the sub-topics and key points the curriculum sets out under each element. The task you set, the tools you issue and the items of evaluation you write all come from that content. Do not require a technique, a machine, a material or a standard the trainees were never taught, and do not reach for your own knowledge of the trade to make the task look more professional: a task nobody was prepared for is not a harder assessment, it is an invalid one. Where no taught content is given for an element, work from the performance criterion alone.

THE CANDIDATE'S TASK BRIEF
One practical task, realistic for a Kenyan workplace, that can genuinely be performed in the time allowed with the tools listed. The brief gives:
- task: what the candidate is to produce or perform.
- conditions: where, with what support, working alone or in a team, what is provided and what is not.
- tools_equipment_materials: what the candidate is issued with.
- safety_requirements: the PPE and safe-working requirements that apply.
- time_allowed: the working time.
THE BRIEF NEVER REVEALS HOW IT IS MARKED. No items of evaluation, no checklist wording, no marks, no weighting, no count of steps, no hint of what the assessor will be watching for. A candidate reading the brief learns the job, not the mark scheme.

TWO CHECKLISTS
- observation_checklist: the PROCESS. What the assessor watches the candidate DO, in the order it is done, while the work is in progress. Actions, sequence, technique, safe and correct use of tools, housekeeping.
- product_checklist: the OUTCOME. What the assessor INSPECTS on the finished work once the candidate has stopped. Dimensions, finish, function, completeness, conformity to specification, records handed in.
An observable action belongs on the observation checklist; anything that can be judged from the finished work alone belongs on the product checklist.

AN ITEM OF EVALUATION IS NOT A PERFORMANCE CRITERION
A performance criterion is general and comes from the occupational standard. An item of evaluation is specific to the task you have set and says what the assessor will actually observe or inspect, in this task, with these tools, on this product. Never restate a PC, reword a PC, or shorten a PC into an item. If an item could be lifted unchanged into another task on the same unit, it is too general - make it concrete.
Weak: "Demonstrates correct use of hand tools."
Strong: "Grips the tenon saw at the knee-height mark and cuts on the waste side of the line without splintering the shoulder."

HOW MANY
Between %(min_items)d and %(max_items)d parent items across the two checklists COMBINED. Where one item has genuine stages, list them as sub_parts of that item; sub-parts do not count towards the total and carry no marks of their own.

ORAL QUESTIONS
Only where a performance criterion needs underpinning knowledge that watching the candidate cannot capture - why a step is done that way, what would be done if a condition changed, a judgement that stays inside the candidate's head. Each carries the response indicators the assessor accepts. Where observation and the product between them cover the PCs, return none.

TERMINOLOGY (Kenya CBET) - MANDATORY
Use: trainee, candidate, assessor, unit of competency, performance criteria, competency, Continuous Assessment Test (CAT).
Never use: student, pupil, learner, teacher, lecturer, instructor, exam, quiz, or test as a noun.
Spelling: British / Kenyan English - organise, practise (as a verb), programme, labelled, capitalised, centred.

Return ONE JSON object and nothing else.""" % {
    "min_items": MIN_CHECKLIST_ITEMS, "max_items": MAX_CHECKLIST_ITEMS}


# --------------------------------------------------------------------------- #
# The data half of the request
# --------------------------------------------------------------------------- #
def _content_block(tool: AssessmentTool) -> str:
    """The taught content, or nothing at all.

    Nothing at all is a real case, not a defect: a unit whose curriculum could
    not be read still gets a paper, written from its performance criteria the
    way the module worked before this existed. Both standing instructions say
    what to do when this section is absent, so the omission is handled rather
    than silently changing what the model thinks it is being asked for.
    """
    rendered = assessment_content.render(tool.content)
    if not rendered:
        return ""
    return ("\n\nCONTENT TAUGHT, from the curriculum - the body of knowledge "
            "this assessment draws on:\n" + rendered)


def _header(tool: AssessmentTool) -> str:
    cat = tool.cat
    return (f"UNIT: {tool.unit_title}\n"
            f"CDACC CODE: {tool.cdacc_code}\n"
            f"ISCED CODE: {tool.isced_code}\n"
            f"PROGRAMME: {tool.programme}\n"
            f"KNQF LEVEL: {tool.knqf_level}\n"
            f"ASSESSMENT: {cat.label} ({cat.assessment_type})\n"
            f"TOTAL MARKS: {cat.total_marks}\n"
            f"DURATION: {cat.duration_minutes} minutes")


def build_written_prompt(tool: AssessmentTool) -> str:
    """The per-paper half: the allocation table and nothing standing.

    The allowed verbs are printed per row rather than as one bank, because the
    model is choosing a verb for THAT row's level and a combined list invites
    it to pick from the wrong one.
    """
    rows = []
    for n, a in enumerate(tool.allocations, start=1):
        verbs = ", ".join(VERB_BANK.get(a.bloom, []))
        rows.append(
            f"{n}. element {a.element_number} ({a.element_title}) | "
            f"PC {a.pc_number}: {a.pc_text} | bloom: {a.bloom} | "
            f"marks: {a.marks} | allowed verbs: {verbs or '(any)'}")
    table = "\n".join(rows)
    level_note = ""
    if str(tool.knqf_level).strip() in CONSTRUCTED_RESPONSE_ONLY_LEVELS:
        level_note = ("\nThis is a KNQF level %s paper: every item is "
                      "constructed response. Selected-response formats are "
                      "not used at this level at all.\n" % tool.knqf_level)
    return f"""{_header(tool)}{_content_block(tool)}

MARK ALLOCATION TABLE - one item per row, in this order, at these marks:
{table}

ITEMS REQUIRED: {len(tool.allocations)}
{level_note}
Write the scenario first, then one item per row above, each one carrying that scenario's id. Return the JSON object now."""


def build_practical_prompt(tool: AssessmentTool) -> str:
    """The per-task half: what each PC is worth, and nothing standing."""
    by_pc: Dict[str, int] = tool.marks_by_pc()
    seen: List[str] = []
    rows = []
    for a in tool.allocations:
        if a.pc_number in seen:
            continue
        seen.append(a.pc_number)
        rows.append(f"- PC {a.pc_number} (element {a.element_number} "
                    f"{a.element_title}): {a.pc_text} | marks: "
                    f"{by_pc.get(a.pc_number, a.marks)}")
    methods = assessment_content.methods(tool.content)
    suggested = ("\n\nASSESSMENT METHODS THE CURRICULUM SUGGESTS FOR THIS "
                 "UNIT:\n" + "\n".join(f"- {m}" for m in methods)
                 if methods else "")
    return f"""{_header(tool)}{_content_block(tool)}{suggested}

PERFORMANCE CRITERIA ASSESSED, with the marks fixed for each:
{chr(10).join(rows)}

TIME ALLOWED: {tool.cat.duration_minutes} minutes

Write the candidate's task brief, then the observation checklist, then the product checklist, then any oral questions. Return the JSON object now."""


# --------------------------------------------------------------------------- #
# Keeping the generation
# --------------------------------------------------------------------------- #
def save_raw(tool: AssessmentTool, payload, suffix: str = "") -> str:
    """Write one generation to RAW_DIR, before anything is made of it.

    `_chat_json` does the JSON decode itself, so the earliest this module can
    take hold of a generation is as the decoded payload - which is dumped here
    verbatim, before any coercion into dataclasses. That is what makes a bad
    paper debuggable: the file says what the model actually returned, not what
    survived our reading of it.

    Never raises. A generation that cannot be filed is worth a log line, not a
    failed paper.
    """
    try:
        os.makedirs(RAW_DIR, exist_ok=True)
        marker = os.path.join(RAW_DIR, ".gitignore")
        if not os.path.exists(marker):
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("*\n")
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        name = "%s-%s-%s%s.json" % (stamp, tool.cat.cat_id,
                                    tool.cat.assessment_type,
                                    ("-" + suffix) if suffix else "")
        path = os.path.join(RAW_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
        _trim_raw()
        return path
    except OSError as e:                      # noqa: BLE001 - logging must not fail a run
        runlog.warn(f"Assessment: could not file the raw generation: {e}")
        return ""


def _trim_raw() -> None:
    try:
        files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))
        for stale in files[:-MAX_RAW_FILES]:
            os.remove(os.path.join(RAW_DIR, stale))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Reading the generation
# --------------------------------------------------------------------------- #
def _text(value) -> str:
    return str(value).strip() if isinstance(value, (str, int, float)) else ""


def _int(value, default: int = 0) -> int:
    """A mark, however the model spelled it, or the default.

    Never guesses at words ('four'): a mark that did not arrive as a number is
    a mark the model did not follow, and the validators are there to say so.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _str_list(value) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [_text(v) for v in value if _text(v)]
    return []


def _scenarios(raw) -> List[Scenario]:
    out: List[Scenario] = []
    seen = set()
    for i, row in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        sid = _text(row.get("id")) or f"S{i}"
        if sid in seen:
            continue
        seen.add(sid)
        out.append(Scenario(id=sid, title=_text(row.get("title")),
                            text=_text(row.get("text"))))
    return out


def _marking_scheme(raw) -> List[MarkingPoint]:
    out: List[MarkingPoint] = []
    for row in raw if isinstance(raw, list) else []:
        if isinstance(row, str):
            if row.strip():
                out.append(MarkingPoint(text=row.strip(), marks=1))
            continue
        if not isinstance(row, dict):
            continue
        text = _text(row.get("text"))
        if text:
            out.append(MarkingPoint(text=text, marks=_int(row.get("marks"), 1)))
    return out


def _items(raw, tool: AssessmentTool) -> List[Item]:
    """Items as returned, numbered by us.

    The element, PC, Bloom and marks are read from the MODEL's answer, not
    stamped from the allocation: an item written to the wrong budget is a
    wrong item, and overwriting the number would leave the wrong question
    sitting under a right total. `assessment_validators` compares the two.

    A row that names nothing at all falls back to the allocation in the same
    position, so an otherwise good paper is not lost to a dropped tag.
    """
    allocs = tool.allocations
    out: List[Item] = []
    for i, row in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(row, dict):
            runlog.warn(f"Assessment: item {i + 1} was not an object; skipped")
            continue
        fallback = allocs[i] if i < len(allocs) else None
        stem = _text(row.get("stem"))
        if not stem:
            runlog.warn(f"Assessment: item {i + 1} arrived with no stem")
        fmt = _text(row.get("item_format")) or "short_response"
        out.append(Item(
            number=len(out) + 1,
            element_number=(_text(row.get("element_number"))
                            or (fallback.element_number if fallback else "")),
            pc_number=(_text(row.get("pc_number"))
                       or (fallback.pc_number if fallback else "")),
            bloom=(_text(row.get("bloom")).lower()
                   or (fallback.bloom if fallback else "")),
            stem=stem,
            marks=_int(row.get("marks"), -1),
            marking_scheme=_marking_scheme(row.get("marking_scheme")),
            item_format=fmt,
            scenario_id=_text(row.get("scenario_id")),
        ))
    return out


def _checklist(raw, start: int) -> List[ChecklistItem]:
    out: List[ChecklistItem] = []
    for row in raw if isinstance(raw, list) else []:
        if isinstance(row, str):
            if row.strip():
                out.append(ChecklistItem(number=start + len(out),
                                         text=row.strip()))
            continue
        if not isinstance(row, dict):
            continue
        text = _text(row.get("text"))
        if not text:
            continue
        out.append(ChecklistItem(
            number=start + len(out),
            text=text,
            pc_numbers=_str_list(row.get("pc_numbers")),
            marks=_int(row.get("marks"), 0),
            sub_parts=_str_list(row.get("sub_parts")),
        ))
    return out


def _task_brief(raw) -> Optional[TaskBrief]:
    if not isinstance(raw, dict):
        return None
    return TaskBrief(
        task=_text(raw.get("task")),
        conditions=_text(raw.get("conditions")),
        tools_equipment_materials=_str_list(raw.get("tools_equipment_materials")),
        safety_requirements=_str_list(raw.get("safety_requirements")),
        time_allowed=_text(raw.get("time_allowed")),
    )


def _oral_questions(raw) -> List[OralQuestion]:
    out: List[OralQuestion] = []
    for row in raw if isinstance(raw, list) else []:
        if not isinstance(row, dict):
            continue
        question = _text(row.get("question"))
        if not question:
            continue
        out.append(OralQuestion(
            number=len(out) + 1,
            pc_numbers=_str_list(row.get("pc_numbers")),
            question=question,
            response_indicators=_str_list(row.get("response_indicators")),
        ))
    return out


def apply_written(tool: AssessmentTool, payload) -> AssessmentTool:
    """Put a written generation onto the tool, tolerating any shape.

    A payload that is not the object asked for yields an empty paper and a log
    line rather than an exception: the run survives, the validators report a
    paper with no items, and the raw file says what arrived.
    """
    data = payload if isinstance(payload, dict) else {}
    if not isinstance(payload, dict):
        runlog.warn("Assessment: the written generation was not a JSON object; "
                    "nothing could be read from it")
    tool.scenarios = _scenarios(data.get("scenarios"))
    tool.items = _items(data.get("items"), tool)
    known = {s.id for s in tool.scenarios}
    only = tool.scenarios[0].id if len(tool.scenarios) == 1 else ""
    for item in tool.items:
        if item.scenario_id and item.scenario_id not in known:
            runlog.warn(f"Assessment: item {item.number} references scenario "
                        f"'{item.scenario_id}', which was not written")
            item.scenario_id = ""
        if not item.scenario_id and only:
            # One scenario means one context, so there is nothing to decide:
            # the item was written to that situation whether or not the tag
            # came back. This is a pointer being restored, not a correction -
            # it changes no wording and hides no fault. Where the paper has
            # SEVERAL scenarios an untagged item is genuinely ambiguous, so it
            # is left alone for the validators to raise and the repair pass to
            # re-anchor.
            runlog.log(f"Assessment: item {item.number} arrived without a "
                       f"scenario id; attached to the paper's only scenario")
            item.scenario_id = only
    return tool


def apply_practical(tool: AssessmentTool, payload) -> AssessmentTool:
    """Put a practical generation onto the tool, tolerating any shape."""
    data = payload if isinstance(payload, dict) else {}
    if not isinstance(payload, dict):
        runlog.warn("Assessment: the practical generation was not a JSON "
                    "object; nothing could be read from it")
    tool.task_brief = _task_brief(data.get("task_brief"))
    tool.observation_checklist = _checklist(data.get("observation_checklist"), 1)
    tool.product_checklist = _checklist(
        data.get("product_checklist"), len(tool.observation_checklist) + 1)
    tool.oral_questions = _oral_questions(data.get("oral_questions"))
    return tool


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
def generate(tool: AssessmentTool, api_key: str = "", model: str = "",
             progress_cb=None) -> AssessmentTool:
    """Write one assessment tool: ONE request, for THIS CAT alone.

    Written path returns scenarios and items; practical path returns the task
    brief, the observation checklist, the product checklist and any oral
    questions. The tool is returned filled in; what came back is not judged
    here - call `assessment_validators.validate` next.

    Batching several CATs into one request is not an optimisation that was
    passed over, it is a correctness failure: the independence rule ("no
    stem gives away another item's key") is a property of one paper, and a
    model writing four papers at once neither holds it nor can be checked
    against it.
    """
    if not tool.allocations:
        raise AIError("Nothing to write: this CAT has no mark allocations. "
                      "Allocate the marks before generating.")

    api_key = api_key or load_api_key()
    if not api_key:
        raise AIError("No GROQ_API_KEY configured, so the assessment items "
                      "cannot be written. Set it in .env and try again.")

    model = resolve_model(api_key, model or load_model_name(), progress_cb)
    practical = tool.is_practical
    prompt = (build_practical_prompt(tool) if practical
              else build_written_prompt(tool))
    schema = practical_schema() if practical else written_schema()
    name = "assessment_practical" if practical else "assessment_written"
    system = AS_PRACTICAL_SYSTEM if practical else AS_WRITTEN_SYSTEM

    _emit_progress(progress_cb,
                   f"Assessment: writing {tool.cat.label} "
                   f"({tool.cat.assessment_type}) - "
                   f"{len(tool.allocations)} allocation(s)")

    payload = _chat_json(prompt, api_key, model, schema, name,
                         progress_cb=progress_cb, temperature=TEMPERATURE,
                         system=system)

    path = save_raw(tool, payload)
    if path:
        runlog.log(f"Assessment: raw generation kept at {path}")

    tool = (apply_practical(tool, payload) if practical
            else apply_written(tool, payload))
    if practical:
        _emit_progress(progress_cb,
                       f"Assessment: {len(tool.observation_checklist)} "
                       f"observation and {len(tool.product_checklist)} product "
                       f"items of evaluation")
    else:
        _emit_progress(progress_cb,
                       f"Assessment: {len(tool.items)} item(s), "
                       f"{len(tool.scenarios)} scenario(s)")
    return tool
