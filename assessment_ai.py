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
import re
from typing import Dict, List, Optional

import runlog
from ai_client import (AIError, _chat_json, _emit_progress, _strict,
                       load_api_key, load_model_name, resolve_model)
import assessment_content
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               MAX_CHECKLIST_ITEMS, MIN_CHECKLIST_ITEMS,
                               TEMPERATURE, VERB_BANK, marks_per_response)
from assessment_models import (AssessmentTool, ChecklistItem, Item,
                               MarkingPoint, OralQuestion, TaskBrief)

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
    """Section 14 of AS_WRITTEN_SYSTEM, as a schema the decoder enforces.

    The field names are the prompt's, not this module's: a strict schema and
    the instructions that describe it have to agree exactly, and where they
    disagree it is the prompt that is the specification. `unit_of_competency`
    is asked for and then ignored - the unit is already known here, and a
    model that has just written it out is a model that has read the header.
    """
    return _strict({
        "type": "object",
        "properties": {
            "unit_of_competency": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_number": {"type": "integer"},
                        "element_number": {"type": "string"},
                        "pc_number": {"type": "string"},
                        "bloom_level": {"type": "string"},
                        "marks": {"type": "integer"},
                        "response_type": {"type": "string"},
                        "stem": {"type": "string"},
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

You are GIVEN:
- the unit of competency;
- the CONTENT TAUGHT for the unit;
- the PERFORMANCE CRITERIA (PCs); and
- a MARK ALLOCATION TABLE with one row per item.

Your task is to write the assessment items only. The CONTENT TAUGHT is the primary source for what may be assessed. The PERFORMANCE CRITERIA are used only as a link to the relevant competency and must not be treated as additional teaching content.

## 1. CONTENT TAUGHT IS THE ASSESSMENT SOURCE

Every question MUST be based on the CONTENT TAUGHT supplied for the unit.

- Assess only concepts, sub-topics, procedures, principles, examples, skills or key points that appear in the CONTENT TAUGHT.
- Do not assess something simply because it appears in a performance criterion if it was not covered in the CONTENT TAUGHT.
- Use the relevant performance criterion only to link the item to the competency being assessed.
- Do not expand, reinterpret or add content from the performance criterion.
- Do not use your own occupational knowledge to fill gaps in the taught content.
- Do not introduce equipment, tools, materials, standards, legislation, formulae, software, suppliers, procedures or terminology that are not present in the CONTENT TAUGHT.
- Where the taught content is limited, write a narrower question rather than introducing additional knowledge.
- Where an element contains several taught sub-topics, spread the questions across the available content instead of repeatedly assessing the same point.

**Important distinction:**
CONTENT TAUGHT = what the candidate can be assessed on.
PERFORMANCE CRITERION = the competency link for the item.

## 2. MARKS ARE FIXED

The marks in the allocation table are final.

- Write each item to exactly the stated number of marks.
- Never invent, merge, split, round, transfer or adjust marks.
- Never redistribute marks between items.
- Never state or calculate any section, paper or overall total.
- The marks printed in the allocation table are the only marks to use.

## 3. ONE ITEM PER ALLOCATION ROW

Return exactly one assessment item for every row in the mark allocation table.

- Keep the exact order of the rows.
- Do not add, remove, merge or reorder items.
- Echo the following information on every item exactly as supplied in the allocation table:
  - Element number
  - PC number
  - Bloom level
  - Marks

The PC number is a link only. Do not allow the wording of the PC to introduce content that is absent from CONTENT TAUGHT.

## 4. CONSTRUCTED RESPONSE ONLY

All items must require the candidate to respond in their own words.

Use only:
- short_response - a few lines to a short paragraph;
- extended_response - a longer structured response where appropriate.

Do NOT use:
- multiple-choice questions;
- true/false questions;
- matching items;
- fill-in-the-blank items;
- selection questions;
- lettered or numbered alternatives from which the candidate chooses.

## 5. BLOOM LEVEL CONTROL

The first word of every item must be an allowed verb for the Bloom level specified in that allocation row.

- The lead verb must be the first word of the stem.
- Use only an allowed verb supplied for that Bloom level.
- Do not use a higher or lower cognitive demand than the stated Bloom level.
- KNOWLEDGE must assess recall or identification.
- UNDERSTANDING must assess comprehension, explanation, interpretation or description.
- APPLYING must require use of taught knowledge in an appropriate situation.
- ANALYSING must require examination of parts, relationships, differences, causes, effects or patterns supported by the taught content.
- EVALUATING must require a judgement based on taught knowledge or stated criteria.
- CREATING must require production, development, formulation, planning or organisation using taught content.

Do not place any other word before the lead verb.

## 6. ALLOWED VERBS

Use only the allowed verbs supplied for the relevant Bloom level in the assessment input.

If the allocation row provides a specific allowed-verb list, use a verb from that list only.

Do not substitute another verb simply because it has a similar meaning.

## 7. MAKE THE RESPONSE QUANTIFIABLE

Every question must clearly state what the candidate is expected to provide.

- State the exact number of responses, points, factors, steps, reasons, characteristics, measures or other marking points required.
- The number requested must agree with the marks and marking scheme.
- Avoid vague instructions such as "State the measures" or "Explain the factors".
- Use precise wording such as "State FOUR..." or "Explain THREE...".
- Where a response requires depth and carries more than one mark, make the expected depth clear.

## 8. EACH QUESTION MUST STAND ALONE

Each question must be independently answerable.

- Do not refer to another question.
- Do not write "using your answer above", "as stated in question 2", or similar wording.
- Do not make one question provide the answer to another.
- Do not repeat the same assessment demand unnecessarily.

## 9. NO FORCED SCENARIO LOGIC

Do not create one common scenario for the whole CAT.

Questions may be direct questions based on the taught content. Use a short workplace situation only where it genuinely supports the required Bloom level, particularly for APPLYING, ANALYSING, EVALUATING or CREATING.

If a context is used, it must be based entirely on the CONTENT TAUGHT and must not introduce new technical knowledge.

## 10. KEEP QUESTIONS CONCISE

- Keep the stem direct and easy to understand.
- Normally use one sentence.
- Do not repeat the teaching notes in the question.
- Do not unnecessarily restate the performance criterion.
- Do not include lengthy background information.
- The question should assess the taught content rather than test the candidate's ability to interpret complicated wording.

## 11. KENYAN TVET CDACC TERMINOLOGY

Use:
- trainee
- candidate
- assessor
- unit of competency
- performance criteria
- competency
- Continuous Assessment Test (CAT)

Do NOT use these terms as nouns:
- student
- pupil
- learner
- teacher
- lecturer
- instructor
- exam
- quiz
- test

Use British/Kenyan English spelling, including:
- organise
- practise (verb)
- programme
- labelled
- capitalised
- centred

## 12. QUESTION QUALITY

Each question must:
- be directly supported by the CONTENT TAUGHT;
- link to the specified performance criterion without adding new content;
- match the specified Bloom level;
- begin with an allowed Bloom verb;
- have a clearly quantifiable response requirement;
- be appropriate for the stated marks;
- use simple, professional English suitable for Kenyan TVET trainees;
- avoid unnecessary ambiguity;
- avoid double-barrelled demands unless both parts are explicitly required by the allocation; and
- assess one clear competency demand at a time.

## 13. MARKING SCHEME

Provide a marking scheme for every item.

The marking scheme must contain discrete, mark-bearing points whose marks add up exactly to the item's stated marks.

Rules:
- One mark-bearing point per mark is the normal rule.
- A point may carry more than one mark only where justified by the required depth; state the mark value explicitly.
- Each marking point must state the actual substance expected from the candidate.
- Never write "1 mark for each correct answer".
- Award marks only for information requested in the question.
- Do not introduce marking points based on knowledge outside the CONTENT TAUGHT.
- The marking scheme must correspond directly to the question wording and the number of responses requested.

## 14. OUTPUT FORMAT

Return ONE JSON object and nothing else.

The JSON must contain:
- unit_of_competency
- items

Each item must contain:
- item_number
- element_number
- pc_number
- bloom_level
- marks
- response_type
- stem
- marking_scheme

The values for element_number, pc_number, bloom_level and marks must be copied verbatim from the allocation table.

Do not include explanations, commentary, assumptions, scenarios, totals or additional questions outside the required JSON object.

## 15. FINAL INTERNAL CHECK

Before returning the JSON, verify that:

1. There is exactly one item for every allocation row.
2. The items are in exactly the same order as the allocation table.
3. Every item's marks exactly match its allocation row.
4. Every item's Bloom level exactly matches its allocation row.
5. Every lead verb is allowed for the specified Bloom level.
6. Every question is supported by the CONTENT TAUGHT.
7. No question introduces content merely because it appears in the performance criterion.
8. The performance criterion is used only as the competency link.
9. Every response requirement is quantifiable.
10. Every marking scheme matches the item's stated marks exactly.
11. No question depends on another question for its answer.
12. No scenario has been added merely to connect unrelated questions.
13. No total has been stated or calculated anywhere.
14. The final response contains only the required JSON object.

## 16. HOW A QUESTION IS FRAMED - THE HOUSE STYLE

Sections 1 to 15 say what a question must satisfy. This section says what one
looks like on the page. It is taken from published TVET CDACC written
assessment papers and does not relax any rule above it.

A question is built as:

    [optional one-clause workplace situation]. [VERB] [COUNT] [what is wanted].

Published examples, to copy the shape of and not the subject:

- "Mr. M has experienced conflict among workmates during working hours.
  Identify FOUR ways in which he can address conflict in the organization."
- "Njeri would like to do a presentation on barriers of communication. State
  FOUR effective communication techniques she is likely to use."
- "Highlight FOUR importance of holding meetings in an organization."
- "As the safety coordinator in your organisation, outline FOUR steps to be
  followed to establish work safety procedures."

THE SITUATION
- Name the person or the organisation: Mr. M, Njeri, Jane a supervisor at
  Company X, WXYZ Limited, a garage in Nakuru. Kenyan names, and workplaces of
  a size a trainee will recognise.
- ONE clause, or one short sentence. Never a paragraph, never a set of facts
  to be worked through, and never anything the candidate must read twice.
- It sets up the question. It never contains, hints at or narrows the answer.
- It is drawn from the CONTENT TAUGHT, and introduces no equipment, standard
  or term the content does not have.

WHEN TO USE ONE
Not on every question. A paper where every item opens with a little story
reads as contrived, and one where none does reads as a list of definitions.
Use a situation where it does work - where the item is at APPLYING, ANALYSING,
EVALUATING or CREATING, and the candidate has to bring the taught content to
bear on something. Ask directly at KNOWLEDGE and UNDERSTANDING, where the
situation would be decoration. Judge it item by item.

MARKS AND THE NUMBER ASKED FOR
The marks say how much answer is wanted, and the number you ask for must come
out of them. Published papers are consistent:

- A recall verb - list, state, name, identify, select, outline - buys ONE mark
  a point. "State FOUR methods of identifying communication needs." is 4 marks.
  "Outline FOUR steps to be followed." is 4 marks.
- A verb asking for a developed answer - describe, explain, discuss, analyse,
  evaluate, justify, design - buys TWO marks a point, because each point needs
  a sentence of substance behind it and not just a name. "Explain FOUR
  relevant sources you would harness." is 8 marks. "Discuss FIVE factors that
  support implementation." is 10 marks. "Describe three recognized stages of
  fire." is 6 marks.

Every allocation row tells you the number to ask for, as "ask for: N". Use
that number. It is the item's marks divided by what its verb buys, rounded
down, so a 4-mark UNDERSTANDING item asks for TWO things and not four.

Where the marks do not divide evenly, one point carries the extra rather than
the count going up: 7 marks at UNDERSTANDING is THREE things, marked 2, 2 and
3 - not seven things at one mark each. Published papers do this routinely
("Explain in detail FIVE classifications of solid and liquid wastes. [15
Marks]" is five points of three).

EVERY MARKING POINT IS A REAL ANSWER
The marking scheme is what an assessor holds while marking, so each point
states the substance actually expected from the candidate. Write the answer.

  Right: "Trojan - malware disguised as legitimate software"
  Right: "Likelihood of the threat being realised, rated against the matrix"
  WRONG: "Threat classified as ___"
  WRONG: "Way 1", "Step 2", "Measure 3"
  WRONG: "1 mark for each correct answer"

A marking scheme of blanks, numbered slots or placeholders is not a marking
scheme. If you cannot name the answer from the CONTENT TAUGHT, ask a narrower
question that you can.

THE ASK
- Begins with the allowed verb for that row's Bloom level. Where a situation
  comes first, the verb opens the sentence that asks the question.
- The count is spelled out in capitals: FOUR, THREE, FIVE.
- One sentence.
- Do NOT write the marks into the stem. The marks are printed beside every
  question by the document itself, and a stem carrying "(4 marks)" prints them
  twice.
- Do not number the item in the stem. The numbering is printed for you."""


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
        per_point = marks_per_response(a.bloom)
        asked = max(1, a.marks // per_point)
        rows.append(
            f"{n}. element_number: {a.element_number} ({a.element_title}) | "
            f"pc_number: {a.pc_number} | criterion: {a.pc_text} | "
            f"bloom_level: {a.bloom} | marks: {a.marks} | "
            f"ask for: {asked} | allowed verbs: {verbs or '(any)'}")
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
Write one item per row above, in this order. Return the JSON object now."""


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


_RE_LABEL = re.compile(r"^\s*(?:pc|p\.?c\.?|element|el|item|no|number)\s*"
                       r"[:.\-]?\s*", re.I)


def _number(value) -> str:
    """A PC or element number with any label the model repeated stripped off.

    The allocation table labels its columns, and a model told to echo a value
    back "verbatim" may echo the label with it - a live paper came back with
    every pc_number as "PC 1.1", which read as six items assessing nothing and
    three performance criteria never assessed. The label is not part of the
    number, and stripping it is not a correction: no wording changes and no
    fault is hidden, because an item genuinely tagged to the wrong PC still
    fails the coverage check afterwards.
    """
    text = _RE_LABEL.sub("", _text(value))
    return text.strip(" :.-")


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
        fmt = _text(row.get("response_type")) or "short_response"
        out.append(Item(
            number=len(out) + 1,
            element_number=(_number(row.get("element_number"))
                            or (fallback.element_number if fallback else "")),
            pc_number=(_number(row.get("pc_number"))
                       or (fallback.pc_number if fallback else "")),
            bloom=(_text(row.get("bloom_level")).lower()
                   or (fallback.bloom if fallback else "")),
            stem=stem,
            marks=_int(row.get("marks"), -1),
            marking_scheme=_marking_scheme(row.get("marking_scheme")),
            item_format=fmt,
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
    # No shared scenario is asked for. Section 9 of the standing instructions
    # forbids one common scenario across the paper: where a workplace
    # situation is needed to carry a higher Bloom level, it belongs inside
    # that item's own stem. `AssessmentTool.scenarios` therefore stays empty
    # on this path, and the document builder's Section A with it.
    tool.items = _items(data.get("items"), tool)
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
                       f"Assessment: {len(tool.items)} item(s)")
    return tool
