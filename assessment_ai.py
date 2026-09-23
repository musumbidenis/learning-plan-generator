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
from fractions import Fraction
from typing import Dict, List, Optional

import runlog
from ai_client import (AIError, _chat_json, _emit_progress, _strict,
                       load_api_key, load_model_name, resolve_model)
from assessment_allocation import _largest_remainder
import assessment_content
import assessment_knowledge
import assessment_research
import assessment_resources
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               MAX_CHECKLIST_ITEMS, MIN_CHECKLIST_ITEMS,
                               SHORT_RESPONSE, TEMPERATURE, VERB_BANK,
                               allowed_verbs, marks_per_response,
                               response_type, sector_for)
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

You are GIVEN some or all of:
- the unit of competency;
- RESOURCES FROM THE TRAINER - their own notes, slides or recording;
- TEACHING NOTES, read from published sources for the gaps;
- the TOPICS TAUGHT, from the curriculum;
- the PERFORMANCE CRITERIA (PCs);
- a MARK ALLOCATION TABLE with one row per item; and
- FURTHER INSTRUCTIONS from the trainer for this particular paper.

Your task is to write the assessment items only. The trainer's RESOURCES are the first source of every question; the TEACHING NOTES fill the gaps they leave; the TOPICS TAUGHT say which topics are in scope; and the PERFORMANCE CRITERIA are the competency link and the source of the marks and the Bloom level, and are never content.

## 1. WHERE THE CONTENT OF A QUESTION COMES FROM

Four sources may be supplied and they do four different jobs. Confusing them
is the single commonest way a paper comes out wrong. They rank, and the
ranking is not a preference - it is the order in which a candidate could
fairly have met the material.

**1. RESOURCES FROM THE TRAINER - the first source, when they are there.**
The trainer's own notes, slides or recording of this unit. This is what the
trainees actually sat through, so it is what they can fairly be asked about.
When RESOURCES are supplied, every question that can come from them DOES come
from them: read the passage, ask something it answers, and write the marking
scheme out of it. Prefer a question grounded in the trainer's own material to
a better-phrased one that is not.

**2. TEACHING NOTES - for the gaps, and only the gaps.**
Read from published reference sources for the taught topics the RESOURCES do
not reach. Use one only where the resources are silent on that topic. Where
both speak, THE RESOURCES WIN - if a note says something the trainer's
material contradicts or does not mention, follow the trainer.

**3. TOPICS TAUGHT - the list of topics, not their content.**
Which sub-topics and key points are in scope. Headings, not answers. Nothing
outside them may be assessed, whatever a resource or a note wanders onto.

**4. PERFORMANCE CRITERIA - these are NOT content.**
A performance criterion says which competency an item is evidence for, and
the allocation table uses it to fix the marks and the Bloom level. That is its
whole job. Do not take the subject of a question from a PC, do not expand a PC
into content, and do not treat its wording as something to assess.

HOW TO WRITE AN ITEM
1  Read the allocation row: the PC to link to, the Bloom level, the marks, how
   many responses to ask for, and the verbs it may open with. All fixed.
2  Find the material for that row's topic - the RESOURCES first, then the
   TEACHING NOTES if the resources do not cover it.
3  Ask a question that material answers, at that level, for those marks. The
   marking scheme names the real answer, taken from that material.

BE CONCRETE
Whatever the source, use the real substance of the trade as it names it - the
tools, standards, settings, figures and terminology a competent practitioner
in Kenya would actually use on that topic, by name. A question on vulnerability
scanning names a real scanner and a real finding; a question on access control
cites the standard by its proper name; a question on cable sizing names the
factor, not "a suitable factor". Never write "a suitable tool" where the
material in front of you names one.

WHAT NOT TO DO
- Do not ask a question your material cannot answer. If you cannot write the
  marking scheme from what is in front of you, ask a different question on the
  same topic that you can.
- Do not hand a line back. A key point that lists its own examples has already
  given the answer: "Types of malware: virus, worm, trojan, ransomware"
  answers "List FOUR types of malware" before the candidate picks up a pen.
  Ask what a competent worker must know ABOUT those things - how each spreads,
  how it is detected, what is done about it.
- Do not assess a topic that is not in the TOPICS TAUGHT. Resources and notes
  go deeper into the taught topics; they never add new ones. If either wanders
  onto something the unit does not cover, leave that part alone.
- Do not reach for your own knowledge of the trade first. Use what you were
  given. Your own knowledge is for judging whether it is true and whether it
  makes sense in a Kenyan workplace, and for the topics nothing supplied
  reaches at all - in which case keep strictly inside the taught topic and
  prefer a narrow question you can mark to a broad one you cannot.
- Never mention the resources, the notes, the unit or the course in a question
  or a marking scheme. The candidate is being assessed on the work.

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

The PC number is a link only. Do not take the subject of a question from a PC, and do not let its wording introduce content that is in neither the TEACHING NOTES nor the TOPICS TAUGHT.

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

Each allocation row's Bloom level comes from the performance criterion's own wording - a criterion reading "Tools and equipment are identified" is a KNOWLEDGE criterion - so the level already fits what is being assessed. Write to it; do not reach above or below it to make the paper look more demanding.

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

If a context is used, it must be based entirely on the TEACHING NOTES and must not introduce new technical knowledge.

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
- be answerable from the TEACHING NOTES, on a topic in the TOPICS TAUGHT;
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
- Take each marking point from the TEACHING NOTES. Where a topic has no note, keep strictly inside the TOPICS TAUGHT.
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
6. Every question sits on a topic in the TOPICS TAUGHT, and its answer comes from that topic's TEACHING NOTES rather than from the heading.
7. No question assesses a topic that is absent from the TOPICS TAUGHT, whether it came from a performance criterion, from a note that wandered, or from your own knowledge.
8. The performance criterion is used only as the competency link.
9. Every response requirement is quantifiable.
10. Every marking scheme matches the item's stated marks exactly.
11. No question depends on another question for its answer.
12. No scenario has been added merely to connect unrelated questions.
13. No total has been stated or calculated anywhere.
14. The final response contains only the required JSON object.

## 16. HOW A QUESTION IS FRAMED - THE HOUSE STYLE

Sections 1 to 15 say what a question must satisfy. This says what one looks
like on the page. Taken from published TVET CDACC papers; it relaxes no rule
above it.

    [optional one-clause workplace situation]. [VERB] [COUNT] [what is wanted].

Published examples - copy the shape, never the subject:

- "Mr. M has experienced conflict among workmates during working hours.
  Identify FOUR ways in which he can address conflict in the organization."
- "Highlight FOUR importance of holding meetings in an organization."
- "As the safety coordinator in your organisation, outline FOUR steps to be
  followed to establish work safety procedures."

THE SITUATION
- Name the person or the organisation: Mr. M, Njeri, WXYZ Limited, a garage in
  Nakuru. Kenyan names, workplaces a trainee will recognise.
- ONE clause or one short sentence. Never a paragraph, never facts to be
  worked through, never anything the candidate must read twice.
- It sets up the question and never hints at the answer.
- It may name the real equipment, tools, standards and figures the teaching
  notes give for that topic - that is what makes it a situation rather than a
  sentence.

WHEN TO USE ONE
Not on every question: a paper of little stories reads as contrived and one
with none reads as a list of definitions. Use one at APPLYING, ANALYSING,
EVALUATING and CREATING, where the candidate has to bring the material to
bear on something. Ask directly at KNOWLEDGE and UNDERSTANDING. Judge it item
by item.

MARKS AND THE NUMBER ASKED FOR
The marks say how much answer is wanted, and published papers are consistent:

- A recall verb - list, state, name, identify, select, outline - buys ONE mark
  a point. "Outline FOUR steps to be followed." is 4 marks.
- A verb asking for a developed answer - describe, explain, discuss, analyse,
  evaluate, justify, design - buys TWO, because each point needs a sentence of
  substance behind it and not just a name. "Explain FOUR relevant sources you
  would harness." is 8 marks.

Every allocation row gives the number as "ask for: N". Use it. Where the marks
do not divide evenly, one point carries the extra rather than the count going
up: 7 marks at UNDERSTANDING is THREE things marked 2, 2 and 3 - not seven
things at one mark each.

EVERY MARKING POINT IS A REAL ANSWER
The marking scheme is what an assessor holds while marking, so each point
states the substance expected. Write the answer, out of the teaching notes.

  Right: "Trojan - malware disguised as legitimate software"
  Right: "Worm - spreads itself across the network without a host file;
          detected by unexplained traffic between hosts"
  WRONG: "Threat classified as ___"
  WRONG: "Way 1", "Step 2", "Measure 3"
  WRONG: "First malware type with propagation method and detection technique"
  WRONG: "1 mark for each correct answer"

This holds however compound the question is: an item asking for three types of
malware with a propagation method for each needs three points naming three
actual types with their actual propagation. A scheme of blanks, numbered slots
or placeholders is not a marking scheme. If you cannot name the answer from
the TEACHING NOTES, ask a narrower question on the same topic that you can.

THE ASK
- Begins with the allowed verb for that row's Bloom level. Where a situation
  comes first, the verb opens the sentence that asks the question.
- The count is spelled out in capitals: FOUR, THREE, FIVE.
- One sentence.
- Do NOT write the marks into the stem. The marks are printed beside every
  question by the document itself, and a stem carrying "(4 marks)" prints them
  twice.
- Do not number the item in the stem. The numbering is printed for you.
- Do not mention the course OR your notes inside the stem. "State FOUR types
  of malware covered in the unit", "as taught in this unit", "described in the
  notes", "as mentioned in the teaching notes" - these belong to a syllabus,
  not to a question. The candidate already knows which unit they are sitting,
  and they have never seen your notes. Ask directly and leave both out:
  "Identify FOUR earthing measures", not "Identify FOUR earthing measures
  described in the notes".
"""


AS_PRACTICAL_SYSTEM = """You are a senior TVET assessor in Kenya, writing the practical assessment for one unit of competency under the TVET CDACC framework.

You are GIVEN the unit, the CAT, and a MARK ALLOCATION TABLE stating the marks fixed for each performance criterion. You do not choose what is assessed or for how many marks. You write the task the candidate is set and the items of evaluation the assessor works from.

THE MARKS ARE NOT YOURS
Give every item of evaluation a marks figure saying how much it is worth RELATIVE to the other items on the SAME performance criterion - 1 for a routine step, more for a step that carries the task. Those figures are scaled to the criterion's real marks afterwards, so they do not have to add up to anything and you must not try to make them. Never state a total for a criterion, a checklist or the assessment.

Each item of evaluation traces to exactly ONE performance criterion. Give that one pc_number, as it is written in the list below and with no label in front of it.

WHAT THE TASK MAY REQUIRE
You are given the TOPICS TAUGHT for this unit - the sub-topics and key points the curriculum sets out under each element. They set the SCOPE of the task: every skill the candidate is asked to perform sits on a topic that appears there.

Within that scope, set a real job rather than a rehearsal of the syllabus. Use the tools, materials, settings, standards and quantities a Kenyan workplace would really use for that work, by name, and make the items of evaluation say what a competent assessor would actually watch for - the things that separate work done properly from work that merely got finished.

REFERENCE NOTES
The TEACHING NOTES are the material the trainer teaches this unit from, and they are what the task is built out of: the real tools, standards, settings, procedures and classifications that make a brief concrete and an item of evaluation checkable come from there, not from your own knowledge and not from the wording of a PC. They go deeper into the taught topics; they never add one, so where a note wanders onto something the TOPICS TAUGHT do not cover, leave that part alone. They can also be wrong or dated: you are the assessor, and anything you take from one must be correct enough to judge a candidate against. Never mention the notes, their source or the course in the brief, the checklists or the oral questions.

The line is the same as the scope: more realism on a taught skill is wanted, a skill nobody covered is not. A task requiring a technique the trainees were never taught is not a harder assessment, it is an invalid one. Where no taught content is given for an element, work from the performance criterion alone.

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

EVERY ITEM OF EVALUATION CARRIES MARKS
There is no such thing as an item worth nothing. Each PC's stated marks are shared out across the items that trace to it, and every one of those items gets at least one mark. If an item is not worth a mark it is not worth the assessor's attention - fold it into another item or leave it out.

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
    return ("\n\nTOPICS TAUGHT, from the curriculum - which topics this "
            "assessment may cover. These are headings, not answers; what is "
            "IN them is in the TEACHING NOTES below:\n" + rendered)


def _resource_block(tool: AssessmentTool, budget: int = 0) -> str:
    """The trainer's own material, or nothing.

    First in the prompt and first in the ranking, because it is the only
    source that is not an approximation: the curriculum lists headings, the
    occupational standard lists criteria, and the teaching notes are a
    reference work's account of the subject. This is what the trainees
    actually sat through.

    `budget` caps it in characters: 0 means the module's own default and a
    negative number means leave it out. It is the LAST thing `_fit` gives up.
    """
    if budget < 0:
        return ""
    chosen = (assessment_resources.fit(tool.resources, budget) if budget
              else tool.resources)
    rendered = assessment_resources.render(chosen)
    if not rendered:
        return ""
    return ("\n\nRESOURCES FROM THE TRAINER - their own notes, slides or "
            "recording for this unit. THIS IS WHAT THE TRAINEES WERE ACTUALLY "
            "TAUGHT FROM, and it is the first source of every question:\n"
            + rendered
            + "\n\nWork from this material wherever it reaches. Read the "
              "passage for your row's topic, ask something it answers, and "
              "write the marking scheme out of it. Where this material and a "
              "TEACHING NOTE disagree, follow this - the trainer taught it. "
              "Where it is silent on a topic, the teaching notes below fill "
              "that gap."
              "\n\nTHE CANDIDATE IS NOT SITTING AN EXAM ON THIS DOCUMENT. "
              "They are being assessed on the work. So no question and no "
              "marking point may mention it - not the file, not a slide, not "
              "a page, and not \"the notes\", \"the class notes\", \"the "
              "handout\" or \"the unit\". Use the material; never refer to "
              "it.\n"
              "  WRITE:     \"List FOUR types of malware.\"\n"
              "  NOT:       \"List FOUR types of malware covered in the class "
              "notes.\"\n"
              "  WRITE:     \"Describe THREE components of role-based access "
              "control.\"\n"
              "  NOT:       \"...as outlined in the notes.\"")


def _instructions_block(tool: AssessmentTool) -> str:
    """What the trainer asked for on top, in their own words.

    Passed through as written rather than summarised or interpreted. It is
    placed LAST, after everything else, because it is the trainer speaking
    about this particular paper and should be read in the light of all of it -
    and because a late instruction is the one a model is most likely to still
    be holding when it starts writing.
    """
    said = (tool.extra_instructions or "").strip()
    if not said:
        return ""
    return ("\n\nFURTHER INSTRUCTIONS FROM THE TRAINER for this paper, in "
            "their own words:\n" + said
            + "\n\nFollow these within the rules above. They may steer what a "
              "question is about, how hard it is, or which topics to favour. "
              "They cannot change the marks, the Bloom level or the number of "
              "items - those are fixed in the allocation table - and they "
              "cannot put a topic in scope that the TOPICS TAUGHT leave out. "
              "If an instruction cannot be followed without breaking one of "
              "those, follow the rule and write the paper anyway.")


def _knowledge_block(tool: AssessmentTool, budget: int = 0) -> str:
    """Real substance on the taught key points, or nothing.

    The companion to `_content_block` and the answer to its weakness. The
    content block says what may be assessed and is made of headings; this says
    what those headings actually contain, so the model has something to be
    deep ABOUT. Absent, the paper is written the way it was before - from the
    headings - which is shallower and is exactly what this exists to fix.

    `budget` caps the notes in characters: 0 means the module's own default
    and a negative number means leave them out altogether. `_fit` lowers it
    when a prompt would otherwise be too large to send.
    """
    if budget < 0:
        return ""
    rendered = (assessment_knowledge.render(tool.knowledge, budget) if budget
                else assessment_knowledge.render(tool.knowledge))
    if not rendered:
        return ""
    return ("\n\nTEACHING NOTES - the material this unit is taught from. THIS "
            "IS WHAT THE QUESTIONS ARE MADE OF. Every allocation row names "
            "the note or notes it is written from:\n" + rendered
            + "\n\nRead the note your row names, then ask something that note "
              "answers, at that row's level and for its marks. Each marking "
              "point is the real answer, out of the note. Do not reach for "
              "your own knowledge first and do not take the subject of a "
              "question from a performance criterion. The notes go deeper "
              "into the taught topics and never add one, so where a note "
              "wanders outside the TOPICS TAUGHT, leave that part alone. They "
              "can also be wrong or dated - you are the assessor.\n\nThe "
              "candidate has never seen these notes. A stem that says "
              "\"described in the notes\" or \"as mentioned in the teaching "
              "notes\" is pointing them at a document they were never given. "
              "Ask the question directly.")


def _exemplar_block(tool: AssessmentTool, limit: int = 0) -> str:
    """Real questions from real papers, or nothing.

    Handed over with the warning that matters: the repositories that publish
    these are individual colleges, so a paper on the same unit may come from a
    health school and be full of clinics. The shape is what is wanted. The
    subject never is.
    """
    if limit < 0:
        return ""
    shown = tool.exemplars[:limit] if limit else tool.exemplars
    rendered = assessment_research.render(shown)
    if not rendered:
        return ""
    return ("\n\nHOW REAL TVET CDACC QUESTIONS ARE WRITTEN - actual questions "
            "from published papers, shown to you as a PATTERN:\n" + rendered
            + "\n\nCopy the SHAPE of these - the length of the lead-in, where "
              "the verb falls, how the count is stated, what a question of "
              "that many marks asks for. Never copy the SUBJECT. These papers "
              "come from other colleges and other trades, and anything they "
              "are about that is not in the TEACHING NOTES is out of bounds."
              "\n\nThese are CANDIDATE papers, so they show you questions and "
              "no marking schemes. Yours still carries one, and every point in "
              "it still names the answer an assessor looks for - never "
              "\"Threat 1 description\", never \"First tool\". Give the scheme "
              "the same attention as the question.")


def _sector_note(tool: AssessmentTool) -> str:
    """How this trade's own setters phrase a question, or nothing.

    The per-row verb list already carries the permission; this says where it
    came from, which is what lets the model reach for the rest of the idiom -
    a Mechanical paper says "with the aid of a sketch" and a Health one asks
    "What do you understand by", and neither is a verb list.
    """
    sector = sector_for(tool.programme, tool.unit_title)
    if sector is None or not sector.verbs:
        return ""
    return ("\n\nSECTOR HOUSE STYLE - this unit sits in " + sector.name
            + ", and published CDACC papers in this sector open their "
              "questions with: " + ", ".join(sector.verbs)
            + ".\nWrite in that idiom. Each allocation row still lists the "
              "verbs allowed at ITS level, and that list wins - these are the "
              "sector's habits, not a licence to use a verb from another "
              "level.")


def _notes_per_row(tool: AssessmentTool, budget: int) -> Dict[int, List[str]]:
    """{row number: the note numbers that row is written from}.

    Dealt out rather than handed round. Giving every row on an element the
    same notes produced three items in one paper asking the same question -
    a criterion worth twelve marks is split into three items of four, each
    row was pointed at both of element 1's notes, and all three came back
    listing the same four threats. A candidate answers once and is paid three
    times, and the paper covers a third of what it claims to.

    So the element's notes are dealt round-robin across the rows that sit on
    it: first row gets the first note, second row the second, and round again
    if there are more rows than notes. Where an element has as many notes as
    rows, no two rows start from the same material.

    Built from the notes that SURVIVED the budget, not from all of them:
    pointing a row at a note the trim dropped is worse than not pointing at
    all, because the model then hunts for something that is not there.
    """
    shown = (assessment_knowledge.fitting(tool.knowledge, budget) if budget > 0
             else assessment_knowledge.fitting(tool.knowledge))
    by_element: Dict[str, List[str]] = {}
    for index, note in enumerate(shown, start=1):
        by_element.setdefault(note.element_number, []).append(f"N{index}")

    rows_of: Dict[str, List[int]] = {}
    for row, a in enumerate(tool.allocations, start=1):
        rows_of.setdefault(a.element_number, []).append(row)

    out: Dict[int, List[str]] = {}
    for element, rows in rows_of.items():
        notes = by_element.get(element, [])
        if not notes:
            continue
        if len(notes) <= len(rows):
            for position, row in enumerate(rows):
                out[row] = [notes[position % len(notes)]]
        else:
            # More notes than rows: share them out so none goes unused.
            for position, row in enumerate(rows):
                out[row] = notes[position::len(rows)]
    return out


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


# The largest prompt worth sending, in characters, system instructions
# included.
#
# Measured rather than guessed, and the guess was badly wrong. Asking Groq for
# the token count of a real assessment prompt: 21,544 characters came to 4,891
# prompt tokens, so this text runs at about 4.4 characters to the token, not
# the 3.1 that was assumed here. The tier's 8000 tokens a minute is therefore
# roughly 35,000 characters, and the full prompt - every teaching note, every
# exemplar - is about 25,000 and goes through with room to spare. It was
# measured doing so: 5,891 prompt tokens in, a 3,316-token paper back.
#
# What had looked like a prompt too large to send was the rolling per-minute
# window, which `ai_client` now waits out (see `_is_window_full`). So this
# ceiling is no longer the thing shaping the prompt. It is a backstop against
# a unit whose curriculum is so large that the prompt really would not fit,
# and it sits below the 35,000 the tier allows because the paper coming back
# is counted in the same minute.
PROMPT_CHAR_CEILING = 30000

# What gives way, in order: (characters of teaching notes, exemplars shown).
#
# The exemplars go first, and go almost entirely, because the notes are now
# what the questions are MADE of while the exemplars only show how a question
# is phrased - and most of what they teach is already written into section 16,
# which was distilled from the same papers. A paper written in a slightly
# plainer style is a paper; a paper with no content to ask about is not.
#
# The allocation table, the topics taught and the standing instructions are
# not on this list at all. Without them there is no paper.
# (characters of the trainer's resources, characters of teaching notes,
#  exemplars shown).
#
# The order of surrender follows the order of authority. The exemplars go
# first and go almost entirely - most of what they teach is already written
# into section 16. Then the teaching notes, which only fill gaps. The
# trainer's own material is last, because it is the only source that is not an
# approximation of what was taught, and a paper written without it is a paper
# written about a different course.
#
# The allocation table, the topics taught and the standing instructions are
# not on this list at all. Without them there is no paper.
_FIT_STEPS = ((6000, 5200, 5), (6000, 4200, 3), (6000, 3400, -1),
              (6000, 2000, -1), (5000, 1200, -1), (4000, -1, -1),
              (2800, -1, -1), (1600, -1, -1), (-1, -1, -1))


def _fit(assemble, system: str) -> str:
    """The prompt, trimmed until the provider will accept it.

    `assemble(notes_budget, exemplar_limit)` builds a candidate prompt, and
    `system` is the standing instructions it will be sent with - they count
    towards the same limit, and they are the larger half.

    In normal use nothing is given up: an ordinary unit's full prompt is well
    inside the ceiling, and the per-minute refusals that this once existed to
    dodge are waited out in `ai_client` instead, which keeps the paper whole.
    This is for the unit whose curriculum really is too big.

    The concessions are walked in order and the first that fits is returned.
    If none does, the smallest is sent anyway: a prompt still too large with
    no notes and no exemplars is one whose taught content alone is oversized,
    and the honest outcome there is the provider's own error rather than a
    paper quietly written from half a curriculum.
    """
    smallest = ""
    for step in ((0, 0, 0),) + _FIT_STEPS:
        smallest = assemble(*step)
        if len(system) + len(smallest) <= PROMPT_CHAR_CEILING:
            if step != (0, 0, 0):
                runlog.log(f"Assessment: the prompt was trimmed to fit - "
                           f"{step[0]} characters of the trainer's resources, "
                           f"{step[1]} of teaching notes and {step[2]} "
                           f"exemplar(s)")
            return smallest
    runlog.warn("Assessment: the prompt is over the provider's size limit "
                "even with no reference notes and no exemplars")
    return smallest


def build_written_prompt(tool: AssessmentTool, resource_budget: int = 0,
                         notes_budget: int = 0,
                         exemplar_limit: int = 0) -> str:
    """The per-paper half: the allocation table and nothing standing.

    The allowed verbs are printed per row rather than as one bank, because the
    model is choosing a verb for THAT row's level and a combined list invites
    it to pick from the wrong one.
    """
    level_note = ""
    if str(tool.knqf_level).strip() in CONSTRUCTED_RESPONSE_ONLY_LEVELS:
        level_note = ("\nThis is a KNQF level %s paper: every item is "
                      "constructed response. Selected-response formats are "
                      "not used at this level at all.\n" % tool.knqf_level)

    # The verbs this trade's own papers open with - see
    # `assessment_config.allowed_verbs`. None when the programme matches no
    # sector, and the generic bank is then used unchanged.
    sector = sector_for(tool.programme, tool.unit_title)

    def rows_for(notes_budget: int) -> str:
        # Built per candidate prompt, because the notes a row may cite are
        # the notes that survived THAT prompt's budget.
        served = _notes_per_row(tool, notes_budget)
        repeated = {a.pc_number for a in tool.allocations
                    if sum(1 for b in tool.allocations
                           if b.pc_number == a.pc_number) > 1}
        rows = []
        for n, a in enumerate(tool.allocations, start=1):
            verbs = ", ".join(allowed_verbs(a.bloom, sector))
            per_point = marks_per_response(a.bloom)
            asked = max(1, a.marks // per_point)
            # Which notes this row is written from. Without it the model has
            # to guess which of eight notes belongs to the criterion in front
            # of it, and on a unit whose elements share vocabulary it guesses
            # wrong.
            notes = served.get(n, [])
            pointer = (f" | write it from: {', '.join(notes)}"
                       if notes else "")
            # A criterion too big for one question is split into several rows.
            # Saying so stops the model writing the same question three times.
            if a.pc_number in repeated:
                pointer += (" | this criterion carries several items - ask "
                            "something the others do not")
            rows.append(
                f"{n}. element_number: {a.element_number} "
                f"({a.element_title}) | pc_number: {a.pc_number} | "
                f"criterion: {a.pc_text} | bloom_level: {a.bloom} | "
                f"marks: {a.marks} | ask for: {asked} | "
                f"allowed verbs: {verbs or '(any)'}{pointer}")
        return "\n".join(rows)

    def assemble(resource_budget: int, notes_budget: int,
                 exemplar_limit: int) -> str:
        return f"""{_header(tool)}{_resource_block(tool, resource_budget)}{_content_block(tool)}{_knowledge_block(tool, notes_budget)}{_exemplar_block(tool, exemplar_limit)}{_sector_note(tool)}

MARK ALLOCATION TABLE - one item per row, in this order, at these marks:
{rows_for(notes_budget)}

ITEMS REQUIRED: {len(tool.allocations)}
{level_note}
Write one item per row above, in this order. Return the JSON object now.{_instructions_block(tool)}"""

    if (resource_budget, notes_budget, exemplar_limit) != (0, 0, 0):
        return assemble(resource_budget, notes_budget, exemplar_limit)
    return _fit(assemble, AS_WRITTEN_SYSTEM)


def build_practical_prompt(tool: AssessmentTool, resource_budget: int = 0,
                           notes_budget: int = 0,
                           exemplar_limit: int = 0) -> str:
    """The per-task half: what each PC is worth, and nothing standing."""
    by_pc: Dict[str, int] = tool.marks_by_pc()
    seen: List[str] = []
    rows = []
    for a in tool.allocations:
        if a.pc_number in seen:
            continue
        seen.append(a.pc_number)
        rows.append(f"- pc_number: {a.pc_number} | element_number: "
                    f"{a.element_number} ({a.element_title}) | criterion: "
                    f"{a.pc_text} | marks: "
                    f"{by_pc.get(a.pc_number, a.marks)}")
    methods = assessment_content.methods(tool.content)
    suggested = ("\n\nASSESSMENT METHODS THE CURRICULUM SUGGESTS FOR THIS "
                 "UNIT:\n" + "\n".join(f"- {m}" for m in methods)
                 if methods else "")
    # One line per PC, in the order the rows were built, so a PC split across
    # two allocations is budgeted once and at its combined figure.
    def assemble(resource_budget: int, notes_budget: int,
                 exemplar_limit: int) -> str:
        return f"""{_header(tool)}{_resource_block(tool, resource_budget)}{_content_block(tool)}{_knowledge_block(tool, notes_budget)}{_exemplar_block(tool, exemplar_limit)}{_sector_note(tool)}{suggested}

PERFORMANCE CRITERIA ASSESSED, with the marks fixed for each:
{chr(10).join(rows)}

HOW THE MARKS ABOVE ARE USED
Each criterion's marks are shared out across the items you write for it, in proportion to the relative figures you give them. You do not need to make anything add up - write the number of items the task genuinely needs and say how much each is worth next to the others on the same criterion.

TIME ALLOWED: {tool.cat.duration_minutes} minutes

Write the candidate's task brief, then the observation checklist, then the product checklist, then any oral questions. Return the JSON object now.{_instructions_block(tool)}"""

    if (resource_budget, notes_budget, exemplar_limit) != (0, 0, 0):
        return assemble(resource_budget, notes_budget, exemplar_limit)
    return _fit(assemble, AS_PRACTICAL_SYSTEM)


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


def _number_list(value) -> List[str]:
    """PC numbers, with any label the model echoed onto them stripped.

    The same fault as `_number` and in the same place - a model told to echo a
    value back gives the label with it - but on the practical path, where the
    numbers arrive as a list. Worth its own helper because it went unnoticed
    when the written path was fixed: one live run returned "1.1" and the next
    returned "PC 1.1" from the same prompt, which is what a warm model does.
    """
    return [n for n in (_number(v) for v in _str_list(value)) if n]


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
        fmt = response_type(_text(row.get("response_type"))) or SHORT_RESPONSE
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
        # One criterion per item. Every check in `assessment_validators`
        # counts an item's marks in full against each PC it names, while the
        # grand total is a plain sum of the items - so an item tracing to two
        # PCs makes the two views of the same paper disagree by its own marks.
        # A CDACC checklist row evidences one criterion anyway.
        traced = _number_list(row.get("pc_numbers"))[:1]
        out.append(ChecklistItem(
            number=start + len(out),
            text=text,
            pc_numbers=traced,
            marks=_int(row.get("marks"), 0),
            sub_parts=_str_list(row.get("sub_parts")),
        ))
    return out


def _apportion(weights: List[int], total: int) -> List[int]:
    """`total` shared across `weights` in proportion, as whole numbers.

    `assessment_allocation._largest_remainder` takes shares that already sum
    to about `total` and settles the rounding; it does not scale. So the
    weights are scaled here first, on exact Fractions, and it settles the
    remainder - the same rule, and the same tie-break, the CAT's own marks
    were allocated with.

    Weights that are all zero fall back to an even split: the model said
    nothing useful about relative worth, and an even share is the honest
    reading of that.
    """
    if total <= 0 or not weights:
        return [0] * len(weights)
    pool = sum(weights)
    if pool <= 0:
        weights, pool = [1] * len(weights), len(weights)
    raw = [Fraction(w * total, pool) for w in weights]
    return _largest_remainder(raw, total, list(range(len(weights))))


def _fund_checklist(tool: AssessmentTool) -> None:
    """Share each PC's allocated marks across the items that evidence it.

    The model proposes how much each item is worth RELATIVE to the others on
    the same criterion; this turns those proposals into the CAT's actual
    marks. The arithmetic was the one thing still asked of the model on this
    path, and it was the one thing it could not be relied on for: three live
    runs at the same prompt gave a clean tool, a tool whose PCs all read as
    unassessed, and a tool of nineteen items totalling 72 marks against an
    allocation of 40. None of that is a wording fault, so no repair pass could
    have cleared it - the trainer just lost the generation.

    Every other mark in this module is computed in plain code before or after
    the model speaks. This brings the practical path into line, and the sums
    stop being a thing that can go wrong at all.

    Each item is given one mark first, so no row is worth nothing, and what is
    left is apportioned by the model's proposals using the same
    largest-remainder rule the CAT's own marks were allocated with. Where a
    criterion has more items than marks there are not enough to go round; the
    proposals are used as they are and `_check_unfunded_items` reports the
    rows that came out at nought.
    """
    by_pc = tool.marks_by_pc()
    items = tool.observation_checklist + tool.product_checklist
    for pc_number, budget in by_pc.items():
        mine = [c for c in items if pc_number in c.pc_numbers]
        if not mine:
            continue
        proposed = [max(0, c.marks) for c in mine]
        if len(mine) <= budget:
            shares = _apportion(proposed, budget - len(mine))
            for item, share in zip(mine, shares):
                item.marks = 1 + share
        else:
            for item, share in zip(mine, _apportion(proposed, budget)):
                item.marks = share
        runlog.log(f"Assessment: PC {pc_number} - {budget} mark(s) over "
                   f"{len(mine)} item(s) of evaluation")

    orphans = [c for c in items if not c.pc_numbers]
    for item in orphans:
        runlog.warn(f"Assessment: item of evaluation {item.number} traces to "
                    f"no performance criterion and carries no marks")
        item.marks = 0


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
            pc_numbers=_number_list(row.get("pc_numbers")),
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
    _fund_checklist(tool)
    return tool


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
def _send(tool: AssessmentTool, practical: bool, api_key: str, model: str,
          schema: dict, name: str, system: str, progress_cb):
    """The request, made smaller and retried if the provider says it is too big.

    `_fit` already keeps the prompt under a size that has been measured to
    work, but it is measuring characters and the provider is counting tokens -
    and counting the schema and its own framing alongside them, by a formula
    that is not published and does not hold still. A character ceiling is
    therefore a good guess and never a guarantee.

    So the refusal itself is used as the measurement. HTTP 413 means only that
    this request was too large, and the honest response is to give something
    up and ask again rather than to fail a paper over a rate limit. What is
    given up is the optional material, in the order `_FIT_STEPS` sets, and
    only then does the error stand.
    """
    build = build_practical_prompt if practical else build_written_prompt
    attempts = [build(tool)]
    for step in _FIT_STEPS:
        attempts.append(build(tool, *step))

    last: Optional[AIError] = None
    for index, prompt in enumerate(attempts):
        if index and prompt == attempts[index - 1]:
            continue                       # this concession changed nothing
        try:
            return _chat_json(prompt, api_key, model, schema, name,
                              progress_cb=progress_cb,
                              temperature=TEMPERATURE, system=system)
        except AIError as e:
            if "413" not in str(e) and "too large" not in str(e).lower():
                raise
            last = e
            runlog.warn(f"Assessment: the provider refused the request as too "
                        f"large; trying again with less reference material "
                        f"({len(attempts) - index - 1} step(s) left)")
            _emit_progress(progress_cb,
                           "Assessment: the request was too large - trying "
                           "again with less reference material")
    raise last if last else AIError("The assessment could not be written.")


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
    schema = practical_schema() if practical else written_schema()
    name = "assessment_practical" if practical else "assessment_written"
    system = AS_PRACTICAL_SYSTEM if practical else AS_WRITTEN_SYSTEM

    _emit_progress(progress_cb,
                   f"Assessment: writing {tool.cat.label} "
                   f"({tool.cat.assessment_type}) - "
                   f"{len(tool.allocations)} allocation(s)")

    payload = _send(tool, practical, api_key, model, schema, name, system,
                    progress_cb)

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
