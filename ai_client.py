"""Module C - THE SINGLE AI CALL (Groq, ONE grounded request).

The ONLY place AI is used in the whole pipeline. It fills the columns AI is good
at - learning_outcomes, trainee_activities, resources, assessments - grounded in
the deterministically-parsed unit + curriculum content. `key_points` are supplied
(authoritative curriculum content); the model keeps/format them, never invents.

Everything before this stage (parsing, planning) and after it (doc building) is
pure Python. Robust safety nets re-stamp the deterministic schedule and backfill
any blank cell, so the output is never empty even if the API misbehaves.

Why Groq: its free tier allows 1,000 requests a day with no card, and its
gpt-oss and qwen3 models honour `response_format: json_schema` with strict
constrained decoding - the guarantee everything downstream is built on. (Mistral
was here before; on a free workspace it allows the capable models ZERO requests
a minute, which left only the small ministral models.)
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests

import runlog
from models import DeliveryStep, PlanInputs, Session, SessionPlan, Unit

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# The Groq key is read from the environment / .env (GROQ_API_KEY) - see
# load_api_key(). It is never hard-coded in source and never entered in the UI.

# Groq's strongest free model that does strict constrained decoding. Only a
# preference: which model actually generates is settled at run time by
# `resolve_model`, because an account may not be able to call this one.
DEFAULT_MODEL = "openai/gpt-oss-120b"

GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"
LIST_MODELS_TIMEOUT = 30
PROBE_TIMEOUT = 30

# Room for the probe's answer. Not 20: the gpt-oss models spend tokens
# reasoning before they write anything, so a tight ceiling cut them off
# mid-thought and Groq rejected the empty result as a schema failure - which
# read as "this model can't do schemas" and disqualified the best models.
PROBE_TOKENS = 512

# How long to allow one grounded request. A model that has never completed one
# gets less rope: it is far more likely to be a model that cannot do the job
# than a slow answer worth waiting for.
REQUEST_TIMEOUT = 180
UNPROVEN_TIMEOUT = 90
# One timeout is usually the API being busy; twice running is the model.
UNPROVEN_ATTEMPTS = 2

# How many models to work through before giving up on a request.
MODEL_ATTEMPTS = 4

# Models that have completed a real structured request in this process.
_PROVEN_MODELS: set = set()

# Ordering for that search, best first, matched as a substring of the model
# name. It decides only what to TRY first; anything not listed is tried in the
# middle, so a model released after this line was written still gets a turn.
# gpt-oss and qwen3 are the families Groq documents as doing strict schema
# decoding, which is why they lead.
_MODEL_FAMILIES = ("gpt-oss", "qwen3", "llama-3.3", "llama", "compound")

# Purpose-built models: they are chat models, and they would answer, but speech,
# safety and embedding models write poor lesson prose.
_RE_SPECIAL_PURPOSE = re.compile(
    r"whisper|orpheus|prompt-guard|safeguard|guard|tts|embed|rerank|ocr"
    r"|moderation", re.I)

# 'openai/gpt-oss-120b' -> 120, so the larger model of a family goes first.
_RE_MODEL_SIZE = re.compile(r"(\d+)\s*b(?:\b|-)")

# 'qwen/qwen3.8-27b' -> 3.8, so the newer of two same-sized siblings goes first.
_RE_MODEL_VERSION = re.compile(r"(\d+\.\d+)")

# The model chosen for this process, once something has answered.
_RESOLVED_MODEL = ""

# Sessions per grounded Learning-Plan call. A full term (e.g. 24 sessions) does
# not fit in one JSON response - it gets truncated to invalid JSON - so we
# generate in batches and concatenate the rows in order.
#
# Eight, because the cost of a batch is mostly the INSTRUCTIONS, not the
# sessions: measured against the live API, the fixed block of this module's
# prompt is ~1,336 tokens and each session adds only ~94. Every extra batch
# therefore resends 1,336 tokens for nothing, and Groq's binding limit is not
# its 1,000 requests a day but 8,000 TOKENS a minute. Halving the batch count
# of an 11-session plan gives a whole minute's allowance back.
#
# It was four while the model still wrote ~900 output tokens a session; asking
# it to think briefly (see REASONING_EFFORT) roughly halved that, which is what
# makes eight fit. A batch that still overflows is split and retried.
LP_SESSION_CHUNK = 8

# Ceiling on one answer. Headroom for a full batch of eight (measured at ~456
# output tokens a session, so ~3,650 plus the model's brief reasoning); a unit
# with unusually rich key points can run longer, and being cut off mid-JSON is
# dearer than a cap that is never reached. Groq bills what is generated rather
# than reserving this, so a generous ceiling costs no allowance.
MAX_OUTPUT_TOKENS = 8000

# Batches that may be in flight at once. The token allowance is per MODEL, and
# it refills continuously rather than in one lump at the top of the minute, so
# firing the batches together spends it as it arrives instead of leaving the
# connection idle between calls. Measured on a real 11-session plan: 142s in a
# row, 6s together. Three, because a fourth only earns a 429 and the backoff
# that follows costs more than the overlap saves.
LP_MAX_PARALLEL = 3

# Reasoning models on Groq think in billed completion tokens before they answer,
# and on this task the thinking is dead weight: measured over one real batch,
# "low" returned MORE prose (1032 words vs 930) in 28% fewer tokens (2568 vs
# 3577) and a quarter less time. Tokens are the binding limit here, so cutting
# them is a speed-up twice over - each call is quicker, and more calls fit in a
# minute's allowance. Only sent to families that accept the field; anything else
# would answer a 400.
REASONING_EFFORT = "low"
_RE_REASONING_MODEL = re.compile(r"gpt-oss|qwen3|deepseek-r1", re.I)


class AIError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Response schema (forces valid structured JSON)
# --------------------------------------------------------------------------- #
def _strict(schema: dict) -> dict:
    """Make a schema satisfy strict structured-output rules, in place.

    A strict schema is accepted only if every object forbids extra properties
    and requires every property it declares. Applying that here keeps the
    schema definitions themselves readable, and keeps the rule in one place.
    """
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "object":
        props = schema.get("properties") or {}
        schema["additionalProperties"] = False
        schema["required"] = list(props)
        for value in props.values():
            _strict(value)
    elif schema.get("type") == "array":
        _strict(schema.get("items") or {})
    return schema


def _response_schema() -> dict:
    """The Learning-Plan rows, wrapped in an object.

    The rows are an array, but a strict schema's ROOT must be an object - so
    they travel under a `sessions` key, which `call_model` unwraps. Groq
    rejects a top-level array outright.

    The generative fields are STRUCTURED rather than pre-formatted strings.
    Asking the model for "a CAPITALISED heading then bullets" and then having
    doc_builder re-detect that formatting by predicate (`_keypoint_bold` bolds
    a line only if it is >=85% uppercase) means a heading the model
    under-capitalised renders silently unbolded. With {heading, points} the
    formatting is ours to apply and cannot drift - see `_flatten_key_points`.

    week / session_no / is_cat are absent on purpose: the deterministic
    skeleton always wins on those, `session_id` carries the identity, and every
    field dropped is tokens saved and one less key the model can get wrong.
    """
    str_array = {"type": "array", "items": {"type": "string"}}
    return _strict({
        "type": "object",
        "properties": {
            "sessions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "session_title": {"type": "string"},
                        "learning_outcomes": {
                            "type": "object",
                            "properties": {
                                "stem": {"type": "string"},
                                "items": str_array,
                            },
                        },
                        "key_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "heading": {"type": "string"},
                                    "points": str_array,
                                },
                            },
                        },
                        "trainee_activities": str_array,
                        "resources": str_array,
                        "assessments": {
                            "type": "object",
                            "properties": {
                                "knowledge_checks": str_array,
                                "skills": str_array,
                                "attitudes": str_array,
                            },
                        },
                    },
                },
            },
        },
    })


# The three assessment groups, in the order the document renders them, paired
# with the heading `_assessment_bold` in doc_builder looks for.
_ASSESSMENT_GROUPS = (("knowledge_checks", "Knowledge Checks:"),
                      ("skills", "Skills:"),
                      ("attitudes", "Attitudes:"))


def _flatten_learning_outcomes(value) -> List[str]:
    """{stem, items} -> the stem on its own line, then a./b./c."""
    if not isinstance(value, dict):
        return _as_list(value)
    stem = str(value.get("stem") or "").strip()
    items = _as_list(value.get("items"))
    return ([stem] if stem else []) + items


def _flatten_key_points(value) -> List[str]:
    """[{heading, points}] -> a CAPITALISED heading, then '- ' bullets.

    The upper() here is what guarantees doc_builder bolds the heading; before
    this the model had to remember to capitalise, and sometimes did not.
    """
    if not isinstance(value, list) or not any(isinstance(b, dict) for b in value):
        return _as_list(value)
    out: List[str] = []
    for block in value:
        if not isinstance(block, dict):
            out.extend(_as_list(block))
            continue
        heading = str(block.get("heading") or "").strip()
        if heading:
            out.append(heading.upper())
        for point in _as_list(block.get("points")):
            out.append(point if point.startswith("- ") else "- " + point)
    return out


# A leading "1. " / "2) " the model may or may not have supplied.
_RE_ITEM_NUMBER = re.compile(r"^\s*\d+\s*[.)]\s*")


def _flatten_assessments(value) -> List[str]:
    """{knowledge_checks, skills, attitudes} -> grouped, headed, numbered lines.

    The numbering is applied here rather than asked for. Whether an item says
    "1. " is presentation, like the CAPITALISED key-point heading, and leaving
    presentation to the model means checking up on it afterwards - measured
    against real output, every single row came back unnumbered.
    """
    if not isinstance(value, dict):
        return _as_list(value)
    if not any(key in value for key, _ in _ASSESSMENT_GROUPS):
        return _as_list(value)
    out: List[str] = []
    for key, heading in _ASSESSMENT_GROUPS:
        items = _as_list(value.get(key))
        if not items:
            continue
        out.append(heading)
        for n, item in enumerate(items, start=1):
            out.append(f"{n}. {_RE_ITEM_NUMBER.sub('', str(item)).strip()}")
    return out



# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# The standing instructions. Identical on every Learning-Plan call, so they live
# in their own message ahead of anything that varies. Groq's prefix cache can
# then recognise them, and cached tokens do not count against the 8,000
# tokens-a-minute limit - but measured on this account the cache hits about one
# warm call in five and tops out near 768 tokens, so treat that as a small
# bonus. The reason for the split is that instructions and data are different
# things and separating them keeps both readable.
#
# CAT sessions are deliberately NOT described here: they never reach the model.
# `merge_ai_into_sessions` rebuilds every CAT row deterministically from the
# sessions it assesses, so asking for one would be paying for discarded output.
LP_SYSTEM = """You are a senior TVET trainer and industry expert in Kenya, producing Learning Plans for units of competency under the TVET CDACC framework.

You are GIVEN the unit, the term schedule, and - for each session - the official Curriculum Learning Key Points. You do NOT invent syllabus content. You repair, organise and expand ONLY from the sources supplied in the user message.

SOURCES OF TRUTH, in priority order
1. The session's supplied key_points (authoritative - preserve their meaning).
2. The unit's Performance Criteria.
3. The unit's Required Knowledge.
4. The unit's Evidence-Guide assessment methods.
Nothing else. If a fact, tool or topic is not present in or clearly implied by sources 1-4, it does not appear in your output. If one of these sources arrives empty, work from the ones that are present and write less, rather than filling the gap from general knowledge.

SOURCE DATA QUALITY - REPAIR RULES
The supplied titles and key points are auto-extracted and may be messy: fragments, truncated phrases, bad casing, duplicates, shallow stubs.
- Repair form, preserve meaning. Fix grammar, spelling, capitalisation and spacing, and complete an obviously truncated sentence - but ONLY when the intended meaning is clear from the fragment itself, the session's other key points, the Performance Criteria, or the Required Knowledge. This is editing, not authoring: never add a topic, tool or fact not already implied by those sources.
- Thin points: where a point is too shallow to facilitate from, add concrete sub-points drawn ONLY from the supplied Performance Criteria and Required Knowledge.
- Restated criteria: if a session's supplied key points only restate the performance criterion, draw concrete sub-points from the Required Knowledge topics for this unit.
- Unrecoverable items: if a fragment cannot be reconstructed from the supplied sources, omit it. Never output a dangling phrase, an incomplete sentence, an ellipsis, or a placeholder.
- Duplicates: merge near-identical supplied points into one.
- Every line you output is a complete, coherent sentence traceable to sources 1-4.

TERMINOLOGY (Kenya CBET) - MANDATORY
Use: trainee, trainer, session, facilitate, unit of competency, learning outcome, performance criteria, competency, assessment, Continuous Assessment Test (CAT).
Never use: student, pupil, learner, teacher, lecturer, instructor, lesson, lecture, class (as a synonym for session), teach, deliver a lecture, exam, quiz, or test as a noun for the final assessment.
Spelling: British / Kenyan English - organise, practise (as a verb), programme (a course of study), labelled, capitalised, centred.

FIELD RULES
session_id: echo the supplied value VERBATIM. Never alter, renumber or omit it, and never return a session that was not supplied.

session_title: the supplied title, repaired per SOURCE DATA QUALITY. Same meaning, same topic - never renamed. Title Case, no trailing punctuation, at most 12 words.

learning_outcomes.stem: exactly "By the end of the session, the trainee should be able to;"
learning_outcomes.items: exactly three strings, prefixed "a. ", "b. ", "c. ". Each is trainee-centred and rewritten from the performance criteria this session addresses. Open each with a DIFFERENT verb from the level-appropriate list supplied in the user message. Banned openers, because they cannot be assessed: understand, know, learn, appreciate, be aware of, be familiar with, grasp. At most 20 words each.

key_points: two or three blocks. Each heading is at most 5 words, drawn from the supplied content. Each block holds 2-4 points; each point is one complete sentence of at most 20 words, traceable to sources 1-4. Do not introduce a topic that is not in the supplied content.

trainee_activities: EXACTLY five strings, in this order.
1. An understanding-level activity.
2. An analysis or discussion activity.
3. A hands-on application activity.
4. The literal string "Follow up Activity:"
5. One assignment, prefixed "1. ".
Each of the first three starts "- ", NAMES an active-learning method, then describes in 1-2 sentences exactly what the trainee does - referencing this session's specific key points: the actual topic, task, tool, sample or scenario. Never generic filler. At most 40 words each.
Method vocabulary: Group Discussion, Think-Pair-Share, Case Study, Jigsaw, Peer Teaching, Round Robin, Demonstration with Participation, KWL Chart, Concept Mapping, Brainstorming, Gallery Walk, Role Play, Guided Practice, Problem-Based Learning, Fishbowl, Practical Exercise.
Vary the methods across the sessions in one response: no method in more than a third of them, and no two consecutive sessions opening with the same method.
Weak: "- Group Discussion: Trainees discuss the topic in groups and present findings."
Strong: "- Think-Pair-Share: Each trainee lists three differences between a compiler and an interpreter, compares the list with a partner, then the pair reports one agreed difference to the room."

resources: two to four real, locatable items relevant to THIS session - textbooks, official documentation, standards, tools or presentations, including the TVET CDACC curriculum and trainer's guide for the unit. Prefer what a Kenyan polytechnic would realistically have. Never invent ISBNs, page numbers, edition years or URLs; if you are unsure of a URL, name the resource without one.

assessments: one to three items per group, derived from the supplied Evidence-Guide methods. Write each as a plain sentence; do not number them.
- knowledge_checks: what the trainee is asked about this session's key points.
- skills: what the trainee is observed doing.
- attitudes: observable professional behaviours shown during this session's own activities - accuracy, safety, teamwork, timeliness, adherence to procedure. Observable, never internal states.

Return ONE JSON object and nothing else."""


def build_prompt(unit: Unit, sessions: List[Session]) -> str:
    """The per-unit half of the request: data only, no instructions.

    Everything standing lives in LP_SYSTEM. Keeping this half free of
    instructions is what makes separating the two worth doing.
    """
    pcs = "\n".join(f"{pc.number} {pc.text}" for pc in unit.all_pcs)
    methods = "; ".join(unit.assessment_methods) or "Observation; Oral assessment; " \
        "Written assessment; Practical assessment; Portfolio of evidence"
    knowledge = "; ".join(unit.required_knowledge)
    skeleton = [s.to_skeleton_dict() for s in sessions]
    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=1)
    level = unit.level or "6"

    verbs = "Identify, Explain, Apply, Demonstrate, Evaluate, Implement" \
        if str(level) >= "6" else "Identify, Explain, Apply, Demonstrate"

    return f"""UNIT: {unit.unit_title}
CODE: {unit.os_code}
LEVEL: {level}

LEVEL-APPROPRIATE VERBS:
{verbs}

OS PERFORMANCE CRITERIA:
{pcs}

OS EVIDENCE-GUIDE ASSESSMENT METHODS: {methods}

OS REQUIRED KNOWLEDGE: {knowledge or "(none listed)"}

SESSIONS (key_points are AUTHORITATIVE - preserve their meaning):
{skeleton_json}

Return the JSON object now."""


# --------------------------------------------------------------------------- #
# HTTP call
# --------------------------------------------------------------------------- #
# How long to wait after each HTTP 429, in order. Groq's tightest limit is per
# minute (8,000 tokens on the gpt-oss models), so these are generous enough to
# clear it and bounded enough that a real exhaustion reports back inside a
# minute.
_RATE_LIMIT_BACKOFF = (3.0, 8.0, 15.0, 30.0)
_RATE_LIMIT_BUDGET = sum(_RATE_LIMIT_BACKOFF)
# Never sit on a server-supplied Retry-After longer than this.
_MAX_RETRY_AFTER = 60.0


class _ModelUnavailable(Exception):
    """This model can't do the job - excluded by the plan, or unable to answer.

    Internal to this module: the caller moves on to the next candidate.
    """

    def __init__(self, model: str, reason: str):
        super().__init__(f"{model}: {reason}")
        self.model = model
        self.reason = reason


# Groq states a reset as a duration: '7.66s', '2m59.56s', '1h2m3.5s'.
_RE_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?"
                          r"(?:(\d+(?:\.\d+)?)m?s)?$")


def _parse_duration(raw: str) -> float:
    """Seconds from '7.66s' / '2m59.56s' / '90' , or 0.0 when unreadable."""
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:                                  # a bare number of seconds
        return float(text)
    except ValueError:
        pass
    m = _RE_DURATION.fullmatch(text)
    if not m or not any(m.groups()):
        return 0.0
    hours, minutes, seconds = (float(g or 0) for g in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def _wait_wanted(resp) -> float:
    """How long the server says to wait, in seconds, uncapped.

    Retry-After first. Failing that, Groq's x-ratelimit-* headers - but only
    the reset of a limit that has actually run out. Every response carries a
    reset for BOTH the per-minute tokens (seconds away) and the per-day
    requests (hours away), so taking the longest would read a busy minute as a
    day's allowance gone.
    """
    headers = getattr(resp, "headers", None) or {}
    wait = _parse_duration(headers.get("Retry-After", ""))
    if wait:
        return wait
    waits = []
    for key, value in headers.items():
        low = key.lower()
        if "ratelimit-remaining" not in low:
            continue
        if _parse_duration(value) > 0:            # something left of this one
            continue
        reset = headers.get(low.replace("remaining", "reset"), "") or \
            headers.get(key.replace("remaining", "reset"), "")
        waits.append(_parse_duration(reset))
    return max(waits or [0.0])


def _is_exhausted(resp) -> bool:
    """Whether this 429 is an allowance used up rather than a busy minute.

    Groq's free tier is 1,000 requests a DAY per model on top of the per-minute
    limits, and the day's reset is hours away. Sitting through the backoff for
    that wastes a minute and then fails anyway - the useful move is to let the
    caller try a different model, which has its own daily allowance.
    """
    return _wait_wanted(resp) > _RATE_LIMIT_BUDGET


def _retry_after(resp) -> float:
    """The server's own wait in seconds, capped at what we will sit through."""
    return max(0.0, min(_wait_wanted(resp), _MAX_RETRY_AFTER))


def _post(model: str, api_key: str, prompt: str, timeout: int = 180,
          schema: Optional[dict] = None,
          schema_name: str = "learning_plan_sessions",
          temperature: float = 0.2,
          max_tokens: int = MAX_OUTPUT_TOKENS,
          system: str = "") -> dict:
    """POST one chat completion, always in strict JSON-schema mode.

    Every call this module makes wants structured output, the probe included -
    a model that chats but won't take the schema is no use here.

    `system` carries the standing instructions, which are identical on every
    call. Keeping them in their own message, ahead of anything that varies,
    is what lets Groq's prefix cache recognise them - see the note on
    LP_SYSTEM for how much that is actually worth.
    """
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema if schema is not None else _response_schema(),
                "strict": True,
            },
        },
    }
    if _RE_REASONING_MODEL.search(model):
        body["reasoning_effort"] = REASONING_EFFORT
    resp = requests.post(GROQ_ENDPOINT,
                         headers={
                             "Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json",
                         },
                         json=body, timeout=timeout)
    return resp


def _extract_text(payload: dict) -> str:
    """Pull the text out of a chat-completions response, defensively.

    The API can return a 200 with no usable assistant content, for example an
    empty choices list or a choice that only carries a finish reason. Each of
    those raises AIError so the caller can surface the error immediately.
    """
    choices = payload.get("choices") or []
    if not choices:
        raise AIError("Groq returned no choices: " + json.dumps(payload)[:300])
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)
    if not text:
        reason = choice.get("finish_reason") or "no assistant content"
        raise AIError(f"Groq returned an empty response (finish_reason: {reason})")
    return text


# The fields a session row may carry. Anything else a model invents is stray.
_ROW_FIELDS = ("session_id", "session_title", "learning_outcomes",
               "key_points", "trainee_activities", "resources", "assessments")
_ROW_FIELD_BY_SHAPE = {re.sub(r"[^a-z0-9]", "", f): f for f in _ROW_FIELDS}


def _rehome_stray_keys(row: dict) -> dict:
    """Put content a model filed under its own invented key back in the row.

    Models sometimes answer the "add a Follow up Activity line" instruction with
    a `follow_up_activity` FIELD rather than another string in the array. The
    writing is good - only its address is wrong - so a key that is a known field
    under another spelling is merged into that field, and anything else is
    appended to trainee_activities under its own heading, which is where the
    only instruction that invites extra lines puts them.
    """
    if not isinstance(row, dict):
        return row
    kept = {k: v for k, v in row.items() if k in _ROW_FIELDS}
    for key, value in row.items():
        if key in _ROW_FIELDS:
            continue
        target = _ROW_FIELD_BY_SHAPE.get(re.sub(r"[^a-z0-9]", "", str(key).lower()))
        if target:
            kept.setdefault(target, [])
            kept[target] = _as_list(kept[target]) + _as_list(value)
            continue
        heading = str(key).replace("_", " ").strip().title() + ":"
        kept["trainee_activities"] = (_as_list(kept.get("trainee_activities", []))
                                      + [heading] + _as_list(value))
    return kept


def _salvage_rejected_generation(resp):
    """Rows from a 400 that refused an otherwise complete answer, or None.

    Only `json_validate_failed` carries a generation; every other 400 (a model
    that won't take the schema at all, a malformed request) genuinely has
    nothing to recover and must fall through to the caller's handling.
    """
    try:
        error = (resp.json() or {}).get("error") or {}
    except ValueError:
        return None
    if error.get("code") != "json_validate_failed":
        return None
    try:
        data = json.loads(error.get("failed_generation") or "")
    except (ValueError, TypeError):
        return None
    rows = data.get("sessions") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return None
    return [_rehome_stray_keys(row) for row in rows]


def _salvage_json_array(text: str):
    """Recover the complete leading objects of a truncated JSON array.

    A response cut off by the token limit looks like
    '[{...},{...},{"x":"unterminated...'. We keep every top-level object that
    closed cleanly and drop the trailing partial one, so a truncated
    learning-plan response still yields usable rows (the deterministic merge
    backfills any sessions the model never reached). Returns a list, or None if
    nothing salvageable is present.

    A smaller model also breaks JSON in the middle of an otherwise complete
    array - an unescaped quote inside a string, say. Taking only the longest
    prefix threw away the good rows in front of the bad one, so each object
    boundary is tried in turn, longest first.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    closes = []                     # every '}' that closed a top-level object
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                closes.append(i)
    for close in reversed(closes):
        candidate = text[start:close + 1].rstrip().rstrip(",") + "]"
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue                # this object is the broken one; try the previous
        if isinstance(data, list):
            return data
    return None


def _emit_progress(progress_cb, message: str) -> None:
    runlog.log(message)
    if progress_cb is not None:
        try:
            progress_cb(message)
        except Exception:  # noqa: BLE001 - never break generation because UI logging failed
            pass


# Models this workspace has already been refused, so a Learning Plan of eight
# batches plus a Session Plan each doesn't re-discover the same refusal every
# call. Process-lived: upgrade the plan and restart the app to try again.
_UNAVAILABLE_MODELS: set = set()


def _chat_json(prompt: str, api_key: str, model: str, schema: dict,
               schema_name: str, progress_cb=None, temperature: float = 0.2,
               system: str = ""):
    """One grounded request, on a model this workspace can actually call.

    The model is settled before any of this by `resolve_model`. If it stops
    answering mid-run - a plan changing under us - the refusal is recorded and
    the next-best model is resolved and used, rather than failing the plan.
    """
    # A model ruled out on an earlier call must not be tried again: every batch
    # of a Learning Plan comes through here, and re-discovering the same hang
    # would cost the timeout over and over.
    if model in _UNAVAILABLE_MODELS:
        model = resolve_model(api_key, progress_cb=progress_cb)

    tried: List[str] = []
    last_reason = ""
    for _ in range(MODEL_ATTEMPTS):
        try:
            return _chat_json_once(prompt, api_key, model, schema,
                                   schema_name, progress_cb=progress_cb,
                                   temperature=temperature, system=system)
        except _ModelUnavailable as e:
            _UNAVAILABLE_MODELS.add(e.model)
            tried.append(e.model)
            last_reason = e.reason
            message = f"AI: {e.model} {e.reason}; trying another model"
            _emit_progress(progress_cb, message)
            runlog.log(message)
            try:
                model = resolve_model(api_key, progress_cb=progress_cb)
            except AIError:
                break

    raise AIError(
        "No Groq model completed the request. Tried %s; the last one %s. "
        "Check the key at console.groq.com/keys, or set GROQ_MODEL in .env to "
        "a model your account allows."
        % (", ".join(tried) or model, last_reason or "failed"))


def _chat_json_once(prompt: str, api_key: str, model: str, schema: dict,
                    schema_name: str, progress_cb=None, temperature: float = 0.2,
                    system: str = ""):
    """One grounded request to ONE model, retrying network errors and rate limits.

    Returns the parsed JSON exactly as the model produced it (a list or a dict);
    callers coerce it to the shape they expect. Raises `_ModelUnavailable` when
    the workspace's plan doesn't include this model, so the caller can move on
    to one it does.
    """
    _emit_progress(progress_cb, f"AI: trying model {model}")
    net_attempt = 0
    waited = 0
    timeouts = 0
    proven = model in _PROVEN_MODELS
    while True:
        try:
            resp = _post(model, api_key, prompt, schema=schema, system=system,
                         schema_name=schema_name, temperature=temperature,
                         timeout=REQUEST_TIMEOUT if proven else UNPROVEN_TIMEOUT)
        except requests.Timeout as e:
            # Some models take a short prompt happily and then never finish a
            # real one - ministral-14b answers a probe in a second and hangs on
            # a full batch under the same schema. A model that has never
            # delivered and keeps timing out is not working; one that has been
            # delivering all along has hit a blip and deserves the retry.
            if not proven:
                timeouts += 1
                if timeouts >= UNPROVEN_ATTEMPTS:
                    raise _ModelUnavailable(
                        model, "didn't answer in time, twice running") from None
                _emit_progress(progress_cb,
                               f"AI: {model} didn't answer in time; trying it "
                               f"once more")
                continue
            net_attempt += 1
            if net_attempt < 3:
                _emit_progress(progress_cb,
                               f"AI: transient network issue for {model}, "
                               f"retrying ({net_attempt}/3)")
                time.sleep(net_attempt * 0.5)
                continue
            raise AIError(f"{model}: network error: {e}") from e
        except requests.ConnectionError as e:
            net_attempt += 1
            if net_attempt < 3:
                _emit_progress(progress_cb,
                               f"AI: transient network issue for {model}, "
                               f"retrying ({net_attempt}/3)")
                time.sleep(net_attempt * 0.5)
                continue
            # Three dropped connections in a row: move to another model rather
            # than lose the batches already generated. If the network itself is
            # down the next model fails the same way, and the caller says so.
            raise _ModelUnavailable(
                model, f"couldn't be reached ({type(e).__name__})") from None
        except requests.RequestException as e:
            raise AIError(f"{model}: network error: {e}") from e

        if resp.status_code == 200:
            _emit_progress(progress_cb, f"AI: {model} answered")
            payload = resp.json()
            try:
                text = _extract_text(payload)
            except AIError as e:
                raise AIError(f"{model}: {e}") from e
            _PROVEN_MODELS.add(model)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                # The response was cut off mid-JSON (usually the token limit).
                # Salvage the complete rows if it is an array; the caller's merge
                # backfills anything missing.
                salvaged = _salvage_json_array(text)
                if salvaged is not None:
                    _emit_progress(progress_cb,
                                   f"AI: {model} returned unusable JSON; "
                                   f"salvaged {len(salvaged)} complete rows")
                    return salvaged
                reason = (payload.get("choices") or [{}])[0].get("finish_reason")
                hint = (" (the response hit the length limit)"
                        if reason == "length" else "")
                raise AIError(f"{model} returned invalid JSON: {e}{hint}") from e

        # Groq validates the finished answer against the schema rather than
        # constraining every token, so a model can generate a perfectly good
        # plan and still be refused for one stray key - and it hands the whole
        # generation back in `failed_generation`. Throwing that away would cost
        # the batch AND the model (a 400 otherwise rules a model out), so the
        # rows are recovered and the stray keys put back where they belong.
        if resp.status_code == 400:
            salvaged = _salvage_rejected_generation(resp)
            if salvaged is not None:
                _PROVEN_MODELS.add(model)
                _emit_progress(progress_cb,
                               f"AI: {model} answered but named a field the "
                               f"schema doesn't have; recovered its content")
                return salvaged

        # A model this account may not call, or one that won't take the schema,
        # answers 400/404. The key is fine, so this is not an auth failure -
        # it means "not this model".
        if resp.status_code in (400, 404):
            raise _ModelUnavailable(
                model, f"wouldn't take the request ({resp.text[:120]})")

        if resp.status_code in (401, 403):
            raise AIError(
                "Groq auth failed (HTTP %d). Your GROQ_API_KEY is invalid, "
                "expired, or lacks access." % resp.status_code)

        # A Learning Plan is several calls in a row and a Session Plan is one
        # more each, so the per-minute token limit is met routinely - and it
        # clears in seconds. Failing the whole generation on it threw away
        # every batch already produced.
        if resp.status_code == 429:
            # The free tier also caps requests per DAY, per model. That reset
            # is hours away, so sitting through the backoff wastes a minute and
            # fails anyway; another model has its own daily allowance.
            if _is_exhausted(resp):
                raise _ModelUnavailable(
                    model, f"has used up its free allowance "
                           f"(resets in {_wait_wanted(resp) / 60:.0f} min)")
            if waited < len(_RATE_LIMIT_BACKOFF):
                pause = _retry_after(resp) or _RATE_LIMIT_BACKOFF[waited]
                waited += 1
                _emit_progress(progress_cb,
                               f"AI: {model} is rate limited; waiting {pause:.0f}s "
                               f"before retry {waited}/{len(_RATE_LIMIT_BACKOFF)}")
                time.sleep(pause)
                continue
            raise AIError(
                f"{model}: still rate limited after {len(_RATE_LIMIT_BACKOFF)} "
                f"retries over {int(sum(_RATE_LIMIT_BACKOFF))}s (HTTP 429). Give "
                f"it a minute, then generate again. {resp.text[:160]}")

        raise AIError(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")


def call_model(prompt: str, api_key: str, model: str,
                 progress_cb=None) -> List[dict]:
    """The Learning-Plan call: returns the JSON array of session rows."""
    data = _chat_json(prompt, api_key, model, _response_schema(),
                      "learning_plan_sessions", progress_cb=progress_cb,
                      system=LP_SYSTEM)
    if isinstance(data, dict):
        data = data.get("sessions") or data.get("data") or [data]
    if not isinstance(data, list):
        raise AIError(f"{model} did not return a JSON array")
    # Not "{model} succeeded" - a fallback may have answered instead, and the
    # log claiming the model that refused had produced the rows was misleading.
    _emit_progress(progress_cb, f"AI: {len(data)} session rows returned")
    return data


# --------------------------------------------------------------------------- #
# Safety nets - coerce + re-stamp + backfill
# --------------------------------------------------------------------------- #
def _as_list(value) -> List[str]:
    """Normalise anything (str / dict / list / None) into a clean list of strings.

    Guards against the char-splitting bug: a bare string must become a ONE-element
    list, never be iterated character-by-character.
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # split on newlines if the model crammed multiple lines into one string
        parts = [p.strip() for p in s.split("\n") if p.strip()]
        return parts or [s]
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, (list, tuple)):
                out.append(f"{k}: " + "; ".join(map(str, v)))
            else:
                out.append(f"{k}: {v}")
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple, dict)):
                out.extend(_as_list(item))
            else:
                # An element may itself hold several lines: gpt-oss returns a
                # key point as "HEADING:\n- one\n- two" where another model
                # returns three elements. The document renders one paragraph
                # per element, and a newline inside one collapses to a space
                # in Word, so split here and both read the same.
                for line in str(item).split("\n"):
                    t = line.strip()
                    if t:
                        out.append(t)
        return out
    return [str(value).strip()]


def _default_learning_outcomes(s: Session) -> List[str]:
    labels = "abc"
    outs = []
    for i, pc in enumerate(s.pcs[:3] or [s.session_title]):
        text = pc.split(" ", 1)[1] if pc[:3].replace(".", "").isdigit() else pc
        outs.append(f"{labels[i]}. Demonstrate the ability to {text.lower().rstrip('.')}.")
    if not outs:
        outs = [f"a. Demonstrate competence in {s.session_title.lower()}."]
    return ["By the end of the session, the trainee should be able to:"] + outs


def _default_activities(s: Session) -> List[str]:
    # CAT rows are handled deterministically before this point; this default
    # only ever fills content (non-CAT) sessions.
    return [
        "- Engage in a Group Discussion to explore the topic.",
        "- Participate in a Think-Pair-Share on key concepts.",
        "- Conduct a Case Study applying the concepts.",
        "Follow up Activity:",
        f"1. Complete a practical exercise on {s.session_title.lower()}. (15 Marks)",
        "Due date:",
    ]


def _default_assessments(unit: Unit, s: Session) -> List[str]:
    methods = unit.assessment_methods or ["Oral assessment", "Written assessment",
                                          "Practical assessment"]
    out = ["Knowledge Checks:"]
    out += [f"{i+1}. {m}." for i, m in enumerate(methods[:2])]
    out += ["Attitudes:", "1. Honesty and integrity.", "2. Self-reflection on progress."]
    return out


def _default_resources(s: Session) -> List[str]:
    # CAT rows are handled deterministically before this point; this default
    # only ever fills content (non-CAT) sessions.
    return [f"- Textbooks on {s.session_title}", "- PowerPoint presentations on the topic",
            "- Relevant online documentation"]


# --------------------------------------------------------------------------- #
# The self-check
#
# The prompt could ask the model to verify its own work, but it is run at
# REASONING_EFFORT "low" precisely so it does not spend tokens thinking - so a
# request to self-check is a request it is configured not to honour. Every rule
# worth checking is mechanical, so it is checked here instead: free, repeatable,
# and able to say exactly what is wrong. What fails goes back to the model once,
# named, rather than being silently patched or silently shipped.
# --------------------------------------------------------------------------- #

# Openers that cannot be assessed, so cannot begin a learning outcome.
_BANNED_OPENERS = ("understand", "know", "learn", "appreciate", "be aware of",
                   "be familiar with", "grasp")

# Words the Kenya CBET register does not use. Matched whole-word, case
# insensitively. "test" is deliberately absent: "Think-Pair-Share" and "testing
# a program" are legitimate, and the noun sense is not separable by regex.
_BANNED_TERMS = ("student", "students", "pupil", "pupils", "learner", "learners",
                 "teacher", "teachers", "lecturer", "lecturers", "instructor",
                 "instructors", "lesson", "lessons", "lecture", "lectures",
                 "exam", "exams", "quiz", "quizzes")
_RE_BANNED_TERM = re.compile(r"\b(?:%s)\b" % "|".join(_BANNED_TERMS), re.I)

# The active-learning methods the prompt offers; an activity should name one.
_METHODS = ("Group Discussion", "Think-Pair-Share", "Case Study", "Jigsaw",
            "Peer Teaching", "Round Robin", "Demonstration with Participation",
            "Demonstrations with Participation", "KWL Chart", "KWL",
            "Concept Mapping", "Brainstorming", "Gallery Walk", "Role Play",
            "Guided Practice", "Problem-Based Learning", "Fishbowl",
            "Practical Exercise")

_FOLLOW_UP_LINE = "Follow up Activity:"
_OUTCOME_LABELS = ("a.", "b.", "c.")
_RE_PLACEHOLDER = re.compile(r"\.\.\.|\u2026|<[^>]+>|\bTBD\b|\bN/?A\b|"
                             r"\bkey concept \d|\blorem\b", re.I)


# Models write "Think\u2011Pair\u2011Share" with a non-breaking hyphen as often as
# a plain one, and a naive match then reports a named method as missing.
_TYPOGRAPHY = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
               "\u2014": "-", "\u2212": "-", "\u00ad": "-", "\u00a0": " ",
               "\u2019": "'"}


def _plain(text: str) -> str:
    """Lower-cased, with fancy dashes and spaces flattened, for matching."""
    out = str(text)
    for fancy, plain in _TYPOGRAPHY.items():
        out = out.replace(fancy, plain)
    return out.lower()


def _words(text: str) -> int:
    return len(str(text).split())


def _check_row(row: dict) -> List[str]:
    """Everything wrong with one generated session, in plain words."""
    problems: List[str] = []

    title = str(row.get("session_title") or "").strip()
    if not title:
        problems.append("session_title is empty")
    elif _words(title) > 12:
        problems.append(f"session_title is {_words(title)} words; the limit is 12")

    lo = row.get("learning_outcomes")
    items = _as_list(lo.get("items")) if isinstance(lo, dict) else _as_list(lo)
    if len(items) != 3:
        problems.append(f"learning_outcomes.items has {len(items)} entries; "
                        f"it must have exactly 3")
    for n, item in enumerate(items[:3]):
        text = str(item).strip()
        if not text.lower().startswith(_OUTCOME_LABELS[n]):
            problems.append(f"learning outcome {n + 1} must start "
                            f"'{_OUTCOME_LABELS[n]} '")
        body = text[2:].strip().lower()
        opener = next((b for b in _BANNED_OPENERS if body.startswith(b)), "")
        if opener:
            problems.append(f"learning outcome {n + 1} opens with '{opener}', "
                            f"which cannot be assessed")
        if _words(text) > 20:
            problems.append(f"learning outcome {n + 1} is {_words(text)} words; "
                            f"the limit is 20")
    verbs = [str(i)[2:].strip().split(" ")[0].lower() for i in items[:3]
             if len(str(i)) > 2]
    if len(set(verbs)) < len(verbs):
        problems.append("two learning outcomes open with the same verb")

    blocks = row.get("key_points")
    blocks = blocks if isinstance(blocks, list) else []
    if not 2 <= len(blocks) <= 3:
        problems.append(f"key_points has {len(blocks)} block(s); "
                        f"it must have 2 or 3")
    for block in blocks:
        if not isinstance(block, dict):
            problems.append("a key_points block is not an object")
            continue
        heading = str(block.get("heading") or "").strip()
        if not heading:
            problems.append("a key_points block has no heading")
        elif _words(heading) > 5:
            # One strong noun - VULNERABILITIES - is a perfectly good heading,
            # and the model produces them; only rambling ones are a problem.
            problems.append(f"key_points heading '{heading}' is {_words(heading)} "
                            f"words; the limit is 5")
        points = _as_list(block.get("points"))
        if not 2 <= len(points) <= 4:
            problems.append(f"key_points block '{heading}' has {len(points)} "
                            f"points; it must have 2 to 4")
        for point in points:
            if _words(point) > 20:
                problems.append(f"a point under '{heading}' is {_words(point)} "
                                f"words; the limit is 20")

    acts = _as_list(row.get("trainee_activities"))
    if len(acts) != 5:
        problems.append(f"trainee_activities has {len(acts)} entries; it must "
                        f"have exactly 5 (3 activities, the follow-up line, "
                        f"the assignment)")
    else:
        if acts[3].strip() != _FOLLOW_UP_LINE:
            problems.append(f"trainee_activities[3] must be exactly "
                            f"'{_FOLLOW_UP_LINE}'")
        if not acts[4].strip().startswith("1. "):
            problems.append("trainee_activities[4], the assignment, must start '1. '")
    for n, act in enumerate(acts[:3]):
        if not any(_plain(m) in _plain(act) for m in _METHODS):
            problems.append(f"trainee activity {n + 1} names no active-learning "
                            f"method from the list")
        if _words(act) > 40:
            problems.append(f"trainee activity {n + 1} is {_words(act)} words; "
                            f"the limit is 40")

    res = _as_list(row.get("resources"))
    if not 2 <= len(res) <= 4:
        problems.append(f"resources has {len(res)} entries; it must have 2 to 4")

    assess = row.get("assessments")
    if not isinstance(assess, dict):
        problems.append("assessments must be grouped into knowledge_checks, "
                        "skills and attitudes")
    else:
        for key, _heading in _ASSESSMENT_GROUPS:
            group = _as_list(assess.get(key))
            if not 1 <= len(group) <= 3:
                problems.append(f"assessments.{key} has {len(group)} items; "
                                f"it must have 1 to 3")

    blob = json.dumps(row, ensure_ascii=False)
    banned = sorted({m.group(0).lower() for m in _RE_BANNED_TERM.finditer(blob)})
    if banned:
        problems.append("uses words the Kenya CBET register forbids: "
                        + ", ".join(banned))
    if _RE_PLACEHOLDER.search(blob):
        problems.append("contains an ellipsis, a placeholder or an unfilled slot")

    return problems


def _check_variety(rows: List[dict]) -> dict:
    """Method repetition across a batch, reported against the offending rows."""
    openers = []
    for row in rows:
        acts = _as_list(row.get("trainee_activities"))
        first = acts[0] if acts else ""
        openers.append(next((m for m in _METHODS
                             if _plain(m) in _plain(first)), ""))

    problems: dict = {}
    allowed = max(1, len(rows) // 3)
    counts: dict = {}
    for method in openers:
        if method:
            counts[method] = counts.get(method, 0) + 1

    for i, row in enumerate(rows):
        found = []
        method = openers[i]
        if method and counts.get(method, 0) > allowed:
            found.append(f"opens with '{method}', which opens "
                         f"{counts[method]} of {len(rows)} sessions in this "
                         f"batch; at most {allowed} may")
        if i and method and method == openers[i - 1]:
            found.append(f"opens with '{method}', the same method as the "
                         f"previous session")
        if found:
            problems.setdefault(_row_id(row, i), []).extend(found)
    return problems


def _row_id(row: dict, index: int) -> str:
    return str(row.get("session_id") or "").strip() or f"row {index + 1}"


def validate_rows(rows: List[dict]) -> dict:
    """{session_id: [what is wrong]} for every row that breaks a rule."""
    problems: dict = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not row:
            continue                       # an absent row is the merge's problem
        found = _check_row(row)
        if found:
            problems[_row_id(row, i)] = found
    for key, extra in _check_variety(
            [r for r in rows if isinstance(r, dict) and r]).items():
        problems.setdefault(key, []).extend(extra)
    return problems


def _repair_prompt(unit: Unit, sessions: List[Session], rows: List[dict],
                   problems: dict) -> str:
    """Ask for the failing sessions again, naming what was wrong with each."""
    wanted = {_row_id(r, i) for i, r in enumerate(rows)} & set(problems)
    failing = [s for s in sessions if s.session_id in wanted]
    faults = "\n".join(
        f"{sid}:\n" + "\n".join(f"  - {p}" for p in problems[sid])
        for sid in sorted(problems) if sid in wanted)
    return (build_prompt(unit, failing)
            + "\n\nYour previous answer for these sessions broke the field rules "
              "below. Return them again, corrected. Change only what is named; "
              "keep everything else as it was.\n\n" + faults)


def _repair_rows(unit: Unit, sessions: List[Session], rows: List[dict],
                 api_key: str, model: str, progress_cb=None) -> List[dict]:
    """One corrective round. Whatever is still wrong afterwards is logged.

    Bounded at a single retry on purpose: a second one costs another batch's
    tokens for diminishing returns, and the deterministic backfill downstream
    means a row that stays imperfect is still a usable row.
    """
    problems = validate_rows(rows)
    if not problems:
        return rows

    _emit_progress(progress_cb,
                   f"AI: {len(problems)} session(s) broke a field rule; "
                   f"asking for them again")
    runlog.log(f"AI: validation found {len(problems)} session(s) to repair: "
               + "; ".join(f"{k} ({len(v)})" for k, v in problems.items()))
    try:
        fixed = call_model(_repair_prompt(unit, sessions, rows, problems),
                           api_key, model, progress_cb=progress_cb)
    except AIError as e:
        runlog.error(f"AI: repair round failed, keeping the original rows: {e}")
        return rows

    by_id = {_row_id(r, i): r for i, r in enumerate(fixed) if isinstance(r, dict)}
    out = []
    for i, row in enumerate(rows):
        replacement = by_id.get(_row_id(row, i))
        out.append(replacement if replacement else row)

    left = validate_rows(out)
    if left:
        runlog.log("AI: still imperfect after one repair round: "
                   + "; ".join(f"{k}: {', '.join(v)}" for k, v in left.items()))
    return out


def merge_ai_into_sessions(sessions: List[Session], ai_rows: List[dict],
                           unit: Unit) -> List[Session]:
    """Apply AI output onto the deterministic skeleton, with full safety nets.

    The deterministic schedule (week / session_no / is_cat / pcs) ALWAYS wins;
    the AI only contributes the generative text columns. Missing fields are
    backfilled so no cell is ever blank.
    """
    # Rows are matched to sessions by the session_id the model echoes back, so a
    # short, reordered or duplicated answer can no longer slide content onto the
    # wrong session - the failure this used to guard against with padding. If a
    # model returns no usable ids at all, pair in order as before.
    rows_by_id = {str(r.get("session_id") or "").strip(): r
                  for r in ai_rows if isinstance(r, dict)}
    rows_by_id.pop("", None)
    generated = [i for i, sess in enumerate(sessions) if not sess.is_cat]
    by_position = {}
    if not rows_by_id:
        by_position = {i: ai_rows[pos] for pos, i in enumerate(generated)
                       if pos < len(ai_rows) and isinstance(ai_rows[pos], dict)}

    for i, s in enumerate(sessions):
        # CAT rows are deterministic - never the AI's output - but CONTEXTUAL:
        # they summarise the content sessions this CAT assesses (those since the
        # previous CAT), which are already filled earlier in this in-order pass.
        # They are never sent to the model at all; see `generate_sessions`.
        if s.is_cat:
            _apply_cat_content(sessions, i)
            continue

        row = rows_by_id.get(s.session_id) or by_position.get(i) or {}

        # session_title: accept the AI's data-quality-repaired title if given,
        # otherwise keep the deterministic skeleton title. Set before the
        # fallbacks below so they build on the cleaned title.
        ai_title = (row.get("session_title") or "").strip()
        if ai_title:
            s.session_title = ai_title

        lo = _flatten_learning_outcomes(row.get("learning_outcomes"))
        ai_kp = _flatten_key_points(row.get("key_points"))
        acts = _as_list(row.get("trainee_activities"))
        res = _as_list(row.get("resources"))
        assess = _flatten_assessments(row.get("assessments"))

        s.learning_outcomes = lo or _default_learning_outcomes(s)
        if ai_kp:
            s.key_points = ai_kp                       # AI already CAPS-formatted
        else:
            # Deterministic formatting keeps the curriculum key points usable
            # when the AI output does not supply a replacement.
            s.key_points = _format_curriculum_keypoints(s.key_points) \
                or [s.session_title.upper()]
        s.trainee_activities = acts if len(acts) >= 3 else _default_activities(s)
        s.resources = res if len(res) >= 2 else _default_resources(s)
        s.assessments = assess or _default_assessments(unit, s)
        # NOTE: week/session_no/is_cat/pcs are NOT touched -> deterministic wins.
    return sessions


def _cat_number(sessions: List[Session], idx: int) -> int:
    """1-based ordinal of the CAT at `idx` among all CAT rows."""
    return sum(1 for s in sessions[:idx + 1] if s.is_cat)


def _covered_sessions(sessions: List[Session], idx: int) -> List[Session]:
    """Content sessions this CAT assesses: those since the previous CAT."""
    covered: List[Session] = []
    for j in range(idx - 1, -1, -1):
        if sessions[j].is_cat:
            break
        covered.append(sessions[j])
    return list(reversed(covered))


def _cat_topic(session: Session) -> str:
    """A session title as a sentence-cased topic phrase (acronyms preserved).

    'Programming Language Types' -> 'programming language types';
    'Apply ICT Skills' -> 'apply ICT skills'. Used mid-sentence in the CAT's
    learning outcomes and activity line.
    """
    words = (session.session_title or "").strip().rstrip(".").split()
    out = []
    for w in words:
        if (len(w) > 1 and w.isupper()) or any(c.isdigit() for c in w):
            out.append(w)                 # keep acronyms / tokens with digits
        else:
            out.append(w.lower())
    return " ".join(out) or "the topic"


def _cat_topics_phrase(covered: List[Session]) -> str:
    topics = [_cat_topic(s) for s in covered if (s.session_title or "").strip()]
    if not topics:
        return "the learning outcomes covered so far"
    if len(topics) == 1:
        return topics[0]
    return ", ".join(topics[:-1]) + " and " + topics[-1]


def _cat_learning_outcomes(covered: List[Session]) -> List[str]:
    """One competency per covered session (what the trainee has learnt)."""
    labels = "abcdefghijklmnop"
    outs = ["By the end of the session, the trainee should be able to:"]
    for k, s in enumerate(covered[:len(labels)]):
        outs.append(f"{labels[k]}. Demonstrate competence in {_cat_topic(s)}.")
    if len(outs) == 1:
        outs.append("a. Demonstrate competence in the learning outcomes covered.")
    return outs


def _cat_keypoints(covered: List[Session]) -> List[str]:
    """'ASSESSMENT COVERAGE' heading + a bullet per covered topic."""
    out = ["ASSESSMENT COVERAGE"]
    for s in covered:
        title = (s.session_title or "").strip()
        if title:
            out.append(f"- {title}")
    if len(out) == 1:
        out.append("- The learning outcomes covered so far.")
    return out


def _cat_activities(cat_no: int, covered: List[Session]) -> List[str]:
    return [f"- Complete the Continuous Assessment Test (CAT {cat_no}) to evaluate "
            f"the trainee's competence in {_cat_topics_phrase(covered)}."]


def _cat_resources() -> List[str]:
    return ["- Assessment tool(s) and marking scheme/rubric",
            "- Observation checklist and assessor guide",
            "- Answer booklets and writing materials"]


def _cat_assessments() -> List[str]:
    return ["Knowledge Checks:",
            "1. Graded assessment of knowledge. and/or",
            "2. Evaluation of practical application in problem-solving.",
            "Attitudes:",
            "1. Honesty and integrity during assessment.",
            "2. Self-reflection on learning progress."]


def _apply_cat_content(sessions: List[Session], idx: int) -> None:
    """Fill a CAT row in place from the content sessions it assesses."""
    s = sessions[idx]
    covered = _covered_sessions(sessions, idx)
    s.learning_outcomes = _cat_learning_outcomes(covered)
    s.key_points = _cat_keypoints(covered)
    s.trainee_activities = _cat_activities(_cat_number(sessions, idx), covered)
    s.resources = _cat_resources()
    s.assessments = _cat_assessments()


def rebuild_cat_session(sessions: List[Session], idx: int) -> Session:
    """Public deterministic recompute of one CAT from its current siblings."""
    _apply_cat_content(sessions, idx)
    return sessions[idx]


def _format_curriculum_keypoints(points: List[str]) -> List[str]:
    """Deterministic CAPS-heading formatting for the offline (no-AI) path.

    Each curriculum content line becomes an uppercase heading; any parenthetical
    '(e.g., a, b, c)' enumerations are split out into '- ' bullets beneath it.
    """
    import re as _re
    out: List[str] = []
    for p in points:
        m = _re.search(r"\(e\.g\.?,?\s*(.+?)\)", p, _re.I)
        heading = _re.sub(r"\s*\(e\.g\.?.*?\)", "", p).strip()
        out.append(heading.upper())
        if m:
            for item in _re.split(r",| and ", m.group(1)):
                item = item.strip()
                if item:
                    out.append(f"- {item}")
    return out


# --------------------------------------------------------------------------- #
# Public orchestration
# --------------------------------------------------------------------------- #
def list_chat_models(api_key: str) -> List[str]:
    """Every model this key can see, one name per model.

    Groq's listing is flat - an id, and `active` saying whether it can be
    called at all. Retired models are dropped here so the model search never
    spends a probe on one.
    """
    try:
        resp = requests.get(GROQ_MODELS_ENDPOINT,
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=LIST_MODELS_TIMEOUT)
    except requests.RequestException as e:
        raise AIError(f"Couldn't list the Groq models: {e}") from None
    if resp.status_code != 200:
        raise AIError(f"Couldn't list the Groq models (HTTP "
                      f"{resp.status_code}). {resp.text[:200]}")

    names: List[str] = []
    seen: set = set()
    for entry in resp.json().get("data", []):
        name = entry.get("id", "")
        if not name or name in seen:
            continue
        if entry.get("active") is False:          # absent means callable
            continue
        seen.add(name)
        names.append(name)
    return names


def _model_rank(name: str) -> tuple:
    """Sort key: how well a model is likely to write a Learning Plan.

    Not a quality league table - just enough ordering to try the capable
    models first. A model the list knows nothing about sorts in the middle,
    ahead of the small ones, so a newly released model is tried on its own
    merits rather than ignored.
    """
    lower = (name or "").lower()
    base = float(len(_MODEL_FAMILIES)) / 2
    for i, family in enumerate(_MODEL_FAMILIES):
        if family in lower:
            base = float(i)
            break
    # within a family, more parameters first (gpt-oss-120b before -20b), then
    # the newer version (qwen3.8 before qwen3.6)
    size = _RE_MODEL_SIZE.search(lower)
    version = _RE_MODEL_VERSION.search(lower)
    return (base,
            -int(size.group(1)) if size else 0,
            -float(version.group(1)) if version else 0.0,
            name)


def _candidate_models(model: str, api_key: str) -> List[str]:
    """The configured model first, then every other one, most capable first."""
    try:
        discovered = list_chat_models(api_key)
    except AIError:
        discovered = []               # can't list: the configured model is all we have
    usable = [m for m in discovered
              if not _RE_SPECIAL_PURPOSE.search(m) and m != model]
    ordered = [model] + sorted(usable, key=_model_rank)
    # Ruling models out must never leave nothing to try: a dropped connection
    # can rule out the only model we know about AND make discovery come back
    # empty, and one more attempt beats failing the generation outright.
    return [m for m in ordered if m not in _UNAVAILABLE_MODELS] or ordered


def _answers(model: str, api_key: str) -> bool:
    """Whether `model` will take a structured request from this workspace.

    The probe uses JSON-schema mode because every real call does: only some of
    Groq's models do strict constrained decoding, and finding that out on the
    first batch would waste a full prompt. A model that can't take the schema
    answers 400 here and rules itself out.
    """
    schema = {"type": "object", "properties": {"ok": {"type": "string"}},
              "required": ["ok"], "additionalProperties": False}
    try:
        resp = _post(model, api_key, 'Reply {"ok":"yes"}', timeout=PROBE_TIMEOUT,
                     schema=schema, schema_name="probe",
                     max_tokens=PROBE_TOKENS)
    except requests.RequestException:
        return True                   # a network blip proves nothing; let it try
    if resp.status_code == 200:
        return True
    if resp.status_code == 429 and not _is_exhausted(resp):
        return True                   # a busy minute, not an exhausted allowance
    _UNAVAILABLE_MODELS.add(model)
    return False


def resolve_model(api_key: str, model: Optional[str] = None,
                  progress_cb=None) -> str:
    """The model to generate with: the configured one, or the best that answers.

    Not every model an account can see will take a strict JSON schema, and a
    model's free allowance can be used up for the day, so 'configured' and
    'usable' are different questions. Rather than carry a hard-coded list of
    second choices, ask the API what exists and try them in order until one
    answers. Settled once per process.
    """
    global _RESOLVED_MODEL
    model = model or load_model_name()
    if _RESOLVED_MODEL and _RESOLVED_MODEL not in _UNAVAILABLE_MODELS:
        return _RESOLVED_MODEL

    candidates = _candidate_models(model, api_key)
    for candidate in candidates:
        if not _answers(candidate, api_key):
            continue
        if candidate != model:
            message = (f"AI: {model} isn't usable on this Groq account; "
                       f"generating with {candidate}")
            _emit_progress(progress_cb, message)
            runlog.log(message)
        _RESOLVED_MODEL = candidate
        return candidate

    raise AIError(
        "No Groq model your key can call will take a Learning Plan request. "
        "Tried %s. Check the key at console.groq.com/keys, or set GROQ_MODEL "
        "in .env to a model your account allows."
        % (", ".join(candidates) or model))


def batch_size_for(model: str) -> int:
    """How many sessions to ask for at once.

    Flat, because on Groq the ceiling is the account's tokens-per-minute rather
    than anything about the model: a bigger model does not buy a bigger batch.
    Kept as a function so the batching code has one place to ask.
    """
    return LP_SESSION_CHUNK


def _generate_batch(unit: Unit, chunk: List[Session], api_key: str,
                    model: str, progress_cb=None) -> List[dict]:
    """Exactly one row per session in `chunk`, whatever the model returns.

    Rows are positional - `merge_ai_into_sessions` pairs row *i* with session
    *i* - so a model that runs out of room mid-array must never simply return a
    short list: every following batch would then slide onto the wrong sessions.
    A short answer is retried in halves (a smaller batch fits where a big one
    didn't), and anything still missing is padded so the deterministic backfill
    lands on the right row.
    """
    rows = call_model(build_prompt(unit, chunk), api_key, model,
                        progress_cb=progress_cb)
    if len(rows) >= len(chunk):
        return rows[:len(chunk)]

    # Some rows but not all means the answer ran out of room, and a smaller
    # batch fits where a big one didn't. NO rows is a different failure - the
    # model gave nothing - and splitting only multiplies it.
    if rows and len(chunk) > 1:
        _emit_progress(progress_cb,
                       f"AI: {len(rows)} of {len(chunk)} sessions came back; "
                       f"retrying them in smaller batches")
        half = len(chunk) // 2
        return (_generate_batch(unit, chunk[:half], api_key, model, progress_cb)
                + _generate_batch(unit, chunk[half:], api_key, model, progress_cb))

    return rows + [{}] * (len(chunk) - len(rows))


def generate_sessions(unit: Unit, sessions: List[Session], api_key: str = "",
                      model: Optional[str] = None, progress_cb=None) -> List[Session]:
    """Run the grounded AI calls and merge the result into the skeleton.

    The model is `model` or `GROQ_MODEL` where the account can call it, and
    otherwise the most capable model that does answer - see `resolve_model`.
    """
    # Key semantics: api_key=None  -> use the configured (.env / hard-coded) key
    if api_key is None:
        api_key = load_api_key()
    if not api_key:
        raise AIError("No Groq API key available.")

    model = resolve_model(api_key, model, progress_cb)
    runlog.log(f"AI: model = {model}")

    # Generate in batches so a long term's JSON never overflows the token limit
    # (a single 24-session response gets truncated -> invalid JSON). Rows are
    # concatenated in order, preserving alignment with the deterministic
    # skeleton, and sized for the model that will actually answer them.
    # CAT rows are rebuilt deterministically by `merge_ai_into_sessions`, so
    # generating them would be paying for output that is thrown away - roughly a
    # fifth of a term's tokens. Only content sessions are sent.
    to_generate = [s for s in sessions if not s.is_cat]
    if not to_generate:
        return merge_ai_into_sessions(sessions, [], unit)

    size = batch_size_for(model)
    chunks = [to_generate[i:i + size] for i in range(0, len(to_generate), size)]

    # Batches don't depend on each other, so they go together rather than in a
    # row - see LP_MAX_PARALLEL. Each worker collects its own progress lines
    # instead of calling progress_cb, because that callback draws into the
    # Streamlit page and only the main thread may do that; the lines are
    # replayed here, in batch order, as each batch lands.
    def run_batch(numbered):
        ci, chunk = numbered
        lines: List[str] = []
        first = sum(len(c) for c in chunks[:ci - 1]) + 1
        lines.append(f"AI: generating sessions {first}-{first + len(chunk) - 1} "
                     f"of {len(to_generate)} (batch {ci}/{len(chunks)})")
        rows = _generate_batch(unit, chunk, api_key, model, lines.append)
        return lines, rows

    ai_rows: List[dict] = []
    numbered = list(enumerate(chunks, start=1))
    with ThreadPoolExecutor(max_workers=min(LP_MAX_PARALLEL, len(chunks))) as pool:
        for lines, rows in pool.map(run_batch, numbered):
            for line in lines:
                _emit_progress(progress_cb, line)
            ai_rows.extend(rows)

    # Checked across the whole plan rather than per batch, because method
    # variety is a property of the plan a trainee sits through, not of however
    # the work happened to be divided.
    ai_rows = _repair_rows(unit, to_generate, ai_rows, api_key, model, progress_cb)
    return merge_ai_into_sessions(sessions, ai_rows, unit)


def regenerate_learning_plan_session(unit: Unit, sessions: List[Session], idx: int,
                                     *, api_key: str = "",
                                     model: Optional[str] = None,
                                     progress_cb=None) -> Session:
    """Regenerate ONE session in `sessions` in place (mutates sessions[idx]).

    Content sessions rerun the grounded call for just that session (higher
    temperature so the result differs) and merge back onto the deterministic
    skeleton, so week / session_no / is_cat / pcs and the authoritative
    key_points survive. CAT rows are deterministic and simply recomputed from the
    content sessions they assess. Raises AIError if the model returns nothing
    usable so the caller can retry.
    """
    session = sessions[idx]

    if session.is_cat:                       # deterministic - no API call needed
        return rebuild_cat_session(sessions, idx)

    if api_key is None:
        api_key = load_api_key()
    if not api_key:
        raise AIError("No Groq API key available.")
    model = resolve_model(api_key, model, progress_cb)

    prompt = build_prompt(unit, [session])
    _emit_progress(progress_cb, f"Regenerating session {session.session_no}")
    data = _chat_json(prompt, api_key, model, _response_schema(),
                      "learning_plan_sessions", progress_cb=progress_cb,
                      temperature=0.6, system=LP_SYSTEM)
    if isinstance(data, dict):
        data = data.get("sessions") or data.get("data") or [data]
    if not isinstance(data, list) or not data:
        raise AIError("AI returned no session content")
    merge_ai_into_sessions([session], data, unit)   # mutates `session` in place
    return session


# =========================================================================== #
# SESSION PLAN - one detailed lesson plan per chosen Learning-Plan session.
#
# The Learning Plan already gives each session its learning outcomes, key points,
# trainee activities, resources and assessments. A Session Plan expands ONE of
# those sessions into the minute-by-minute KTTC delivery breakdown (Introduction
# -> Session Delivery steps -> Session Review). Only that breakdown is generated;
# everything else is carried over verbatim, and a deterministic fallback fills
# the breakdown when no API is available, so a session plan never hard-fails.
# =========================================================================== #
def _delivery_steps_schema() -> dict:
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "step_label": {"type": "string"},
                "minutes": {"type": "integer"},
                "trainer_activity": str_array,
                "trainee_activity": str_array,
                "learning_check": str_array,
            },
        },
    }


def _session_plan_schema() -> dict:
    str_array = {"type": "array", "items": {"type": "string"}}
    return _strict({
        "type": "object",
        "properties": {
            "introduction": str_array,
            "delivery_steps": _delivery_steps_schema(),
            "review": str_array,
            "assignment": {"type": "string"},
            "lln_requirements": {"type": "string"},
            "safety_requirements": {"type": "string"},
        },
    })


# The Session Plan's standing instructions, split from its data for the same
# reasons as LP_SYSTEM.
SP_SYSTEM = """You are a senior TVET trainer and industry expert in Kenya, writing ONE detailed SESSION PLAN (a single lesson) for a session taken from an approved Learning Plan. Ground everything in the supplied content; do NOT invent new topics.

Write every activity line in the imperative present tense (base verb form): "Take roll call", "Lead a group discussion", "Demonstrate the tool" - NOT "Takes roll call", "Leads", "Demonstrates".

TERMINOLOGY (Kenya CBET) - MANDATORY
Use: trainee, trainer, session, facilitate, unit of competency, learning outcome, performance criteria, competency, assessment, Continuous Assessment Test (CAT).
Never use: student, pupil, learner, teacher, lecturer, instructor, lesson, lecture, class (as a synonym for session), teach, deliver a lecture, exam, quiz, or test as a noun for the final assessment.
Spelling: British / Kenyan English - organise, practise (as a verb), programme (a course of study), labelled, capitalised, centred.

FIELD RULES

introduction: 3-4 short bullets for the 5-minute opening. Start with "Trainer:" then what the trainer does - take roll call; review the previous session; state this session's title and expected learning outcomes.

delivery_steps: 3-4 steps that together fill EXACTLY the delivery minutes given in the user message. Every step and sub-step "minutes" must sum to that figure.
- step_label: "Step 1", "Step 2", "Step 3", in order. If one step is involving enough to need its own breakdown, split it into consecutive sub-steps labelled "Step 1(a)", "Step 1(b)", "Step 1(c)" - each a full step object with its own minutes, trainer_activity, trainee_activity and learning_check. Only break a step down when it genuinely warrants it; otherwise keep one row per step.
- minutes: integer. The longest step is the hands-on or practice step.
- trainer_activity: 1-3 short lines describing what the TRAINER does this step. Name the CBET active-learning method - Group Discussion, Think-Pair-Share, Demonstration, Guided Practice, Case Study and the like - and reference THIS session's specific key points, tools or tasks. Concrete, never generic. Facilitate the session; never lecture.
- trainee_activity: 1-3 short lines describing what the TRAINEES do in response, referencing the same specific content.
- learning_check: grouped assessment lines for this step - a CAPITALISED group word ("Knowledge", "Skills" or "Attitudes") followed by numbered items such as "1. Oral questioning", "2. Observation of developed work", drawn from the session's assessments and the evidence-guide methods. Early steps lean on Knowledge; hands-on steps add Skills.
Order the steps so they progress from understanding, to guided practice, to independent application.

review: 2-3 short bullets for the 5-minute close. Start with "Trainer:" - summarise key points; answer questions; preview the next session.

assignment: ONE concrete take-home task grounded in this session's content, written as a full sentence.

lln_requirements: one sentence on how trainees with Language, Literacy and Numeracy or other special needs are catered for in THIS session - simplified handouts, a sign-language interpreter, extra time and so on.

safety_requirements: one sentence on the workplace SOPs and safety precautions relevant to THIS session's topic.

Return ONE JSON object and nothing else."""


def build_session_plan_prompt(unit: Unit, session: Session,
                              duration_minutes: int) -> str:
    """The per-session half: this session's content, no instructions.

    The Learning Plan has already been generated and approved by the time this
    runs, so its outcomes, key points, activities and assessments are handed
    over as source material - the Session Plan elaborates them into a delivery
    breakdown rather than deriving the content afresh.
    """
    delivery = max(10, int(duration_minutes) - 10)   # 5' intro + 5' review fixed
    los = "\n".join(session.learning_outcomes) or "(derive from the title)"
    kps = "\n".join(session.key_points) or "(none supplied)"
    acts = "\n".join(session.trainee_activities) or "(none supplied)"
    assess = "\n".join(session.assessments) or "(none supplied)"
    level = unit.level or "6"
    methods = "; ".join(unit.assessment_methods) or ("Observation; Oral "
        "questioning; Written assessment; Practical assessment; Portfolio of evidence")

    return f"""UNIT: {unit.unit_title}
CODE: {unit.os_code}
LEVEL: {level}
SESSION TITLE: {session.session_title}
DELIVERY MINUTES (the delivery_steps must sum to exactly this): {delivery}

THIS SESSION'S LEARNING OUTCOMES:
{los}

THIS SESSION'S LEARNING KEY POINTS (authoritative content to facilitate):
{kps}

THIS SESSION'S PLANNED TRAINEE ACTIVITIES (active-learning methods to operationalise):
{acts}

THIS SESSION'S ASSESSMENTS:
{assess}

OS EVIDENCE-GUIDE ASSESSMENT METHODS: {methods}

Return the JSON object now."""


def _group_assessments(assessments: List[str]) -> "dict":
    """Split a session's assessment lines into Knowledge / Skills / Attitudes."""
    groups: dict = {"Knowledge": [], "Skills": [], "Attitudes": []}
    current = "Knowledge"
    for raw in assessments:
        line = str(raw).strip()
        low = line.lower()
        if low.startswith("knowledge"):
            current = "Knowledge"
            continue
        if low.startswith("skill"):
            current = "Skills"
            continue
        if low.startswith("attitude"):
            current = "Attitudes"
            continue
        if line:
            groups[current].append(line)
    return groups


def _check_lines(groups: "dict", keys: List[str]) -> List[str]:
    """Render selected assessment groups as 'Heading' + numbered items."""
    out: List[str] = []
    for key in keys:
        items = groups.get(key) or []
        if not items:
            continue
        out.append(key)
        for i, it in enumerate(items[:3], start=1):
            txt = re.sub(r"^\s*\d+[\.\)]\s*", "", it)
            out.append(f"{i}. {txt}")
    return out


def _split_minutes(total: int, n: int) -> List[int]:
    """Split `total` minutes across `n` steps, giving the middle step the most."""
    n = max(1, n)
    base = max(5, total // n)
    mins = [base] * n
    mins[len(mins) // 2] += total - sum(mins)     # absorb the remainder centrally
    if mins[len(mins) // 2] < 5:                   # keep every step >= 5'
        mins[len(mins) // 2] = 5
    return mins


def _default_delivery(unit: Unit, session: Session, duration_minutes: int) -> dict:
    """Deterministic Introduction / steps / Review built from the session itself.

    Used as the no-API fallback AND as the per-field safety net for thin AI output,
    so a session plan is always complete and grounded in the Learning-Plan session.
    """
    # method bullets vs. follow-up lines in the planned trainee activities
    method_bullets, followup = [], ""
    capture_followup = False
    for raw in session.trainee_activities:
        line = str(raw).strip()
        if not line:
            continue
        if line.lower().startswith("follow up activity"):
            capture_followup = True
            continue
        if line.lower().startswith("due date"):
            capture_followup = False
            continue
        if capture_followup and not followup:
            followup = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
            continue
        if line.startswith("-"):
            method_bullets.append(line.lstrip("-").strip())
    if not method_bullets:
        method_bullets = [f"Explore {session.session_title.lower()} through guided discussion.",
                          f"Apply the key points of {session.session_title.lower()} in a short practical task."]

    groups = _group_assessments(session.assessments)
    delivery = max(10, int(duration_minutes) - 10)
    n = min(4, max(2, len(method_bullets)))
    method_bullets = method_bullets[:n]
    mins = _split_minutes(delivery, n)

    steps = []
    for i, activity in enumerate(method_bullets):
        is_practice = (i == n - 1)
        steps.append({
            "step_label": f"Step {i + 1}",
            "minutes": mins[i],
            "trainer_activity": [
                "Trainer:",
                f"Facilitate the activity and guide trainees through {session.session_title.lower()}.",
            ],
            "trainee_activity": ["Trainee(s):", activity],
            "learning_check": _check_lines(
                groups, ["Knowledge", "Skills"] if is_practice else ["Knowledge"])
            or ["Knowledge", "1. Oral questioning"],
        })

    return {
        "introduction": [
            "Trainer:",
            "Take roll call.",
            "Review the previous session.",
            "State the session title and the expected learning outcomes.",
        ],
        "delivery_steps": steps,
        "review": [
            "Trainer:",
            "Summarize the key points covered in the session.",
            "Respond to trainee questions and clarify difficult areas.",
            "Preview the next session.",
        ],
        "assignment": followup or f"Complete a short practical exercise on {session.session_title.lower()}.",
        "lln_requirements": ("Identify any trainees with Language, Literacy or Numeracy "
                             "or other special needs and put measures in place (e.g. simplified "
                             "handouts, extra time, or a sign-language interpreter)."),
        "safety_requirements": ("Comply with the workplace standard operating procedures (SOPs) "
                                "applicable to this session."),
    }


def _coerce_steps(raw_steps, fallback_steps: List[dict]) -> List[DeliveryStep]:
    steps: List[DeliveryStep] = []
    for raw in (raw_steps if isinstance(raw_steps, list) else []):
        if not isinstance(raw, dict):
            continue
        trainer = _as_list(raw.get("trainer_activity"))
        trainee = _as_list(raw.get("trainee_activity"))
        check = _as_list(raw.get("learning_check"))
        if not (trainer or trainee):
            continue
        try:
            minutes = int(raw.get("minutes") or 0)
        except (TypeError, ValueError):
            minutes = 0
        steps.append(DeliveryStep(
            step_label=str(raw.get("step_label") or f"Step {len(steps) + 1}").strip(),
            minutes=max(0, minutes),
            trainer_activity=trainer,
            trainee_activity=trainee,
            learning_check=check or ["Knowledge", "1. Oral questioning"],
        ))
    if not steps:
        steps = [DeliveryStep(step_label=s["step_label"], minutes=s["minutes"],
                              trainer_activity=s["trainer_activity"],
                              trainee_activity=s["trainee_activity"],
                              learning_check=s["learning_check"])
                 for s in fallback_steps]
    return steps


def _assemble_session_plan(unit: Unit, session: Session, inputs: PlanInputs,
                           body: dict, *, display_code: str, trainer_number: str,
                           session_date: str, session_time: str) -> SessionPlan:
    return SessionPlan(
        unit_title=unit.unit_title,
        unit_code=display_code or unit.os_code or unit.isced_code,
        session_title=session.session_title,
        trainer_name=inputs.trainer_name,
        trainer_number=trainer_number,
        institution=inputs.institution,
        level=inputs.level or unit.level,
        class_code=inputs.class_code,
        num_trainees=str(inputs.num_trainees),
        session_date=session_date,
        session_time=session_time,
        learning_outcomes=list(session.learning_outcomes),
        resources=list(session.resources),
        lln_requirements=body.get("lln_requirements", ""),
        safety_requirements=body.get("safety_requirements", ""),
        introduction=_as_list(body.get("introduction")),
        delivery_steps=body["delivery_steps"],
        review=_as_list(body.get("review")),
        assignment=str(body.get("assignment") or "").strip(),
    )


def _merge_session_plan_body(ai: dict, unit: Unit, session: Session,
                             duration_minutes: int) -> dict:
    """AI body with deterministic safety nets for every thin/missing field."""
    defaults = _default_delivery(unit, session, duration_minutes)
    ai = ai if isinstance(ai, dict) else {}

    intro = _as_list(ai.get("introduction")) or defaults["introduction"]
    review = _as_list(ai.get("review")) or defaults["review"]
    steps = _coerce_steps(ai.get("delivery_steps"), defaults["delivery_steps"])
    assignment = str(ai.get("assignment") or "").strip() or defaults["assignment"]
    lln = str(ai.get("lln_requirements") or "").strip() or defaults["lln_requirements"]
    safety = str(ai.get("safety_requirements") or "").strip() or defaults["safety_requirements"]

    return {
        "introduction": intro,
        "delivery_steps": steps,
        "review": review,
        "assignment": assignment,
        "lln_requirements": lln,
        "safety_requirements": safety,
    }


def build_session_plan_offline(unit: Unit, session: Session, inputs: PlanInputs, *,
                               display_code: str = "", trainer_number: str = "",
                               session_date: str = "", session_time: str = "",
                               duration_minutes: int = 120) -> SessionPlan:
    """A complete SessionPlan built deterministically from the session (no AI)."""
    body = _default_delivery(unit, session, duration_minutes)
    body["delivery_steps"] = _coerce_steps(body["delivery_steps"], body["delivery_steps"])
    return _assemble_session_plan(
        unit, session, inputs, body, display_code=display_code,
        trainer_number=trainer_number, session_date=session_date,
        session_time=session_time)


def generate_session_plan(unit: Unit, session: Session, inputs: PlanInputs, *,
                          display_code: str = "", trainer_number: str = "",
                          session_date: str = "", session_time: str = "",
                          duration_minutes: int = 120, api_key: str = "",
                          model: Optional[str] = None, progress_cb=None) -> SessionPlan:
    """One grounded AI call for the chosen session's delivery breakdown.

    Falls back to the deterministic build if no key is configured or the call
    fails, so the user always gets a usable session plan.
    """
    if api_key is None:
        api_key = load_api_key()

    if not api_key:
        _emit_progress(progress_cb, "Session plan: no API key - building offline.")
        return build_session_plan_offline(
            unit, session, inputs, display_code=display_code,
            trainer_number=trainer_number, session_date=session_date,
            session_time=session_time, duration_minutes=duration_minutes)

    model = resolve_model(api_key, model, progress_cb)
    prompt = build_session_plan_prompt(unit, session, duration_minutes)
    _emit_progress(progress_cb, "Session plan: prompt prepared")
    ai = _chat_json(prompt, api_key, model, _session_plan_schema(),
                    "session_plan", progress_cb=progress_cb, system=SP_SYSTEM)
    body = _merge_session_plan_body(ai if isinstance(ai, dict) else {},
                                    unit, session, duration_minutes)
    _emit_progress(progress_cb,
                   f"Session plan: {len(body['delivery_steps'])} delivery steps")
    return _assemble_session_plan(
        unit, session, inputs, body, display_code=display_code,
        trainer_number=trainer_number, session_date=session_date,
        session_time=session_time)


def _config(name: str) -> str:
    """Read a config value from the environment / .env, then Streamlit secrets.

    Delegates to `config.get` so the Drive layer and this module resolve settings
    the same way. Kept as a module-level name because the rest of this file (and
    its tests) call it.
    """
    import config
    return config.get(name)


def load_api_key() -> str:
    """The Groq key, read from GROQ_API_KEY (env / .env, then st.secrets).

    Returns '' when unset; callers treat an empty key as an error. No key is
    hard-coded in source.
    """
    return _config("GROQ_API_KEY")


def load_model_name() -> str:
    """Preferred model, overridable via `.env`."""
    primary = _config("GROQ_MODEL")
    return primary or DEFAULT_MODEL
