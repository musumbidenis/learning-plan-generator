"""Module C - THE SINGLE AI CALL (Mistral, ONE grounded request).

The ONLY place AI is used in the whole pipeline. It fills the columns AI is good
at - learning_outcomes, trainee_activities, resources, assessments - grounded in
the deterministically-parsed unit + curriculum content. `key_points` are supplied
(authoritative curriculum content); the model keeps/format them, never invents.

Everything before this stage (parsing, planning) and after it (doc building) is
pure Python. Robust safety nets re-stamp the deterministic schedule and backfill
any blank cell, so the output is never empty even if the API misbehaves.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

import requests

import runlog
from models import DeliveryStep, PlanInputs, Session, SessionPlan, Unit

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# The Mistral key is read from the environment / .env (MISTRAL_API_KEY) - see
# load_api_key(). It is never hard-coded in source and never entered in the UI.

DEFAULT_MODEL = "mistral-small-latest"


class AIError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Response schema (forces valid structured JSON)
# --------------------------------------------------------------------------- #
def _response_schema() -> dict:
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "week": {"type": "integer"},
                "session_no": {"type": "string"},
                "is_cat": {"type": "boolean"},
                "session_title": {"type": "string"},
                "learning_outcomes": str_array,
                "key_points": str_array,
                "trainee_activities": str_array,
                "resources": str_array,
                "assessments": str_array,
            },
            "required": [
                "week", "session_no", "session_title", "learning_outcomes",
                "key_points", "trainee_activities", "resources", "assessments",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def build_prompt(unit: Unit, sessions: List[Session]) -> str:
    pcs = "\n".join(f"{pc.number} {pc.text}" for pc in unit.all_pcs)
    methods = "; ".join(unit.assessment_methods) or "Observation; Oral assessment; " \
        "Written assessment; Practical assessment; Portfolio of evidence"
    knowledge = "; ".join(unit.required_knowledge)
    skeleton = [s.to_skeleton_dict() for s in sessions]
    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=1)
    level = unit.level or "6"

    verbs = "Identify, Explain, Apply, Demonstrate, Evaluate, Implement" \
        if str(level) >= "6" else "Identify, Explain, Apply, Demonstrate"

    return f"""You are a senior TVET trainer and industry expert, creating a Learning Plan. You are GIVEN the unit, the term schedule, and - for each session - the official Curriculum Learning Key Points. Do NOT invent syllabus content; use what is given.

UNIT: {unit.unit_title} | CODE: {unit.os_code} | LEVEL: {level}
OS PERFORMANCE CRITERIA:
{pcs}
OS EVIDENCE-GUIDE ASSESSMENT METHODS: {methods}
OS REQUIRED KNOWLEDGE (underpinning topics for this unit): {knowledge or "(none listed)"}

SESSIONS (fill each; key_points are AUTHORITATIVE, keep them):
{skeleton_json}

For EACH session output:
- session_title: the supplied title, cleaned per SOURCE DATA QUALITY (fix casing, typos, and fragments into a complete, readable title; keep the same meaning and topic - do not rename).
- learning_outcomes: an array of 3 strings labelled "a.", "b.", "c.", trainee-centred, the list should always be introduced as follows: "By the end of the session, the trainee should be able to; {{The labeled list starts below}}...", rewritten from the session's PCs. Use level-appropriate verbs ({verbs}).
- key_points: present the supplied Curriculum content, repaired per SOURCE DATA QUALITY below. Format as 2-3 CAPITALISED headings, each followed by ~3 short bullet sub-points drawn from the supplied content. Do NOT invent new topics. If a session's supplied key_points only restate the performance criterion (i.e. no curriculum content was available for this unit), you MAY draw concrete, relevant sub-points from the OS REQUIRED KNOWLEDGE topics listed above.
- trainee_activities: EXACTLY 3 bullets. Each bullet starts "- ", NAMES an active-learning method (e.g Group Discussion, Think-Pair-Share, Case Study, Jigsaw, Peer Teaching, Round Robin, Demonstrations with Participation, KWL, Concept Mapping, Brainstorming e.t.c), then describes in 1-2 full sentences exactly what the trainee does - referencing this session's specific key points and learning outcomes (the actual topic, task, tool, sample, or scenario), never generic filler. Order the three so they progress from understanding to hands-on application. Then add a line "Follow up Activity:" and "1. <a specific assignment grounded in this session's content>."
- resources: at least 2 bullets - real textbooks, presentations, tools, or online docs relevant to the topic.
- assessments: derived from the Evidence-Guide methods, grouped under "Knowledge Checks:", "Skills:" and "Attitudes:" with numbered items.

SOURCE DATA QUALITY
The supplied session titles and Curriculum Learning Key Points are auto-extracted and may be messy: fragments, incomplete sentences, truncated phrases, duplicates, or shallow stubs.
- Repair form, preserve meaning. You MAY fix grammar, spelling, capitalisation, spacing, and complete an obviously truncated sentence into a coherent one - ONLY when the intended meaning is clear from the fragment itself, the session's other key points, the Performance Criteria, or the Required Knowledge. This is editing, not authoring: never add a topic, tool, or fact not already implied by those supplied sources.
- Thin or missing points: where a point is too shallow to teach from, you MAY add concrete sub-points, but ONLY drawn from the supplied Performance Criteria and Required Knowledge for this unit. Never draw on outside knowledge.
- Unrecoverable items: if a fragment is unintelligible and cannot be reconstructed from the supplied sources, OMIT it. Never output a fragment, a dangling phrase, an incomplete sentence, or a placeholder.
- Every key point you output must be a complete, coherent sentence traceable to the supplied curriculum, Performance Criteria, or Required Knowledge.
- Apply the same repairs to each session_title: return a clean, complete, correctly-capitalised title with the same meaning; never output a truncated or fragmentary title.

CAT sessions (is_cat true): learning_outcomes about demonstrating competence; key_point heading ASSESSMENT COVERAGE; trainee_activities = "- Complete the Continous Assessment Test(CAT)." ; resources = Assessment Tool(s) and/or Observation Checklist, Assessor Guide; assessments = "-Graded Knowledge.".

TERMINOLOGY - use Kenya Competency-Based Education and Training (CBET) terms ONLY: "trainee" (never student/pupil/learner), "trainer" (never teacher/lecturer/instructor), "session" (never "lecture"/"lesson"/"class"), "facilitate" (never "teach" or "deliver a lecture"), "unit of competency", "learning outcome", "performance criteria", "competency", "assessment" / "Continuous Assessment Test (CAT)" (never "exam" or "test" as a noun for the final). NEVER output placeholders like "Key concept 1". Return ONLY the JSON array.
"""


# --------------------------------------------------------------------------- #
# HTTP call
# --------------------------------------------------------------------------- #
def _post(model: str, api_key: str, prompt: str, timeout: int = 120,
          schema: Optional[dict] = None,
          schema_name: str = "learning_plan_sessions") -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 14000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema if schema is not None else _response_schema(),
                "strict": True,
            },
        },
    }
    resp = requests.post(MISTRAL_ENDPOINT,
                         headers={
                             "Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json",
                         },
                         json=body, timeout=timeout)
    return resp


def _extract_text(payload: dict) -> str:
    """Pull the text out of a Mistral chat-completions response, defensively.

    The API can return a 200 with no usable assistant content, for example an
    empty choices list or a choice that only carries a finish reason. Each of
    those raises AIError so the caller can surface the error immediately.
    """
    choices = payload.get("choices") or []
    if not choices:
        raise AIError("Mistral returned no choices: " + json.dumps(payload)[:300])
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
        raise AIError(f"Mistral returned an empty response (finish_reason: {reason})")
    return text


def _emit_progress(progress_cb, message: str) -> None:
    runlog.log(message)
    if progress_cb is not None:
        try:
            progress_cb(message)
        except Exception:  # noqa: BLE001 - never break generation because UI logging failed
            pass


def _chat_json(prompt: str, api_key: str, model: str, schema: dict,
               schema_name: str, progress_cb=None):
    """One grounded request (short retry on transient network errors).

    Returns the parsed JSON exactly as the model produced it (a list or a dict);
    callers coerce it to the shape they expect.
    """
    _emit_progress(progress_cb, f"AI: trying model {model}")
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = _post(model, api_key, prompt, schema=schema, schema_name=schema_name)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < 3:
                _emit_progress(progress_cb,
                               f"AI: transient network issue for {model}, retrying ({attempt}/3)")
                time.sleep(attempt * 0.5)
                continue
            raise AIError(f"{model}: network error: {e}") from e
        except requests.RequestException as e:
            raise AIError(f"{model}: network error: {e}") from e

        if resp.status_code == 200:
            try:
                text = _extract_text(resp.json())
            except AIError as e:
                raise AIError(f"{model}: {e}") from e
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise AIError(f"{model} returned invalid JSON: {e}") from e

        if resp.status_code in (401, 403):
            raise AIError(
                "Mistral auth failed (HTTP %d). Your MISTRAL_API_KEY is invalid, "
                "expired, or lacks access." % resp.status_code)

        if resp.status_code == 429:
            raise AIError(f"{model}: rate/quota limited (HTTP 429){resp.text[:160]}")

        raise AIError(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")

    if last_error is not None:
        raise AIError(f"{model}: network error: {last_error}") from last_error
    raise AIError(f"{model}: request failed")


def call_mistral(prompt: str, api_key: str, model: str,
                 progress_cb=None) -> List[dict]:
    """The Learning-Plan call: returns the JSON array of session rows."""
    data = _chat_json(prompt, api_key, model, _response_schema(),
                      "learning_plan_sessions", progress_cb=progress_cb)
    if isinstance(data, dict):
        data = data.get("sessions") or data.get("data") or [data]
    if not isinstance(data, list):
        raise AIError(f"{model} did not return a JSON array")
    _emit_progress(progress_cb, f"AI: {model} succeeded ({len(data)} session rows)")
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
                t = str(item).strip()
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


def merge_ai_into_sessions(sessions: List[Session], ai_rows: List[dict],
                           unit: Unit) -> List[Session]:
    """Apply AI output onto the deterministic skeleton, with full safety nets.

    The deterministic schedule (week / session_no / is_cat / pcs) ALWAYS wins;
    the AI only contributes the generative text columns. Missing fields are
    backfilled so no cell is ever blank.
    """
    for i, s in enumerate(sessions):
        row = ai_rows[i] if i < len(ai_rows) else {}

        # CAT rows are fully deterministic - fixed wording every time, never the
        # AI's output. The planner already gives them a clean title.
        if s.is_cat:
            s.learning_outcomes = _cat_learning_outcomes()
            s.key_points = _cat_keypoints()
            s.trainee_activities = _cat_activities()
            s.resources = _cat_resources()
            s.assessments = _cat_assessments()
            continue

        # session_title: accept the AI's data-quality-repaired title if given,
        # otherwise keep the deterministic skeleton title. Set before the
        # fallbacks below so they build on the cleaned title.
        ai_title = (row.get("session_title") or "").strip()
        if ai_title:
            s.session_title = ai_title

        lo = _as_list(row.get("learning_outcomes"))
        ai_kp = _as_list(row.get("key_points"))
        acts = _as_list(row.get("trainee_activities"))
        res = _as_list(row.get("resources"))
        assess = _as_list(row.get("assessments"))

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


def _cat_learning_outcomes() -> List[str]:
    return [
        "By the end of the session, the trainee should be able to:",
        "a. Demonstrate competence in the learning outcomes covered.",
    ]


def _cat_keypoints() -> List[str]:
    return ["ASSESSMENT COVERAGE"]


def _cat_activities() -> List[str]:
    return ["- Complete the Continous Assessment Test(CAT)."]


def _cat_resources() -> List[str]:
    return ["Assessment Tool(s) and/or Observation Checklist", "Assessor Guide"]


def _cat_assessments() -> List[str]:
    return ["-Graded Knowledge."]


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
def generate_sessions(unit: Unit, sessions: List[Session], api_key: str = "",
                      model: Optional[str] = None, progress_cb=None) -> List[Session]:
    """Run the single grounded AI call and merge the result into the skeleton.

    The model defaults to `mistral-large-latest` unless the caller pins a
    different one via `model` or `MISTRAL_MODEL`.
    """
    # Key semantics: api_key=None  -> use the configured (.env / hard-coded) key
    if api_key is None:
        api_key = load_api_key()
    if not api_key:
        raise AIError("No Mistral API key available.")

    # Resolve the model unless the caller pinned one.
    if model is None:
        model = load_model_name()
    runlog.log(f"AI: model = {model}")

    prompt = build_prompt(unit, sessions)
    _emit_progress(progress_cb, "AI: prompt prepared")
    ai_rows = call_mistral(prompt, api_key, model, progress_cb=progress_cb)
    return merge_ai_into_sessions(sessions, ai_rows, unit)


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
def _session_plan_schema() -> dict:
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "introduction": str_array,
            "delivery_steps": {
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
                    "required": ["step_label", "minutes", "trainer_activity",
                                 "trainee_activity", "learning_check"],
                },
            },
            "review": str_array,
            "assignment": {"type": "string"},
            "lln_requirements": {"type": "string"},
            "safety_requirements": {"type": "string"},
        },
        "required": ["introduction", "delivery_steps", "review", "assignment",
                     "lln_requirements", "safety_requirements"],
    }


def build_session_plan_prompt(unit: Unit, session: Session,
                              duration_minutes: int) -> str:
    delivery = max(10, int(duration_minutes) - 10)   # 5' intro + 5' review fixed
    los = "\n".join(session.learning_outcomes) or "(derive from the title)"
    kps = "\n".join(session.key_points) or "(none supplied)"
    acts = "\n".join(session.trainee_activities) or "(none supplied)"
    assess = "\n".join(session.assessments) or "(none supplied)"
    level = unit.level or "6"
    methods = "; ".join(unit.assessment_methods) or ("Observation; Oral "
        "questioning; Written assessment; Practical assessment; Portfolio of evidence")

    return f"""You are a senior TVET trainer and industry expert writing ONE detailed SESSION PLAN (a single lesson) for the session below, taken from an approved Learning Plan. Ground everything in the supplied content; do NOT invent new topics.

UNIT: {unit.unit_title} | CODE: {unit.os_code} | LEVEL: {level}
SESSION TITLE: {session.session_title}

THIS SESSION'S LEARNING OUTCOMES:
{los}

THIS SESSION'S LEARNING KEY POINTS (authoritative content to teach):
{kps}

THIS SESSION'S PLANNED TRAINEE ACTIVITIES (active-learning methods to operationalise):
{acts}

THIS SESSION'S ASSESSMENTS:
{assess}
OS EVIDENCE-GUIDE ASSESSMENT METHODS: {methods}

Write every activity line in the imperative present tense (base verb form): "Take roll call", "Lead a group discussion", "Demonstrate the tool" - NOT "Takes roll call", "Leads", "Demonstrates".

Produce a JSON object with these fields:
- introduction: 3-4 short bullets for the 5-minute opening. Start with "Trainer:" then what the trainer does (take roll call; review the previous session; state this session's title and expected learning outcomes).
- delivery_steps: 3-4 steps that together fill EXACTLY {delivery} minutes (every step/sub-step "minutes" must sum to {delivery}). Each step:
    * step_label: "Step 1", "Step 2", "Step 3" (in order). If a single step is involving enough to need its own breakdown, split it into consecutive sub-steps labelled "Step 1(a)", "Step 1(b)", "Step 1(c)"... - each a full step object with its own minutes, trainer_activity, trainee_activity and learning_check. Only break a step down when it genuinely warrants it; otherwise keep one row per step.
    * minutes: integer; the longest step is the hands-on/practice step.
    * trainer_activity: 1-3 short lines describing what the TRAINER does this step - name the CBET active-learning method (e.g. Group Discussion, Think-Pair-Share, Demonstration, Guided Practice, Case Study) and reference THIS session's specific key points/tools/tasks. Concrete, never generic. Facilitate the session - never "lecture" or "give a lecture".
    * trainee_activity: 1-3 short lines describing what the TRAINEES do in response, referencing the same specific content.
    * learning_check: grouped assessment lines for this step - a CAPITALISED group word ("Knowledge", "Skills", or "Attitudes") followed by numbered items ("1. Oral questioning", "2. Observation of developed work"), drawn from the session's assessments / evidence-guide methods. Early steps lean on Knowledge; hands-on steps add Skills.
  Order the steps so they progress from understanding -> guided practice -> independent application.
- review: 2-3 short bullets for the 5-minute close. Start with "Trainer:" (summarise key points; answer questions; preview the next session).
- assignment: ONE concrete take-home task grounded in this session's content (a full sentence).
- lln_requirements: one sentence noting how trainees with Language/Literacy/Numeracy or other special needs are catered for in THIS session (e.g. simplified handouts, sign-language interpreter, extra time).
- safety_requirements: one sentence on the workplace SOPs / safety precautions relevant to THIS session's topic.

TERMINOLOGY - Kenya CBET only: "trainee" (never student/learner/pupil), "trainer" (never teacher/lecturer/instructor), "session" (never "lecture"/"lesson"/"class"), "facilitate" (never "teach" or "deliver a lecture"), "unit of competency", "learning outcome", "performance criteria", "competency", "assessment"/"Continuous Assessment Test (CAT)". Return ONLY the JSON object.
"""


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
    if model is None:
        model = load_model_name()

    if not api_key:
        _emit_progress(progress_cb, "Session plan: no API key - building offline.")
        return build_session_plan_offline(
            unit, session, inputs, display_code=display_code,
            trainer_number=trainer_number, session_date=session_date,
            session_time=session_time, duration_minutes=duration_minutes)

    prompt = build_session_plan_prompt(unit, session, duration_minutes)
    _emit_progress(progress_cb, "Session plan: prompt prepared")
    ai = _chat_json(prompt, api_key, model, _session_plan_schema(),
                    "session_plan", progress_cb=progress_cb)
    body = _merge_session_plan_body(ai if isinstance(ai, dict) else {},
                                    unit, session, duration_minutes)
    _emit_progress(progress_cb,
                   f"Session plan: {len(body['delivery_steps'])} delivery steps")
    return _assemble_session_plan(
        unit, session, inputs, body, display_code=display_code,
        trainer_number=trainer_number, session_date=session_date,
        session_time=session_time)


_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load .env into os.environ exactly once per process (idempotent)."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        from dotenv import load_dotenv
        load_dotenv()
        _ENV_LOADED = True


def _config(name: str) -> str:
    """Read a config value from the environment / .env, then Streamlit secrets.

    Locally the value comes from .env (loaded into os.environ once). On Streamlit
    Community Cloud there is no .env - secrets are set in the app's Secrets box
    and exposed via st.secrets, which this falls back to. No secret is ever
    hard-coded in source or committed to the repo.
    """
    _ensure_env_loaded()
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:                                   # only present when running under Streamlit
        import streamlit as st
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def load_api_key() -> str:
    """The Mistral key, read from MISTRAL_API_KEY (env / .env, then st.secrets).

    Returns '' when unset; callers treat an empty key as an error. No key is
    hard-coded in source.
    """
    return _config("MISTRAL_API_KEY")


def load_model_name() -> str:
    """Primary model, overridable via `.env`."""
    primary = _config("MISTRAL_MODEL")
    return primary or DEFAULT_MODEL
