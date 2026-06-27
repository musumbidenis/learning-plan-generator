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
from typing import List, Optional

import requests

import runlog
from models import Session, Unit

MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# The Mistral key is read from the environment / .env (MISTRAL_API_KEY) - see
# load_api_key(). It is never hard-coded in source and never entered in the UI.

DEFAULT_MODEL = "mistral-large-latest"


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

    return f"""You are a KSTVET internal verifier completing a Learning Plan to PASS REF KTTC/TP/LP/F07, RVNP style. You are GIVEN the unit, the term schedule, and - for each session - the official Curriculum Learning Key Points. Do NOT invent syllabus content; use what is given.

UNIT: {unit.unit_title} | CODE: {unit.os_code} | LEVEL: {level}
OS PERFORMANCE CRITERIA:
{pcs}
OS EVIDENCE-GUIDE ASSESSMENT METHODS: {methods}
OS REQUIRED KNOWLEDGE (underpinning topics for this unit): {knowledge or "(none listed)"}

SESSIONS (fill each; key_points are AUTHORITATIVE, keep them):
{skeleton_json}

For EACH session output:
- learning_outcomes: an array of 3 strings labelled "a.", "b.", "c.", trainee-centred, each completing "By the end of the session, the trainee should be able to ...", rewritten from the session's PCs. Use level-appropriate verbs ({verbs}).
- key_points: keep the supplied Curriculum content. Format as 2-3 CAPITALISED headings, each followed by ~3 short bullet sub-points drawn from the supplied content. Do NOT invent new topics. If a session's supplied key_points only restate the performance criterion (i.e. no curriculum content was available for this unit), you MAY draw concrete, relevant sub-points from the OS REQUIRED KNOWLEDGE topics listed above.
- trainee_activities: EXACTLY 3 bullets, each starting "- " and NAMING an active-learning method (Group Discussion, Think-Pair-Share, Case Study, Jigsaw, Peer Teaching, Round Robin, Demonstrations with Participation, KWL, Concept Mapping, Brainstorming), then a line "Follow up Activity:", then "1. <assignment>. (N Marks)" (content sessions 10-25 marks, CAT 0 marks), then "Due date:".
- resources: at least 2 bullets - real textbooks, presentations, tools, or online docs relevant to the topic.
- assessments: derived from the Evidence-Guide methods, grouped under "Knowledge Checks:", "Skills:" and "Attitudes:" with numbered items.

CAT sessions (is_cat true): learning_outcomes about demonstrating competence; key_points headings ASSESSMENT COVERAGE and ASSESSMENT STRATEGY; trainee_activities = a Quiz Game review bullet + a Q&A bullet + "- Complete the CAT." + "Follow up Activity:" + "1. <review task>. (0 Marks)" + "Due date:"; resources = CAT paper, writing materials, course notes; assessments = graded Knowledge plus Attitudes (honesty, self-reflection).

TERMINOLOGY - use Kenya Competency-Based Education and Training (CBET) terms ONLY: "trainee" (never student/pupil/learner), "trainer" (never teacher/lecturer), "unit of competency", "learning outcome", "performance criteria", "competency", "assessment" / "Continuous Assessment Test (CAT)" (never "exam" or "test" as a noun for the final). NEVER output placeholders like "Key concept 1". Return ONLY the JSON array.
"""


# --------------------------------------------------------------------------- #
# HTTP call
# --------------------------------------------------------------------------- #
def _post(model: str, api_key: str, prompt: str, timeout: int = 180) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.35,
        "max_tokens": 32000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "learning_plan_sessions",
                "schema": _response_schema(),
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


def call_mistral(prompt: str, api_key: str, model: str) -> List[dict]:
    """One grounded request with no fallback chain."""
    runlog.log(f"AI: trying model {model}")
    try:
        resp = _post(model, api_key, prompt)
    except requests.RequestException as e:
        raise AIError(f"{model}: network error: {e}") from e

    if resp.status_code == 200:
        try:
            text = _extract_text(resp.json())
        except AIError as e:
            raise AIError(f"{model}: {e}") from e
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise AIError(f"{model} returned invalid JSON: {e}") from e
        if isinstance(data, dict):
            data = data.get("sessions") or data.get("data") or [data]
        if not isinstance(data, list):
            raise AIError(f"{model} did not return a JSON array")
        runlog.log(f"AI: {model} succeeded ({len(data)} session rows)")
        return data

    if resp.status_code in (401, 403):
        raise AIError(
            "Mistral auth failed (HTTP %d). Your MISTRAL_API_KEY is invalid, "
            "expired, or lacks access." % resp.status_code)

    if resp.status_code == 429:
        raise AIError(f"{model}: rate/quota limited (HTTP 429){resp.text[:160]}")

    raise AIError(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")


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
    marks = 0 if s.is_cat else 15
    if s.is_cat:
        return [
            "- Engage in a Quiz Game to review key concepts.",
            "- Participate in Q&A Sessions to clarify doubts.",
            "- Complete the CAT.",
            "Follow up Activity:",
            "1. Review feedback and identify areas for improvement. (0 Marks)",
            "Due date:",
        ]
    return [
        "- Engage in a Group Discussion to explore the topic.",
        "- Participate in a Think-Pair-Share on key concepts.",
        "- Conduct a Case Study applying the concepts.",
        "Follow up Activity:",
        f"1. Complete a practical exercise on {s.session_title.lower()}. ({marks} Marks)",
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
    if s.is_cat:
        return ["- CAT question paper", "- Writing materials", "- Course notes and textbooks"]
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

        lo = _as_list(row.get("learning_outcomes"))
        ai_kp = _as_list(row.get("key_points"))
        acts = _as_list(row.get("trainee_activities"))
        res = _as_list(row.get("resources"))
        assess = _as_list(row.get("assessments"))

        s.learning_outcomes = lo or _default_learning_outcomes(s)
        if ai_kp:
            s.key_points = ai_kp                       # AI already CAPS-formatted
        elif s.is_cat:
            s.key_points = _cat_keypoints()
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


def _cat_keypoints() -> List[str]:
    return ["ASSESSMENT COVERAGE", "ASSESSMENT STRATEGY"]


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
                      model: Optional[str] = None) -> List[Session]:
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
    ai_rows = call_mistral(prompt, api_key, model)
    return merge_ai_into_sessions(sessions, ai_rows, unit)


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
