"""Module C - THE SINGLE AI CALL (Google Gemini, ONE grounded request).

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
import time
from typing import List, Optional

import requests

import runlog
from models import Session, Unit

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Hard-coded Gemini key (per request) so it is NOT entered in the UI.
# SECURITY NOTE: this key is now embedded in source - do NOT commit/share this
# file publicly or the key leaks. An env var GEMINI_API_KEY (in .env) overrides it.
DEFAULT_API_KEY = "AQ.Ab8RN6JsVvoaxpg8WEMZIIg1BEo2Q4Qajxd5qhHKysHxVzCwvg"

# Ordered model "switch case": the single call walks this chain top-to-bottom and
# advances to the NEXT model the instant the current one is unavailable (quota
# exhausted / HTTP 429 / transient error), returning the first one that responds.
#
# Gemini 2.0 is listed FIRST (preferred) so it is used the moment a billing-enabled
# key makes it available. NOTE: on free-tier keys Google currently reports
# `quota limit: 0` for the entire 2.0 family AND the *-pro tiers, so those are
# skipped instantly and the call lands on the first working flash model (2.5+).
# Override the whole chain with GEMINI_MODEL / GEMINI_FALLBACK_MODELS in .env.
MODEL_CHAIN = [
    "gemini-2.0-flash",          # preferred; quota-limited on free-tier keys
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",          # first model that works on a free-tier key
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",            # high-quality last resorts (quota permitting)
    "gemini-pro-latest",
]
DEFAULT_MODEL = MODEL_CHAIN[0]
DEFAULT_FALLBACK_MODELS = MODEL_CHAIN[1:]
DEFAULT_FALLBACK_MODEL = DEFAULT_FALLBACK_MODELS[0]  # back-compat


class AIError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Response schema (forces valid structured JSON)
# --------------------------------------------------------------------------- #
def _response_schema() -> dict:
    str_array = {"type": "ARRAY", "items": {"type": "STRING"}}
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "week": {"type": "INTEGER"},
                "session_no": {"type": "STRING"},
                "is_cat": {"type": "BOOLEAN"},
                "session_title": {"type": "STRING"},
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
    url = GEMINI_ENDPOINT.format(model=model)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 32000,
            "responseMimeType": "application/json",
            "responseSchema": _response_schema(),
        },
    }
    resp = requests.post(url, params={"key": api_key},
                         headers={"Content-Type": "application/json"},
                         json=body, timeout=timeout)
    return resp


def _extract_text(payload: dict) -> str:
    cands = payload.get("candidates") or []
    if not cands:
        raise AIError("Gemini returned no candidates: " + json.dumps(payload)[:300])
    parts = cands[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _quota_reason(resp) -> str:
    """Short human note for a 429: flag the 'free-tier quota is 0' case so the log
    explains WHY a model is being skipped (vs a transient per-minute rate limit)."""
    body = resp.text or ""
    if "limit: 0" in body or '"limit": 0' in body or '"limit":0' in body:
        return " - free-tier quota is 0 for this model (billing required)"
    return ""


def call_gemini(prompt: str, api_key: str, model: str,
                fallback_model=None, max_retries: int = 3) -> List[dict]:
    """One grounded request that walks an ordered model chain ("switch case").

    Tries `model`, then each entry of `fallback_model` (a name or list of names),
    advancing to the NEXT model the moment the current one is unavailable - quota
    exhausted (429), invalid output, or a transient/server error. Returns the JSON
    rows from the first model that responds. Each attempt is logged so the UI Log
    viewer shows the chain progressing. A 401/403 aborts immediately (the key,
    not the model, is the problem). `max_retries` is accepted for back-compat.
    """
    if fallback_model is None:
        fallbacks = []
    elif isinstance(fallback_model, (list, tuple)):
        fallbacks = list(fallback_model)
    else:
        fallbacks = [fallback_model]
    models = [model] + fallbacks
    last_err: Optional[str] = None

    for mdl in models:
        runlog.log(f"AI: trying model {mdl}")
        try:
            resp = _post(mdl, api_key, prompt)
        except requests.RequestException as e:
            last_err = f"{mdl}: network error: {e}"
            runlog.warn(f"AI: {mdl} network error ({e}) - trying next model")
            continue

        if resp.status_code == 200:
            text = _extract_text(resp.json())
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                last_err = f"{mdl} returned invalid JSON: {e}"
                runlog.warn(f"AI: {mdl} returned invalid JSON - trying next model")
                continue
            if isinstance(data, dict):
                data = data.get("sessions") or data.get("data") or [data]
            if not isinstance(data, list):
                last_err = f"{mdl} did not return a JSON array"
                runlog.warn(f"AI: {mdl} returned non-array JSON - trying next model")
                continue
            runlog.log(f"AI: {mdl} succeeded ({len(data)} session rows)")
            return data

        if resp.status_code in (401, 403):
            raise AIError(
                "Gemini auth failed (HTTP %d). Your GEMINI_API_KEY is invalid, "
                "expired, or lacks access. Note: AI-Studio keys (incl. AQ.* "
                "OAuth-style keys) are passed via ?key= and CAN expire - "
                "regenerate one at aistudio.google.com." % resp.status_code)

        if resp.status_code == 429:
            reason = _quota_reason(resp)
            last_err = f"{mdl}: rate/quota limited (HTTP 429){reason}"
            runlog.warn(f"AI: {mdl} unavailable (429{reason}) - trying next model")
            continue

        # any other error (4xx/5xx): skip to the next model
        last_err = f"{mdl}: HTTP {resp.status_code} {resp.text[:200]}"
        runlog.warn(f"AI: {mdl} HTTP {resp.status_code} - trying next model")

    raise AIError(last_err or "All Gemini models in the fallback chain were "
                  "unavailable. Add a billing-enabled GEMINI_API_KEY to .env.")


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
            # offline fallback: render the curriculum content as CAPS headings so
            # the no-AI path still matches the RVNP Learning-Key-Points style.
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
                      model: Optional[str] = None,
                      fallback_model=None,
                      offline_ok: bool = False) -> List[Session]:
    """Run the single grounded AI call and merge the result into the skeleton.

    If `offline_ok` and no key is present, fall back to deterministic defaults so
    the pipeline still yields a complete document (used for tests / no-key demos).
    The model chain (`model` + `fallback_model`) defaults to the .env-overridable
    MODEL_CHAIN when the caller does not pass explicit values.
    """
    # Key semantics: api_key=None  -> use the configured (.env / hard-coded) key
    #                api_key=""    -> caller FORCES offline (no API call)
    # (Previously the key was only loaded when offline_ok was False, so the app's
    #  online call - api_key=None, offline_ok=True - never loaded a key and always
    #  fell back to offline defaults. That was the real "AI never goes through" bug.)
    if api_key is None:
        api_key = load_api_key()
    if not api_key:
        if offline_ok:
            runlog.warn("AI: no API key - using grounded offline defaults")
            return merge_ai_into_sessions(sessions, [], unit)
        raise AIError("No Gemini API key available.")

    # Resolve the model chain (env-overridable) unless the caller pinned one.
    if model is None or fallback_model is None:
        chain = load_model_chain()
        if model is None:
            model = chain[0]
        if fallback_model is None:
            fallback_model = chain[1:]
    runlog.log(f"AI: model chain = {[model] + list(fallback_model)}")

    prompt = build_prompt(unit, sessions)
    ai_rows = call_gemini(prompt, api_key, model, fallback_model)
    return merge_ai_into_sessions(sessions, ai_rows, unit)


def load_api_key() -> str:
    """The hard-coded key, overridable by GEMINI_API_KEY in the environment/.env."""
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("GEMINI_API_KEY", "").strip() or DEFAULT_API_KEY


def load_model_chain() -> List[str]:
    """Ordered model chain, overridable via .env.

    GEMINI_MODEL sets the primary model; GEMINI_FALLBACK_MODELS (comma-separated,
    or the singular GEMINI_FALLBACK_MODEL) sets the rest. Omit both to use the
    built-in MODEL_CHAIN. The primary is always tried first, so listing 2.0 there
    keeps it preferred whenever a billing-enabled key makes it available.
    """
    from dotenv import load_dotenv
    load_dotenv()
    primary = os.getenv("GEMINI_MODEL", "").strip()
    fb_raw = (os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
              or os.getenv("GEMINI_FALLBACK_MODEL", "").strip())
    if not primary and not fb_raw:
        return list(MODEL_CHAIN)
    chain: List[str] = [primary] if primary else [MODEL_CHAIN[0]]
    chain += [m.strip() for m in fb_raw.split(",") if m.strip()]
    # de-duplicate while preserving order
    seen, out = set(), []
    for m in chain:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out or list(MODEL_CHAIN)


# --------------------------------------------------------------------------- #
# Tiny AI fallback for OS parsing (ONLY when deterministic parsing yields zero
# units, e.g. a scanned / garbled OS). This is a safety hatch, not the norm.
# --------------------------------------------------------------------------- #
def ai_extract_unit_fallback(raw_text: str, api_key: str,
                             model: str = "gemini-2.5-flash",
                             fallback_model: str = "gemini-2.5-flash-lite") -> Unit:
    from models import Element, PerformanceCriterion
    schema = {
        "type": "OBJECT",
        "properties": {
            "unit_title": {"type": "STRING"},
            "os_code": {"type": "STRING"},
            "level": {"type": "STRING"},
            "description": {"type": "STRING"},
            "assessment_methods": {"type": "ARRAY", "items": {"type": "STRING"}},
            "elements": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "number": {"type": "STRING"},
                        "title": {"type": "STRING"},
                        "performance_criteria": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {"number": {"type": "STRING"},
                                               "text": {"type": "STRING"}},
                            },
                        },
                    },
                },
            },
        },
    }
    prompt = ("Extract the Kenya CDACC unit of competency from this Occupational "
              "Standard text. Return ONLY JSON with unit_title, os_code, level, "
              "description, assessment_methods[], and elements[] each with number, "
              "title and performance_criteria[] (number like '1.1', text).\n\n"
              + raw_text[:14000])
    body_schema = schema
    # reuse the HTTP path but with this object schema
    url = GEMINI_ENDPOINT.format(model=model)
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8000,
                                 "responseMimeType": "application/json",
                                 "responseSchema": body_schema}}
    resp = requests.post(url, params={"key": api_key},
                         headers={"Content-Type": "application/json"},
                         json=body, timeout=120)
    if resp.status_code != 200:
        raise AIError(f"OS fallback failed: HTTP {resp.status_code} {resp.text[:200]}")
    data = json.loads(_extract_text(resp.json()))
    unit = Unit(unit_title=data.get("unit_title", ""), os_code=data.get("os_code", ""),
                level=str(data.get("level", "")), description=data.get("description", ""),
                assessment_methods=_as_list(data.get("assessment_methods")))
    unit.skill_task = unit.description
    for el in data.get("elements", []):
        e = Element(number=str(el.get("number", "")), title=el.get("title", ""))
        for pc in el.get("performance_criteria", []):
            e.performance_criteria.append(
                PerformanceCriterion(number=str(pc.get("number", "")),
                                     text=pc.get("text", "")))
        unit.elements.append(e)
    return unit
