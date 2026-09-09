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

# The model chosen for this process, once something has answered.
_RESOLVED_MODEL = ""

# Sessions per grounded Learning-Plan call. A full term (e.g. 24 sessions) does
# not fit in one JSON response - it gets truncated to invalid JSON - so we
# generate in batches and concatenate the rows in order.
#
# Four, not eight, because Groq's binding limit is not its 1,000 requests a day
# but 8,000 TOKENS a minute on the gpt-oss models: eight sessions (~2k in, ~6k
# out) spend a whole minute's allowance in one request.
LP_SESSION_CHUNK = 4

# Ceiling on one answer, for the same reason. Above this a single request can
# exceed the per-minute token allowance and be refused outright.
MAX_OUTPUT_TOKENS = 6000


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

    Retry-After first; Groq also puts the same information in its
    x-ratelimit-reset-* headers, and sends those when Retry-After is absent.
    """
    headers = getattr(resp, "headers", None) or {}
    wait = _parse_duration(headers.get("Retry-After", ""))
    if wait:
        return wait
    resets = [_parse_duration(v) for k, v in headers.items()
              if "ratelimit-reset" in k.lower()]
    return max([w for w in resets if w] or [0.0])


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
          max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """POST one chat completion, always in strict JSON-schema mode.

    Every call this module makes wants structured output, the probe included -
    a model that chats but won't take the schema is no use here.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
               schema_name: str, progress_cb=None, temperature: float = 0.2):
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
                                   temperature=temperature)
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
                    schema_name: str, progress_cb=None, temperature: float = 0.2):
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
            resp = _post(model, api_key, prompt, schema=schema,
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
                      "learning_plan_sessions", progress_cb=progress_cb)
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

        # CAT rows are deterministic - never the AI's output - but CONTEXTUAL:
        # they summarise the content sessions this CAT assesses (those since the
        # previous CAT), which are already filled earlier in this in-order pass.
        if s.is_cat:
            _apply_cat_content(sessions, i)
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
    # within a family, more parameters first: ministral-14b before ministral-3b
    size = _RE_MODEL_SIZE.search(lower)
    return (base, -int(size.group(1)) if size else 0, name)


def _candidate_models(model: str, api_key: str) -> List[str]:
    """The configured model first, then every other one, most capable first."""
    try:
        discovered = list_chat_models(api_key)
    except AIError:
        discovered = []               # can't list: the configured model is all we have
    usable = [m for m in discovered
              if not _RE_SPECIAL_PURPOSE.search(m) and m != model]
    ordered = [model] + sorted(usable, key=_model_rank)
    return [m for m in ordered if m not in _UNAVAILABLE_MODELS]


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
                     schema=schema, schema_name="probe", max_tokens=20)
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
    size = batch_size_for(model)
    chunks = [sessions[i:i + size] for i in range(0, len(sessions), size)]
    ai_rows: List[dict] = []
    for ci, chunk in enumerate(chunks, start=1):
        first, last = len(ai_rows) + 1, len(ai_rows) + len(chunk)
        _emit_progress(progress_cb,
                       f"AI: generating sessions {first}-{last} of {len(sessions)} "
                       f"(batch {ci}/{len(chunks)})")
        # a batch may have moved us to another model; the rest follow it there
        model = resolve_model(api_key, model if model not in _UNAVAILABLE_MODELS
                              else None, progress_cb)
        ai_rows.extend(_generate_batch(unit, chunk, api_key, model, progress_cb))
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
                      temperature=0.6)
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
            "required": ["step_label", "minutes", "trainer_activity",
                         "trainee_activity", "learning_check"],
        },
    }


def _session_plan_schema() -> dict:
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "introduction": str_array,
            "delivery_steps": _delivery_steps_schema(),
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
                    "session_plan", progress_cb=progress_cb)
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
