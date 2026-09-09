"""Robustness of the Mistral response path. ZERO real API calls - the HTTP layer
(`ai_client._post`) is monkeypatched to return canned responses.

Covers the case that motivated the hardening: the provider returns HTTP 200 but
the choice carries only a finish reason and no assistant content.
That must raise a clean AIError and stop, not fall back to another model.
"""

import pytest
import requests

import json

import ai_client
from ai_client import AIError, _extract_text, call_mistral
from models import Session, Unit


class FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _ok_payload(text):
    return {"choices": [{"message": {"content": text}}]}


# The fixture below stubs discovery out; this keeps a handle on the real one.
REAL_LIST_CHAT_MODELS = ai_client.list_chat_models

DISCOVERED = ["mistral-medium-latest", "mistral-small-latest",
              "ministral-14b-latest", "ministral-8b-latest", "codestral-latest"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Nothing in this module may touch the API - model discovery included.

    `generate_sessions` asks Mistral which models exist and probes one before
    generating. Tests that stub only `call_mistral` would otherwise have let
    that out onto the network.
    """
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(200, _ok_payload("[]")))
    monkeypatch.setattr(ai_client, "list_chat_models", lambda key: list(DISCOVERED))
    # what one test learns about the workspace's plan must not leak into another
    monkeypatch.setattr(ai_client, "_UNAVAILABLE_MODELS", set())
    monkeypatch.setattr(ai_client, "_PROVEN_MODELS", set())
    monkeypatch.setattr(ai_client, "_RESOLVED_MODEL", "")


# --------------------------------------------------------------------------- #
# _extract_text
# --------------------------------------------------------------------------- #
def test_extract_text_returns_joined_parts():
    assert _extract_text(_ok_payload("hello")) == "hello"


def test_extract_text_no_choices_raises_aierror():
    with pytest.raises(AIError):
        _extract_text({"choices": []})


def test_extract_text_null_content_raises_aierror_not_attributeerror():
    # content: null must be AIError, not AttributeError.
    with pytest.raises(AIError):
        _extract_text({"choices": [{"message": {"content": None},
                                    "finish_reason": "content_filter"}]})


def test_extract_text_surfaces_finish_reason():
    with pytest.raises(AIError) as ei:
        _extract_text({"choices": [{"finish_reason": "length"}]})
    assert "length" in str(ei.value)


# --------------------------------------------------------------------------- #
# call_mistral behaviour
# --------------------------------------------------------------------------- #
def test_call_mistral_raises_aierror_on_empty_choice(monkeypatch):
    monkeypatch.setattr(
        ai_client, "_post",
        lambda model, api_key, prompt, timeout=180, **kwargs:
            FakeResp(200, {"choices": [{"finish_reason": "content_filter"}]}))
    with pytest.raises(AIError):
        call_mistral("prompt", "key", "m1")


def test_call_mistral_raises_aierror_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        ai_client, "_post",
        lambda model, api_key, prompt, timeout=180, **kwargs:
            FakeResp(200, _ok_payload("not json at all")))
    with pytest.raises(AIError):
        call_mistral("p", "k", "m1")


def test_call_mistral_aborts_immediately_on_403(monkeypatch):
    """A 403 is a key problem, not a model problem."""
    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        return FakeResp(403, text="PERMISSION_DENIED")

    monkeypatch.setattr(ai_client, "_post", fake_post)
    with pytest.raises(AIError):
        call_mistral("p", "k", "m1")


def test_one_timeout_does_not_condemn_a_model(monkeypatch):
    """ministral-8b answers a batch in seconds; a single slow minute must not
    rule it out for the rest of the session."""
    seen = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        seen.append(model)
        if len(seen) == 1:
            raise requests.Timeout("busy")
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    assert call_mistral("p", "k", "ministral-8b-latest") == []
    assert seen == ["ministral-8b-latest", "ministral-8b-latest"]
    assert "ministral-8b-latest" not in ai_client._UNAVAILABLE_MODELS


def test_a_proven_model_retries_a_transient_timeout(monkeypatch):
    """Once a model has delivered, a timeout is a blip - keep the model."""
    attempts = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        attempts.append(model)
        if len(attempts) < 3:
            raise requests.Timeout("timed out")
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ai_client, "_PROVEN_MODELS", {"m1"})

    assert call_mistral("p", "k", "m1") == []
    assert attempts == ["m1", "m1", "m1"]


def test_a_model_that_has_never_delivered_and_hangs_is_replaced(monkeypatch):
    """ministral-14b answers a probe in a second and then never finishes a
    real batch under the same schema. That is not a model that works."""
    seen = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        seen.append((model, timeout))
        if model == "mistral-small-latest":
            raise requests.Timeout("timed out")
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    assert call_mistral("p", "k", "mistral-small-latest") == []
    # tried it twice - one timeout is usually the API being busy - then moved on
    assert [m for m, _ in seen].count("mistral-small-latest") ==         ai_client.UNPROVEN_ATTEMPTS
    # and gave the unproven model less rope than a proven one gets
    assert seen[0][1] == ai_client.UNPROVEN_TIMEOUT
    assert seen[-1][0] != "mistral-small-latest"


def test_call_mistral_retries_connection_reset(monkeypatch):
    attempts = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise requests.ConnectionError(
                "Connection aborted.",
                ConnectionResetError(10054, "An existing connection was forcibly closed", None, 10054, None),
            )
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    assert call_mistral("p", "k", "m1") == []
    assert len(attempts) == 3


# --------------------------------------------------------------------------- #
# Rate limiting (HTTP 429)
# --------------------------------------------------------------------------- #
def test_a_rate_limit_is_waited_out_rather_than_failing_the_generation(monkeypatch):
    """A Learning Plan is several calls in a row, so Mistral's per-minute limit
    is met routinely - and it clears in seconds. Failing threw away every batch
    already generated."""
    attempts = []
    slept = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        attempts.append(model)
        if len(attempts) < 3:
            return FakeResp(429, text='{"message": "Rate limit exceeded"}')
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", slept.append)

    assert call_mistral("p", "k", "m1") == []
    assert len(attempts) == 3
    assert slept == list(ai_client._RATE_LIMIT_BACKOFF[:2])


def test_the_servers_own_retry_after_wins_over_our_backoff(monkeypatch):
    slept = []
    calls = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        calls.append(model)
        if len(calls) == 1:
            resp = FakeResp(429, text="slow down")
            resp.headers = {"Retry-After": "7"}
            return resp
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", slept.append)

    assert call_mistral("p", "k", "m1") == []
    assert slept == [7.0]


def test_an_absurd_retry_after_is_capped(monkeypatch):
    resp = FakeResp(429, text="")
    resp.headers = {"Retry-After": "86400"}
    assert ai_client._retry_after(resp) == ai_client._MAX_RETRY_AFTER


def test_a_retry_after_date_falls_back_to_our_own_backoff():
    resp = FakeResp(429, text="")
    resp.headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert ai_client._retry_after(resp) == 0.0


def test_a_response_without_headers_does_not_break_the_retry():
    assert ai_client._retry_after(FakeResp(429, text="")) == 0.0


def test_a_persistent_rate_limit_gives_up_with_something_actionable(monkeypatch):
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(429, text="Rate limit exceeded"))
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    with pytest.raises(AIError) as excinfo:
        call_mistral("p", "k", "m1")
    message = str(excinfo.value)
    assert "429" in message
    assert "minute" in message.lower()


# --------------------------------------------------------------------------- #
# A model the workspace's plan doesn't include
# --------------------------------------------------------------------------- #
def _no_allowance_resp():
    """What Mistral sends for a model a free workspace may not call: a rate
    limit whose allowance is zero, which no amount of waiting will clear."""
    resp = FakeResp(429, text='{"message": "Rate limit exceeded"}')
    resp.headers = {"x-ratelimit-limit-req-minute": "0",
                    "x-ratelimit-remaining-req-minute": "0"}
    return resp


def test_a_zero_allowance_is_told_apart_from_a_busy_minute():
    assert ai_client._has_no_allowance(_no_allowance_resp())
    busy = FakeResp(429, text="")
    busy.headers = {"x-ratelimit-limit-req-minute": "188",
                    "x-ratelimit-remaining-req-minute": "0"}
    assert not ai_client._has_no_allowance(busy)
    assert not ai_client._has_no_allowance(FakeResp(429, text=""))


def _plan_allows(*allowed):
    """A fake _post where only `allowed` models answer; the rest are excluded
    from the plan the way Mistral does it - zero requests a minute."""
    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        if model in allowed:
            return FakeResp(200, _ok_payload('{"ok": "yes"}'))
        return _no_allowance_resp()
    return fake_post


def test_the_configured_model_is_used_when_the_plan_allows_it(monkeypatch):
    monkeypatch.setattr(ai_client, "_post", _plan_allows("mistral-small-latest"))
    assert ai_client.resolve_model("k", "mistral-small-latest") ==         "mistral-small-latest"


def test_a_model_the_plan_excludes_gives_way_to_one_that_answers(monkeypatch):
    """Mistral allows an excluded model zero requests a minute, so waiting can
    never clear it - the only way on is a model the plan does include."""
    monkeypatch.setattr(ai_client, "_post", _plan_allows("ministral-14b-latest"))
    assert ai_client.resolve_model("k", "mistral-small-latest") ==         "ministral-14b-latest"


def test_the_candidates_come_from_the_api_most_capable_first(monkeypatch):
    """The list is whatever the key can see, not a list written in here."""
    order = ai_client._candidate_models("mistral-small-latest", "k")
    assert order[0] == "mistral-small-latest"          # the configured one leads
    assert order[1:] == ["mistral-medium-latest", "ministral-14b-latest",
                         "ministral-8b-latest"]


def test_a_code_model_is_never_chosen_to_write_lesson_content(monkeypatch):
    """codestral answers perfectly well; it just writes poor lesson prose."""
    assert "codestral-latest" not in ai_client._candidate_models("m", "k")
    for name in ("voxtral-small-latest", "codestral-2508", "labs-leanstral-1-5",
                 "mistral-ocr-latest", "mistral-vibe-cli-fast"):
        assert ai_client._RE_SPECIAL_PURPOSE.search(name), name


def test_a_model_this_code_has_never_heard_of_is_still_tried(monkeypatch):
    """Ranked in the middle - ahead of the small models, behind the big ones -
    so a model released after this was written gets its turn."""
    monkeypatch.setattr(ai_client, "list_chat_models",
                        lambda key: ["ministral-8b-latest", "mistral-zeta-latest",
                                     "mistral-medium-latest"])
    order = ai_client._candidate_models("mistral-small-latest", "k")
    assert order == ["mistral-small-latest", "mistral-medium-latest",
                     "mistral-zeta-latest", "ministral-8b-latest"]


def test_the_bigger_model_of_a_family_goes_first():
    ranked = sorted(["ministral-3b-latest", "ministral-14b-latest",
                     "ministral-8b-latest"], key=ai_client._model_rank)
    assert ranked == ["ministral-14b-latest", "ministral-8b-latest",
                      "ministral-3b-latest"]


def test_aliases_of_one_model_are_not_eight_candidates(monkeypatch):
    """Mistral lists mistral-medium-latest under nine names."""
    payload = {"data": [
        {"id": "mistral-medium-2604", "capabilities": {"completion_chat": True},
         "aliases": ["mistral-medium-latest", "mistral-medium"]},
        {"id": "mistral-medium-latest", "capabilities": {"completion_chat": True},
         "aliases": ["mistral-medium-2604", "mistral-medium"]},
        {"id": "mistral-embed", "capabilities": {"completion_chat": False},
         "aliases": []},
    ]}

    class Resp:
        status_code = 200

        def json(self):
            return payload

    monkeypatch.setattr(ai_client.requests, "get", lambda *a, **k: Resp())
    assert REAL_LIST_CHAT_MODELS("k") == ["mistral-medium-latest"]


def test_a_busy_minute_does_not_disqualify_a_model(monkeypatch):
    """A used-up allowance is not the same as no allowance."""
    busy = FakeResp(429, text="Rate limit exceeded")
    busy.headers = {"x-ratelimit-limit-req-minute": "188",
                    "x-ratelimit-remaining-req-minute": "0"}
    monkeypatch.setattr(ai_client, "_post", lambda *a, **k: busy)
    assert ai_client.resolve_model("k", "mistral-small-latest") ==         "mistral-small-latest"


def test_the_model_is_settled_once_for_the_process(monkeypatch):
    """A Learning Plan is many calls; re-choosing on each wasted a round trip."""
    probes = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        probes.append(model)
        return _plan_allows("ministral-14b-latest")(model, api_key, prompt)

    monkeypatch.setattr(ai_client, "_post", fake_post)
    first = ai_client.resolve_model("k", "mistral-small-latest")
    before = len(probes)
    second = ai_client.resolve_model("k", "mistral-small-latest")
    assert first == second == "ministral-14b-latest"
    assert len(probes) == before          # settled, not asked again


def test_a_model_that_stops_answering_mid_run_is_replaced(monkeypatch):
    """The plan can change under a running generation."""
    calls = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        calls.append(model)
        if model == "mistral-small-latest":
            return _no_allowance_resp()
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client, "_RESOLVED_MODEL", "mistral-small-latest")
    assert call_mistral("p", "k", "mistral-small-latest") == []
    assert calls[0] == "mistral-small-latest"
    assert calls[-1] != "mistral-small-latest"


def test_when_nothing_answers_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setattr(ai_client, "_post", _plan_allows())      # nothing answers
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    with pytest.raises(AIError) as excinfo:
        ai_client.resolve_model("k", "mistral-small-latest")
    message = str(excinfo.value)
    assert "console.mistral.ai" in message
    assert "MISTRAL_MODEL" in message
    assert "ministral-14b-latest" in message      # names what it tried


def test_batches_are_sized_for_the_model_that_answers():
    assert ai_client.batch_size_for("mistral-small-latest") ==         ai_client.LP_SESSION_CHUNK
    assert ai_client.batch_size_for("mistral-medium-latest") ==         ai_client.LP_SESSION_CHUNK
    for small in ("ministral-14b-latest", "ministral-8b-latest",
                  "open-mistral-nemo"):
        assert ai_client.batch_size_for(small) == ai_client.SMALL_MODEL_CHUNK


def test_a_real_auth_failure_is_still_reported_as_one(monkeypatch):
    """A 403 that isn't about the tier must not be read as a missing model."""
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(403, text="Unauthorized"))
    with pytest.raises(AIError, match="auth failed"):
        call_mistral("p", "k", "m1")


# --------------------------------------------------------------------------- #
# Truncated-response salvage (the token-limit / unterminated-JSON case)
# --------------------------------------------------------------------------- #
def test_salvage_recovers_complete_objects_from_truncated_array():
    text = '[{"a": 1}, {"b": 2}, {"c": "unterminated string...'
    assert ai_client._salvage_json_array(text) == [{"a": 1}, {"b": 2}]


def test_salvage_ignores_braces_and_brackets_inside_strings():
    text = '[{"x": "a } b", "y": [1, 2]}, {"z": "trunc'
    assert ai_client._salvage_json_array(text) == [{"x": "a } b", "y": [1, 2]}]


def test_salvage_keeps_the_rows_in_front_of_a_broken_one():
    """A smaller model breaks JSON mid-array - an unescaped quote inside a
    string - and taking only the longest prefix threw away the good rows."""
    text = '[{"a": 1}, {"b": 2}, {"c": "he said "no" loudly"}, {"d": 4}]'
    assert ai_client._salvage_json_array(text) == [{"a": 1}, {"b": 2}]


def test_salvage_returns_none_when_no_complete_object():
    assert ai_client._salvage_json_array('[{"a": "unterminated') is None
    assert ai_client._salvage_json_array("no array here") is None


def test_chat_json_salvages_truncated_learning_plan(monkeypatch):
    # a 3rd session object is cut off mid-string -> keep the first two
    truncated = ('[{"week": 1, "session_title": "Intro"}, '
                 '{"week": 2, "session_title": "Tools"}, '
                 '{"week": 3, "session_title": "Trunca')
    monkeypatch.setattr(
        ai_client, "_post",
        lambda model, api_key, prompt, timeout=180, **kwargs:
            FakeResp(200, _ok_payload(truncated)))
    rows = call_mistral("p", "k", "m1")
    assert [r["week"] for r in rows] == [1, 2]


# --------------------------------------------------------------------------- #
# Per-session Learning-Plan regeneration
# --------------------------------------------------------------------------- #
def _lp_unit_session():
    unit = Unit(unit_title="Apply Market Research", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    session = Session(week=2, session_no="1", is_cat=False,
                      session_title="Market Research Tools",
                      pcs=["1.1 tools are identified"], key_points=["TOOLS"],
                      learning_outcomes=["old"], trainee_activities=["old"],
                      resources=["old"], assessments=["old"])
    return unit, session


def test_regenerate_learning_plan_session_updates_in_place(monkeypatch):
    unit, session = _lp_unit_session()
    row = {"week": 2, "session_no": "1", "session_title": "Market Research Tools",
           "learning_outcomes": ["By the end...", "a. Identify tools."],
           "key_points": ["TOOLS"],
           "trainee_activities": ["- Group Discussion.", "- Case Study.",
                                  "- Demonstration."],
           "resources": ["- Kotler (2021).", "- Slides."],
           "assessments": ["Knowledge Checks:", "1. Oral questioning"]}
    monkeypatch.setattr(
        ai_client, "_post",
        lambda *a, **k: FakeResp(200, _ok_payload(json.dumps([row]))))
    out = ai_client.regenerate_learning_plan_session(
        unit, [session], 0, api_key="k", model="m")
    assert out is session                                   # mutated in place
    assert any("Identify tools" in x for x in session.learning_outcomes)
    assert any("Kotler" in x for x in session.resources)
    # deterministic skeleton is preserved
    assert session.week == 2 and session.pcs == ["1.1 tools are identified"]


def test_regenerate_cat_session_is_deterministic_and_contextual(monkeypatch):
    unit, content = _lp_unit_session()
    content.session_title = "Programming Languages"
    cat = Session(week=4, session_no="1", is_cat=True,
                  session_title="CAT 1 (Continuous Assessment Test)")
    sessions = [content, cat]
    # even if the API is called it must NOT be used for a CAT row
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(200, _ok_payload("[]")))
    out = ai_client.regenerate_learning_plan_session(
        unit, sessions, 1, api_key="k", model="m")
    assert out.key_points[0] == "ASSESSMENT COVERAGE"
    assert "- Programming Languages" in out.key_points            # contextual coverage
    assert out.learning_outcomes[0].startswith("By the end")
    assert any("programming languages" in x.lower() for x in out.learning_outcomes)
    assert out.trainee_activities[0].startswith(
        "- Complete the Continuous Assessment Test (CAT 1)")
    assert out.assessments[0] == "Knowledge Checks:"


def test_regenerate_learning_plan_session_empty_raises(monkeypatch):
    unit, session = _lp_unit_session()
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(200, _ok_payload("[]")))
    with pytest.raises(AIError):
        ai_client.regenerate_learning_plan_session(
            unit, [session], 0, api_key="k", model="m")


def test_cat_content_is_scoped_to_sessions_since_previous_cat():
    unit = Unit(unit_title="U", os_code="X/OS/1", level="6")

    def content(title):
        return Session(is_cat=False, session_title=title, pcs=["1.1 x"],
                       key_points=[title.upper()])

    def cat(n):
        return Session(is_cat=True,
                       session_title=f"CAT {n} (Continuous Assessment Test)")

    sessions = [content("Alpha"), content("Beta"), cat(1),
                content("Gamma"), cat(2)]
    ai_client.merge_ai_into_sessions(sessions, [], unit)   # no AI -> deterministic
    cat1, cat2 = sessions[2], sessions[4]

    assert cat1.key_points == ["ASSESSMENT COVERAGE", "- Alpha", "- Beta"]
    assert cat2.key_points == ["ASSESSMENT COVERAGE", "- Gamma"]   # only since CAT 1
    assert cat1.trainee_activities[0].startswith(
        "- Complete the Continuous Assessment Test (CAT 1)")
    assert cat2.trainee_activities[0].startswith(
        "- Complete the Continuous Assessment Test (CAT 2)")
    # every CAT carries the fixed Knowledge-checks / Attitudes template
    assert cat1.assessments[0] == "Knowledge Checks:"
    assert "Attitudes:" in cat1.assessments
    # learning outcomes list what was covered, one competency per covered session
    assert cat1.learning_outcomes[0].startswith("By the end")
    assert len([x for x in cat1.learning_outcomes if x[:2] in ("a.", "b.")]) == 2


def test_generate_sessions_batches_long_terms(monkeypatch):
    """A 20-session term must be split into ceil(20/8)=3 grounded calls."""
    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do the thing"],
                        key_points=["KEY POINT"]) for i in range(20)]
    calls = []
    monkeypatch.setattr(
        ai_client, "call_mistral",
        lambda prompt, api_key, model, progress_cb=None: (calls.append(1) or []))
    out = ai_client.generate_sessions(unit, sessions, api_key="k", model="m")
    assert len(calls) == 3                 # 3 batches for 20 sessions at chunk 8
    assert len(out) == 20                  # every session still present (backfilled)


def test_a_short_batch_is_retried_smaller_and_never_slides_the_rest(monkeypatch):
    """Rows are positional, so a truncated batch used to shift every later
    session's content onto the wrong row."""
    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do the thing"],
                        key_points=["KEY POINT"]) for i in range(16)]
    sizes = []

    def fake_call(prompt, api_key, model, progress_cb=None):
        # the model manages at most 3 rows before it runs out of room
        asked = prompt.count('"session_title"')
        sizes.append(asked)
        return [{"session_title": f"row {i}"} for i in range(min(asked, 3))]

    monkeypatch.setattr(ai_client, "call_mistral", fake_call)
    out = ai_client.generate_sessions(unit, sessions, api_key="k", model="m")

    assert len(out) == 16
    assert min(sizes) <= 3                  # it did split down to a size that fits
    # every session got a generated title, none left on the skeleton's
    assert all(s.session_title.startswith("row ") for s in out)


def test_a_batch_that_returns_nothing_is_not_split(monkeypatch):
    """No rows is a different failure from a truncated answer, and halving it
    only multiplies the wasted calls."""
    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do it"],
                        key_points=["KEY POINT"]) for i in range(8)]
    calls = []
    monkeypatch.setattr(
        ai_client, "call_mistral",
        lambda prompt, api_key, model, progress_cb=None: (calls.append(1) or []))
    out = ai_client.generate_sessions(unit, sessions, api_key="k", model="m")
    assert len(calls) == 1
    assert len(out) == 8


def test_the_batches_are_sized_for_the_model_that_will_answer(monkeypatch):
    """Discovering the refusal on the first real batch wasted a full
    eight-session prompt, and sized that batch for a model that never ran."""
    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do it"],
                        key_points=["KEY POINT"]) for i in range(9)]
    batch_sizes = []

    monkeypatch.setattr(ai_client, "_post", _plan_allows("ministral-14b-latest"))

    def fake_call(prompt, api_key, model, progress_cb=None):
        asked = prompt.count('"session_title"')
        batch_sizes.append(asked)
        return [{"session_title": f"row {i}"} for i in range(asked)]

    monkeypatch.setattr(ai_client, "call_mistral", fake_call)
    ai_client.generate_sessions(unit, sessions, api_key="k",
                                model="mistral-small-latest")
    assert batch_sizes == [3, 3, 3]        # not [8, 1]


def test_a_usable_model_keeps_the_full_batch_size(monkeypatch):
    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do it"],
                        key_points=["KEY POINT"]) for i in range(9)]
    sizes = []

    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(200, _ok_payload("[]")))
    monkeypatch.setattr(
        ai_client, "call_mistral",
        lambda prompt, api_key, model, progress_cb=None:
            (sizes.append(prompt.count('"session_title"')) or []))
    ai_client.generate_sessions(unit, sessions, api_key="k",
                                model="mistral-small-latest")
    assert sizes == [8, 1]


def test_default_model_prefers_smaller_variant():
    assert ai_client.DEFAULT_MODEL == "mistral-small-latest"


def test_build_prompt_includes_curriculum_only_instruction():
    unit = Unit(unit_title="Install and Configure Software", os_code="IT/OS/123",
                level="5", assessment_methods=["Observation"],
                required_knowledge=["Troubleshooting basics"])
    session = Session(week=1, session_no="1", is_cat=False,
                      session_title="Software installation",
                      pcs=["1.1 Install software"],
                      key_points=["Install software safely"])

    prompt = ai_client.build_prompt(unit, [session])

    assert "Do NOT invent syllabus content; use what is given." in prompt
    assert "Return ONLY the JSON array." in prompt
    assert "ASSESSMENT COVERAGE" in prompt
