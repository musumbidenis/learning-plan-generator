"""Robustness of the Groq response path. ZERO real API calls - the HTTP layer
(`ai_client._post`) is monkeypatched to return canned responses.

Covers the case that motivated the hardening: the provider returns HTTP 200 but
the choice carries only a finish reason and no assistant content.
That must raise a clean AIError and stop, not fall back to another model.
"""

import re

import pytest
import requests

import json

import ai_client
import doc_builder
import resource_finder
from ai_client import (AIError, _extract_text, _response_schema,
                       call_model)
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


# The fixture below stubs these out; these keep handles on the real ones.
REAL_LIST_CHAT_MODELS = ai_client.list_chat_models
REAL_POST = ai_client._post
REAL_REPAIR_ROWS = ai_client._repair_rows

DISCOVERED = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
              "llama-3.3-70b-versatile", "whisper-large-v3"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Nothing in this module may touch the API - model discovery included.

    `generate_sessions` asks Groq which models exist and probes one before
    generating. Tests that stub only `call_model` would otherwise have let
    that out onto the network.
    """
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: FakeResp(200, _ok_payload("[]")))
    monkeypatch.setattr(ai_client, "list_chat_models", lambda key: list(DISCOVERED))
    # Resource discovery reaches the internet - YouTube and every candidate URL
    # - so it is stubbed out here. Its own behaviour is covered in
    # tests/test_resource_finder.py against canned responses.
    monkeypatch.setattr(resource_finder, "find_for_unit",
                        lambda *a, **k: resource_finder.ResourcePool())
    # The corrective round is an extra call_model, which would otherwise show up
    # in every batching test's call count. It is exercised directly instead, in
    # the validation section below; `validate_rows` itself stays real.
    monkeypatch.setattr(ai_client, "_repair_rows",
                        lambda unit, sessions, rows, api_key, model,
                               progress_cb=None, pool=None: rows)
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
# call_model behaviour
# --------------------------------------------------------------------------- #
def test_call_model_raises_aierror_on_empty_choice(monkeypatch):
    monkeypatch.setattr(
        ai_client, "_post",
        lambda model, api_key, prompt, timeout=180, **kwargs:
            FakeResp(200, {"choices": [{"finish_reason": "content_filter"}]}))
    with pytest.raises(AIError):
        call_model("prompt", "key", "m1")


def test_call_model_raises_aierror_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        ai_client, "_post",
        lambda model, api_key, prompt, timeout=180, **kwargs:
            FakeResp(200, _ok_payload("not json at all")))
    with pytest.raises(AIError):
        call_model("p", "k", "m1")


def test_call_model_aborts_immediately_on_403(monkeypatch):
    """A 403 is a key problem, not a model problem."""
    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        return FakeResp(403, text="PERMISSION_DENIED")

    monkeypatch.setattr(ai_client, "_post", fake_post)
    with pytest.raises(AIError):
        call_model("p", "k", "m1")


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

    assert call_model("p", "k", "ministral-8b-latest") == []
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

    assert call_model("p", "k", "m1") == []
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

    assert call_model("p", "k", "mistral-small-latest") == []
    # tried it twice - one timeout is usually the API being busy - then moved on
    assert [m for m, _ in seen].count("mistral-small-latest") ==         ai_client.UNPROVEN_ATTEMPTS
    # and gave the unproven model less rope than a proven one gets
    assert seen[0][1] == ai_client.UNPROVEN_TIMEOUT
    assert seen[-1][0] != "mistral-small-latest"


def test_call_model_retries_connection_reset(monkeypatch):
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

    assert call_model("p", "k", "m1") == []
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

    assert call_model("p", "k", "m1") == []
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

    assert call_model("p", "k", "m1") == []
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
        call_model("p", "k", "m1")
    message = str(excinfo.value)
    assert "429" in message
    assert "minute" in message.lower()


# --------------------------------------------------------------------------- #
# A model this account can't use
# --------------------------------------------------------------------------- #
def _refusal_resp():
    """What Groq sends for a model that won't take the request: a 400 naming
    the reason. The key is fine, so this is not an auth failure."""
    return FakeResp(400, text='{"error": {"message": "response_format '
                              '`json_schema` is not supported for this model"}}')


def _exhausted_resp():
    """A 429 for the free tier's DAILY request cap: none left, reset hours
    away, so no amount of backing off will clear it."""
    resp = FakeResp(429, text='{"error": {"message": "Rate limit reached"}}')
    resp.headers = {"x-ratelimit-limit-requests": "1000",
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "2h13m0s",
                    "x-ratelimit-limit-tokens": "8000",
                    "x-ratelimit-remaining-tokens": "7821",
                    "x-ratelimit-reset-tokens": "1.342s"}
    return resp


def _busy_resp():
    """A 429 for the per-minute TOKEN limit, which clears in seconds.

    Note the trap in these headers, taken from a real response: the day's
    request reset is always present and minutes away, so reading the longest
    reset would call a busy minute a day's allowance gone.
    """
    resp = FakeResp(429, text='{"error": {"message": "Rate limit reached"}}')
    resp.headers = {"x-ratelimit-limit-requests": "1000",
                    "x-ratelimit-remaining-requests": "998",
                    "x-ratelimit-reset-requests": "2m52.8s",
                    "x-ratelimit-limit-tokens": "8000",
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "7.66s"}
    return resp


def test_groq_states_a_reset_as_a_duration():
    assert ai_client._parse_duration("7.66s") == 7.66
    assert ai_client._parse_duration("2m59.56s") == 179.56
    assert ai_client._parse_duration("1h2m3.5s") == 3723.5
    assert ai_client._parse_duration("90") == 90.0          # plain seconds
    assert ai_client._parse_duration("") == 0.0
    assert ai_client._parse_duration("soon") == 0.0


def test_a_used_up_allowance_is_told_apart_from_a_busy_minute():
    """The per-minute limit clears while you wait; the daily one does not.

    The reset that matters is the one whose allowance is actually gone - every
    response carries a reset for both, so the longest is the wrong answer.
    """
    assert ai_client._is_exhausted(_exhausted_resp())
    assert not ai_client._is_exhausted(_busy_resp())
    assert ai_client._retry_after(_busy_resp()) == 7.66     # waits the tokens out
    assert not ai_client._is_exhausted(FakeResp(429, text=""))


def test_a_retry_after_header_still_wins_when_present():
    resp = FakeResp(429, text="")
    resp.headers = {"Retry-After": "7", "x-ratelimit-reset-requests": "2h"}
    assert ai_client._retry_after(resp) == 7.0


def _account_allows(*allowed):
    """A fake _post where only `allowed` models answer; the rest refuse."""
    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        if model in allowed:
            return FakeResp(200, _ok_payload('{"ok": "yes"}'))
        return _refusal_resp()
    return fake_post


def test_the_configured_model_is_used_when_the_account_can_call_it(monkeypatch):
    monkeypatch.setattr(ai_client, "_post", _account_allows("openai/gpt-oss-120b"))
    assert ai_client.resolve_model("k", "openai/gpt-oss-120b") == \
        "openai/gpt-oss-120b"


def test_a_model_that_refuses_the_schema_gives_way_to_one_that_takes_it(monkeypatch):
    """Only some Groq models do strict constrained decoding; the rest answer
    400 to the probe and rule themselves out."""
    monkeypatch.setattr(ai_client, "_post", _account_allows("qwen/qwen3.8-27b"))
    assert ai_client.resolve_model("k", "openai/gpt-oss-120b") == "qwen/qwen3.8-27b"


def test_the_candidates_come_from_the_api_most_capable_first(monkeypatch):
    """The list is whatever the key can see, not a list written in here."""
    order = ai_client._candidate_models("openai/gpt-oss-120b", "k")
    assert order[0] == "openai/gpt-oss-120b"           # the configured one leads
    assert order[1:] == ["openai/gpt-oss-20b", "qwen/qwen3.8-27b",
                         "llama-3.3-70b-versatile"]


def test_a_speech_or_safety_model_is_never_chosen_to_write_lesson_content():
    """They answer perfectly well; they just write poor lesson prose."""
    assert "whisper-large-v3" not in ai_client._candidate_models("m", "k")
    for name in ("whisper-large-v3-turbo", "meta-llama/llama-prompt-guard-2-86m",
                 "openai/gpt-oss-safeguard-20b", "canopylabs/orpheus-v1-english",
                 "text-embedding-3-small"):
        assert ai_client._RE_SPECIAL_PURPOSE.search(name), name


def test_a_model_this_code_has_never_heard_of_is_still_tried(monkeypatch):
    """Ranked in the middle - ahead of the small ones, behind the big ones - so
    a model released after this was written gets its turn."""
    monkeypatch.setattr(ai_client, "list_chat_models",
                        lambda key: ["llama-3.3-70b-versatile", "moonshot/kimi-k2",
                                     "openai/gpt-oss-20b"])
    order = ai_client._candidate_models("openai/gpt-oss-120b", "k")
    assert order == ["openai/gpt-oss-120b", "openai/gpt-oss-20b",
                     "llama-3.3-70b-versatile", "moonshot/kimi-k2"]


def test_the_bigger_model_of_a_family_goes_first():
    ranked = sorted(["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
                    key=ai_client._model_rank)
    assert ranked == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_a_retired_model_is_not_offered_as_a_candidate(monkeypatch):
    """Groq's listing is flat, with `active` saying what can still be called."""
    payload = {"data": [
        {"id": "openai/gpt-oss-120b", "object": "model", "active": True},
        {"id": "llama-3.1-70b-versatile", "object": "model", "active": False},
        {"id": "qwen/qwen3.8-27b", "object": "model"},        # absent = callable
    ]}

    class Resp:
        status_code = 200

        def json(self):
            return payload

    monkeypatch.setattr(ai_client.requests, "get", lambda *a, **k: Resp())
    assert REAL_LIST_CHAT_MODELS("k") == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_a_busy_minute_does_not_disqualify_a_model(monkeypatch):
    """A minute's tokens used up is not the same as the day's requests used up."""
    monkeypatch.setattr(ai_client, "_post", lambda *a, **k: _busy_resp())
    assert ai_client.resolve_model("k", "openai/gpt-oss-120b") == \
        "openai/gpt-oss-120b"


def test_a_daily_cap_moves_to_a_model_with_its_own_allowance(monkeypatch):
    """The free tier's 1,000 a day is per model, so another model is the way on
    - and waiting hours for the reset is not."""
    slept = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        if model == "openai/gpt-oss-120b":
            return _exhausted_resp()
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", slept.append)
    monkeypatch.setattr(ai_client, "_RESOLVED_MODEL", "openai/gpt-oss-120b")

    assert call_model("p", "k", "openai/gpt-oss-120b") == []
    assert slept == []                    # no point waiting out a daily cap
    assert "openai/gpt-oss-120b" in ai_client._UNAVAILABLE_MODELS


def test_the_model_is_settled_once_for_the_process(monkeypatch):
    """A Learning Plan is many calls; re-choosing on each wasted a round trip."""
    probes = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        probes.append(model)
        return _account_allows("qwen/qwen3.8-27b")(model, api_key, prompt)

    monkeypatch.setattr(ai_client, "_post", fake_post)
    first = ai_client.resolve_model("k", "openai/gpt-oss-120b")
    before = len(probes)
    second = ai_client.resolve_model("k", "openai/gpt-oss-120b")
    assert first == second == "qwen/qwen3.8-27b"
    assert len(probes) == before          # settled, not asked again


def test_a_model_that_stops_answering_mid_run_is_replaced(monkeypatch):
    """An allowance can run out under a running generation."""
    calls = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        calls.append(model)
        if model == "openai/gpt-oss-120b":
            return _refusal_resp()
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client, "_RESOLVED_MODEL", "openai/gpt-oss-120b")
    assert call_model("p", "k", "openai/gpt-oss-120b") == []
    assert calls[0] == "openai/gpt-oss-120b"
    assert calls[-1] != "openai/gpt-oss-120b"


def test_when_nothing_answers_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setattr(ai_client, "_post", _account_allows())   # nothing answers
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    with pytest.raises(AIError) as excinfo:
        ai_client.resolve_model("k", "openai/gpt-oss-120b")
    message = str(excinfo.value)
    assert "console.groq.com" in message
    assert "GROQ_MODEL" in message
    assert "qwen/qwen3.8-27b" in message          # names what it tried


def test_a_multi_line_list_item_becomes_one_paragraph_each():
    """gpt-oss returns a key point as one string holding a heading and its
    bullets; another model returns them as separate elements. The document
    writes one paragraph per element and Word collapses a newline inside one
    into a space, so they are split apart here and both render the same."""
    assert ai_client._as_list(["THREAT CATEGORIES:\n- Malware\n- Phishing",
                               "SECURITY CONTROLS:"]) == \
        ["THREAT CATEGORIES:", "- Malware", "- Phishing", "SECURITY CONTROLS:"]
    # a plain list is untouched, and a bare string is still never char-split
    assert ai_client._as_list(["a", "b"]) == ["a", "b"]
    assert ai_client._as_list("solo") == ["solo"]


def test_the_batch_is_sized_for_the_token_allowance_not_the_model():
    """8,000 tokens a minute is the ceiling, and a bigger model doesn't raise it."""
    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"):
        assert ai_client.batch_size_for(model) == ai_client.LP_SESSION_CHUNK
    assert ai_client.LP_SESSION_CHUNK == 8
    assert ai_client.MAX_OUTPUT_TOKENS == 8000


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
    rows = call_model("p", "k", "m1")
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
        ai_client, "call_model",
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

    monkeypatch.setattr(ai_client, "call_model", fake_call)
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
                        key_points=["KEY POINT"])
                for i in range(ai_client.LP_SESSION_CHUNK)]
    calls = []
    monkeypatch.setattr(
        ai_client, "call_model",
        lambda prompt, api_key, model, progress_cb=None: (calls.append(1) or []))
    out = ai_client.generate_sessions(unit, sessions, api_key="k", model="m")
    assert len(calls) == 1
    assert len(out) == ai_client.LP_SESSION_CHUNK


def test_a_term_is_asked_for_in_batches(monkeypatch):
    """Twenty sessions go out as 8 + 8 + 4, never as one prompt: Groq's ceiling
    is 8,000 tokens a minute and one big batch spends it all at once."""
    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do it"],
                        key_points=["KEY POINT"]) for i in range(20)]
    sizes = []

    def fake_call(prompt, api_key, model, progress_cb=None):
        asked = prompt.count('"session_title"')
        sizes.append(asked)
        return [{"session_title": f"row {i}"} for i in range(asked)]

    monkeypatch.setattr(ai_client, "call_model", fake_call)
    ai_client.generate_sessions(unit, sessions, api_key="k",
                                model="openai/gpt-oss-120b")
    # sorted, because the batches go out together and may land in any order
    assert sorted(sizes, reverse=True) == [8, 8, 4]


def test_the_default_model_is_groqs_strongest_free_one():
    assert ai_client.DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_the_standing_instructions_forbid_inventing_syllabus_content():
    assert "You do NOT invent syllabus content" in ai_client.LP_SYSTEM
    assert "Return ONE JSON object and nothing else." in ai_client.LP_SYSTEM


def test_the_data_half_carries_data_and_no_instructions():
    """The split only pays off if nothing static leaks into the per-unit half:
    one interpolated value ahead of the rules would break the cacheable prefix,
    and it is the reason the two halves are readable apart."""
    unit = Unit(unit_title="Install and Configure Software", os_code="IT/OS/123",
                level="5", assessment_methods=["Observation"],
                required_knowledge=["Troubleshooting basics"])
    session = Session(week=1, session_no="1", is_cat=False,
                      session_title="Software installation",
                      pcs=["1.1 Install software"],
                      key_points=["Install software safely"])

    prompt = ai_client.build_prompt(unit, [session])

    assert "Install and Configure Software" in prompt
    assert "1.1 Install software" in prompt
    assert "Troubleshooting basics" in prompt
    assert "W1-S1" in prompt                       # the id the model echoes back
    # level 5 drops the two highest verbs
    assert "Evaluate" not in prompt
    # none of the standing rules are duplicated here
    for rule in ("SOURCES OF TRUTH", "TERMINOLOGY", "FIELD RULES",
                 "Method vocabulary"):
        assert rule not in prompt


# --------------------------------------------------------------------------- #
# speed: low reasoning effort, and batches that overlap
# --------------------------------------------------------------------------- #
def test_a_reasoning_model_is_told_not_to_think_at_length(monkeypatch):
    """gpt-oss and qwen3 bill their thinking as completion tokens, and tokens a
    minute is the binding limit - so the thinking is asked to be brief."""
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return FakeResp(200, _ok_payload('{"sessions": []}'))

    monkeypatch.setattr(requests, "post", fake_post)
    REAL_POST("openai/gpt-oss-120b", "k", "prompt")

    assert sent["reasoning_effort"] == "low"


def test_a_model_without_a_thinking_budget_is_not_sent_one(monkeypatch):
    """The field is not universal; a model that has no use for it answers a
    400, which would cost us a working model."""
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return FakeResp(200, _ok_payload('{"sessions": []}'))

    monkeypatch.setattr(requests, "post", fake_post)
    REAL_POST("llama-3.3-70b-versatile", "k", "prompt")

    assert "reasoning_effort" not in sent


def test_the_batches_of_a_plan_go_out_together(monkeypatch):
    """The token allowance refills continuously and is per model, so batches
    waiting in a queue leave it unspent. Three at a time, measured at 6s for a
    plan that took 142s one after another."""
    import threading

    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do it"],
                        key_points=["KEY POINT"]) for i in range(24)]
    in_flight = threading.Barrier(3, timeout=10)

    def fake_call(prompt, api_key, model, progress_cb=None):
        in_flight.wait()          # times out unless three run at once
        asked = prompt.count('"session_title"')
        return [{"session_title": f"row {i}"} for i in range(asked)]

    monkeypatch.setattr(ai_client, "call_model", fake_call)
    out = ai_client.generate_sessions(unit, sessions, api_key="k",
                                      model="openai/gpt-oss-120b")

    assert len(out) == 24


def test_rows_stay_with_their_own_session_when_batches_land_out_of_order(monkeypatch):
    """Rows are positional. Batches finishing in a different order than they
    were sent must not shuffle the plan."""
    import threading
    import time as _time

    unit = Unit(unit_title="U", os_code="X/OS/1", level="5",
                assessment_methods=["Observation"])
    sessions = [Session(week=i + 1, session_no="1", is_cat=False,
                        session_title=f"S{i}", pcs=["1.1 do it"],
                        key_points=["KEY POINT"]) for i in range(16)]
    first = threading.Event()

    def fake_call(prompt, api_key, model, progress_cb=None):
        titles = re.findall(r'"session_title": "(S\d+)"', prompt)
        if titles and titles[0] == "S0":
            first.wait(5)          # the opening batch answers last
        else:
            _time.sleep(0.05)
            first.set()
        return [{"session_title": f"ai {t}"} for t in titles]

    monkeypatch.setattr(ai_client, "call_model", fake_call)
    out = ai_client.generate_sessions(unit, sessions, api_key="k",
                                      model="openai/gpt-oss-120b")

    assert [s.session_title for s in out] == [f"ai S{i}" for i in range(16)]


# --------------------------------------------------------------------------- #
# A generation the validator refused, but which is perfectly good content
# --------------------------------------------------------------------------- #
def _rejected(generation):
    """Groq's shape for 'you generated it, but it broke the schema'."""
    return FakeResp(400, {"error": {
        "code": "json_validate_failed",
        "message": "Generated JSON does not match the expected schema.",
        "failed_generation": generation}})


def test_a_generation_refused_over_one_stray_key_is_not_thrown_away(monkeypatch):
    """Groq validates the FINISHED answer, so a complete plan can be refused for
    a single invented field - and it returns the whole generation with the 400.
    Discarding it would cost the batch and, worse, rule the model out."""
    generation = json.dumps({"sessions": [
        {"session_title": "Threats", "trainee_activities": ["- Group Discussion: name threats."],
         "follow_up_activity": "1. Scan the lab workstation and report."},
        {"session_title": "Controls", "trainee_activities": ["- Case Study: pick a control."]}]})
    monkeypatch.setattr(ai_client, "_post", lambda *a, **k: _rejected(generation))

    rows = call_model("prompt", "k", "openai/gpt-oss-120b")

    assert [r["session_title"] for r in rows] == ["Threats", "Controls"]


def test_content_filed_under_an_invented_key_is_put_back_where_it_belongs():
    """The model was asked for a 'Follow up Activity:' LINE and answered with a
    FIELD. The writing is good; only its address is wrong."""
    row = ai_client._rehome_stray_keys(
        {"session_title": "Threats",
         "trainee_activities": ["- Group Discussion: name threats."],
         "follow_up_activity": "1. Scan the lab workstation and report."})

    assert row["trainee_activities"] == [
        "- Group Discussion: name threats.",
        "Follow Up Activity:",
        "1. Scan the lab workstation and report."]
    assert "follow_up_activity" not in row


def test_a_known_field_under_another_spelling_is_merged_not_appended():
    row = ai_client._rehome_stray_keys(
        {"resources": ["Textbook"], "Resources ": ["Lab manual"]})

    assert row["resources"] == ["Textbook", "Lab manual"]
    assert list(row) == ["resources"]


def test_a_400_with_nothing_to_recover_still_rules_the_model_out(monkeypatch):
    """A model that won't take the schema at all has no generation attached;
    that must still move us to one that will."""
    refusals = []

    def fake_post(model, api_key, prompt, **kw):
        refusals.append(model)
        if model == "openai/gpt-oss-120b":
            return FakeResp(400, {"error": {"code": "invalid_request_error",
                                            "message": "unsupported response_format"}})
        return FakeResp(200, _ok_payload('{"sessions": [{"session_title": "ok"}]}'))

    monkeypatch.setattr(ai_client, "_post", fake_post)

    rows = call_model("prompt", "k", "openai/gpt-oss-120b")

    assert rows == [{"session_title": "ok"}]
    assert refusals[0] == "openai/gpt-oss-120b" and len(refusals) > 1


def test_a_refused_generation_that_is_not_json_falls_through(monkeypatch):
    """Nothing to salvage means the ordinary 'not this model' path."""
    monkeypatch.setattr(ai_client, "_post",
                        lambda *a, **k: _rejected("{not json at all"))

    with pytest.raises(AIError):
        call_model("prompt", "k", "openai/gpt-oss-120b")


def test_the_prompt_says_the_follow_up_line_is_not_its_own_field():
    """The salvage is the net; this is the fix - the instruction that produced
    the stray key, and the schema that leaves nowhere else to put it."""
    assert "The literal string \"Follow up Activity:\"" in ai_client.LP_SYSTEM
    # and there is no longer anywhere to put it but the array: the schema gives
    # trainee_activities as a flat list of strings
    acts = (_response_schema()["properties"]["sessions"]["items"]
            ["properties"]["trainee_activities"])
    assert acts == {"type": "array", "items": {"type": "string"}}


# --------------------------------------------------------------------------- #
# Structured answers, flattened for the document
# --------------------------------------------------------------------------- #
def _good_row(session_id="W1-S1", opener="Group Discussion"):
    """A row that breaks none of the field rules."""
    return {
        "session_id": session_id,
        "session_title": "ICT Security Threats",
        "learning_outcomes": {
            "stem": "By the end of the session, the trainee should be able to;",
            "items": ["a. Identify common ICT security threats.",
                      "b. Explain how a threat exploits a vulnerability.",
                      "c. Apply controls that reduce exposure."]},
        "key_points": [
            {"heading": "THREAT CATEGORIES",
             "points": ["Malware spreads through infected files.",
                        "Phishing targets people rather than systems."]},
            {"heading": "VULNERABILITY AND IMPACT",
             "points": ["Outdated software widens the attack surface.",
                        "Weak passwords enable account takeover."]}],
        "trainee_activities": [
            f"- {opener}: trainees name three threats seen locally and rank them.",
            "- Think-Pair-Share: each trainee matches a threat to a control, "
            "compares with a partner, then reports one agreed pair.",
            "- Guided Practice: each trainee inspects a sample phishing email "
            "and records the indicators found.",
            "Follow up Activity:",
            "1. Prepare a one-page threat profile for a chosen organisation."],
        "resources": ["TVET CDACC curriculum and trainer guide for this unit",
                      "NIST Cybersecurity Framework documentation"],
        "assessments": {
            "knowledge_checks": ["1. Oral questioning on threat categories."],
            "skills": ["1. Observation of the trainee identifying indicators."],
            "attitudes": ["1. Observation of accuracy when recording findings."]},
    }


def test_a_key_point_heading_is_capitalised_by_us_not_by_the_model():
    """doc_builder bolds a line only if it is >=85% uppercase. That used to
    depend on the model remembering; now the heading arrives as its own field
    and we capitalise it, so the bolding cannot drift."""
    lines = ai_client._flatten_key_points(
        [{"heading": "threat categories",
          "points": ["Malware spreads through infected files."]}])

    assert lines == ["THREAT CATEGORIES", "- Malware spreads through infected files."]
    assert doc_builder._keypoint_bold(lines[0])
    assert not doc_builder._keypoint_bold(lines[1])


def test_assessments_are_grouped_under_the_headings_the_document_bolds():
    lines = ai_client._flatten_assessments(
        {"knowledge_checks": ["1. Oral questioning."],
         "skills": ["1. Observation."],
         "attitudes": ["1. Punctuality observed."]})

    assert lines[0] == "Knowledge Checks:"
    assert all(doc_builder._assessment_bold(h)
               for h in ("Knowledge Checks:", "Skills:", "Attitudes:"))


def test_an_empty_assessment_group_leaves_out_its_heading():
    lines = ai_client._flatten_assessments(
        {"knowledge_checks": ["1. Oral questioning."], "skills": [], "attitudes": []})

    assert lines == ["Knowledge Checks:", "1. Oral questioning."]


def test_the_outcome_stem_gets_its_own_line_above_the_items():
    lines = ai_client._flatten_learning_outcomes(
        {"stem": "By the end of the session, the trainee should be able to;",
         "items": ["a. Identify threats.", "b. Explain impact."]})

    assert lines[0].startswith("By the end of the session")
    assert lines[1:] == ["a. Identify threats.", "b. Explain impact."]


def test_a_model_that_answers_in_the_old_flat_shape_still_merges():
    """Belt and braces: the adapters fall back to plain list coercion, so an
    answer in the previous shape is not lost."""
    assert ai_client._flatten_key_points(["THREATS", "- Malware."]) == \
        ["THREATS", "- Malware."]
    assert ai_client._flatten_assessments(["Knowledge Checks:", "1. Oral."]) == \
        ["Knowledge Checks:", "1. Oral."]


# --------------------------------------------------------------------------- #
# Rows find their own session, and CATs never leave the building
# --------------------------------------------------------------------------- #
def _sessions(n, cats=()):
    out = []
    for i in range(n):
        out.append(Session(week=i + 1, session_no="1", is_cat=(i in cats),
                           session_title=f"S{i}", pcs=["1.1 do it"],
                           key_points=["KEY POINT"]))
    return out


def test_rows_are_matched_by_identity_not_by_arrival_order():
    """The safety net that used to need padding: rows can come back in any
    order, and each still lands on its own session."""
    sessions = _sessions(3)
    rows = [{"session_id": "W3-S1", "session_title": "third"},
            {"session_id": "W1-S1", "session_title": "first"},
            {"session_id": "W2-S1", "session_title": "second"}]

    out = ai_client.merge_ai_into_sessions(sessions, rows, Unit())

    assert [s.session_title for s in out] == ["first", "second", "third"]


def test_a_row_for_a_session_that_was_never_sent_is_ignored():
    sessions = _sessions(2)
    rows = [{"session_id": "W1-S1", "session_title": "mine"},
            {"session_id": "W9-S9", "session_title": "not from this plan"}]

    out = ai_client.merge_ai_into_sessions(sessions, rows, Unit())

    assert out[0].session_title == "mine"
    assert "not from this plan" not in [s.session_title for s in out]


def test_rows_without_ids_still_pair_in_order():
    """An older or sloppier model that drops session_id must not lose the plan."""
    sessions = _sessions(2)
    rows = [{"session_title": "first"}, {"session_title": "second"}]

    out = ai_client.merge_ai_into_sessions(sessions, rows, Unit())

    assert [s.session_title for s in out] == ["first", "second"]


def test_cat_sessions_are_never_sent_to_the_model(monkeypatch):
    """Their content is rebuilt deterministically, so generating it would be
    paying for output that goes straight in the bin."""
    sessions = _sessions(6, cats=(2, 5))
    sent = []

    def fake_call(prompt, api_key, model, progress_cb=None):
        sent.extend(re.findall(r'"session_id": "(W\d+-S\d+)"', prompt))
        return []

    monkeypatch.setattr(ai_client, "call_model", fake_call)
    ai_client.generate_sessions(sessions[0].__class__ and Unit(), sessions,
                                api_key="k", model="openai/gpt-oss-120b")

    assert "W3-S1" not in sent and "W6-S1" not in sent    # the two CATs
    assert sorted(sent) == ["W1-S1", "W2-S1", "W4-S1", "W5-S1"]


def test_a_plan_of_nothing_but_cats_makes_no_call_at_all(monkeypatch):
    calls = []
    monkeypatch.setattr(ai_client, "call_model",
                        lambda *a, **k: calls.append(1) or [])
    sessions = _sessions(2, cats=(0, 1))

    out = ai_client.generate_sessions(Unit(), sessions, api_key="k", model="m")

    assert calls == []
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# The self-check (v2 section 7), as code
# --------------------------------------------------------------------------- #
def test_a_row_that_follows_the_rules_reports_nothing():
    assert ai_client.validate_rows([_good_row()]) == {}


@pytest.mark.parametrize("mutate, expected", [
    (lambda r: r["learning_outcomes"]["items"].append("d. One too many."),
     "exactly 3"),
    (lambda r: r["learning_outcomes"].__setitem__(
        "items", ["a. Understand threats.", "b. Explain them.", "c. Apply them."]),
     "cannot be assessed"),
    (lambda r: r["key_points"].pop(),
     "it must have 2 or 3"),
    (lambda r: r["trainee_activities"].__setitem__(3, "Follow-up:"),
     "must be exactly 'Follow up Activity:'"),
    (lambda r: r["trainee_activities"].__setitem__(
        0, "- Trainees talk about it a bit."),
     "names no active-learning method"),
    (lambda r: r["resources"].clear(),
     "it must have 2 to 4"),
    (lambda r: r["assessments"].__setitem__("skills", []),
     "assessments.skills has 0 items"),
    (lambda r: r.__setitem__("session_title", "Students Learn About Threats"),
     "Kenya CBET register forbids"),
    (lambda r: r["key_points"][0]["points"].__setitem__(0, "Malware is ..."),
     "ellipsis, a placeholder"),
])
def test_the_self_check_catches_what_the_model_is_asked_not_to_do(mutate, expected):
    row = _good_row()
    mutate(row)

    problems = ai_client.validate_rows([row])

    assert problems, f"expected a complaint containing {expected!r}"
    assert any(expected in p for p in problems["W1-S1"]), problems["W1-S1"]


def test_the_same_method_opening_every_session_is_flagged():
    """A plan where every session opens with a Group Discussion is a worse plan,
    and no per-row check can see it."""
    rows = [_good_row(f"W{i}-S1", opener="Group Discussion") for i in range(1, 7)]

    problems = ai_client.validate_rows(rows)

    assert problems, "repetition across the batch went unnoticed"
    assert any("opens" in p for v in problems.values() for p in v)


def test_varied_openings_across_a_batch_pass():
    methods = ["Group Discussion", "Case Study", "Jigsaw",
               "Role Play", "Fishbowl", "Gallery Walk"]
    rows = [_good_row(f"W{i + 1}-S1", opener=m) for i, m in enumerate(methods)]

    assert ai_client.validate_rows(rows) == {}


# --------------------------------------------------------------------------- #
# The corrective round
# --------------------------------------------------------------------------- #
def test_only_the_failing_sessions_are_sent_back(monkeypatch):
    sessions = _sessions(3)
    # distinct openers, or the variety check would fault all three
    rows = [_good_row("W1-S1", opener="Case Study"),
            _good_row("W2-S1", opener="Jigsaw"),
            _good_row("W3-S1", opener="Role Play")]
    rows[1]["resources"] = ["only one"]                 # W2-S1 alone is wrong
    prompts = []

    def fake_call(prompt, api_key, model, progress_cb=None):
        prompts.append(prompt)
        return [_good_row("W2-S1", opener="Jigsaw")]

    monkeypatch.setattr(ai_client, "call_model", fake_call)
    out = REAL_REPAIR_ROWS(Unit(), sessions, rows, "k", "m")

    assert len(prompts) == 1
    assert "W2-S1" in prompts[0]
    assert "W1-S1" not in prompts[0] and "W3-S1" not in prompts[0]
    assert "resources has 1" in prompts[0]              # it is told what was wrong
    assert ai_client.validate_rows(out) == {}


def test_a_clean_batch_costs_no_extra_call(monkeypatch):
    calls = []
    monkeypatch.setattr(ai_client, "call_model",
                        lambda *a, **k: calls.append(1) or [])

    out = REAL_REPAIR_ROWS(Unit(), _sessions(1), [_good_row()], "k", "m")

    assert calls == []
    assert out == [_good_row()]


def test_a_failed_repair_keeps_the_rows_we_already_have(monkeypatch):
    """A corrective round that errors must not cost us the batch."""
    rows = [_good_row()]
    rows[0]["resources"] = ["only one"]

    def boom(*a, **k):
        raise AIError("no model answered")

    monkeypatch.setattr(ai_client, "call_model", boom)
    out = REAL_REPAIR_ROWS(Unit(), _sessions(1), rows, "k", "m")

    assert out == rows


def test_the_repair_round_is_not_repeated(monkeypatch):
    """Bounded at one: a second round costs another batch for diminishing
    returns, and the deterministic backfill makes an imperfect row usable."""
    rows = [_good_row()]
    rows[0]["resources"] = ["only one"]
    calls = []

    def still_wrong(prompt, api_key, model, progress_cb=None):
        calls.append(1)
        bad = _good_row()
        bad["resources"] = ["still only one"]
        return [bad]

    monkeypatch.setattr(ai_client, "call_model", still_wrong)
    out = REAL_REPAIR_ROWS(Unit(), _sessions(1), rows, "k", "m")

    assert len(calls) == 1
    assert out[0]["resources"] == ["still only one"]     # kept, and logged


def test_assessment_items_are_numbered_by_us_not_asked_for():
    """Measured against real output, every row came back unnumbered. Numbering
    is presentation, like the CAPITALISED heading, so it is applied here."""
    lines = ai_client._flatten_assessments(
        {"knowledge_checks": ["Oral questioning on threat types.",
                              "Written short-answer items."],
         "skills": ["Observation at the workstation."], "attitudes": []})

    assert lines == ["Knowledge Checks:",
                     "1. Oral questioning on threat types.",
                     "2. Written short-answer items.",
                     "Skills:",
                     "1. Observation at the workstation."]


def test_a_model_that_numbers_them_itself_is_renumbered_not_doubled():
    lines = ai_client._flatten_assessments(
        {"knowledge_checks": ["3. Out of order.", "1. Second."],
         "skills": [], "attitudes": []})

    assert lines == ["Knowledge Checks:", "1. Out of order.", "2. Second."]


def test_a_one_word_key_point_heading_is_allowed():
    """VULNERABILITIES is a good heading; the rule that banned it was wrong."""
    row = _good_row()
    row["key_points"][0]["heading"] = "VULNERABILITIES"

    assert ai_client.validate_rows([row]) == {}


def test_a_rambling_key_point_heading_is_still_caught():
    row = _good_row()
    row["key_points"][0]["heading"] = "A Very Long Heading That Rambles On Somewhat"

    problems = ai_client.validate_rows([row])["W1-S1"]

    assert any("the limit is 5" in p for p in problems)


def test_a_method_written_with_a_fancy_hyphen_still_counts():
    """Models write Think-Pair-Share with a non-breaking hyphen as often as a
    plain one; reporting that as 'no method named' is a false alarm."""
    row = _good_row()
    row["trainee_activities"][0] = (
        "- Think‑Pair‑Share: each trainee lists two threats, compares "
        "with a partner, then the pair reports one to the room.")

    assert ai_client.validate_rows([row]) == {}


# --------------------------------------------------------------------------- #
# A 413 that is really the per-minute window
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, text):
        self.text = text
        self.headers = {}


def test_a_413_about_tokens_per_minute_is_rate_limiting():
    """Groq says "Request too large ... on tokens per minute (TPM): Limit
    8000, Requested 8174" when the last minute is simply spent. Measured: the
    same request, unchanged, goes through once the window clears."""
    resp = _Resp("Request too large for model `openai/gpt-oss-120b` in "
                 "organization `org_x` service tier `on_demand` on tokens "
                 "per minute (TPM): Limit 8000, Requested 8174")

    assert ai_client._is_window_full(resp)


def test_a_413_about_anything_else_is_left_to_fail():
    """A request that genuinely will never fit is not helped by waiting."""
    assert not ai_client._is_window_full(_Resp("Payload too large"))
    assert not ai_client._is_window_full(_Resp(""))
