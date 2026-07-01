"""Robustness of the Mistral response path. ZERO real API calls - the HTTP layer
(`ai_client._post`) is monkeypatched to return canned responses.

Covers the case that motivated the hardening: the provider returns HTTP 200 but
the choice carries only a finish reason and no assistant content.
That must raise a clean AIError and stop, not fall back to another model.
"""

import pytest
import requests

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


def test_call_mistral_retries_transient_timeouts(monkeypatch):
    attempts = []

    def fake_post(model, api_key, prompt, timeout=180, **kwargs):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise requests.Timeout("timed out")
        return FakeResp(200, _ok_payload("[]"))

    monkeypatch.setattr(ai_client, "_post", fake_post)
    monkeypatch.setattr(ai_client.time, "sleep", lambda *_: None)

    assert call_mistral("p", "k", "m1") == []
    assert len(attempts) == 3


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
# Truncated-response salvage (the token-limit / unterminated-JSON case)
# --------------------------------------------------------------------------- #
def test_salvage_recovers_complete_objects_from_truncated_array():
    text = '[{"a": 1}, {"b": 2}, {"c": "unterminated string...'
    assert ai_client._salvage_json_array(text) == [{"a": 1}, {"b": 2}]


def test_salvage_ignores_braces_and_brackets_inside_strings():
    text = '[{"x": "a } b", "y": [1, 2]}, {"z": "trunc'
    assert ai_client._salvage_json_array(text) == [{"x": "a } b", "y": [1, 2]}]


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
