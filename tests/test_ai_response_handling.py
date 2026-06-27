"""Robustness of the Mistral response path. ZERO real API calls - the HTTP layer
(`ai_client._post`) is monkeypatched to return canned responses.

Covers the case that motivated the hardening: the provider returns HTTP 200 but
the choice carries only a finish reason and no assistant content.
That must NOT crash with a KeyError/AttributeError - it should raise a clean
AIError so `call_mistral` can log it and advance to the next model in the chain.
"""

import pytest

import ai_client
from ai_client import AIError, _extract_text, call_mistral


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
# call_mistral chain behaviour
# --------------------------------------------------------------------------- #
def test_call_mistral_advances_past_empty_choice(monkeypatch):
    """First model returns 200-but-blocked; the chain should fall through to the
    second model and return its rows instead of failing."""
    responses = {
        "m1": FakeResp(200, {"choices": [{"finish_reason": "content_filter"}]}),
        "m2": FakeResp(200, _ok_payload('[{"week": 1}]')),
    }
    monkeypatch.setattr(ai_client, "_post",
                        lambda model, api_key, prompt, timeout=180: responses[model])
    rows = call_mistral("prompt", "key", "m1", fallback_model=["m2"])
    assert rows == [{"week": 1}]


def test_call_mistral_raises_aierror_when_all_models_empty(monkeypatch):
    monkeypatch.setattr(
        ai_client, "_post",
        lambda model, api_key, prompt, timeout=180:
            FakeResp(200, {"choices": [{"finish_reason": "content_filter"}]}))
    with pytest.raises(AIError):
        call_mistral("prompt", "key", "m1", fallback_model=["m2"])


def test_call_mistral_advances_past_invalid_json(monkeypatch):
    responses = {
        "m1": FakeResp(200, _ok_payload("not json at all")),
        "m2": FakeResp(200, _ok_payload('[{"week": 2}]')),
    }
    monkeypatch.setattr(ai_client, "_post",
                        lambda model, api_key, prompt, timeout=180: responses[model])
    assert call_mistral("p", "k", "m1", fallback_model=["m2"]) == [{"week": 2}]


def test_call_mistral_aborts_immediately_on_403(monkeypatch):
    """A 403 is a key problem, not a model problem - abort, do not try fallbacks."""
    calls = []

    def fake_post(model, api_key, prompt, timeout=180):
        calls.append(model)
        return FakeResp(403, text="PERMISSION_DENIED")

    monkeypatch.setattr(ai_client, "_post", fake_post)
    with pytest.raises(AIError):
        call_mistral("p", "k", "m1", fallback_model=["m2"])
    assert calls == ["m1"]      # never advanced to m2
