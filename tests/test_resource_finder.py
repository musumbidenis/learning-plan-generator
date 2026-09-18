"""Resource discovery, against canned responses - ZERO real network calls.

The point of this module is that nothing unverified reaches a document, so
these tests are mostly about what gets REJECTED: an invented video id, a dead
link, a resource a model typed instead of choosing.
"""

import json

import pytest
import requests

import ai_client
import resource_finder
from models import Session, Unit


class FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Nothing here may reach the internet; every test says what it expects."""
    def refuse(*a, **k):
        raise AssertionError("a test tried to make a real request")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests, "head", refuse)
    monkeypatch.setattr(requests, "post", refuse)


# --------------------------------------------------------------------------- #
# Videos: a plain fetch is not enough
# --------------------------------------------------------------------------- #
def test_a_dead_video_is_rejected_even_though_the_page_answers(monkeypatch):
    """A deleted or invented YouTube id still serves a 200 'unavailable' page,
    so the watch URL proves nothing. oEmbed is what actually discriminates -
    and this is the exact failure seen live: three proposed videos, three
    well-formed ids, none of which existed."""
    def fake_get(url, **kw):
        if "oembed" in url:
            return FakeResp(404, text="Not Found")
        return FakeResp(200, text="<html>Video unavailable</html>")

    monkeypatch.setattr(requests, "get", fake_get)

    assert resource_finder.verify_video(
        "https://www.youtube.com/watch?v=aB3xK9zQ1mP") is None


def test_a_real_video_is_named_by_youtube_not_by_the_model(monkeypatch):
    """The title we print is the video's own, so a right URL with a wrong
    description cannot get through either."""
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(
        200, {"title": "DDoS Attack Explained",
              "author_name": "PowerCert Animated Videos"}))

    got = resource_finder.verify_video("https://www.youtube.com/watch?v=ilhGh9CEIwM")

    assert got.title == "DDoS Attack Explained"
    assert got.channel == "PowerCert Animated Videos"
    assert got.kind == "video"


def test_video_search_reads_real_ids_off_the_results_page(monkeypatch):
    """Ids come from an index, so a real id is never guessed to begin with."""
    page = ('junk"videoId":"bDAY-oUP0DQ"more"videoId":"awhqnSskWjU"'
            '"videoId":"bDAY-oUP0DQ"')

    def fake_get(url, **kw):
        if "oembed" in url:
            vid = kw["params"]["url"].rsplit("=", 1)[1]
            return FakeResp(200, {"title": f"Video {vid}", "author_name": "Chan"})
        return FakeResp(200, text=page)

    monkeypatch.setattr(requests, "get", fake_get)

    found = resource_finder.search_videos("ddos", want=5)

    assert [r.url for r in found] == [
        "https://www.youtube.com/watch?v=bDAY-oUP0DQ",
        "https://www.youtube.com/watch?v=awhqnSskWjU"]   # deduplicated


def test_a_youtube_search_that_fails_yields_nothing_rather_than_raising(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda url, **kw: FakeResp(429, text="slow down"))

    assert resource_finder.search_videos("anything") == []


# --------------------------------------------------------------------------- #
# Pages: proposed, then made to prove it
# --------------------------------------------------------------------------- #
def test_a_page_that_404s_never_becomes_a_resource(monkeypatch):
    monkeypatch.setattr(requests, "head", lambda url, **kw: FakeResp(404))
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(404))

    assert resource_finder.verify_page("https://example.com/invented.pdf") is False


def test_a_host_that_refuses_head_is_retried_with_get(monkeypatch):
    """Some servers 405 a HEAD they would happily answer as a GET; treating
    that as dead would throw away real resources."""
    monkeypatch.setattr(requests, "head", lambda url, **kw: FakeResp(405))
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(200))

    assert resource_finder.verify_page("https://www.sans.org/policies") is True


def test_a_connection_error_is_a_dead_resource_not_a_crash(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("no such host")

    monkeypatch.setattr(requests, "head", boom)
    monkeypatch.setattr(requests, "get", boom)

    assert resource_finder.verify_page("https://example.invalid/x") is False


def test_verify_all_keeps_only_what_exists(monkeypatch):
    def fake_head(url, **kw):
        return FakeResp(200 if "real" in url else 404)

    monkeypatch.setattr(requests, "head", fake_head)
    monkeypatch.setattr(requests, "get", fake_head)

    kept = resource_finder.verify_all([
        resource_finder.Resource("Real one", "https://a.test/real"),
        resource_finder.Resource("Invented", "https://a.test/made-up"),
        resource_finder.Resource("Not a url", "ask your trainer"),
    ])

    assert [r.title for r in kept] == ["Real one"]


def test_proposed_pages_are_parsed_out_of_whatever_wraps_them():
    text = ('Here you go:\n```json\n'
            '{"resources":[{"title":"NIST CSF","url":"https://nist.gov/x",'
            '"kind":"standard"},{"title":"","url":"https://no.title/"}]}\n```')

    got = resource_finder._parse_resources(text)

    assert len(got) == 1 and got[0].title == "NIST CSF"


def test_a_proposed_video_is_reclassified_because_videos_are_searched():
    got = resource_finder._parse_resources(json.dumps(
        {"resources": [{"title": "Some talk", "url": "https://x.test/",
                        "kind": "video"}]}))

    assert got[0].kind == "documentation"


# --------------------------------------------------------------------------- #
# How the pool reaches the plan
# --------------------------------------------------------------------------- #
def _pool():
    return resource_finder.ResourcePool(resources=[
        resource_finder.Resource("DDoS Attack Explained",
                                 "https://www.youtube.com/watch?v=ilhGh9CEIwM",
                                 "video", "PowerCert Animated Videos"),
        resource_finder.Resource("NIST Cybersecurity Framework",
                                 "https://www.nist.gov/cyberframework", "standard"),
        resource_finder.Resource("SANS Security Policy Templates",
                                 "https://www.sans.org/security-resources/policies"),
    ])


def test_a_video_line_names_its_channel_and_link():
    assert _pool().resources[0].as_line() == (
        "- Video: DDoS Attack Explained, PowerCert Animated Videos - "
        "https://www.youtube.com/watch?v=ilhGh9CEIwM")


def test_the_pool_is_offered_to_the_model_as_a_numbered_list():
    unit = Unit(unit_title="Manage ICT Security", os_code="IT/OS/1", level="6")
    session = Session(week=1, session_no="1", session_title="Threats",
                      pcs=["1.1 x"], key_points=["KP"])

    prompt = ai_client.build_prompt(unit, [session], _pool())

    assert "VERIFIED RESOURCES" in prompt
    assert "1. [video] DDoS Attack Explained" in prompt
    assert "https://www.nist.gov/cyberframework" in prompt


def test_chosen_numbers_become_the_exact_verified_lines():
    """The model picks numbers; the lines are rendered from the pool. It never
    types a URL, so it can neither mistype nor invent one."""
    lines = ai_client._flatten_resources(["2", "1"], _pool())

    assert lines == [
        "- NIST Cybersecurity Framework - https://www.nist.gov/cyberframework",
        "- Video: DDoS Attack Explained, PowerCert Animated Videos - "
        "https://www.youtube.com/watch?v=ilhGh9CEIwM"]


def test_a_number_outside_the_pool_is_dropped():
    assert ai_client._flatten_resources(["99"], _pool()) == []


def test_a_typed_title_is_ignored_when_a_pool_exists():
    """If the model writes prose instead of choosing, that prose is exactly the
    unverifiable thing we are refusing to print."""
    assert ai_client._flatten_resources(
        ["Cisco Networking Academy, ict-security.pdf"], _pool()) == []


def test_without_a_pool_the_words_are_kept():
    """Offline, or when nothing verified, resources are named without links."""
    assert ai_client._flatten_resources(["- A textbook on ICT security"], None) \
        == ["- A textbook on ICT security"]


def test_the_self_check_rejects_a_resource_that_is_not_a_pool_number():
    row = {"session_id": "W1-S1",
           "resources": ["https://www.cisco.com/invented-page.html", "2"]}

    problems = ai_client.validate_rows([row], _pool())

    assert any("is not a number" in p for p in problems["W1-S1"])


def test_the_self_check_rejects_a_number_off_the_end_of_the_list():
    row = {"session_id": "W1-S1", "resources": ["1", "9"]}

    problems = ai_client.validate_rows([row], _pool())

    assert any("is not on the VERIFIED RESOURCES list" in p
               for p in problems["W1-S1"])


def test_a_session_that_picked_nothing_usable_still_gets_real_resources():
    """Better an adjacent verified resource than the old generic filler."""
    sessions = [Session(week=1, session_no="1", session_title="Threats",
                        pcs=["1.1 x"], key_points=["KP"])]

    out = ai_client.merge_ai_into_sessions(
        sessions, [{"session_id": "W1-S1", "resources": ["99"]}], Unit(), _pool())

    assert len(out[0].resources) == 3
    assert all("http" in line for line in out[0].resources)
