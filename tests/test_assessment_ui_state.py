"""Session state for the assessment path is seeded per session, not per import.

The crash this guards against: `st.session_state has no attribute "at_raw"`.
`assessment_ui` is an imported module, so anything it does at import time runs
once for the whole server process. Seeding defaults there serves whichever
browser session happened to be first and leaves every later one to fall over
on the first read.
"""
import re

import pytest

assessment_ui = pytest.importorskip("assessment_ui")

SOURCE = open(assessment_ui.__file__, encoding="utf-8").read()


class FakeState(dict):
    """Enough of st.session_state to show the seeding works."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def test_importing_the_module_seeds_nothing():
    """If it seeded at import, a second session would find the keys missing
    and the module would look fine in every test."""
    assert not re.search(r"^ss\.setdefault", SOURCE, re.M)
    assert "_ensure_state()" in SOURCE


def test_ensure_state_fills_a_fresh_session(monkeypatch):
    state = FakeState()
    monkeypatch.setattr(assessment_ui, "ss", state)

    assessment_ui._ensure_state()

    assert set(state) == set(assessment_ui._STATE)
    assert state.at_raw == ""
    assert state.at_weighting is None


def test_ensure_state_leaves_a_live_session_alone():
    """It runs on every rerun, so it must not wipe work in progress."""
    state = FakeState(at_raw="1.1\tTools are identified\t4\t4")
    original = dict(assessment_ui._STATE)

    for key, default in original.items():
        state.setdefault(key, default)

    assert state.at_raw.startswith("1.1")


def test_render_seeds_before_it_reads():
    """The seeding call comes first in render - a step that reads state before
    it is placed is the bug again."""
    body = SOURCE[SOURCE.index("def render("):]
    first = next(line.strip() for line in body.splitlines()[1:]
                 if line.strip() and not line.strip().startswith('"""')
                 and not line.strip().startswith("The whole path"))
    assert first == "_ensure_state()"


def test_every_state_key_the_module_reads_has_a_default():
    """A new `ss.at_something` without an entry in _STATE is the same crash."""
    read = set(re.findall(r"\bss\.(at_\w+)", SOURCE))
    assert read <= set(assessment_ui._STATE), sorted(read - set(assessment_ui._STATE))
