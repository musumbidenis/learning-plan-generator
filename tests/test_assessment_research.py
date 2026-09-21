"""Scouting real CDACC papers for how a question is actually written.

ZERO network calls: every test either works on canned paper text or
monkeypatches the HTTP layer, so a missed patch fails loudly rather than
reaching out to a university's repository from a test run.
"""

import io
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assessment_research as research
from assessment_models import Exemplar


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Nothing here may reach the network, and nothing may write to the real
    cache."""
    def refuse(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(research.requests, "get", refuse)
    monkeypatch.setattr(research, "CACHE_DIR", str(tmp_path / "research"))


# A page of a real CDACC paper, as pdfplumber hands it over - dotted answer
# lines, page furniture and all.
PAPER = """
©2023 TVET CDACC
SECTION A: (40 MARKS)
Answer All questions in this section
1. As a good communicator it is very important to understand needs. State FOUR
methods of identifying communication needs. (4 Marks)
...............................................................................
2. Mr. M has experienced conflict among workmates during working hours. Identify
FOUR ways in which he can address conflict in the organization. (4 Marks)
...............................................................................
3. Highlight FOUR importance of holding meetings in an organization. (4 Marks)
Page 2 of 4
SECTION B: (60 MARKS)
11. Jane a supervisor in company X is implementing communication strategy.
a) Discuss FIVE factors that support implementation. (10 Marks)
"""


# --------------------------------------------------------------------------- #
# Reading a paper
# --------------------------------------------------------------------------- #
def test_the_questions_come_out_with_their_marks():
    found = research.questions(PAPER)

    assert [e.marks for e in found] == [4, 4, 4, 10]
    assert found[0].text.startswith("As a good communicator")


def test_the_answer_lines_and_page_furniture_are_stripped():
    found = research.questions(PAPER)

    assert all("...." not in e.text for e in found)
    assert all("Page 2 of 4" not in e.text for e in found)


def test_a_paper_with_nothing_in_it_yields_nothing():
    assert research.questions("") == []
    assert research.questions("SECTION A\nAnswer all questions.") == []


def test_a_fragment_too_short_to_be_a_question_is_dropped():
    assert research.questions("1. State FOUR. (4 Marks)") == []


def test_a_run_on_block_too_long_to_be_one_question_is_dropped():
    long_one = "1. " + ("a very long preamble " * 40) + "(4 Marks)"

    assert research.questions(long_one) == []


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
def test_the_questions_nearest_the_unit_come_first():
    """A wildly off-subject exemplar still drags a paper sideways: a model
    shown a spreadsheet question while writing on ICT security reaches for
    spreadsheets."""
    pool = [Exemplar(text="Outline FOUR uses of a spreadsheet package"),
            Exemplar(text="State FOUR types of malware and how each spreads"),
            Exemplar(text="Describe THREE methods of mixing concrete")]

    ranked = research.rank(pool, ["malware", "security threats"])

    assert "malware" in ranked[0].text


def test_ranking_without_topics_keeps_what_it_was_given():
    pool = [Exemplar(text="a"), Exemplar(text="b")]

    assert research.rank(pool, []) == pool


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def test_a_unit_title_with_slashes_still_makes_one_file():
    research._remember("Manage ICT Security / Level 6", [Exemplar(text="q")])

    assert os.path.isfile(research.cache_path("Manage ICT Security / Level 6"))


def test_what_was_scouted_is_read_back():
    research._remember("Manage ICT Security",
                       [Exemplar(text="State FOUR threats", marks=4,
                                 source="A paper", repository="amref")])

    back = research._cached("Manage ICT Security")

    assert [(e.text, e.marks, e.source) for e in back] == [
        ("State FOUR threats", 4, "A paper")]


def test_a_unit_never_scouted_reads_back_as_nothing_known():
    assert research._cached("Never Seen") is None


def test_a_stale_cache_is_ignored_rather_than_trusted():
    research._remember("Manage ICT Security", [Exemplar(text="q")])
    path = research.cache_path("Manage ICT Security")
    with io.open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["fetched"] = time.time() - (research.CACHE_DAYS + 1) * 86400
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)

    assert research._cached("Manage ICT Security") is None


def test_a_unit_with_no_papers_is_remembered_as_none_rather_than_re_scouted():
    """An empty result is a result. Without it every generation on a unit
    nobody has published pays the full search again."""
    research._remember("Obscure Unit", [])

    assert research._cached("Obscure Unit") == []


def test_a_corrupt_cache_costs_the_exemplars_not_the_run():
    os.makedirs(research.CACHE_DIR, exist_ok=True)
    with io.open(research.cache_path("Broken"), "w", encoding="utf-8") as fh:
        fh.write('{"exemplars": [')

    assert research._cached("Broken") is None


# --------------------------------------------------------------------------- #
# Degrading
# --------------------------------------------------------------------------- #
def test_a_repository_that_will_not_answer_costs_no_generation(monkeypatch):
    def broken(*a, **k):
        raise research.requests.RequestException("connection refused")
    monkeypatch.setattr(research, "_get", broken)

    assert research.search("anything") == []
    assert research.exemplars_for("Some Unit") == []


def test_a_repository_answering_with_junk_is_survived(monkeypatch):
    class Resp:
        status_code = 200
        def json(self):
            return {"nothing": "expected here"}
    monkeypatch.setattr(research, "_get", lambda *a, **k: Resp())

    assert research.search("anything") == []


def test_an_empty_unit_title_is_not_searched_for(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("searched for nothing")
    monkeypatch.setattr(research, "search", refuse)

    assert research.exemplars_for("") == []
    assert research.exemplars_for("   ") == []


# --------------------------------------------------------------------------- #
# What the model is shown
# --------------------------------------------------------------------------- #
def test_the_block_carries_the_marks_each_question_was_worth():
    rendered = research.render([
        Exemplar(text="State FOUR methods", marks=4, source="Comm Skills L6")])

    assert '"State FOUR methods" (4 marks)' in rendered
    assert "Comm Skills L6" in rendered


def test_nothing_found_renders_to_nothing_at_all():
    assert research.render([]) == ""
