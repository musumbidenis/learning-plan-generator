"""Reference notes: choosing the right article, and surviving a bad network.

Nothing here touches the network. Every test either works on canned article
text or stubs `_api`, because a test that needs Wikipedia is a test that fails
on a train.
"""
import json
import os
import time

import pytest

import assessment_knowledge as know
from assessment_models import ContentTopic, ElementContent, KnowledgeNote


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _ict_content():
    return [
        ElementContent(
            element_number="1", element_title="Identify ICT security threats",
            topics=[
                ContentTopic("1.1", "Identification of ICT security threats",
                             ["Definition of terms",
                              "Types of malware: virus, worm, trojan",
                              "Social engineering attacks"]),
                ContentTopic("1.2", "Assessment of vulnerabilities",
                             ["Vulnerability scanning tools",
                              "Risk rating matrix"])]),
        ElementContent(
            element_number="2", element_title="Apply ICT security controls",
            topics=[
                ContentTopic("2.1", "Configuration of access controls",
                             ["Role based access control",
                              "Password policy settings"])]),
    ]


MALWARE = """Malware is any software intentionally designed to cause disruption
to a computer or network. Researchers classify malware into sub-types such as
computer viruses, worms and trojan horses. Symantec reported a rise in variants.

== History ==
The first worm spread in 1988. Fred Cohen described the theory.

== Types ==
A virus attaches itself to a host file. A worm needs no host.

== Detection ==
Antivirus software and CVE identifiers are used. NIST publishes guidance.

== Prevention ==
Organisations apply patching schedules, least privilege and staff awareness
training. The Center for Internet Security publishes benchmarks that many
organisations follow, and a firewall is configured to restrict traffic between
segments of a network so that an infection cannot spread freely from one host
to the next once it has arrived. Backups are kept offline so that ransomware
cannot reach them, and recovery is rehearsed rather than assumed to work.

== See also ==
Computer security

== References ==
Bickel, J. and Bratvold, R. wrote on the subject. Thomas, P. added to it.
"""


# --------------------------------------------------------------------------- #
# Reading the key point
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key_point,expected", [
    ("Types of malware: virus, worm, trojan", "malware virus worm trojan"),
    ("Methods of cooking vegetables", "cooking vegetables"),
    ("Configuration of access controls", "access controls"),
    ("Earthing", "Earthing"),
    ("Risk rating matrix", "Risk rating matrix"),
])
def test_the_subject_is_pulled_out_of_the_curriculum_heading(key_point,
                                                             expected):
    assert know.subject_of(key_point) == expected


def test_a_key_point_that_is_only_scaffolding_is_skipped():
    """"Definition of terms" reduces to "terms", which fetched Terms of
    Endearment. It is curriculum housekeeping, not a subject."""
    assert know.subject_of("Definition of terms") == "terms"
    assert know.note_for("Definition of terms", {"security", "malware"}) is None


def test_the_shortened_phrasing_is_offered_to_the_title_search():
    """The curriculum's trailing word is not the subject's: the article is
    called Password policy, not Password policy settings."""
    assert know._phrasings("Password policy settings") == [
        "Password policy settings", "Password policy"]
    assert know._phrasings("Earthing") == ["Earthing"]


# --------------------------------------------------------------------------- #
# Believing an article
# --------------------------------------------------------------------------- #
def test_the_unit_vocabulary_comes_from_the_whole_content():
    vocab = know.vocabulary("Manage ICT security", _ict_content())

    assert {"malware", "trojan", "vulnerability", "password", "access"} <= vocab
    # three words of one key point could never tell two subjects apart
    assert len(vocab) > 15


def test_an_article_in_the_wrong_sense_is_rejected():
    """Writing system reads perfectly and is about alphabets."""
    electrical = know.vocabulary(
        "Perform electrical installation",
        [ElementContent(element_number="1", element_title="Install wiring",
                        topics=[ContentTopic("1.1", "Wiring systems", [
                            "Cable and conduit", "Socket outlets",
                            "Earthing and circuit protection"])])])
    writing = ("A writing system is a conventional system for representing a "
               "language using symbols called a script.") * 20

    assert know.coverage(writing, electrical) < know.MIN_COVERAGE


def test_an_exact_title_beats_a_longer_article_that_shares_more_words():
    """Role-based access control lost to Relationship-based access control,
    which is longer and therefore holds more of a security vocabulary."""
    vocab = know.vocabulary("Manage ICT security", _ict_content())
    right, _ = know._score("Role-based access control", "roles and access",
                           "Role based access control", vocab)
    wrong, _ = know._score("Relationship-based access control",
                           "access control security password malware "
                           "vulnerability risk threat " * 5,
                           "Role based access control", vocab)

    assert right > wrong


def test_a_title_match_is_held_to_a_lower_bar_than_a_stranger():
    """Vulnerability scanner is short and exactly on subject; it was thrown
    out for being short."""
    assert (know._gate_for("Vulnerability scanner", "Vulnerability scanning")
            == know.MIN_COVERAGE_TITLED)
    assert (know._gate_for("Dynamic application security testing",
                           "Vulnerability scanning") == know.MIN_COVERAGE)


def test_a_disambiguation_page_is_not_an_article():
    assert know._is_signpost("Social engineering may refer to: " + "x" * 2000)
    assert know._is_signpost("short stub")
    assert not know._is_signpost(MALWARE)


# --------------------------------------------------------------------------- #
# Turning an article into a note
# --------------------------------------------------------------------------- #
def test_the_breakdown_is_the_articles_own_headings_without_the_apparatus():
    covers = know._covers(MALWARE)

    assert covers == ["History", "Types", "Detection", "Prevention"]


def test_names_exclude_sentence_openings_and_ordinary_capitalised_words():
    named = know._named(MALWARE)

    assert "CVE" in named and "NIST" in named
    assert "Types" not in named           # a heading, and a common word
    assert "Organisations" not in named   # written in lower case further down


def test_names_stop_before_the_bibliography():
    """Reading names out of a reference list produced Bickel and Bratvold."""
    named = know._named(MALWARE)

    assert "Bickel" not in named
    assert "Bratvold" not in named


def test_the_summary_is_the_lead_and_not_the_whole_article():
    summary = know._summary(MALWARE)

    assert summary.startswith("Malware is any software")
    assert "History" not in summary
    assert len(summary) <= know.SUMMARY_CHARS + 3


# --------------------------------------------------------------------------- #
# Which key points get looked up
# --------------------------------------------------------------------------- #
def test_key_points_are_spread_across_the_elements():
    """Taken in order, element 1's five key points would spend the whole
    budget and element 2's questions would be the shallow ones."""
    chosen = know.key_points(_ict_content(), limit=4)
    elements = [element for element, _topic, _point in chosen]

    assert elements.count("2") >= 1
    assert len(chosen) == 4


def test_a_sub_topic_with_no_key_points_stands_in_for_itself():
    content = [ElementContent(element_number="1", element_title="E",
                              topics=[ContentTopic("1.1", "Earthing", [])])]

    assert know.key_points(content) == [("1", "1.1", "Earthing")]


def test_an_empty_curriculum_asks_for_nothing():
    assert know.key_points([]) == []
    assert know.notes_for("Anything", []) == []


# --------------------------------------------------------------------------- #
# The prompt budget
# --------------------------------------------------------------------------- #
def _note(n):
    return KnowledgeNote(key_point=f"Key point {n}", topic_number="1.1",
                         summary="S" * 300, covers=["A", "B"],
                         named=["NIST", "CVE"], source_title="T")


def test_the_rendered_notes_stay_inside_their_budget():
    """The first version of this block pushed a request to 8070 tokens against
    a limit of 8000, and the provider refused the paper outright."""
    rendered = know.render([_note(i) for i in range(10)], budget=900)

    assert len(rendered) < 1400           # one note may cross the line


def test_a_budget_too_small_for_one_note_still_gives_one():
    """Better a single note than a block that says nothing."""
    assert know.render([_note(1)], budget=10)


def test_no_notes_renders_to_nothing():
    assert know.render([]) == ""
    assert know.summarise([]).startswith("no reference notes")


# --------------------------------------------------------------------------- #
# Failure is ordinary
# --------------------------------------------------------------------------- #
def test_a_dead_api_costs_the_notes_and_not_the_paper(monkeypatch):
    monkeypatch.setattr(know, "_api", lambda params: {})
    monkeypatch.setattr(know, "_cached", lambda unit: None)
    monkeypatch.setattr(know, "_remember", lambda unit, notes: None)

    assert know.notes_for("Manage ICT security", _ict_content()) == []


def test_an_api_that_raises_costs_one_key_point_and_not_the_rest(monkeypatch):
    calls = {"n": 0}

    def explode(key_point, vocab, element="", topic=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("network on fire")
        return KnowledgeNote(key_point=key_point, summary="s" * 40)

    monkeypatch.setattr(know, "note_for", explode)
    monkeypatch.setattr(know, "_cached", lambda unit: None)
    monkeypatch.setattr(know, "_remember", lambda unit, notes: None)

    notes = know.notes_for("Manage ICT security", _ict_content())

    assert len(notes) == calls["n"] - 1
    assert notes


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def test_notes_are_read_back_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(know, "CACHE_DIR", str(tmp_path))
    written = [KnowledgeNote(key_point="Malware", summary="what it is",
                             covers=["Types"], named=["CVE"],
                             source_title="Malware")]
    know._remember("Manage ICT security", written)

    read = know._cached("Manage ICT security")

    assert [n.key_point for n in read] == ["Malware"]
    assert read[0].named == ["CVE"]


def test_a_stale_cache_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(know, "CACHE_DIR", str(tmp_path))
    path = know.cache_path("Manage ICT security")
    os.makedirs(tmp_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"fetched": time.time() - know.CACHE_DAYS * 86400 - 10,
                   "notes": [{"key_point": "Malware"}]}, fh)

    assert know._cached("Manage ICT security") is None


def test_a_corrupt_cache_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(know, "CACHE_DIR", str(tmp_path))
    os.makedirs(tmp_path, exist_ok=True)
    with open(know.cache_path("Manage ICT security"), "w",
              encoding="utf-8") as fh:
        fh.write("{not json")

    assert know._cached("Manage ICT security") is None


def test_a_definition_of_terms_key_point_is_housekeeping():
    """"Definition of terms: threat, vulnerability, risk, asset" searched to
    the Common Vulnerability Scoring System - real, about vulnerabilities, and
    nothing to do with defining four words. The trainer defines them; there is
    nothing to read up."""
    assert know._is_housekeeping(
        know.subject_of("Definition of terms: threat, vulnerability, risk"))
    assert know._is_housekeeping(know.subject_of("Overview of the topic"))


@pytest.mark.parametrize("key_point", [
    "Types of malware: virus, worm, trojan",
    "Principle of least privilege",
    "Role based access control",
    "Risk rating matrix: likelihood against impact",
])
def test_a_real_subject_is_not_mistaken_for_housekeeping(key_point):
    assert not know._is_housekeeping(know.subject_of(key_point))
