"""The verbs a sector's own papers open with.

Read off published CDACC papers sector by sector. They widen what a STEM may
open with; they must never change how a CRITERION is read.
"""
import pytest

import assessment_ai
import assessment_validators as av
from assessment_config import (ANALYSING, APPLYING, CREATING, EVALUATING,
                               KNOWLEDGE, UNDERSTANDING, HOUSE_VERBS,
                               VERB_BANK, allowed_verbs, level_of_verb,
                               natural_level, sector_for)


# --------------------------------------------------------------------------- #
# Finding the sector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("programme,expected", [
    ("ICT Technician Level 6", "Computing & Informatics"),
    ("Cyber Security Level 6", "Computing & Informatics"),
    ("Electrical Installation Level 5", "Electrical & Electronic Engineering"),
    ("Solar PV Installation Level 5", "Electrical & Electronic Engineering"),
    ("Automotive Engineering Level 6", "Mechanical & Automotive Engineering"),
    ("Welding and Fabrication Level 5", "Mechanical & Automotive Engineering"),
    ("Building Technology Level 6", "Building & Civil Engineering"),
    ("Food and Beverage Level 5", "Hospitality"),
    ("Community Health Level 6", "Health Sciences"),
    ("Social Work and Community Development", "Social Work"),
    ("Horticulture Level 5", "Agriculture & Aquaculture"),
    ("Accountancy Level 6", "Business Studies"),
    ("Hairdressing Level 5", "Fashion Design & Cosmetology"),
    ("Analytical Chemistry Level 6", "Applied Sciences"),
])
def test_a_programme_finds_its_sector(programme, expected):
    sector = sector_for(programme, "Some unit")

    assert sector is not None and sector.name == expected


def test_an_unknown_programme_has_no_sector():
    """A wrong sector is worse than none - it would put "Inscribe" in front of
    a catering trainee."""
    assert sector_for("Something Else Entirely", "A unit") is None


def test_a_basic_unit_keeps_its_own_style_inside_any_programme():
    """Communication is taught in an engineering programme and in a
    hospitality one, and is set the same way in both."""
    for programme in ("Automotive Engineering Level 6", "Baking Level 5"):
        sector = sector_for(programme, "Demonstrate Communication Skills")
        assert sector.name == "Basic & common units"


def test_the_longest_programme_match_wins():
    """"agricultural engineering" must not lose to "engineering" matching
    inside it."""
    sector = sector_for("Agricultural Engineering Level 5", "Operate a tractor")

    assert sector.name == "Agriculture & Aquaculture"


# --------------------------------------------------------------------------- #
# What a stem may open with
# --------------------------------------------------------------------------- #
def test_a_sector_gains_only_the_verbs_it_actually_uses():
    electrical = sector_for("Electrical Installation Level 5", "Wiring")

    applying = allowed_verbs(APPLYING, electrical)

    assert "calculate" in applying and "determine" in applying
    assert "factorize" in applying
    assert "inscribe" not in applying      # that is Mechanical's, not theirs
    assert "draw" not in applying          # nor theirs


def test_a_sector_gains_nothing_at_a_level_its_papers_do_not_use():
    electrical = sector_for("Electrical Installation Level 5", "Wiring")

    assert allowed_verbs(KNOWLEDGE, electrical) == VERB_BANK[KNOWLEDGE]


def test_the_generic_bank_always_survives():
    for sector_name in ("ICT Technician", "Automotive Engineering",
                        "Community Health", "Analytical Chemistry"):
        sector = sector_for(sector_name, "A unit")
        for level in VERB_BANK:
            assert set(VERB_BANK[level]) <= set(allowed_verbs(level, sector))


def test_a_sector_with_no_paper_found_uses_the_bank_alone():
    """Applied Sciences: nothing published to copy, so nothing is invented."""
    applied = sector_for("Analytical Chemistry Level 6", "Titration")

    assert applied.verbs == ()
    for level in VERB_BANK:
        assert allowed_verbs(level, applied) == VERB_BANK[level]


def test_no_sector_at_all_uses_the_bank_alone():
    for level in VERB_BANK:
        assert allowed_verbs(level, None) == VERB_BANK[level]


# --------------------------------------------------------------------------- #
# The line that must not be crossed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("criterion", [
    "Advice is given to the client as per the policy",
    "Records are written up according to the procedure",
    "Materials are found on site as per the schedule",
    "Readings are determined using the instrument",
    "Drawings are printed as per the standard",
])
def test_house_verbs_never_change_how_a_criterion_is_read(criterion):
    """These words appear constantly in criteria in senses that have nothing
    to do with cognitive level. Feeding them to the criterion reader would
    re-level half the library overnight."""
    assert natural_level(criterion) == ""


def test_the_criterion_reader_still_reads_the_bank():
    assert natural_level("Tools and equipment are identified") == KNOWLEDGE
    assert natural_level("Faults are diagnosed by procedure") == ANALYSING
    assert natural_level("A schedule is developed as per the manual") == CREATING


# --------------------------------------------------------------------------- #
# A house verb is a real verb to the checker
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("verb,level", [
    ("calculate", APPLYING), ("determine", APPLYING), ("inscribe", APPLYING),
    ("mention", KNOWLEDGE), ("highlight", KNOWLEDGE), ("discuss", UNDERSTANDING),
    ("advise", EVALUATING), ("what do you understand by", UNDERSTANDING),
])
def test_every_house_verb_has_a_level(verb, level):
    assert level_of_verb(verb) == level


def test_a_multi_word_opening_is_recognised():
    """"What do you understand by" is five words and its first word is not a
    verb at all."""
    assert av._verb_at("What do you understand by a risk matrix?") == \
        "what do you understand by"
    assert av._verb_at("Illustrate and label the parts") == \
        "illustrate and label"
    assert av._verb_at("Carry out the procedure") == "carry out"
    assert av._verb_at("State FOUR threats") == "state"


def test_every_house_verb_is_reachable_from_some_sector():
    """A verb in the table that no sector uses is dead weight, and a sign the
    two tables have drifted apart."""
    from assessment_config import SECTORS, BASIC_SECTOR
    used = set(BASIC_SECTOR.verbs)
    for sector in SECTORS:
        used |= set(sector.verbs)
    for level, verbs in HOUSE_VERBS.items():
        for verb in verbs:
            assert verb in used or verb.endswith("ise") or verb.endswith("ize"), \
                f"{verb} ({level}) belongs to no sector"
