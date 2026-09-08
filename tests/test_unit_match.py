"""Pairing OS units against Curriculum units - pure functions, no I/O."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_match
from models import UnitRef


def os_ref(title, isced="", code=""):
    return UnitRef(title=title, isced_code=isced, code=code, source="OS")


def cu_ref(title, isced="", code=""):
    return UnitRef(title=title, isced_code=isced, code=code, source="CU")


# --------------------------------------------------------------------------- #
# The join keys
# --------------------------------------------------------------------------- #
def test_pairs_on_the_isced_code():
    pairs = unit_match.pair_units(
        [os_ref("APPLY COMPUTER PROGRAMMING SKILLS", isced="0611 151 05 A")],
        [cu_ref("Computer Programming", isced="0611 151 05 A")])
    assert len(pairs) == 1
    assert pairs[0].match == unit_match.MATCH_ISCED
    assert pairs[0].is_matched and pairs[0].is_reliable


def test_pairs_across_the_os_cu_code_families():
    # IT/OS/... and IT/CU/... name the same unit in the two documents
    pairs = unit_match.pair_units(
        [os_ref("APPLY COMMUNICATION SKILLS", code="IT/OS/ICTA/CC/02/5/MA")],
        [cu_ref("COMMUNICATION SKILLS", code="IT/CU/ICTA/CC/02/5/MA")])
    assert pairs[0].match == unit_match.MATCH_CODE
    assert pairs[0].is_reliable


def test_code_family_matching_is_not_limited_to_it_codes():
    pairs = unit_match.pair_units(
        [os_ref("WELDING", code="ENG/OS/MECH/CC/01/4/MA")],
        [cu_ref("WELDING PRACTICE", code="ENG/CU/MECH/CC/01/4/MA")])
    assert pairs[0].match == unit_match.MATCH_CODE


def test_pairs_on_the_title_when_no_code_is_available():
    pairs = unit_match.pair_units([os_ref("APPLY COMMUNICATION SKILLS")],
                                  [cu_ref("Apply Communication Skills")])
    assert pairs[0].match == unit_match.MATCH_TITLE
    assert pairs[0].is_matched
    # a title match is a weaker claim than a code match, and says so
    assert not pairs[0].is_reliable


def test_pairs_on_a_near_title_when_wording_differs_slightly():
    pairs = unit_match.pair_units([os_ref("APPLY ENTREPRENEURIAL SKILLS")],
                                  [cu_ref("APPLY ENTREPRENEURSHIP SKILLS")])
    assert pairs[0].match == unit_match.MATCH_FUZZY
    assert not pairs[0].is_reliable


def test_unrelated_titles_are_left_unpaired():
    pairs = unit_match.pair_units([os_ref("APPLY WELDING TECHNIQUES")],
                                  [cu_ref("COMPUTER NETWORKING")])
    assert len(pairs) == 2
    assert all(p.match == unit_match.MATCH_NONE for p in pairs)
    assert all(not p.is_matched for p in pairs)


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #
def test_the_isced_code_beats_a_more_flattering_title():
    """The codes are authoritative; identical titles must not override them."""
    o = [os_ref("COMMUNICATION SKILLS", isced="0611 151 05 A")]
    c = [cu_ref("COMPUTER PROGRAMMING", isced="0611 151 05 A"),
         cu_ref("COMMUNICATION SKILLS", isced="0999 999 99 Z")]
    pairs = unit_match.pair_units(o, c)
    matched = [p for p in pairs if p.is_matched]
    assert len(matched) == 1
    assert matched[0].match == unit_match.MATCH_ISCED
    assert matched[0].cu_ref.title == "COMPUTER PROGRAMMING"


def test_the_stronger_fuzzy_pairing_wins_over_a_weaker_one():
    o = [os_ref("APPLY DIGITAL LITERACY SKILLS")]
    c = [cu_ref("APPLY DIGITAL LITERACY"), cu_ref("APPLY DIGITAL LITERACY SKILL")]
    pairs = unit_match.pair_units(o, c)
    matched = [p for p in pairs if p.is_matched]
    assert matched[0].cu_ref.title == "APPLY DIGITAL LITERACY SKILL"


# --------------------------------------------------------------------------- #
# Nothing is lost
# --------------------------------------------------------------------------- #
def test_a_unit_present_in_only_one_document_still_gets_a_row():
    o = [os_ref("SHARED UNIT", isced="1"), os_ref("OS ONLY UNIT", isced="2")]
    c = [cu_ref("SHARED UNIT", isced="1"), cu_ref("CURRICULUM ONLY UNIT", isced="3")]
    pairs = unit_match.pair_units(o, c)

    assert len(pairs) == 3
    os_only = [p for p in pairs if p.os_ref and not p.cu_ref]
    cu_only = [p for p in pairs if p.cu_ref and not p.os_ref]
    assert [p.os_ref.title for p in os_only] == ["OS ONLY UNIT"]
    assert [p.cu_ref.title for p in cu_only] == ["CURRICULUM ONLY UNIT"]


def test_every_ref_appears_exactly_once():
    o = [os_ref(f"UNIT {i}", isced=str(i)) for i in range(5)]
    c = [cu_ref(f"UNIT {i}", isced=str(i)) for i in range(3, 8)]
    pairs = unit_match.pair_units(o, c)

    seen_os = [id(p.os_ref) for p in pairs if p.os_ref is not None]
    seen_cu = [id(p.cu_ref) for p in pairs if p.cu_ref is not None]
    assert sorted(seen_os) == sorted(id(r) for r in o)
    assert sorted(seen_cu) == sorted(id(r) for r in c)


def test_one_curriculum_unit_is_never_claimed_by_two_os_units():
    o = [os_ref("APPLY COMMUNICATION SKILLS"), os_ref("APPLY COMMUNICATION SKILLS")]
    c = [cu_ref("APPLY COMMUNICATION SKILLS")]
    pairs = unit_match.pair_units(o, c)
    assert sum(1 for p in pairs if p.is_matched) == 1
    assert len(pairs) == 2


def test_os_ordering_is_preserved_with_curriculum_only_units_appended():
    o = [os_ref("FIRST", isced="1"), os_ref("SECOND", isced="2")]
    c = [cu_ref("SECOND", isced="2"), cu_ref("ORPHAN", isced="9")]
    pairs = unit_match.pair_units(o, c)
    assert [p.title for p in pairs] == ["FIRST", "SECOND", "ORPHAN"]


def test_empty_inputs_are_handled():
    assert unit_match.pair_units([], []) == []
    assert len(unit_match.pair_units([os_ref("A")], [])) == 1
    assert len(unit_match.pair_units([], [cu_ref("A")])) == 1


# --------------------------------------------------------------------------- #
# Display helpers used by the table
# --------------------------------------------------------------------------- #
def test_pair_reports_a_code_for_display_preferring_isced():
    pair = unit_match.pair_units(
        [os_ref("UNIT", isced="0611 151 05 A", code="IT/OS/ICTA/CC/02/5/MA")],
        [cu_ref("UNIT", isced="0611 151 05 A")])[0]
    assert pair.code == "0611 151 05 A"


def test_pair_falls_back_to_the_tvet_code_and_to_the_curriculum_side():
    assert unit_match.pair_units(
        [os_ref("UNIT", code="IT/OS/ICTA/CC/02/5/MA")], [])[0].code \
        == "IT/OS/ICTA/CC/02/5/MA"
    assert unit_match.pair_units(
        [], [cu_ref("UNIT", code="IT/CU/ICTA/CC/02/5/MA")])[0].code \
        == "IT/CU/ICTA/CC/02/5/MA"


# --------------------------------------------------------------------------- #
# Suggestions for the manual matching tables
# --------------------------------------------------------------------------- #
def test_suggestions_map_indices_in_both_directions():
    o = [os_ref("FIRST", isced="1"), os_ref("SECOND", isced="2")]
    c = [cu_ref("SECOND", isced="2"), cu_ref("FIRST", isced="1")]
    os_to_cu, cu_to_os = unit_match.suggest_counterparts(o, c)
    assert os_to_cu[0][0] == 1 and os_to_cu[1][0] == 0
    assert cu_to_os[1][0] == 0 and cu_to_os[0][0] == 1


def test_suggestions_carry_the_match_kind_so_the_ui_can_flag_weak_ones():
    o = [os_ref("APPLY COMMUNICATION SKILLS", isced="1")]
    c = [cu_ref("COMMUNICATION SKILLS", isced="1")]
    os_to_cu, _ = unit_match.suggest_counterparts(o, c)
    assert os_to_cu[0][1] == unit_match.MATCH_ISCED

    o = [os_ref("APPLY COMMUNICATION SKILLS")]
    c = [cu_ref("Apply Communication Skills")]
    os_to_cu, _ = unit_match.suggest_counterparts(o, c)
    assert os_to_cu[0][1] in (unit_match.MATCH_TITLE, unit_match.MATCH_FUZZY)


def test_units_with_no_counterpart_get_no_suggestion():
    o = [os_ref("ONLY IN THE OS", isced="1")]
    c = [cu_ref("ONLY IN THE CURRICULUM", isced="2")]
    os_to_cu, cu_to_os = unit_match.suggest_counterparts(o, c)
    assert os_to_cu == {} and cu_to_os == {}


def test_suggestions_are_one_to_one():
    o = [os_ref("SHARED"), os_ref("SHARED")]
    c = [cu_ref("SHARED")]
    os_to_cu, cu_to_os = unit_match.suggest_counterparts(o, c)
    assert len(os_to_cu) == 1 and len(cu_to_os) == 1
    assert len(set(j for j, _ in os_to_cu.values())) == len(os_to_cu)


def test_suggestions_handle_an_empty_side():
    assert unit_match.suggest_counterparts([], [cu_ref("A")]) == ({}, {})
    assert unit_match.suggest_counterparts([os_ref("A")], []) == ({}, {})
