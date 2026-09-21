"""Remembering which PCs each CAT has already assessed.

The ledger's whole job is to be right weeks later: the run that sets CAT 1 is
rarely the run that sets CAT 2, so these tests write to disk and read back
rather than exercising an in-memory object.
"""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assessment_ledger as ledger_store
from assessment_models import (CAT_1, CAT_2, CAT_3, FINAL_CAT, PRACTICAL,
                               THEORY, Ledger)

UNIT = "TO/OS/TTM/CR/01/6/MA"
ALL_PCS = ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3"]


@pytest.fixture(autouse=True)
def ledger_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_store, "LEDGER_DIR", str(tmp_path / "ledger"))


def _after_cat1(pcs=("1.1", "1.2"), assessment_type=THEORY):
    led = Ledger(unit_code=UNIT)
    led.record(CAT_1, assessment_type, list(pcs))
    return led


# --------------------------------------------------------------------------- #
# What the next CAT opens with
# --------------------------------------------------------------------------- #
def test_the_next_cat_defaults_to_what_the_last_one_left():
    assert ledger_store.default_selection(
        _after_cat1(), CAT_2, THEORY, ALL_PCS) == ["1.3", "2.1", "2.2", "2.3"]


def test_a_final_cat_re_enables_the_whole_unit():
    """It is comprehensive, which is why it is marked out of the full unit
    where the earlier three cover a subset."""
    assert ledger_store.default_selection(
        _after_cat1(), FINAL_CAT, THEORY, ALL_PCS) == ALL_PCS


def test_the_two_types_are_remembered_apart():
    """A PC assessed in a written CAT has not been assessed practically."""
    led = _after_cat1(assessment_type=THEORY)

    assert ledger_store.default_selection(
        led, CAT_2, PRACTICAL, ALL_PCS) == ALL_PCS


def test_a_cat_only_excludes_the_cats_before_it():
    """Re-opening CAT 2 must not exclude what CAT 3 went on to assess, or
    editing an earlier CAT would silently empty it."""
    led = _after_cat1()
    led.record(CAT_3, THEORY, ["2.1", "2.2"])

    assert ledger_store.default_selection(
        led, CAT_2, THEORY, ALL_PCS) == ["1.3", "2.1", "2.2", "2.3"]


def test_nothing_left_to_assess_opens_with_everything():
    """An empty selection is never a useful starting point - the trainer
    would have to tick every box to get anywhere."""
    led = _after_cat1(pcs=ALL_PCS)

    assert ledger_store.default_selection(led, CAT_2, THEORY, ALL_PCS) == ALL_PCS


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_coverage_names_what_no_cat_has_ever_assessed():
    led = _after_cat1()
    led.record(CAT_2, THEORY, ["1.3"])

    assert ledger_store.coverage(led, ALL_PCS)[THEORY] == ["2.1", "2.2", "2.3"]
    assert ledger_store.coverage(led, ALL_PCS)[PRACTICAL] == ALL_PCS


# --------------------------------------------------------------------------- #
# Surviving the gap between sessions
# --------------------------------------------------------------------------- #
def test_a_ledger_survives_being_written_and_read_back():
    led = _after_cat1()
    led.record(CAT_1, PRACTICAL, ["2.3"])
    ledger_store.save(led)

    back = ledger_store.load(UNIT)

    assert back.already_assessed(THEORY) == ["1.1", "1.2"]
    assert back.already_assessed(PRACTICAL) == ["2.3"]


def test_a_unit_code_with_slashes_still_makes_one_file():
    """Unit codes are IT/OS/ICTA/CR/01/5/MA - a path, not a file name."""
    ledger_store.save(_after_cat1())

    assert os.path.isfile(ledger_store.path_for(UNIT))


def test_two_units_do_not_share_a_ledger():
    ledger_store.save(_after_cat1())
    other = Ledger(unit_code="IT/OS/ICTA/CR/01/5/MA")
    other.record(CAT_1, THEORY, ["3.1"])
    ledger_store.save(other)

    assert ledger_store.load(UNIT).already_assessed(THEORY) == ["1.1", "1.2"]


def test_an_unread_unit_starts_empty_rather_than_failing():
    assert ledger_store.load("NEVER/SEEN/BEFORE").already_assessed(THEORY) == []


def test_a_corrupt_ledger_costs_a_default_not_the_run():
    """Half a JSON file is what a crash mid-write leaves. The trainer gets a
    worse default; they do not get a stack trace."""
    os.makedirs(ledger_store.LEDGER_DIR, exist_ok=True)
    with io.open(ledger_store.path_for(UNIT), "w", encoding="utf-8") as fh:
        fh.write('{"consumed": {"theory":')

    assert ledger_store.load(UNIT).already_assessed(THEORY) == []


def test_junk_inside_a_readable_ledger_is_ignored_not_trusted():
    os.makedirs(ledger_store.LEDGER_DIR, exist_ok=True)
    with io.open(ledger_store.path_for(UNIT), "w", encoding="utf-8") as fh:
        json.dump({"consumed": {"theory": {CAT_1: ["1.1"]},
                                "nonsense": {CAT_1: ["9.9"]},
                                "practical": "not a mapping"}}, fh)

    back = ledger_store.load(UNIT)

    assert back.already_assessed(THEORY) == ["1.1"]
    assert back.already_assessed(PRACTICAL) == []
