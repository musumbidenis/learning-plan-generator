"""Shared pytest fixtures. Adds the project root to sys.path and locates the
sample documents that ship with the repo (used for integration-style parser
tests that make ZERO API calls)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OS_PDF = os.path.join(ROOT, "Occupational Standard (3).pdf")
CURR_PDF = os.path.join(ROOT, "Curriculum  (3).pdf")

PROG_ISCED = "0613 451 05A"
PROG_OS_CODE = "IT/OS/ICTA/CC/02/5/MA"


@pytest.fixture(scope="session")
def os_units():
    import os_parser
    if not os.path.exists(OS_PDF):
        pytest.skip("sample Occupational Standard PDF not present")
    return os_parser.parse_os(OS_PDF)


@pytest.fixture(scope="session")
def curr_units():
    import curriculum_parser as cp
    if not os.path.exists(CURR_PDF):
        pytest.skip("sample Curriculum PDF not present")
    return cp.parse_curriculum(CURR_PDF)


@pytest.fixture(scope="session")
def prog_os_unit(os_units):
    import os_parser
    u = os_parser.find_unit(os_units, PROG_OS_CODE)
    assert u is not None
    return u


@pytest.fixture(scope="session")
def prog_curr_unit(curr_units):
    import curriculum_parser as cp
    u = cp.find_unit(curr_units, isced_code=PROG_ISCED)
    assert u is not None
    return u
