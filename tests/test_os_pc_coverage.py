"""Regression: no OS unit may have elements but ZERO performance criteria.

This is the invariant the `detect_column_split` bug violated: on units whose PC
column starts far left (e.g. the Computer-Science OS 'Computer Organisation' and
'Artificial Intelligence'), the boundary was placed inside the PC column, so the
elements parsed but every PC was dropped. `_pc_anchored_split` fixes it. These
tests run against the sample OS PDFs bundled in the repo root (ZERO API calls).
"""

import os

import pytest

import os_parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Both Occupational-Standard samples shipped in the repo. (3) is the Level-5 ICTA
# set used elsewhere by conftest; (4) is the Computer-Science Level-6 set where
# the column-split bug originally surfaced.
OS_SAMPLES = ["Occupational Standard (3).pdf", "Occupational Standard (4).pdf"]


@pytest.mark.parametrize("sample", OS_SAMPLES)
def test_every_unit_with_elements_has_pcs(sample):
    path = os.path.join(ROOT, sample)
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {sample}")
    units = os_parser.parse_os(path)
    assert units, f"no units parsed from {sample}"
    offenders = [u.unit_title for u in units if u.elements and not u.all_pcs]
    assert not offenders, f"{sample}: units with elements but no PCs: {offenders}"


def test_computer_science_os_formerly_broken_units():
    """The two units that parsed 0 PCs before the fix now parse their full set."""
    path = os.path.join(ROOT, "Occupational Standard (4).pdf")
    if not os.path.exists(path):
        pytest.skip("Computer-Science OS sample not present")
    units = os_parser.parse_os(path)

    def find(substr):
        return next((u for u in units if substr in u.unit_title.upper()), None)

    comp_org = find("COMPUTER ORGANISATION")
    ai = find("ARTIFICIAL INTELLIGENCE")
    assert comp_org is not None and ai is not None
    assert len(comp_org.all_pcs) > 0, "Computer Organisation parsed no PCs"
    assert len(ai.all_pcs) > 0, "Artificial Intelligence parsed no PCs"
