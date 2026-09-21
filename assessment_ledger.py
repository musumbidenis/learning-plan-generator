"""Which performance criteria each CAT has already assessed, remembered.

A unit is assessed four times - CAT 1, 2, 3 and a Final CAT - and each of
those has a written and a practical record. The point of remembering is that
CAT 2 should not silently re-assess what CAT 1 already covered, and that
before a Final CAT somebody should be told which PCs the unit has never
assessed at all.

Kept on disk rather than in `st.session_state`, because the run that writes
CAT 1 is rarely the run that writes CAT 2 - a trainer comes back to it weeks
later, after the Streamlit session is long gone. One JSON file per unit, named
from its code, under a gitignored directory beside the other caches.

The Final CAT is comprehensive and ignores all of this: it re-enables every PC
in the unit, which is why it is marked out of the whole unit where the earlier
three cover a subset.
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Dict, List

from assessment_models import (ASSESSMENT_TYPES, CAT_SEQUENCE, FINAL_CAT,
                               Ledger)

LEDGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".assessment_ledger")


def _safe_name(unit_code: str) -> str:
    """A file name from a unit code. 'IT/OS/ICTA/CR/01/5/MA' has slashes in it."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", unit_code or "").strip("_")
    return (cleaned or "unit")[:80] + ".json"


def path_for(unit_code: str) -> str:
    return os.path.join(LEDGER_DIR, _safe_name(unit_code))


def load(unit_code: str) -> Ledger:
    """This unit's ledger, or an empty one. Never raises: a ledger that cannot
    be read costs a good default, not the run."""
    ledger = Ledger(unit_code=unit_code)
    try:
        with io.open(path_for(unit_code), encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return ledger
    consumed = raw.get("consumed")
    if not isinstance(consumed, dict):
        return ledger
    for assessment_type, by_cat in consumed.items():
        if assessment_type not in ASSESSMENT_TYPES or not isinstance(by_cat, dict):
            continue
        for cat_id, pcs in by_cat.items():
            if isinstance(pcs, list):
                ledger.record(cat_id, assessment_type,
                              [str(p) for p in pcs if p])
    return ledger


def save(ledger: Ledger) -> None:
    """Write it back. A ledger that cannot be saved costs the next default."""
    try:
        os.makedirs(LEDGER_DIR, exist_ok=True)
        with io.open(path_for(ledger.unit_code), "w", encoding="utf-8") as fh:
            json.dump({"unit_code": ledger.unit_code,
                       "consumed": ledger.consumed}, fh, indent=2)
    except OSError:
        pass


def default_selection(ledger: Ledger, cat_id: str, assessment_type: str,
                      all_pc_numbers: List[str]) -> List[str]:
    """What this CAT should open with selected.

    The earlier CATs' leftovers, in the unit's own order - or everything, for
    a Final CAT, and for a CAT that would otherwise open with nothing left to
    assess. An empty selection is never a useful starting point: the trainer
    would have to tick twelve boxes to get anywhere.
    """
    if cat_id == FINAL_CAT:
        return list(all_pc_numbers)
    already = set(ledger.already_assessed(assessment_type,
                                          before=CAT_SEQUENCE.get(cat_id, 99)))
    remaining = [n for n in all_pc_numbers if n not in already]
    return remaining or list(all_pc_numbers)


def coverage(ledger: Ledger, all_pc_numbers: List[str]) -> Dict[str, List[str]]:
    """{assessment_type: the PCs no CAT of that type has assessed}.

    What a trainer needs before setting a Final CAT: a unit is not fully
    assessed just because four CATs happened.
    """
    out: Dict[str, List[str]] = {}
    for assessment_type in ASSESSMENT_TYPES:
        seen = set(ledger.already_assessed(assessment_type))
        out[assessment_type] = [n for n in all_pc_numbers if n not in seen]
    return out
