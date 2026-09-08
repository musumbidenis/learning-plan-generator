"""Pair the units indexed from an Occupational Standard with those from a
Curriculum, so the UI can show ONE table instead of two unrelated dropdowns.

The two documents describe the same units from different angles, and a trainer
picking from two independent lists is one slip away from planning an OS unit
against the wrong curriculum unit. Pairing them up front makes a mismatch
visible before anything is generated.

Matching cascades from most to least reliable, and stops at the first hit:

    isced  - identical ISCED unit code (the documents' real join key)
    code   - TVET code ignoring the OS/CU segment (IT/OS/... == IT/CU/...)
    title  - identical or containing unit title, once normalised
    fuzzy  - closest title above a similarity floor
    none   - no counterpart found in the other document

Every ref that goes in comes out exactly once. A unit present in only one of
the two documents becomes a half-empty row rather than disappearing - the
library holds plenty of programmes where the two files don't line up yet, and
silently dropping those units would hide the problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from curriculum_parser import norm, norm_code_loose
from models import UnitRef

# Below this title similarity a pair is more likely noise than a match. Titles
# in these documents are near-identical when they correspond at all, so the bar
# sits high on purpose.
FUZZY_THRESHOLD = 0.72

MATCH_ISCED = "isced"
MATCH_CODE = "code"
MATCH_TITLE = "title"
MATCH_FUZZY = "fuzzy"
MATCH_NONE = "none"

# Ordered best-first; used for the UI badge and for sorting.
_MATCH_RANK = {MATCH_ISCED: 0, MATCH_CODE: 1, MATCH_TITLE: 2,
               MATCH_FUZZY: 3, MATCH_NONE: 4}


@dataclass
class UnitPair:
    """One row of the matched table: an OS unit, its curriculum twin, or one of
    the two on its own."""
    os_ref: Optional[UnitRef] = None
    cu_ref: Optional[UnitRef] = None
    match: str = MATCH_NONE

    @property
    def is_matched(self) -> bool:
        return self.os_ref is not None and self.cu_ref is not None

    @property
    def is_reliable(self) -> bool:
        """True when the pairing rests on a code, not on title similarity."""
        return self.match in (MATCH_ISCED, MATCH_CODE)

    @property
    def title(self) -> str:
        """Whichever side has a title, preferring the Occupational Standard."""
        for ref in (self.os_ref, self.cu_ref):
            if ref is not None and ref.title:
                return ref.title
        return ""

    @property
    def code(self) -> str:
        """The best identifier available for display."""
        for ref in (self.os_ref, self.cu_ref):
            if ref is None:
                continue
            if ref.isced_code:
                return ref.isced_code
            if ref.code:
                return ref.code
        return ""


def _title_key(title: str) -> str:
    return norm(title)


def _title_words(title: str) -> str:
    """Whitespace-normalised lowercase title, for sequence comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))


def _similarity(a: str, b: str) -> float:
    wa, wb = _title_words(a), _title_words(b)
    if not wa or not wb:
        return 0.0
    return SequenceMatcher(None, wa, wb).ratio()


def _index_by(refs: List[UnitRef], key_fn, free: set) -> Dict[str, int]:
    """Map key -> index for still-unmatched refs. First occurrence wins, so a
    duplicated code never steals a later unit's counterpart."""
    out: Dict[str, int] = {}
    for i, ref in enumerate(refs):
        if i not in free:
            continue
        key = key_fn(ref)
        if key and key not in out:
            out[key] = i
    return out


def pair_units(os_refs: List[UnitRef],
               cu_refs: List[UnitRef]) -> List[UnitPair]:
    """Pair the two unit lists, best matches first within each OS unit's turn.

    The result preserves the Occupational Standard's ordering, with any
    curriculum-only units appended at the end.
    """
    os_refs = list(os_refs or [])
    cu_refs = list(cu_refs or [])

    partner: Dict[int, int] = {}          # os index -> cu index
    reason: Dict[int, str] = {}
    free_cu = set(range(len(cu_refs)))

    def run_pass(key_fn, label: str) -> None:
        lookup = _index_by(cu_refs, key_fn, free_cu)
        if not lookup:
            return
        for i, ref in enumerate(os_refs):
            if i in partner:
                continue
            j = lookup.get(key_fn(ref))
            if j is not None and j in free_cu:
                partner[i], reason[i] = j, label
                free_cu.discard(j)

    run_pass(lambda r: norm(r.isced_code), MATCH_ISCED)
    run_pass(lambda r: norm_code_loose(r.code), MATCH_CODE)
    run_pass(lambda r: _title_key(r.title), MATCH_TITLE)

    # Fuzzy pass: score every remaining combination and take the strongest
    # pairings first, so a good match is never consumed by a weaker one.
    remaining_os = [i for i in range(len(os_refs)) if i not in partner]
    if remaining_os and free_cu:
        scored = []
        for i in remaining_os:
            for j in sorted(free_cu):
                score = _similarity(os_refs[i].title, cu_refs[j].title)
                if score >= FUZZY_THRESHOLD:
                    scored.append((score, i, j))
        for _score, i, j in sorted(scored, key=lambda t: -t[0]):
            if i in partner or j not in free_cu:
                continue
            partner[i], reason[i] = j, MATCH_FUZZY
            free_cu.discard(j)

    pairs = [UnitPair(os_ref=ref,
                      cu_ref=cu_refs[partner[i]] if i in partner else None,
                      match=reason.get(i, MATCH_NONE))
             for i, ref in enumerate(os_refs)]
    pairs.extend(UnitPair(os_ref=None, cu_ref=cu_refs[j], match=MATCH_NONE)
                 for j in sorted(free_cu))
    return pairs


def match_rank(pair: UnitPair) -> int:
    """Sort key exposing how trustworthy a pairing is (0 = best)."""
    return _MATCH_RANK.get(pair.match, len(_MATCH_RANK))


def suggest_counterparts(os_refs: List[UnitRef], cu_refs: List[UnitRef]):
    """Index-to-index suggestions in both directions, for manual matching.

    The UI shows the two documents' units as separate tables and lets the
    trainer pair them by hand; these suggestions are what it pre-fills and
    marks, so the matching stays a helper rather than a decision made for them.

    Returns (os_to_cu, cu_to_os), each mapping an index to (index, match_kind).
    """
    os_pos = {id(r): i for i, r in enumerate(os_refs)}
    cu_pos = {id(r): i for i, r in enumerate(cu_refs)}
    os_to_cu, cu_to_os = {}, {}
    for pair in pair_units(os_refs, cu_refs):
        if not pair.is_matched:
            continue
        i, j = os_pos.get(id(pair.os_ref)), cu_pos.get(id(pair.cu_ref))
        if i is None or j is None:
            continue
        os_to_cu[i] = (j, pair.match)
        cu_to_os[j] = (i, pair.match)
    return os_to_cu, cu_to_os
