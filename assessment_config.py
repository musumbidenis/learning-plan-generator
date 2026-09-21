"""Everything about assessment tools that is a policy rather than a rule.

Bloom profiles, verb banks and thresholds live here as data because they are
institutional choices, not facts about the domain: a college changes its mix
of levels without anyone touching the allocation engine. Nothing in this file
is hard-coded anywhere else.

One deliberate divergence from the generic taxonomies: this institution's
guideline places "compare" under UNDERSTANDING, where most published Bloom
lists put it under ANALYSING. The guideline wins. Do not "fix" it.
"""

from __future__ import annotations

from typing import Dict, List

from assessment_models import (ANALYSING, APPLYING, BLOOM_LEVELS, CREATING,
                               EVALUATING, KNOWLEDGE, UNDERSTANDING)

# --------------------------------------------------------------------------- #
# How a paper should spread across the levels
# --------------------------------------------------------------------------- #
# Keyed by KNQF level as a string, as `models.Unit.level` stores it. The
# fallback is used for any level not named here.
BLOOM_PROFILES: Dict[str, Dict[str, float]] = {
    "3": {KNOWLEDGE: 0.30, UNDERSTANDING: 0.30, APPLYING: 0.25,
          ANALYSING: 0.10, EVALUATING: 0.03, CREATING: 0.02},
    "4": {KNOWLEDGE: 0.25, UNDERSTANDING: 0.25, APPLYING: 0.25,
          ANALYSING: 0.15, EVALUATING: 0.05, CREATING: 0.05},
    "5": {KNOWLEDGE: 0.20, UNDERSTANDING: 0.20, APPLYING: 0.25,
          ANALYSING: 0.20, EVALUATING: 0.08, CREATING: 0.07},
    "6": {KNOWLEDGE: 0.15, UNDERSTANDING: 0.20, APPLYING: 0.25,
          ANALYSING: 0.20, EVALUATING: 0.10, CREATING: 0.10},
}
DEFAULT_PROFILE = BLOOM_PROFILES["5"]


def profile_for(knqf_level: str) -> Dict[str, float]:
    """The Bloom mix for a level, falling back rather than failing."""
    return BLOOM_PROFILES.get(str(knqf_level or "").strip(), DEFAULT_PROFILE)


# --------------------------------------------------------------------------- #
# The verbs an item may open with
# --------------------------------------------------------------------------- #
VERB_BANK: Dict[str, List[str]] = {
    KNOWLEDGE: ["list", "state", "define", "name", "identify", "select",
                "outline"],
    UNDERSTANDING: ["describe", "explain", "summarise", "summarize",
                    "classify", "compare", "distinguish", "interpret"],
    APPLYING: ["demonstrate", "apply", "compute", "illustrate", "use",
               "solve", "carry out"],
    ANALYSING: ["analyse", "analyze", "differentiate", "examine",
                "investigate", "diagnose", "break down"],
    EVALUATING: ["evaluate", "justify", "critique", "recommend", "assess",
                 "defend"],
    CREATING: ["design", "develop", "formulate", "compose", "construct",
               "propose"],
}


def level_of_verb(verb: str) -> str:
    """The level a lead verb belongs to, or '' when it is in no bank."""
    want = (verb or "").strip().lower()
    for level in BLOOM_LEVELS:
        if want in VERB_BANK[level]:
            return level
    return ""


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
# A PC allocated more than this may be split into two items at different
# levels, which is how a paper reaches all six without disturbing any PC's
# total.
SPLIT_ITEM_THRESHOLD = 5

# Marks per selected PC. Below the floor a PC allocates to zero and cannot be
# assessed at all; below the viable figure every PC gets one mark, which forces
# the whole paper to the lowest levels.
MIN_MARKS_PER_PC = 1
VIABLE_MARKS_PER_PC = 2
MAX_TOTAL_MARKS = 200

# A practical tool's items of evaluation, parent items only - sub-parts do not
# count towards it.
MIN_CHECKLIST_ITEMS = 10
MAX_CHECKLIST_ITEMS = 25

# How alike two pieces of text may be before they are treated as the same
# thing: an item of evaluation that merely restates its PC, or an item whose
# stem gives away another item's marking key. Deliberately conservative to
# start with - tune against real output rather than guessing.
ITEM_VS_PC_SIMILARITY = 0.75
ITEM_INDEPENDENCE_OVERLAP = 0.60

# Levels at which a candidate must supply the response. Selected-response
# formats are not used here.
CONSTRUCTED_RESPONSE_ONLY_LEVELS = ("5", "6")

# The model writes prose under tight constraints, so it is kept cold.
TEMPERATURE = 0.3
MAX_REPAIR_PASSES = 2
