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

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _participles(verb: str) -> set:
    """The forms a verb takes in a performance criterion.

    CDACC writes its criteria in the passive - "Tools and equipment ARE
    IDENTIFIED according to workplace procedures" - so the bank's infinitives
    never appear as written. The forms are generated from the bank rather than
    listed in a second table, because a second table is a second thing to keep
    in step with the first.
    """
    forms = {verb, verb + "ed", verb + "d"}
    if verb.endswith("y"):
        forms.add(verb[:-1] + "ied")
    if verb.endswith("e"):
        forms.add(verb + "d")
    if re.match(r".*[aeiou][bcdfgklmnprstvz]$", verb):
        forms.add(verb + verb[-1] + "ed")     # plan -> planned
    return forms


_NATURAL: Dict[str, str] = {}
for _level in BLOOM_LEVELS:
    for _verb in VERB_BANK[_level]:
        for _form in _participles(_verb):
            _NATURAL.setdefault(_form, _level)


def natural_level(text: str) -> str:
    """The Bloom level a performance criterion asks for, or '' if it is silent.

    A criterion reading "Tools and equipment are identified according to
    workplace procedures" is a KNOWLEDGE criterion. It cannot be assessed at
    CREATING however much a target distribution would like it to be, and a
    paper that tries produces a question nobody can answer honestly.

    The first bank verb found wins - a criterion states its demand once, at
    the front. A criterion using a verb from no bank ("Access controls are
    CONFIGURED", "Work area is PREPARED") returns '', and the caller is free
    to place it wherever the paper needs it, because nothing in its wording
    says otherwise.
    """
    for word in re.findall(r"[a-z]+", (text or "").lower()):
        level = _NATURAL.get(word)
        if level:
            return level
    return ""


# --------------------------------------------------------------------------- #
# The verbs a SECTOR actually uses
# --------------------------------------------------------------------------- #
# Read off published CDACC papers, sector by sector. They are the openings a
# real setter in that trade writes, and they are not the same set as the Bloom
# bank: an Electrical paper says "Calculate the current", a Mechanical one says
# "Inscribe a circle", a Health one says "What do you understand by", and none
# of those is in any generic taxonomy list. A paper that cannot use its own
# trade's language reads as though it came from another sector, which is
# exactly what it did.
#
# THESE WIDEN WHAT A STEM MAY OPEN WITH. THEY DO NOT FEED `natural_level`.
# That distinction is the whole reason they live in a second table. A
# criterion is read for its own demand - "Faults are diagnosed" is ANALYSING -
# and the words below appear constantly in criteria in senses that have
# nothing to do with cognitive level: "Advice is GIVEN to the client",
# "Records are WRITTEN up", "Materials are FOUND on site". Feeding them into
# the criterion reader would re-level half the criteria in the library
# overnight. So `_NATURAL` is built from VERB_BANK alone, above, and stays
# that way.
HOUSE_VERBS: Dict[str, List[str]] = {
    KNOWLEDGE: ["give", "mention", "highlight", "choose"],
    UNDERSTANDING: ["discuss", "explain with examples", "explain using "
                    "sketches", "describe with the aid of a sketch",
                    "what do you understand by"],
    APPLYING: ["calculate", "determine", "convert", "change", "factorize",
               "factorise", "simplify", "find", "show", "perform", "write",
               "draw", "sketch", "copy", "inscribe", "print",
               "illustrate and label"],
    EVALUATING: ["advise"],
    CREATING: [],
    ANALYSING: [],
}


@dataclass(frozen=True)
class Sector:
    """One TVET sector, its programmes, and the verbs its papers really use."""
    name: str
    programmes: Tuple[str, ...]
    verbs: Tuple[str, ...]


# Ordered as given. `sector_for` matches on the programme name, and on the unit
# title for the basic units, which are taught in every programme and keep their
# own house style wherever they appear.
SECTORS: Tuple[Sector, ...] = (
    Sector("Agriculture & Aquaculture",
           ("agricultural extension", "agripreneurship", "horticulture",
            "dairy farm", "dairy plant", "aquaculture", "animal production",
            "agricultural engineering"),
           ("state", "identify", "name", "list", "outline", "explain",
            "discuss")),
    # No open written paper was found for this sector, so there is nothing to
    # copy and the generic bank is used. An empty tuple says "not known",
    # which is different from "no verbs", and `allowed_verbs` treats it so.
    Sector("Applied Sciences",
           ("analytical chemistry", "industrial chemistry", "applied biology",
            "applied statistics", "science laboratory", "biomedical",
            "environmental technician", "library", "information science"),
           ()),
    Sector("Building & Civil Engineering",
           ("building technology", "civil engineering", "architecture",
            "quantity surveying", "land survey", "plumbing",
            "carpentry", "joinery", "masonry", "interior design"),
           ("list", "name", "define", "state", "convert", "outline",
            "describe", "explain", "discuss",
            "describe with the aid of a sketch")),
    Sector("Business Studies",
           ("accountancy", "banking", "finance", "business management",
            "human resource", "supply chain", "procurement",
            "project management", "office administration", "marketing",
            "cooperative management"),
           ("highlight", "define", "mention", "list", "state", "write",
            "explain")),
    Sector("Computing & Informatics",
           ("ict technician", "computer science", "software development",
            "network system", "cyber security", "graphic design",
            "information communication technology"),
           ("name", "list", "state", "identify", "define", "outline",
            "explain", "describe", "differentiate", "distinguish",
            "calculate", "determine", "perform", "write", "draw",
            "simplify")),
    Sector("Electrical & Electronic Engineering",
           ("electrical installation", "electrical engineering", "electronics",
            "telecommunication", "instrumentation", "control",
            "solar pv", "solar photovoltaic"),
           ("change", "solve", "calculate", "determine", "factorize",
            "factorise")),
    Sector("Fashion Design & Cosmetology",
           ("fashion design", "beauty therapy", "hairdressing", "cosmetology",
            "leather technology", "footwear", "textile"),
           ("define", "name", "list", "state", "give", "identify", "mention",
            "highlight", "outline", "explain", "differentiate", "discuss")),
    Sector("Health Sciences",
           ("community health", "health records", "medical laboratory",
            "nutrition", "dietetics", "perioperative", "theatre technology"),
           ("what do you understand by", "name", "state", "list", "identify",
            "outline", "describe", "explain", "distinguish",
            "illustrate and label")),
    Sector("Hospitality",
           ("food production", "culinary", "food and beverage",
            "food & beverage", "baking", "housekeeping", "tour and travel",
            "tour & travel", "tour guiding", "food technology"),
           ("define", "list", "name", "state", "identify", "describe",
            "explain with examples", "evaluate")),
    Sector("Mechanical & Automotive Engineering",
           ("automotive", "autobody", "mechatronics", "mechanical production",
            "welding", "fabrication", "refrigeration", "air conditioning",
            "plant technology"),
           ("copy", "construct", "inscribe", "draw", "sketch",
            "explain using sketches", "print", "find", "determine", "show",
            "solve")),
    Sector("Social Work",
           ("social work", "community development", "counselling psychology",
            "counseling psychology", "child protection"),
           ("name", "list", "state", "identify", "explain", "justify")),
)

# Taught in every programme, and they keep this style wherever they appear -
# so the UNIT decides here, not the programme.
BASIC_UNITS: Tuple[str, ...] = (
    "communication", "digital literacy", "employability", "entrepreneur",
    "environmental literacy", "numeracy", "occupational safety",
    "safety and health", "osh",
)
BASIC_SECTOR = Sector(
    "Basic & common units",
    BASIC_UNITS,
    ("select", "choose", "list", "state", "identify", "highlight", "outline",
     "describe", "explain", "discuss", "calculate", "advise"))


def sector_for(programme: str = "", unit_title: str = "") -> Optional[Sector]:
    """The sector whose house style this paper should be written in.

    The unit is asked first. Communication and Digital Literacy are taught
    inside an engineering programme and inside a hospitality one, and they are
    set the same way in both - a basic unit carries its own style, not its
    host programme's.

    Returns None when nothing matches, and the caller then uses the generic
    bank. A wrong sector is worse than none: it would put "Inscribe" in front
    of a catering trainee.
    """
    unit = _norm_text(unit_title)
    for name in BASIC_UNITS:
        if name in unit:
            return BASIC_SECTOR
    haystack = _norm_text(f"{programme} {unit_title}")
    best: Optional[Sector] = None
    best_len = 0
    for sector in SECTORS:
        for keyword in sector.programmes:
            # Longest keyword wins: "electrical engineering" beats
            # "engineering" appearing inside "agricultural engineering".
            if keyword in haystack and len(keyword) > best_len:
                best, best_len = sector, len(keyword)
    return best


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9& ]+", " ", (text or "").lower())


def allowed_verbs(bloom: str, sector: Optional[Sector] = None) -> List[str]:
    """What an item at `bloom` may open with, in this sector.

    The generic bank always applies. On top of it come only the house verbs
    this sector actually uses AND that sit at this level - so an Electrical
    paper gains "calculate" and "determine" at APPLYING and gains nothing at
    KNOWLEDGE, because its published papers open no recall question that way.

    A sector with no verbs recorded (see Applied Sciences) gets the bank
    alone, which is what the generator did before any of this existed.
    """
    bank = list(VERB_BANK.get(bloom, []))
    if sector is None or not sector.verbs:
        return bank
    wanted = set(sector.verbs)
    return bank + [v for v in HOUSE_VERBS.get(bloom, [])
                   if v in wanted and v not in bank]


_ALL_VERBS: Dict[str, str] = {}
for _level in BLOOM_LEVELS:
    for _verb in list(VERB_BANK[_level]) + list(HOUSE_VERBS.get(_level, [])):
        _ALL_VERBS.setdefault(_verb, _level)

# The longest opening phrase in either table, so a reader knows how many words
# to try: "what do you understand by" is five.
MAX_VERB_WORDS = max(len(v.split()) for v in _ALL_VERBS)


def level_of_verb(verb: str) -> str:
    """The level a lead verb belongs to, or '' when it is in no bank.

    Both tables, because this answers "where does this verb sit", and the
    answer does not change with the sector - only whether the sector uses it
    does, which is `allowed_verbs`.
    """
    return _ALL_VERBS.get((verb or "").strip().lower(), "")


# --------------------------------------------------------------------------- #
# What one response is worth
# --------------------------------------------------------------------------- #
# Published CDACC papers are consistent about this, and it is the difference
# between a question that can be answered and one that cannot:
#
#     "State FOUR methods of identifying communication needs."      4 marks
#     "Outline FOUR steps to be followed."                          4 marks
#     "Explain FOUR relevant sources you would harness."            8 marks
#     "Discuss FIVE factors that support implementation."          10 marks
#     "Describe three recognized stages of fire."                   6 marks
#
# A recall verb buys one mark a point - the candidate names a thing. A verb
# asking for a developed answer buys two, because each point needs a sentence
# of substance behind it. Keyed by Bloom level rather than by verb because
# every verb in a level's bank makes the same demand.
#
# It is also the floor on an item: an item must afford at least one response,
# so "Explain ... (1 mark)" is not a hard question, it is an unanswerable one.
MARKS_PER_RESPONSE: Dict[str, int] = {
    KNOWLEDGE: 1,
    UNDERSTANDING: 2,
    APPLYING: 2,
    ANALYSING: 2,
    EVALUATING: 2,
    CREATING: 2,
}


def marks_per_response(level: str) -> int:
    """What one point is worth at `level`, and so the smallest item it allows."""
    return MARKS_PER_RESPONSE.get((level or "").strip().lower(), 1)


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
# A PC allocated more than this may be split into two items at different
# levels. Used only for criteria whose own wording names no level - the rest
# are split for size alone, by MAX_RESPONSES_PER_ITEM.
SPLIT_ITEM_THRESHOLD = 5

# The most responses one question may ask for. Published CDACC papers sit at
# three to five - "State FOUR methods", "Discuss FIVE factors" - and a
# question asking for twelve is a list-writing exercise, not an assessment
# item. A criterion carrying more marks than this allows is split into as many
# questions as it needs, all at its own level.
MAX_RESPONSES_PER_ITEM = 5

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

# The two response types an item may carry, and the other words a model uses
# for them. A repair pass came back saying "constructed_response", which is the
# right KIND of item named in the wrong words - the format check then rejected
# a perfectly good item, and two repair passes could not converge because
# nothing was actually wrong with it.
#
# Only synonyms are mapped. "multiple_choice" is NOT a wrong word for the same
# thing, it is a different thing, so it is left exactly as returned for the
# format check to catch. A model that wrote a selected-response item must be
# reported, never quietly relabelled.
SHORT_RESPONSE = "short_response"
EXTENDED_RESPONSE = "extended_response"

_FORMAT_SYNONYMS: Dict[str, str] = {
    "constructed_response": SHORT_RESPONSE,
    "constructed response": SHORT_RESPONSE,
    "short": SHORT_RESPONSE,
    "short_answer": SHORT_RESPONSE,
    "short answer": SHORT_RESPONSE,
    "structured_response": SHORT_RESPONSE,
    "structured": SHORT_RESPONSE,
    "extended": EXTENDED_RESPONSE,
    "extended_answer": EXTENDED_RESPONSE,
    "essay": EXTENDED_RESPONSE,
    "long_answer": EXTENDED_RESPONSE,
    "long response": EXTENDED_RESPONSE,
}


def response_type(value: str) -> str:
    """A response type as this module names it, where the model meant one."""
    text = (value or "").strip().lower().replace("-", "_")
    if text in (SHORT_RESPONSE, EXTENDED_RESPONSE):
        return text
    return _FORMAT_SYNONYMS.get(text, (value or "").strip())

# How alike an element's title and a learning outcome's title must be before
# the two are taken to be the same thing. The two documents are written by
# different committees and the titles are usually identical, so a match this
# far below 1.0 is generous on purpose - it absorbs a dropped plural or a
# reordered phrase without letting two genuinely different outcomes pair up.
OUTCOME_MATCH_SIMILARITY = 0.62

# Writing is a generative job and a cold model does it badly: at 0.3 every
# paper on a unit came back with the same scenario, the same worked example
# and the same four marking points, because the highest-probability wording is
# the same wording every time. The constraints that matter - the marks, the
# Bloom levels, which PC each item sits on - are not entrusted to sampling at
# all; they are computed before the call and checked after it. So the
# temperature buys variety and depth in the prose and risks nothing that the
# validators are not already watching.
TEMPERATURE = 0.85

# A repair is the opposite job: one item, one named fault, everything else
# frozen. Here the likeliest correction IS the wanted one, and warmth only
# invites the model to rewrite more than it was asked to.
REPAIR_TEMPERATURE = 0.35

MAX_REPAIR_PASSES = 2
