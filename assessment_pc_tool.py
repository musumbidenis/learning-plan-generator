"""Module B0 - a unit's PC weighting tool, BUILT from its OS (NO AI).

The alternative to pasting CDACC's published table (`assessment_weighting`):
where CDACC has not published a weighting for the programme, or the trainer
needs one for formative CATs, it is built here by CDACC's own method - the
Secretariat's *Performance Criteria Weighting* (May 2022), as the
*Understanding CBET* series works it through:

    1  map the evidence guide's CRITICAL ASPECTS onto the PCs they cover
    2  rank the elements by critical aspects, ties broken - in this order -
       by product/process, creativity, number of PCs, knowledge/performance;
       elements still tied share the average of their ranks ("ideal rank")
    3  weight % = rank / sum of ranks x 100, rounded to total exactly 100
    4  multiply by a FACTOR only when the lowest weight cannot be shared out
       across its PCs; the factor is that element's number of PCs
    5  split each element theory : practical by the unit's ratio
    6  share each side out among the element's PCs - knowledge-only PCs take
       no practical, performance-only PCs no theory, and a PC takes more the
       more critical aspects it covers

Every judgement the method leaves to the developer - which PCs an aspect
covers, whether a PC is knowledge or performance, product or process, creative
or to an SOP - is made here from the OS's own wording, and every one of them
is returned editable, with its reason, so the trainer can correct it. The
arithmetic is not editable: it follows from the judgements.

The result converts to the same `UnitWeighting` a pasted table parses to, so
nothing downstream knows which way the weighting arrived.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from assessment_models import (PRACTICAL, THEORY, Problem, UnitWeighting,
                               WeightedElement, WeightedPC)
from models import Unit

KNOWLEDGE_PC, PERFORMANCE_PC, BOTH_PC = "K", "P", "K+P"
NATURES = (KNOWLEDGE_PC, PERFORMANCE_PC, BOTH_PC)

# CDACC's 2022 theory:practical ratios by KNQF level. The 2026 ICT weighting
# already departs from them (4:6 at Level 6, 3:7 at 5, 1:9 at 4), and the ratio
# really belongs to the unit, so this is only the starting value the trainer
# is asked to confirm.
DEFAULT_RATIOS: Dict[str, Tuple[int, int]] = {
    "6": (1, 1), "5": (2, 3), "4": (3, 7), "3": (1, 4)}
FALLBACK_RATIO = (2, 3)

MAX_FACTOR = 20

# --------------------------------------------------------------------------- #
# Reading a PC's wording
# --------------------------------------------------------------------------- #
# CDACC writes PCs in the passive ('Hardware devices are disassembled'), so the
# verbs are listed as participles. A PC using both kinds - 'Request form is
# received AND INTERPRETED' - needs knowledge and performance; one using
# neither says nothing either way and is treated the same.
_KNOWLEDGE_VERBS = set("""
identified explained described defined determined interpreted outlined listed
stated recognised recognized analysed analyzed evaluated reviewed discussed
understood compared classified distinguished observed assessed interpreted
""".split())
_PERFORMANCE_VERBS = set("""
applied performed prepared carried conducted installed configured assembled
disassembled connected tested cleaned used demonstrated maintained operated set
fixed repaired mounted produced fabricated constructed built developed designed
created implemented documented recorded measured calculated drawn replaced
handled stored deployed uploaded integrated updated coded programmed executed
adjusted calibrated inspected serviced managed organised organized modelled
formulated generated presented written drafted filled completed administered
monitored checked cut welded painted wired loaded launched packaged received
obtained collected sorted labelled labeled
""".split())
# A PC that ends in something that can be handed in: a report, a drawing, a
# built item. CDACC ranks product above process.
_PRODUCT_WORDS = set("""
report reports result results record records document documents drawing
drawings plan plans schedule schedules budget budgets program programs
programme prototype product products model models design designs website
application applications circuit circuits specification specifications
produced fabricated constructed built generated drafted written drawn
""".split())
# Innovation, 'thinking outside the box', as against following an SOP.
_CREATIVE_WORDS = set("""
designed developed created formulated innovated composed customised customized
improvised modified proposed generated devised invented modelled adapted
""".split())


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def pc_nature(text: str) -> str:
    """'K', 'P' or 'K+P', from the verbs the PC is written in."""
    words = set(_words(text))
    knowledge = bool(words & _KNOWLEDGE_VERBS)
    performance = bool(words & _PERFORMANCE_VERBS)
    if knowledge and not performance:
        return KNOWLEDGE_PC
    if performance and not knowledge:
        return PERFORMANCE_PC
    return BOTH_PC


def is_product(text: str) -> bool:
    return bool(set(_words(text)) & _PRODUCT_WORDS)


def needs_creativity(text: str) -> bool:
    return bool(set(_words(text)) & _CREATIVE_WORDS)


# --------------------------------------------------------------------------- #
# Step 1 - critical aspects onto PCs
# --------------------------------------------------------------------------- #
_SUFFIXES = ("ations", "ation", "ments", "ment", "ities", "ity", "ings", "ing",
             "ied", "ies", "ed", "es", "s")
_STOP = set("""
the and are is as per of to in for on with by be been its their a an or at
from that this these those into based according line all any other
""".split())


def _stem(word: str) -> str:
    """A crude stem: enough that 'Applied' meets 'apply', 'maintenance' meets
    'maintained'. Five letters of what is left after the commonest suffix."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[:-len(suffix)]
            break
    return word[:5]


def _stems(text: str) -> set:
    return {_stem(w) for w in _words(text) if len(w) > 2 and w not in _STOP}


@dataclass
class CriticalAspect:
    """One 'critical aspect of competency' and the PCs it was mapped onto."""
    number: str                          # its position in the evidence guide
    text: str
    pcs: List[str] = field(default_factory=list)


def map_aspects(unit: Unit,
                aspects: Optional[Sequence[str]] = None) -> List[CriticalAspect]:
    """Each critical aspect, mapped to the PC or PCs whose wording it shares.

    An aspect paraphrases its PCs in the past tense ('Performed matrix
    operations' for 'Matrix operations are performed ...'), so the words they
    share find it. Each shared word counts for less the more PCs use it, which
    is what stops 'workplace' and 'procedures' - in every PC - from deciding
    anything. The element's title is read as part of each of its PCs, because
    an aspect often paraphrases the element rather than one PC in it.

    An aspect that shares nothing distinctive with any PC is left unmapped for
    the trainer; guessing there would put marks on a PC for no reason.
    """
    texts = list(aspects if aspects is not None
                 else getattr(unit, "critical_aspects", []))
    docs: List[Tuple[str, set, set]] = []
    for el in unit.elements:
        title = _stems(el.title)
        for pc in el.performance_criteria:
            docs.append((pc.number, _stems(pc.text), title))
    frequency: Dict[str, int] = {}
    for _num, own, title in docs:
        for s in own | title:
            frequency[s] = frequency.get(s, 0) + 1

    out: List[CriticalAspect] = []
    for i, text in enumerate(texts, 1):
        want = _stems(text)
        scores = []
        for number, own, title in docs:
            score = (sum(1 / frequency[s] for s in want & own)
                     + 0.5 * sum(1 / frequency[s] for s in want & title - own))
            scores.append((score, number))
        best = max((s for s, _n in scores), default=0)
        pcs = ([n for s, n in scores if s >= best - 1e-9]
               if best >= 0.3 else [])
        out.append(CriticalAspect(number=str(i), text=text, pcs=pcs))
    return out


# --------------------------------------------------------------------------- #
# The judgements, per PC
# --------------------------------------------------------------------------- #
@dataclass
class PCFacets:
    """What the method needs to know about a PC. All of it is editable."""
    number: str
    element_number: str
    text: str
    nature: str = BOTH_PC
    product: bool = False
    creativity: bool = False


def read_facets(unit: Unit) -> List[PCFacets]:
    return [PCFacets(number=pc.number, element_number=el.number, text=pc.text,
                     nature=pc_nature(pc.text), product=is_product(pc.text),
                     creativity=needs_creativity(pc.text))
            for el in unit.elements for pc in el.performance_criteria]


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #
@dataclass
class ElementRank:
    """One row of the ranking sheet (Sheet 1 of the tool)."""
    number: str
    title: str
    n_pcs: int
    critical: int
    product: bool
    creativity: bool
    performance_only: bool
    rank: float = 0.0
    weight: int = 0                      # % of 100
    marks: int = 0                       # weight x factor
    theory: int = 0
    practical: int = 0
    reason: str = ""


@dataclass
class PCWeight:
    """One row of the PC weighting tool (Sheet 2)."""
    number: str
    element_number: str
    text: str
    nature: str
    critical: int                        # aspects this PC covers
    product: bool
    creativity: bool
    theory: int = 0
    practical: int = 0
    reason: str = ""


@dataclass
class WeightingTool:
    unit_title: str = ""
    cdacc_code: str = ""
    isced_code: str = ""
    knqf_level: str = ""
    ratio: Tuple[int, int] = FALLBACK_RATIO
    factor: int = 1
    factor_reason: str = ""
    aspects: List[CriticalAspect] = field(default_factory=list)
    elements: List[ElementRank] = field(default_factory=list)
    pcs: List[PCWeight] = field(default_factory=list)

    @property
    def rank_total(self) -> float:
        return sum(e.rank for e in self.elements)

    def pcs_of(self, element_number: str) -> List[PCWeight]:
        return [p for p in self.pcs if p.element_number == element_number]

    def to_weighting(self) -> UnitWeighting:
        """The same object a pasted CDACC table parses to."""
        weighting = UnitWeighting(unit_title=self.unit_title,
                                  cdacc_code=self.cdacc_code,
                                  isced_code=self.isced_code,
                                  knqf_level=self.knqf_level)
        for el in self.elements:
            element = WeightedElement(number=el.number, title=el.title)
            element.pcs = [WeightedPC(number=p.number,
                                      element_number=el.number, text=p.text,
                                      theory_weight=p.theory,
                                      practical_weight=p.practical)
                           for p in self.pcs_of(el.number)]
            element.stated_theory_total = element.total(THEORY)
            element.stated_practical_total = element.total(PRACTICAL)
            weighting.elements.append(element)
        theory = sum(p.theory for p in self.pcs)
        practical = sum(p.practical for p in self.pcs)
        weighting.stated_grand_theory = theory
        weighting.stated_grand_practical = practical
        divisor = math.gcd(theory, practical) or 1
        weighting.ratio = ((theory // divisor, practical // divisor)
                           if theory or practical else None)
        return weighting


def default_ratio(knqf_level: str) -> Tuple[int, int]:
    return DEFAULT_RATIOS.get(str(knqf_level or "").strip(), FALLBACK_RATIO)


def _largest_remainder(raw: Sequence[float], total: int) -> List[int]:
    """Whole numbers summing to `total`, nearest to `raw`; ties to the earlier."""
    floors = [math.floor(x + 1e-9) for x in raw]
    short = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - floors[i]), i))
    for i in order[:max(short, 0)]:
        floors[i] += 1
    return floors


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# -- step 2 ------------------------------------------------------------------ #
_TIE_BREAKS = ("critical aspects", "product over process", "creativity",
               "number of PCs", "performance over knowledge")


def _key(el: ElementRank) -> Tuple:
    return (el.critical, el.product, el.creativity, el.n_pcs,
            el.performance_only)


def _rank(elements: List[ElementRank]) -> None:
    """Ranks n (highest) down to 1; fully tied elements share the average."""
    ordered = sorted(elements, key=_key)            # lowest first
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and _key(ordered[j + 1]) == _key(ordered[i]):
            j += 1
        shared = (i + 1 + j + 1) / 2                 # mean of positions i..j
        for el in ordered[i:j + 1]:
            el.rank = shared
        i = j + 1
    for el in elements:
        el.reason = _rank_reason(el, ordered)


def _describe(el: ElementRank) -> str:
    bits = [f"{el.critical} critical aspect{'s' * (el.critical != 1)}",
            "product" if el.product else "process",
            "needs creativity" if el.creativity else "to procedure",
            f"{el.n_pcs} PC{'s' * (el.n_pcs != 1)}",
            "performance only" if el.performance_only
            else "knowledge and performance"]
    return "; ".join(bits)


def _rank_reason(el: ElementRank, ordered: List[ElementRank]) -> str:
    """Why this element sits where it does, in the method's own terms."""
    reason = _describe(el) + "."
    same = [o.number for o in ordered if o is not el and _key(o) == _key(el)]
    if same:
        return (reason + f" Tied on every factor with element "
                f"{', '.join(same)}; the tied ranks are averaged.")
    below = [o for o in ordered if o.rank < el.rank]
    if below:
        nearest = below[-1]
        for name, mine, theirs in zip(_TIE_BREAKS, _key(el), _key(nearest)):
            if mine != theirs:
                if name != _TIE_BREAKS[0]:
                    reason += (f" Level with element {nearest.number} on "
                               f"critical aspects; ranked above it on {name}.")
                break
    return reason


# -- step 4 ------------------------------------------------------------------ #
def _realistic(el: ElementRank, factor: int, ratio: Tuple[int, int]) -> bool:
    """Can this element's marks be shared out across its PCs, on each side?"""
    t, p = ratio
    marks = el.weight * factor
    return all(marks * side / (t + p) >= el.n_pcs for side in (t, p) if side)


def choose_factor(elements: List[ElementRank],
                  ratio: Tuple[int, int]) -> Tuple[int, str]:
    """The factor, and the reason to write down for it.

    None is needed if the lowest weight already shares out across its PCs.
    Otherwise it is that element's number of PCs - CDACC's 'guided by the
    number of PC' - raised only as far as it takes to make the share real.
    """
    if not elements:
        return 1, ""
    lowest = min(elements, key=lambda e: (e.weight, -e.n_pcs))
    if _realistic(lowest, 1, ratio):
        return 1, (f"No factor: the lowest weight ({lowest.weight}, element "
                   f"{lowest.number}) already shares out across its "
                   f"{lowest.n_pcs} PCs.")
    factor = max(lowest.n_pcs, 2)
    while factor < MAX_FACTOR and not _realistic(lowest, factor, ratio):
        factor += 1
    return factor, (f"Factor {factor}: element {lowest.number} has the lowest "
                    f"weight ({lowest.weight}), too few marks for its "
                    f"{lowest.n_pcs} PCs on both theory and practical; the "
                    f"factor is its number of PCs, giving "
                    f"{lowest.weight * factor} marks. Total "
                    f"{100 * factor} marks for the term.")


# -- step 6 ------------------------------------------------------------------ #
def _share(marks: int, pcs: List[PCWeight], shares: List[float],
           eligible: List[bool]) -> List[int]:
    """`marks` across `pcs`, by share, only to eligible PCs (all if none are),
    and at least 1 to each eligible PC whenever there are marks enough."""
    if marks <= 0 or not pcs:
        return [0] * len(pcs)
    if not any(eligible):
        eligible = [True] * len(pcs)
    weights = [s if ok else 0.0 for s, ok in zip(shares, eligible)]
    total = sum(weights)
    out = _largest_remainder([marks * w / total for w in weights], marks)
    idx = [i for i, ok in enumerate(eligible) if ok]
    if marks >= len(idx):
        for i in idx:
            if out[i] == 0:
                donor = max(idx, key=lambda k: (out[k], -k))
                out[donor] -= 1
                out[i] += 1
    return out


def _pc_reason(pc: PCWeight) -> str:
    bits = []
    if pc.critical:
        bits.append(f"critical - covers {pc.critical} critical aspect"
                    f"{'s' * (pc.critical != 1)}")
    else:
        bits.append("not named in the critical aspects")
    # Said against the marks actually given: an element with no performance
    # PC still has practical marks to place, and they go to every PC in it.
    if pc.nature == KNOWLEDGE_PC:
        bits.append("knowledge only, so no practical marks" if not pc.practical
                    else "knowledge only, but its element has no performance "
                         "PC to take the practical marks")
    elif pc.nature == PERFORMANCE_PC:
        bits.append("performance only, so no theory marks" if not pc.theory
                    else "performance only, but its element has no knowledge "
                         "PC to take the theory marks")
    else:
        bits.append("knowledge and performance")
    bits.append("produces a product" if pc.product else "process")
    if pc.creativity:
        bits.append("needs creativity")
    return "; ".join(bits).capitalize() + "."


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def check(unit: Unit) -> List[Problem]:
    """What in the OS as read would make the tool wrong.

    The tool is only as good as the element/PC table it is built on. A PC
    number read twice, or an element read with no PCs, means the OS did not
    parse cleanly, and weighting it would hand marks to rows that are not the
    document's. Those block; the trainer pastes CDACC's table instead.
    """
    problems: List[Problem] = []
    if not unit.elements:
        problems.append(Problem("No elements were read from this unit's OS."))
    seen: Dict[str, int] = {}
    for el in unit.elements:
        if not el.performance_criteria:
            problems.append(Problem(
                f"Element {el.number} was read with no performance criteria.",
                where=el.number))
        for pc in el.performance_criteria:
            seen[pc.number] = seen.get(pc.number, 0) + 1
    for number, count in seen.items():
        if count > 1:
            problems.append(Problem(
                f"PC {number} was read {count} times from the OS.",
                where=number))
    if not getattr(unit, "critical_aspects", []):
        problems.append(Problem(
            "No critical aspects were found in the evidence guide, so the "
            "elements can only be ranked on the tie-breakers. Add them below "
            "if the OS has them.", blocking=False))
    return problems


def build(unit: Unit, aspects: List[CriticalAspect],
          facets: List[PCFacets], ratio: Tuple[int, int],
          factor: Optional[int] = None) -> WeightingTool:
    """The whole tool, from the trainer-confirmed judgements.

    `factor=None` chooses it by the method's rule; a number overrides it.
    """
    t, p = ratio
    if t < 0 or p < 0 or t + p == 0:
        t, p = FALLBACK_RATIO
    tool = WeightingTool(unit_title=unit.unit_title, cdacc_code=unit.os_code,
                         isced_code=unit.isced_code, knqf_level=unit.level,
                         ratio=(t, p), aspects=aspects)

    covers: Dict[str, int] = {}
    for aspect in aspects:
        for number in aspect.pcs:
            covers[number] = covers.get(number, 0) + 1
    by_number = {f.number: f for f in facets}

    for el in unit.elements:
        mine = [by_number[pc.number] for pc in el.performance_criteria
                if pc.number in by_number]
        numbers = {f.number for f in mine}
        critical = sum(1 for a in aspects if numbers & set(a.pcs))
        tool.elements.append(ElementRank(
            number=el.number, title=el.title, n_pcs=len(mine),
            critical=critical,
            product=any(f.product for f in mine),
            creativity=any(f.creativity for f in mine),
            performance_only=bool(mine) and all(
                f.nature == PERFORMANCE_PC for f in mine)))
        for f in mine:
            tool.pcs.append(PCWeight(
                number=f.number, element_number=el.number, text=f.text,
                nature=f.nature, critical=covers.get(f.number, 0),
                product=f.product, creativity=f.creativity))
    if not tool.elements:
        return tool

    # Steps 2-3
    _rank(tool.elements)
    total = tool.rank_total
    for el, w in zip(tool.elements, _largest_remainder(
            [el.rank / total * 100 for el in tool.elements], 100)):
        el.weight = w

    # Step 4
    if factor is None:
        tool.factor, tool.factor_reason = choose_factor(tool.elements, (t, p))
    else:
        tool.factor = max(1, int(factor))
        tool.factor_reason = (f"Factor {tool.factor}, set by the developer. "
                              f"Total {100 * tool.factor} marks for the term.")

    # Step 5: the unit's theory total is fixed by the ratio first, then shared
    # out across elements, so rounding cannot drift the grand total.
    grand = 100 * tool.factor
    for el in tool.elements:
        el.marks = el.weight * tool.factor
    theories = _largest_remainder(
        [el.marks * t / (t + p) for el in tool.elements],
        _half_up(grand * t / (t + p)))
    for el, theory in zip(tool.elements, theories):
        el.theory = min(theory, el.marks)
        el.practical = el.marks - el.theory

    # Step 6
    for el in tool.elements:
        pcs = tool.pcs_of(el.number)
        theory = _share(el.theory, pcs,
                        [1 + pc.critical for pc in pcs],
                        [pc.nature != PERFORMANCE_PC for pc in pcs])
        practical = _share(el.practical, pcs,
                           [1 + pc.critical + pc.product + 0.5 * pc.creativity
                            for pc in pcs],
                           [pc.nature != KNOWLEDGE_PC for pc in pcs])
        for pc, th, pr in zip(pcs, theory, practical):
            pc.theory, pc.practical = th, pr
            pc.reason = _pc_reason(pc)
    return tool
