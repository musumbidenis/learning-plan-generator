"""Data the Assessment Tool module passes between its parts.

The module turns a unit's performance criteria into a CAT: a written paper or
a practical task, with its marking scheme, table of specifications and PC
distribution table. It is a sibling of the Learning Plan and Session Plan
generators and shares their conventions - dataclasses here, extraction in
plain code, the model used only for prose.

THE DIVISION OF LABOUR, which every part of this module depends on:

    assessment_weighting   the pasted PC weighting table -> UnitWeighting
    assessment_allocation  UnitWeighting + a CatDefinition -> [Allocation]
    assessment_ai          [Allocation] -> the written items, as prose
    assessment_docs        all of the above -> four .docx documents

Every mark value is computed in `assessment_allocation`, in plain arithmetic,
before the model is called at all. The model receives fixed numbers and writes
words around them. It never computes, totals or adjusts a mark - the same rule
that keeps the Learning Plan's assessment numbering deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
THEORY = "theory"
PRACTICAL = "practical"
ASSESSMENT_TYPES = (THEORY, PRACTICAL)

# The fixed set. A unit has four assessments, each with a theory and a
# practical record, so a unit can carry up to eight generated tools.
CAT_1, CAT_2, CAT_3, FINAL_CAT = "CAT_1", "CAT_2", "CAT_3", "FINAL_CAT"

CAT_LABELS = {CAT_1: "CAT 1", CAT_2: "CAT 2", CAT_3: "CAT 3",
              FINAL_CAT: "Final CAT"}
CAT_SEQUENCE = {CAT_1: 1, CAT_2: 2, CAT_3: 3, FINAL_CAT: 4}

KNOWLEDGE = "knowledge"
UNDERSTANDING = "understanding"
APPLYING = "applying"
ANALYSING = "analysing"
EVALUATING = "evaluating"
CREATING = "creating"

# In order. The table of specifications lays its columns out this way, and a
# paper has to reach all six.
BLOOM_LEVELS = (KNOWLEDGE, UNDERSTANDING, APPLYING,
                ANALYSING, EVALUATING, CREATING)


# --------------------------------------------------------------------------- #
# The weighting table, as pasted and parsed
# --------------------------------------------------------------------------- #
@dataclass
class WeightedPC:
    """One performance criterion with the marks the unit weights it at.

    `number` matches `models.PerformanceCriterion.number` ('1.1'), which is
    what ties this table back to the Occupational Standard the app already
    read. The weights are the unit's own, not this CAT's - a CAT's marks are
    computed from them by `assessment_allocation`.
    """
    number: str                        # '1.1'
    element_number: str                # '1'
    text: str = ""
    theory_weight: int = 0
    practical_weight: int = 0

    def weight_for(self, assessment_type: str) -> int:
        return (self.theory_weight if assessment_type == THEORY
                else self.practical_weight)


@dataclass
class WeightedElement:
    """An element and its weighted PCs, with the sub-total the table states.

    The stated sub-total is kept rather than recomputed: it is a checksum the
    parser verifies against the PC rows, and a mismatch means the paste is
    wrong and has to be corrected before anything is computed from it.
    """
    number: str                        # '1'
    title: str = ""
    pcs: List[WeightedPC] = field(default_factory=list)
    stated_theory_total: Optional[int] = None
    stated_practical_total: Optional[int] = None

    def total(self, assessment_type: str) -> int:
        return sum(pc.weight_for(assessment_type) for pc in self.pcs)


@dataclass
class UnitWeighting:
    """A whole unit's weighting table, ready to allocate from."""
    unit_title: str = ""
    cdacc_code: str = ""
    isced_code: str = ""
    knqf_level: str = ""               # '3'..'6', as models.Unit.level stores it
    elements: List[WeightedElement] = field(default_factory=list)
    stated_grand_theory: Optional[int] = None
    stated_grand_practical: Optional[int] = None
    # From the grand totals, reduced - (2, 3) for 40:60, (1, 9) for 10:90.
    ratio: Optional[Tuple[int, int]] = None

    @property
    def pcs(self) -> List[WeightedPC]:
        return [pc for el in self.elements for pc in el.pcs]

    def pc(self, number: str) -> Optional[WeightedPC]:
        return next((p for p in self.pcs if p.number == number), None)

    def element_of(self, pc_number: str) -> Optional[WeightedElement]:
        return next((el for el in self.elements
                     if any(p.number == pc_number for p in el.pcs)), None)

    def grand_total(self, assessment_type: str) -> int:
        return sum(el.total(assessment_type) for el in self.elements)


@dataclass
class Problem:
    """Something wrong with a pasted table, or with what the user asked for.

    `blocking` decides whether the user may go on. A sub-total that does not
    add up is blocking because everything downstream is computed from it; a PC
    weighted zero is a warning, because the document may really say that.
    """
    message: str
    blocking: bool = True
    where: str = ""                    # a PC or element number, when it has one


# --------------------------------------------------------------------------- #
# What the user asked for
# --------------------------------------------------------------------------- #
@dataclass
class CatDefinition:
    """One assessment: which CAT, written or practical, out of how many marks.

    `total_marks` is entered by the user and is NOT derived from the weighting
    table: institutions mark the same unit out of different totals. The Final
    CAT is comprehensive - it re-enables every PC in the unit - which is why it
    is marked out of the full unit rather than a subset.
    """
    cat_id: str = CAT_1
    assessment_type: str = THEORY
    total_marks: int = 0
    selected_pcs: List[str] = field(default_factory=list)   # PC numbers
    duration_minutes: int = 90

    @property
    def label(self) -> str:
        return CAT_LABELS.get(self.cat_id, self.cat_id)

    @property
    def sequence(self) -> int:
        return CAT_SEQUENCE.get(self.cat_id, 0)

    @property
    def comprehensive(self) -> bool:
        return self.cat_id == FINAL_CAT


@dataclass
class Allocation:
    """One PC's share of this CAT, in marks that are final by the time the
    model sees them.

    A PC allocated enough marks may be split into two entries with different
    Bloom levels, which is how a paper reaches all six levels without
    disturbing any PC's total. Two entries therefore can carry the same
    `pc_number`.
    """
    element_number: str
    element_title: str
    pc_number: str
    pc_text: str
    weight: int
    marks: int
    bloom: str = ""                    # one level; the ToS cell this fills


@dataclass
class Ledger:
    """Which PCs each CAT has already consumed, per assessment type.

    Kept per unit so CAT 2 can default to what CAT 1 left, and so the coverage
    view can say what the unit has never assessed. The Final CAT ignores it.
    """
    unit_code: str = ""
    consumed: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def record(self, cat_id: str, assessment_type: str,
               pc_numbers: List[str]) -> None:
        self.consumed.setdefault(assessment_type, {})[cat_id] = list(pc_numbers)

    def already_assessed(self, assessment_type: str,
                         before: int = 99) -> List[str]:
        """Every PC consumed by an earlier CAT of this type, in order."""
        seen: List[str] = []
        by_cat = self.consumed.get(assessment_type, {})
        for cat_id in sorted(by_cat, key=lambda c: CAT_SEQUENCE.get(c, 0)):
            if CAT_SEQUENCE.get(cat_id, 0) < before:
                seen.extend(n for n in by_cat[cat_id] if n not in seen)
        return seen


# --------------------------------------------------------------------------- #
# What the trainees were actually taught
# --------------------------------------------------------------------------- #
@dataclass
class ContentTopic:
    """One curriculum sub-topic and the key points taught under it."""
    number: str                                  # '1.2'
    title: str
    key_points: List[str] = field(default_factory=list)


@dataclass
class ElementContent:
    """The curriculum content behind one element of the occupational standard.

    The performance criteria say what competence looks like; they are one line
    each and deliberately general. The curriculum says what was taught - the
    sub-topics and the key points under them - and that is the only honest map
    of what a paper may ask about. Without it the model assesses a PC from its
    own knowledge of the trade and invents equipment, standards and
    terminology the trainees never met.

    Built in `assessment_content` by matching each element to its learning
    outcome; carried on the tool so the prompt, and only the prompt, uses it.
    """
    element_number: str
    element_title: str = ""
    outcome_number: str = ""
    outcome_title: str = ""
    duration_hours: int = 0
    topics: List[ContentTopic] = field(default_factory=list)
    suggested_methods: List[str] = field(default_factory=list)

    @property
    def key_point_count(self) -> int:
        return sum(len(t.key_points) for t in self.topics)


@dataclass
class KnowledgeNote:
    """What a taught key point actually contains, read from a reference source.

    DEPTH on a topic the curriculum already lists, never a new topic. The
    curriculum writes "Vulnerability scanning tools" and stops; the note
    carries what a scanner is, the dimensions the subject is normally broken
    down along, and the names a practitioner uses - which is the difference
    between asking a trainee to repeat a heading and asking them to use what
    they were taught. Gathered in `assessment_knowledge`.

    It stays attached to its key point so it can never be read as permission
    to assess something outside the taught scope.
    """
    key_point: str                   # the curriculum line this backs
    element_number: str = ""
    topic_number: str = ""           # the sub-topic it sits under, '1.2'
    summary: str = ""                # a few sentences of real substance
    covers: List[str] = field(default_factory=list)   # the topic's dimensions
    named: List[str] = field(default_factory=list)    # tools, standards, cases
    source_title: str = ""
    source_url: str = ""

    @property
    def label(self) -> str:
        return self.source_title or ""


@dataclass
class Exemplar:
    """One real question from a real past paper, kept with its source.

    A STYLE reference and nothing else. It shows how a TVET CDACC question is
    phrased - how much situation goes in front of the verb, what a
    "differentiate" question is worth - which is not written down in any
    curriculum. It never says what a paper is about; the CONTENT TAUGHT does
    that. Gathered in `assessment_research`.
    """
    text: str
    marks: int = 0
    source: str = ""                 # the paper it came from
    repository: str = ""

    @property
    def label(self) -> str:
        return f"{self.source} ({self.repository})" if self.source else ""


# --------------------------------------------------------------------------- #
# What the model returns, once validated
# --------------------------------------------------------------------------- #
@dataclass
class MarkingPoint:
    text: str
    marks: int = 1


@dataclass
class Item:
    """One question on a written paper."""
    number: int
    element_number: str
    pc_number: str
    bloom: str
    stem: str
    marks: int
    marking_scheme: List[MarkingPoint] = field(default_factory=list)
    item_format: str = "short_response"        # or 'extended_response'
    scenario_id: str = ""


@dataclass
class Scenario:
    id: str
    title: str
    text: str


@dataclass
class ChecklistItem:
    """One item of evaluation on a practical checklist.

    NOT a performance criterion restated: a PC is general and comes from the
    occupational standard, an item of evaluation is specific to the task set
    and says what the assessor will actually watch or inspect.
    """
    number: int
    text: str
    pc_numbers: List[str] = field(default_factory=list)
    marks: int = 0
    sub_parts: List[str] = field(default_factory=list)


@dataclass
class TaskBrief:
    task: str = ""
    conditions: str = ""
    tools_equipment_materials: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    time_allowed: str = ""


@dataclass
class OralQuestion:
    number: int
    pc_numbers: List[str] = field(default_factory=list)
    question: str = ""
    response_indicators: List[str] = field(default_factory=list)


@dataclass
class AssessmentTool:
    """Everything needed to build the four documents for one CAT."""
    unit_title: str = ""
    cdacc_code: str = ""
    isced_code: str = ""
    knqf_level: str = ""
    programme: str = ""
    cat: CatDefinition = field(default_factory=CatDefinition)
    allocations: List[Allocation] = field(default_factory=list)
    # The curriculum content behind the elements being assessed. Empty when no
    # curriculum was read: the paper is then written from the PCs alone, which
    # is what it used to be and is shallower for it.
    content: List[ElementContent] = field(default_factory=list)
    # Real questions from real papers, shown to the model as a pattern to
    # follow. Empty when nothing was found or the repositories were down,
    # which costs the paper its exemplars and not its generation.
    exemplars: List[Exemplar] = field(default_factory=list)
    # Reference notes on the taught key points, giving the model something to
    # be deep ABOUT. Empty when nothing was reachable, which costs the paper
    # its depth and not its generation.
    knowledge: List[KnowledgeNote] = field(default_factory=list)
    # written
    scenarios: List[Scenario] = field(default_factory=list)
    items: List[Item] = field(default_factory=list)
    # practical
    task_brief: Optional[TaskBrief] = None
    observation_checklist: List[ChecklistItem] = field(default_factory=list)
    product_checklist: List[ChecklistItem] = field(default_factory=list)
    oral_questions: List[OralQuestion] = field(default_factory=list)

    @property
    def is_practical(self) -> bool:
        return self.cat.assessment_type == PRACTICAL

    def marks_by_pc(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.allocations:
            out[a.pc_number] = out.get(a.pc_number, 0) + a.marks
        return out
