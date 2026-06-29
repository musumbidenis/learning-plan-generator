"""Shared data structures for the Learning Plan Generator.

These are plain dataclasses that flow through the pipeline:

    parsers (A)  ->  Unit + CurriculumUnit
    planner (B)  ->  list[Session]  (the deterministic skeleton)
    ai_client (C) ->  the same Sessions, with generative fields filled
    doc_builder (D) -> .docx

Everything here is pure data; no parsing or I/O logic lives in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Occupational Standard (Module A1 output)
# --------------------------------------------------------------------------- #
@dataclass
class PerformanceCriterion:
    """A single PC, e.g. number='1.1', text='Programming language types are ...'."""
    number: str
    text: str


@dataclass
class Element:
    """An element (key workplace outcome) with its performance criteria."""
    number: str                       # e.g. '1'
    title: str                        # e.g. 'Apply computer programming skills'
    performance_criteria: List[PerformanceCriterion] = field(default_factory=list)


@dataclass
class Unit:
    """A parsed Occupational-Standard unit of competency."""
    unit_title: str = ""
    os_code: str = ""                 # TVET CDACC CODE (IT/OS/...)
    isced_code: str = ""              # ISCED UNIT CODE (join key across docs)
    level: str = ""
    description: str = ""
    skill_task: str = ""              # description sentence from first action verb
    elements: List[Element] = field(default_factory=list)
    # Evidence-Guide "Methods of Assessment" list (feeds Knowledge/Skills/Attitudes)
    assessment_methods: List[str] = field(default_factory=list)
    # "Required Knowledge" bullets - underpinning theory; used to enrich the plan
    # (esp. the OS-only fallback when no curriculum content exists for the unit).
    required_knowledge: List[str] = field(default_factory=list)

    @property
    def all_pcs(self) -> List[PerformanceCriterion]:
        out: List[PerformanceCriterion] = []
        for el in self.elements:
            out.extend(el.performance_criteria)
        return out


# --------------------------------------------------------------------------- #
# Curriculum (Module A2 output)
# --------------------------------------------------------------------------- #
@dataclass
class SubTopic:
    """A curriculum 'x.y' content sub-topic -> becomes one teaching session.

    `key_points` are the 'x.y.z' child lines (the authoritative syllabus content
    that becomes the Learning Key Points column).
    """
    number: str                       # e.g. '1.1'
    title: str                        # e.g. 'Identification of Programming Languages'
    key_points: List[str] = field(default_factory=list)


@dataclass
class LearningOutcome:
    number: str                       # e.g. '1'
    title: str                        # e.g. 'Apply computer programming skills'
    sub_topics: List[SubTopic] = field(default_factory=list)
    suggested_methods: List[str] = field(default_factory=list)


@dataclass
class UnitRef:
    """A lightweight pointer to a unit produced by the cheap 'index' pass.

    Holds just the identity (title + codes) and the page span, so the UI can list
    units for selection BEFORE any heavy deterministic extraction runs.
    """
    title: str = ""
    isced_code: str = ""          # join key across OS and Curriculum
    code: str = ""                # os_code (IT/OS/...) or curriculum_code (IT/CU/...)
    source: str = ""              # 'OS' or 'CU'
    start_page: int = 0
    end_page: int = 0


@dataclass
class CurriculumUnit:
    unit_title: str = ""
    curriculum_code: str = ""         # TVET CDACC UNIT CODE (IT/CU/...)
    isced_code: str = ""              # join key
    description: str = ""
    learning_outcomes: List[LearningOutcome] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Planner / AI (Modules B & C)
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    """One row of the 9-column session table.

    Fields above the divider are DETERMINISTIC (set by the planner and re-stamped
    after the AI call). Fields below are GENERATIVE (filled by the single AI call,
    except key_points which come from the curriculum and the AI only formats).
    """
    # --- deterministic skeleton -------------------------------------------- #
    week: int = 0
    session_no: str = "1"
    is_cat: bool = False
    session_title: str = ""
    pcs: List[str] = field(default_factory=list)          # full PC strings (x.y text)

    # --- content / generative ---------------------------------------------- #
    key_points: List[str] = field(default_factory=list)   # curriculum content (authoritative)
    learning_outcomes: List[str] = field(default_factory=list)   # a./b./c. trainee-centred
    trainee_activities: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    assessments: List[str] = field(default_factory=list)

    def to_skeleton_dict(self) -> dict:
        """The minimal grounded payload handed to the AI for this session."""
        return {
            "week": self.week,
            "session_no": self.session_no,
            "is_cat": self.is_cat,
            "session_title": self.session_title,
            "pcs": self.pcs,
            "key_points": self.key_points,
        }


@dataclass
class PlanInputs:
    """UI-supplied metadata for the header table and scheduling."""
    trainer_name: str = ""
    institution: str = ""
    level: str = ""
    num_trainees: str = ""
    class_code: str = ""
    date_of_preparation: str = ""
    term_weeks: int = 12
    cat_weeks: List[int] = field(default_factory=lambda: [4, 8, 12])
    sessions_per_week: int = 2
