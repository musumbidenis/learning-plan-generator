"""Module B - DETERMINISTIC session planner (NO AI).

Builds the session skeleton straight from the curriculum sub-topics (one session
per 'x.y' content sub-topic), maps each session to its OS performance criteria,
distributes sessions across the term, and stamps the deterministic schedule
fields (week / session_no / is_cat). The AI never controls the schedule.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from models import CurriculumUnit, PlanInputs, Session, Unit


def _pc_lookup(unit: Optional[Unit]) -> Dict[str, str]:
    """Map PC number -> 'number text' (e.g. '1.1' -> '1.1 Programming language ...')."""
    out: Dict[str, str] = {}
    if not unit:
        return out
    for pc in unit.all_pcs:
        out[pc.number] = f"{pc.number} {pc.text}"
    return out


def _element_pcs(unit: Optional[Unit], major: str) -> List[str]:
    if not unit:
        return []
    for el in unit.elements:
        if el.number == major:
            return [f"{pc.number} {pc.text}" for pc in el.performance_criteria]
    return []


def build_content_sessions(curriculum_unit: CurriculumUnit,
                           os_unit: Optional[Unit]) -> List[Session]:
    """One Session per curriculum sub-topic, mapped to its matching OS PC(s)."""
    pc_map = _pc_lookup(os_unit)
    sessions: List[Session] = []
    for lo in curriculum_unit.learning_outcomes:
        for st in lo.sub_topics:
            # curriculum sub-topic number aligns 1:1 with the OS PC number
            if st.number in pc_map:
                pcs = [pc_map[st.number]]
            else:
                pcs = _element_pcs(os_unit, st.number.split(".")[0])
            sessions.append(Session(
                is_cat=False,
                session_title=st.title,
                pcs=pcs,
                key_points=list(st.key_points),
            ))
    return sessions


def _make_cat(label_no: int) -> Session:
    return Session(
        is_cat=True,
        session_title=f"CAT {label_no} (Continuous Assessment Test)",
        pcs=[],
        key_points=[],
    )


def plan_sessions(curriculum_unit: CurriculumUnit, os_unit: Optional[Unit],
                  inputs: PlanInputs) -> List[Session]:
    """Lay content sessions + CAT rows onto a week grid.

    Rules:
      * `sessions_per_week` slots per week, weeks 1..term_weeks.
      * A CAT consumes one slot in each configured CAT week (the first slot,
        except in the final term week where it becomes the last slot, so the
        term ends on the final assessment - matching the RVNP gold).
      * Content sessions fill the remaining slots in curriculum order; any
        overflow is appended round-robin so nothing is ever dropped.
    """
    content = build_content_sessions(curriculum_unit, os_unit)
    spw = max(1, inputs.sessions_per_week)
    term_weeks = max(1, inputs.term_weeks)

    cat_weeks = sorted({w for w in inputs.cat_weeks if 1 <= w <= term_weeks})
    cat_label = {w: i + 1 for i, w in enumerate(cat_weeks)}

    weeks: List[List[Session]] = [[] for _ in range(term_weeks)]

    # 1) place CATs (front of week, deferred-to-back handled at stamping time)
    for w in cat_weeks:
        weeks[w - 1].append(_make_cat(cat_label[w]))

    # 2) fill content up to spw per week
    ptr = 0
    for wi in range(term_weeks):
        while len(weeks[wi]) < spw and ptr < len(content):
            weeks[wi].append(content[ptr])
            ptr += 1

    # 3) overflow: distribute any remaining content round-robin
    wi = 0
    while ptr < len(content):
        weeks[wi % term_weeks].append(content[ptr])
        ptr += 1
        wi += 1

    # 4) flatten + stamp deterministic fields
    planned: List[Session] = []
    session_no = "1&2" if spw == 2 else None
    for wi, week_sessions in enumerate(weeks, start=1):
        is_final_week = (wi == term_weeks)
        # In the final week, push any CAT to the end of the week.
        if is_final_week:
            cats = [s for s in week_sessions if s.is_cat]
            non = [s for s in week_sessions if not s.is_cat]
            week_sessions = non + cats
        for slot, s in enumerate(week_sessions, start=1):
            s.week = wi
            if session_no:
                s.session_no = session_no
            else:
                # e.g. spw=1 -> "1"; spw=3 -> "1&2&3" style not needed, label slot
                s.session_no = str(slot)
            planned.append(s)
    return planned
