"""The Assessment Tool path: from a unit's performance criteria to four files.

A sibling of the Learning Plan path, and it runs in the same order the work
actually happens in:

    1  pick the performance criteria this assessment covers
    2  say which CAT it is, written or practical, and out of how many marks
    3  paste the unit's PC weighting table, and correct what the paste lost
    4  read the computed distribution - every mark is final at this point
    5  read the curriculum content the paper will be set from
    6  generate, validate, repair
    7  download

The performance criteria say what is assessed; the curriculum says what was
taught. Both go to the model, because a PC on its own is one general line and
a paper written from it alone is written from the model's own knowledge of the
trade - plausible, and about things nobody covered.

Steps 4 and 5 are kept apart on purpose. All the arithmetic happens in step 4,
in `assessment_allocation`, in plain code. By the time the model is called it
receives fixed numbers and writes prose around them, and it never computes,
totals or adjusts a mark - the same rule that keeps the Learning Plan's
assessment numbering deterministic.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

import assessment_allocation as alloc
import assessment_content as content_builder
import assessment_docs as docs
import assessment_ledger as ledger_store
import assessment_validators as validators
import assessment_weighting as weighting_parser
import runlog
from assessment_models import (CAT_1, CAT_2, CAT_3, CAT_LABELS, FINAL_CAT,
                               PRACTICAL, THEORY, Allocation, AssessmentTool,
                               CatDefinition, ElementContent, Problem,
                               UnitWeighting)
from models import Unit

ss = st.session_state

_STATE = dict(at_weighting=None, at_raw="", at_tool=None, at_problems=[])
for _k, _v in _STATE.items():
    ss.setdefault(_k, _v)

_CAT_ORDER = [CAT_1, CAT_2, CAT_3, FINAL_CAT]
_TYPE_LABEL = {THEORY: "Written (theory)", PRACTICAL: "Practical"}


def _show(problems: List[Problem]) -> bool:
    """Render problems; True when none of them blocks going on."""
    for p in problems:
        where = f"**{p.where}** - " if p.where else ""
        (st.error if p.blocking else st.warning)(f"{where}{p.message}")
    return not any(p.blocking for p in problems)


# --------------------------------------------------------------------------- #
# 3. the pasted weighting table
# --------------------------------------------------------------------------- #
def _weighting_step(os_unit: Unit) -> Optional[UnitWeighting]:
    """Paste the table, then correct it.

    Paste out of a PDF is lossy and the trainer has to be able to fix a cell,
    so what is parsed is shown as an editable grid rather than accepted
    silently. Everything downstream is computed from these numbers.
    """
    st.markdown("#### 1. The unit's PC weighting table")
    st.caption("Paste it from the Word or PDF document - tabs, pipes or "
               "spaces all read. Correct anything the paste lost in the grid "
               "below before going on.")
    raw = st.text_area("Weighting table", value=ss.at_raw, height=180,
                       key="at_paste",
                       placeholder="1\tManage tourist arrival and departures\n"
                                   "1.1\tTour transfer resources are assembled"
                                   "\t4\t4\n...")
    if raw != ss.at_raw:
        ss.at_raw, ss.at_weighting = raw, None
    if not raw.strip():
        return None

    if ss.at_weighting is None:
        parsed, problems = weighting_parser.parse(raw)
        ss.at_weighting = parsed
        ss.at_problems = problems
    weighting = ss.at_weighting

    rows = [{"Element": el.number, "PC": pc.number, "Text": pc.text,
             "Theory": pc.theory_weight, "Practical": pc.practical_weight}
            for el in weighting.elements for pc in el.pcs]
    if not rows:
        st.error("Nothing in that paste looked like a weighting table.")
        _show(ss.at_problems)
        return None

    edited = st.data_editor(rows, key="at_grid", width="stretch",
                            num_rows="fixed")
    # Edits go back onto the parsed object rather than into a second copy of
    # the table: one source of truth for every later step.
    for row in edited:
        pc = weighting.pc(str(row["PC"]))
        if pc is not None:
            pc.text = str(row["Text"])
            pc.theory_weight = int(row["Theory"] or 0)
            pc.practical_weight = int(row["Practical"] or 0)

    if not weighting.unit_title:
        weighting.unit_title = os_unit.unit_title
    if not weighting.cdacc_code:
        weighting.cdacc_code = os_unit.os_code
    if not weighting.isced_code:
        weighting.isced_code = os_unit.isced_code
    if not weighting.knqf_level:
        weighting.knqf_level = os_unit.level

    ok = _show(weighting_parser.validate(weighting))
    st.caption(f"{len(weighting.pcs)} performance criteria · "
               f"theory {weighting.grand_total(THEORY)} · "
               f"practical {weighting.grand_total(PRACTICAL)}"
               + (f" · ratio {weighting.ratio[0]}:{weighting.ratio[1]}"
                  if weighting.ratio else ""))
    return weighting if ok else None


# --------------------------------------------------------------------------- #
# 1-2. what to assess, and how much it is worth
# --------------------------------------------------------------------------- #
def _cat_step(weighting: UnitWeighting) -> Optional[CatDefinition]:
    st.markdown("#### 2. The assessment")
    c1, c2, c3 = st.columns(3)
    cat_id = c1.selectbox("Assessment", _CAT_ORDER, index=0,
                          format_func=lambda c: CAT_LABELS[c], key="at_cat")
    assessment_type = c2.selectbox(
        "Type", [THEORY, PRACTICAL],
        format_func=lambda t: _TYPE_LABEL[t], key="at_type")
    total = c3.number_input("Out of (marks)", min_value=1, max_value=400,
                            value=50, step=1, key="at_total")
    duration = st.number_input("Duration (minutes)", min_value=15,
                               max_value=480, value=90, step=15,
                               key="at_duration")

    ledger = ledger_store.load(weighting.cdacc_code)
    all_numbers = [pc.number for pc in weighting.pcs]
    default = ledger_store.default_selection(ledger, cat_id, assessment_type,
                                             all_numbers)

    st.markdown("#### 3. Performance criteria to assess")
    if cat_id == FINAL_CAT:
        st.caption("A Final CAT is comprehensive: every PC is back in play.")
    else:
        done = ledger.already_assessed(assessment_type)
        if done:
            st.caption("Already assessed by an earlier CAT of this type: "
                       + ", ".join(done))

    selected: List[str] = []
    for el in weighting.elements:
        with st.expander(f"{el.number}. {el.title or 'Element ' + el.number}",
                         expanded=True):
            for pc in el.pcs:
                label = (f"{pc.number} {pc.text[:96]} "
                         f"({pc.weight_for(assessment_type)} marks)")
                if st.checkbox(label, value=pc.number in default,
                               key=f"at_pc_{assessment_type}_{pc.number}"):
                    selected.append(pc.number)

    left = ledger_store.coverage(ledger, all_numbers)[assessment_type]
    if cat_id == FINAL_CAT and left:
        st.warning("Never assessed by any earlier CAT of this type: "
                   + ", ".join(left))

    cat = CatDefinition(cat_id=cat_id, assessment_type=assessment_type,
                        total_marks=int(total), selected_pcs=selected,
                        duration_minutes=int(duration))
    if len(selected) < 2:
        st.info("Select at least two performance criteria.")
        return None
    return cat if _show(alloc.check_total(weighting, cat)) else None


# --------------------------------------------------------------------------- #
# 4. the distribution - the last point at which a mark can change
# --------------------------------------------------------------------------- #
def _distribution_step(weighting: UnitWeighting,
                       cat: CatDefinition) -> List[Allocation]:
    allocations = alloc.allocate(weighting, cat)
    allocations = alloc.assign_bloom(allocations, weighting.knqf_level,
                                     cat.total_marks)
    st.markdown("#### 4. Mark distribution")
    st.caption("Computed here, in plain arithmetic. The model is given these "
               "numbers and never changes them.")
    st.dataframe(
        [{"Element": a.element_number, "PC": a.pc_number,
          "Performance criterion": a.pc_text[:80], "Weight": a.weight,
          f"{cat.label} marks": a.marks, "Bloom": a.bloom.title()}
         for a in allocations],
        width="stretch", hide_index=True)
    st.caption(f"Total {sum(a.marks for a in allocations)} of "
               f"{cat.total_marks} marks")
    # Said here rather than after generating: an item too small for its own
    # Bloom level is fixed by changing the selection, not by rewriting a
    # question, and the trainer is looking at the selection right now.
    _show(alloc.check_items(allocations))
    return allocations


def _content_step(os_unit: Unit, curr_unit, weighting: UnitWeighting,
                  allocations: List[Allocation]) -> List[ElementContent]:
    """The taught content behind the elements this CAT covers.

    Shown rather than used silently, because it is the single biggest
    influence on whether the questions are any good and the trainer is the
    only person who can tell at a glance that an element has drawn the wrong
    outcome. Element titles come from the occupational standard where it was
    read - they are what the curriculum's outcome titles are matched against,
    and the pasted weighting table's titles are a lossier copy of the same
    thing.
    """
    elements = os_unit.elements or weighting.elements
    content = content_builder.content_for(
        curr_unit, elements, {a.element_number for a in allocations})

    st.markdown("#### 5. What the paper is set from")
    if not content:
        st.warning(
            "No curriculum content matched these elements, so the questions "
            "will be written from the performance criteria alone. They will "
            "be shallower, and nothing keeps them to what was actually "
            "taught.")
        return content

    st.caption(content_builder.summarise(content)
               + " - the model may assess this and nothing else.")
    for block in content:
        head = f"{block.element_number}. {block.element_title or block.outcome_title}"
        if block.duration_hours:
            head += f"  ({block.duration_hours} hours)"
        with st.expander(head, expanded=False):
            if (block.outcome_title
                    and block.outcome_title != block.element_title):
                st.caption(f"Curriculum learning outcome "
                           f"{block.outcome_number}: {block.outcome_title}")
            for topic in block.topics:
                st.markdown(f"**{topic.number} {topic.title}**")
                for point in topic.key_points:
                    st.markdown(f"- {point}")
    return content


# --------------------------------------------------------------------------- #
# 6-7. generate, then hand over the files
# --------------------------------------------------------------------------- #
def _documents(weighting: UnitWeighting, tool: AssessmentTool) -> None:
    stem = (tool.cdacc_code or tool.unit_title or "assessment").replace("/", "_")
    name = f"{stem}_{tool.cat.cat_id}_{tool.cat.assessment_type}"
    for label, data, suffix in (
            ("PC distribution table",
             docs.build_pc_distribution(weighting, tool), "distribution"),
            ("Table of specifications",
             docs.build_table_of_specifications(tool), "tos"),
            ("Candidate's tool", docs.build_candidates_tool(tool), "candidate"),
            ("Assessor's tool", docs.build_assessors_tool(tool), "assessor")):
        st.download_button(f"⬇ {label}", data=data,
                           file_name=f"{name}_{suffix}.docx",
                           mime=("application/vnd.openxmlformats-officedocument"
                                 ".wordprocessingml.document"),
                           key=f"at_dl_{suffix}")


def render(os_unit: Unit, curr_unit=None, programme: str = "") -> None:
    """The whole path, top to bottom."""
    st.subheader("Assessment tool")
    weighting = _weighting_step(os_unit)
    if weighting is None:
        return
    cat = _cat_step(weighting)
    if cat is None:
        return
    allocations = _distribution_step(weighting, cat)
    content = _content_step(os_unit, curr_unit, weighting, allocations)

    if st.button("Generate assessment tool", type="primary", key="at_go"):
        tool = AssessmentTool(
            unit_title=weighting.unit_title, cdacc_code=weighting.cdacc_code,
            isced_code=weighting.isced_code, knqf_level=weighting.knqf_level,
            programme=programme, cat=cat, allocations=allocations,
            content=content)
        messages: List[str] = []
        box = st.empty()

        def progress(msg: str) -> None:
            messages.append(str(msg))
            box.code("\n".join(messages[-20:]), language="text")

        try:
            with st.spinner("Writing the assessment..."):
                with runlog.timed("Assessment: generate"):
                    tool = assessment_ai_generate(tool, progress)
                problems = validators.validate(tool)
                if validators.repairable(problems):
                    with runlog.timed("Assessment: repair"):
                        tool = validators.repair(tool, problems,
                                                 progress_cb=progress)
                    problems = validators.validate(tool)
            ss.at_tool, ss.at_problems = tool, problems
        except Exception as e:                  # noqa: BLE001
            runlog.error(f"Assessment generation failed: {e}")
            st.error(f"Generation failed: {e}")
            return

    tool = ss.at_tool
    if tool is None:
        return
    st.markdown("#### 6. Result")
    remaining = validators.blocking(ss.at_problems)
    warnings = [p for p in ss.at_problems if not p.blocking]
    if remaining:
        st.error("These could not be repaired automatically and need editing "
                 "by hand before the documents are used:")
        _show(remaining)
    elif not warnings:
        st.success("Every check passed.")
    # Warnings used to be computed and then thrown away here, so a paper with
    # no scenario at all reported "Every check passed". Anything the checks
    # found is shown; only the blocking half decides the wording above it.
    if warnings and not remaining:
        st.info("Nothing here stops the documents being used - read it and "
                "decide whether to generate again.")
    _show(warnings)
    _documents(weighting, tool)


def assessment_ai_generate(tool: AssessmentTool, progress) -> AssessmentTool:
    """Kept separate so a test can stand in for the one call that costs money."""
    import assessment_ai
    return assessment_ai.generate(tool, progress_cb=progress)
