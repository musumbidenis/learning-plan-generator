"""The Assessment Tool path: from a unit's performance criteria to four files.

A sibling of the Learning Plan path, and it runs in the same order the work
actually happens in:

    1  paste the unit's PC weighting table, and correct what the paste lost
    2  say which CAT it is, written or practical, and out of how many marks
    3  pick the performance criteria this assessment covers
    4  read the computed distribution - every mark is final at this point
    5  read the curriculum content the paper will be set from
    6  attach what the trainer actually taught from, and say anything else
       about this paper
    7  look up only the topics those resources do not reach
    8  generate, validate, repair, download

WHERE A QUESTION'S CONTENT COMES FROM, IN ORDER
The performance criteria say what is assessed and what it is worth. They are
never content: a PC is one general line, and a paper written from it is
written from the model's own knowledge of the trade - plausible, and about
things nobody covered.

The curriculum says which topics are in scope. It is a list of headings, and
a paper written from headings is a paper of headings.

The trainer's own resources say what was actually taught, and that is what a
candidate can fairly be asked about - so when step 6 has anything in it, the
questions come from there. Step 7 reads up on whatever the resources miss,
and nothing else fills a gap the trainer has already filled.

Steps 4 and 5 are kept apart on purpose. All the arithmetic happens in step 4,
in `assessment_allocation`, in plain code. By the time the model is called it
receives fixed numbers and writes prose around them, and it never computes,
totals or adjusts a mark - the same rule that keeps the Learning Plan's
assessment numbering deterministic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

import assessment_allocation as alloc
import assessment_content as content_builder
import assessment_docs as docs
import assessment_knowledge as knowledge
import assessment_ledger as ledger_store
import assessment_research as research
import assessment_resources as resource_reader
import assessment_validators as validators
import assessment_weighting as weighting_parser
import runlog
from assessment_models import (CAT_1, CAT_2, CAT_3, CAT_LABELS, FINAL_CAT,
                               PRACTICAL, THEORY, Allocation, AssessmentTool,
                               CatDefinition, ElementContent, Exemplar,
                               KnowledgeNote, Problem,
                               ResourceChunk, UnitWeighting)
from models import Unit

ss = st.session_state

_STATE = dict(at_weighting=None, at_raw="", at_tool=None,
              at_problems=[], at_resources={})


def _ensure_state() -> None:
    """Put the defaults in place for whichever session is running now.

    This module is imported once per server process, but session state belongs
    to a browser session. Setting the defaults at import time serves the first
    session to arrive and nobody else: the next visitor, or the same one after
    a state reset, reaches `_weighting_step` with no `at_raw` and Streamlit
    raises AttributeError. `app.py` gets away with the import-time loop because
    it is the entry script and is re-executed on every rerun; an imported
    module is not, so it has to seed its own keys on each pass.
    """
    for key, default in _STATE.items():
        ss.setdefault(key, default)

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


def _resource_step(content: List[ElementContent]
                   ) -> Tuple[List[ResourceChunk], str]:
    """The trainer's own material, and anything else they want to say.

    The most important step on the page and the only optional one. Everything
    else the generator has is an approximation of what was taught - topic
    headings, criteria, a reference work's account. This is the thing itself,
    so when it is here the questions come from it.

    Each file is read once and kept in session state under a signature of its
    contents, because a Streamlit rerun happens on every keystroke in the
    instructions box and re-reading a PowerPoint - or worse, re-transcribing a
    recording - on each one would be unusable.
    """
    st.markdown("#### 6. The trainer's own notes and resources")
    st.caption("Attach what you actually taught from and the questions will "
               "be written from it: notes, slides, a handout, a recording of "
               "the session. Anything a topic is not covered by is looked up "
               "instead. PDF, Word, PowerPoint, text, audio or video.")
    uploaded = st.file_uploader(
        "Resources", accept_multiple_files=True, key="at_files",
        type=["pdf", "docx", "doc", "odt", "rtf", "pptx", "txt", "md", "csv",
              "mp3", "m4a", "wav", "webm", "mp4", "ogg", "flac"])

    resources: List = []
    for upload in uploaded or []:
        data = upload.getvalue()
        signature = f"{upload.name}:{len(data)}"
        if signature not in ss.at_resources:
            with st.spinner(f"Reading {upload.name}..."):
                ss.at_resources[signature] = resource_reader.read(
                    upload.name, data)
        resources.append(ss.at_resources[signature])

    for resource in resources:
        if resource.ok:
            st.caption(f"✔ **{resource.name}** - {len(resource.chunks)} "
                       f"piece(s) read")
        else:
            st.warning(f"**{resource.name}** could not be used: "
                       f"{resource.note}")

    chosen = resource_reader.select(resources, content) if resources else []
    if chosen:
        st.caption(resource_reader.summarise(resources, chosen)
                   + " - the questions are written from these.")
        with st.expander("What the model will read", expanded=False):
            for chunk in chosen:
                st.markdown(f"**{chunk.label}**")
                st.caption(chunk.text[:600]
                           + ("..." if len(chunk.text) > 600 else ""))
    elif resources:
        st.info("Nothing in those files matched the topics this CAT covers, "
                "so the questions will be written from the curriculum and "
                "looked-up notes instead.")

    extra = st.text_area(
        "Anything else for this paper (optional)", key="at_extra", height=90,
        placeholder="e.g. Favour the practical side - they struggled with "
                    "earthing.\nKeep the language simple, this is a first "
                    "attempt.\nUse the workshop's own tools by name.")
    return chosen, (extra or "").strip()


def _knowledge_step(unit_title: str, content: List[ElementContent],
                    resources: Optional[List[ResourceChunk]] = None
                    ) -> List[KnowledgeNote]:
    """The material the paper is written from, read from reference sources.

    This is the step to read before generating. These notes are what the
    questions are MADE of - not a hint, not extra colour - so a key point that
    has drawn the wrong article is a question about the wrong subject, and the
    trainer is the only person who can see that at a glance.

    Scouted once per unit and cached, like the exemplars.
    """
    if not content:
        return []
    # With resources attached, only the topics they miss are looked up - and
    # which topics those are depends on what was attached THIS time, so the
    # per-unit cache is bypassed.
    covered = ((lambda point: resource_reader.covered(resources, point))
               if resources else None)
    cached = covered is None and knowledge._cached(unit_title) is not None
    if cached:
        notes = knowledge.notes_for(unit_title, content)
    else:
        with st.spinner("Looking up the topics your resources do not cover..."
                        if resources else "Reading up on the taught topics..."):
            notes = knowledge.notes_for(unit_title, content, covered=covered)

    st.markdown("#### 7. Looked up, for the topics the resources miss")
    if not notes:
        st.caption("Your resources cover every assessed topic, so nothing "
                   "needed looking up." if resources else
                   "No teaching notes were found for these topics, so the "
                   "questions are written from the curriculum's key points "
                   "alone. They will be shallower for it.")
        return []
    st.caption(knowledge.summarise(notes)
               + " - the questions are made of this. Read it before "
                 "generating: a note on the wrong subject is a question on "
                 "the wrong subject.")
    with st.expander(f"{len(notes)} teaching note(s)", expanded=False):
        for index, note in enumerate(knowledge.fitting(notes), start=1):
            label = knowledge.label_of(note, index)
            st.markdown(f"**{label}**")
            st.markdown(note.summary)
            for fact in note.facts:
                st.markdown(f"- {fact}")
            # Shown to the trainer, deliberately NOT sent to the model - a
            # bare list of names is raw material for invention. See
            # `assessment_knowledge.render`.
            if note.covers:
                st.caption("The source also covers: " + " | ".join(note.covers))
            if note.named:
                st.caption("Names in the source (not sent to the model): "
                           + ", ".join(note.named))
            if note.source_url:
                st.caption(f"[{note.source_title}]({note.source_url})")
            st.divider()
    return notes


def _research_step(unit_title: str,
                   content: List[ElementContent]) -> List[Exemplar]:
    """Real questions from published CDACC papers on this unit.

    Scouted once per unit and cached on disk, so the first assessment on a
    unit waits about half a minute and every one after it reads the file. A
    repository that is down costs the exemplars and not the paper.
    """
    if not unit_title:
        return []
    topics = [t.title for block in content for t in block.topics]
    cached = research._cached(unit_title) is not None
    if cached:
        found = research.exemplars_for(unit_title, topics)
    else:
        with st.spinner("Looking for real CDACC papers on this unit..."):
            found = research.exemplars_for(unit_title, topics)

    if not found:
        st.caption("No published papers were found for this unit, so the "
                   "questions are written without a style reference.")
        return []
    with st.expander(f"{len(found)} real question(s) found in published "
                     f"papers - used as a style reference only", expanded=False):
        st.caption("The model copies the shape of these - how much situation "
                   "sits in front of the verb, what a question of that many "
                   "marks asks for. It is told explicitly not to copy the "
                   "subject: these come from other colleges and other trades.")
        for ex in found:
            marks = f"  — *{ex.marks} marks*" if ex.marks else ""
            st.markdown(f"- {ex.text}{marks}")
        sources = sorted({e.source for e in found if e.source})
        if sources:
            st.caption("From: " + "; ".join(sources))
    return found


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
    _ensure_state()
    st.subheader("Assessment tool")
    weighting = _weighting_step(os_unit)
    if weighting is None:
        return
    cat = _cat_step(weighting)
    if cat is None:
        return
    allocations = _distribution_step(weighting, cat)
    content = _content_step(os_unit, curr_unit, weighting, allocations)
    unit_title = weighting.unit_title or os_unit.unit_title
    resources, extra = _resource_step(content)
    notes = _knowledge_step(unit_title, content, resources)
    exemplars = _research_step(unit_title, content)

    if st.button("Generate assessment tool", type="primary", key="at_go"):
        tool = AssessmentTool(
            unit_title=weighting.unit_title, cdacc_code=weighting.cdacc_code,
            isced_code=weighting.isced_code, knqf_level=weighting.knqf_level,
            programme=programme, cat=cat, allocations=allocations,
            content=content, exemplars=exemplars, knowledge=notes,
            resources=resources, extra_instructions=extra)
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
    st.markdown("#### 8. Result")
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
