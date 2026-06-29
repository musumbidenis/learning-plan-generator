"""Streamlit UI - Learning Plan Generator (KSTVET REF KTTC/TP/LP/F07, RVNP).

A 4-step FORM WIZARD (one focus per screen, validated before advancing):

  1. Occupational Standard  -> upload -> auto-extract units -> pick the OS unit
  2. Curriculum             -> upload -> auto-extract units -> pick the CU unit
     (leaving step 2 deterministically extracts both selected units)
  3. Plan details           -> unit summary + trainer/class/schedule form
  4. Generate               -> one grounded Mistral call -> download .docx

Navigation is a `streamlit-antd-components` stepper plus Back/Next buttons,
driven by `ss.step`. Future steps stay locked until the current one validates;
the generated plan is cached (keyed by the units + inputs) so going Back and
Next again never repeats the paid AI call unless something actually changed.

The Mistral key + model are configured in ai_client.py (not entered in the UI).
Kenya CBET terminology throughout (trainee/trainer, assessment, CAT, competency).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import inspect
import os
import re
import tempfile
from typing import List, Optional

import streamlit as st
import streamlit_antd_components as sac

import ai_client
import curriculum_parser as cp
import doc_builder
import os_parser
import planner
import runlog
from models import CurriculumUnit, PlanInputs, Unit
from pdf_utils import load_document

st.set_page_config(page_title="Learning Plan Generator", layout="wide")

ss = st.session_state
_DEFAULTS = dict(
    # wizard position
    step=1,
    # Occupational Standard side
    os_sig=None, os_path=None, os_pages=None, os_refs=[], os_ref=None,
    # Curriculum side
    cu_sig=None, cu_path=None, cu_pages=None, cu_refs=[], cu_ref=None,
    # extraction output
    extracted=False, extracted_key=None,
    os_unit=None, curr_unit=None, display_code="",
    # generation cache
    gen_key=None, docx_bytes=None, sessions=None,
)
for _k, _v in _DEFAULTS.items():
    ss.setdefault(_k, _v)

DOCX_MIME = ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document")

STEP_DEFS = [
    ("Occupational Standard", "Upload & pick the OS unit"),
    ("Curriculum", "Upload & pick the curriculum unit"),
    ("Plan details", "Trainer, class & schedule"),
    ("Generate", "Create & download the plan"),
]
N_STEPS = len(STEP_DEFS)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _save_upload(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Words kept lowercase inside a title (never at the start); acronyms kept upper.
_TITLE_SMALL = {"a", "an", "and", "as", "at", "by", "for", "in", "of",
                "on", "or", "the", "to", "with"}
_TITLE_ACRONYMS = {"ICT", "IT", "AI", "OS", "CPU", "I/O", "SQL", "HTML", "CSS"}


def _pretty_name(title: str) -> str:
    """Readable Title Case from an ALL-CAPS CDACC unit title.

    'APPLY COMMUNICATION SKILLS' -> 'Apply Communication Skills';
    'COMPUTER ORGANISATION AND ARCHITECTURE' -> 'Computer Organisation and
    Architecture'. Connector words are lowercased (except first), acronyms and
    tokens containing digits are left untouched.
    """
    words = (title or "").split()
    out = []
    for i, w in enumerate(words):
        up = w.upper()
        if up in _TITLE_ACRONYMS:
            out.append(up)
        elif any(ch.isdigit() for ch in w):
            out.append(w)
        elif i != 0 and w.lower() in _TITLE_SMALL:
            out.append(w.lower())
        else:
            out.append(w.lower().capitalize())
    return " ".join(out) or (title or "")


def _ref_label(ref) -> str:
    # show only the readable unit name in the dropdown (no codes)
    return _pretty_name(ref.title)


def _no_units_message(pages, doc: str) -> str:
    """Explain why a successfully-loaded document yielded zero units, pointing at
    the likely cause: a scanned/image-only PDF (no extractable text) vs. a
    text-based file whose layout the parser did not recognise."""
    has_text = any(getattr(p, "words", None) for p in (pages or []))
    if not has_text:
        return (f"Couldn't read any text from the {doc} - it looks scanned or "
                "image-only. Upload a text-based PDF (or run OCR on it first).")
    return (f"No units found in the {doc}. The file may use an unexpected layout "
            "or may not be a CDACC unit document. Try the official PDF.")


# --------------------------------------------------------------------------- #
# State invalidation (a changed choice clears everything downstream of it)
# --------------------------------------------------------------------------- #
def _reset_extraction_and_generation() -> None:
    ss.extracted = False
    ss.extracted_key = None
    ss.os_unit = None
    ss.curr_unit = None
    ss.display_code = ""
    ss.gen_key = None
    ss.docx_bytes = None
    ss.sessions = None


def _reset_os_downstream() -> None:
    """A new OS file invalidates the OS units, the curriculum choice, and below."""
    ss.os_refs = []
    ss.os_pages = None
    ss.os_ref = None
    ss.cu_sig = None
    ss.cu_path = None
    ss.cu_pages = None
    ss.cu_refs = []
    ss.cu_ref = None
    _reset_extraction_and_generation()


def _reset_cu_downstream() -> None:
    """A new Curriculum file invalidates the curriculum units and below."""
    ss.cu_refs = []
    ss.cu_pages = None
    ss.cu_ref = None
    _reset_extraction_and_generation()


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def _goto(step: int) -> None:
    ss.step = max(1, min(N_STEPS, step))
    st.rerun()


def _validate_step(step: int) -> Optional[str]:
    """Return an error message if `step` is not yet complete, else None."""
    if step == 1 and ss.os_ref is None:
        return "Select an Occupational Standard unit to continue."
    if step == 2 and ss.cu_ref is None:
        return "Select a Curriculum unit to continue."
    if step == 3 and not str(ss.get("f_course", "")).strip():
        return "Enter the Course before continuing."
    return None


def _leave_step(step: int) -> bool:
    """Side effects when advancing past `step`. Returns False to block advancing."""
    if step == 2:
        return _ensure_extracted()
    return True


def _ensure_extracted() -> bool:
    """Deterministically parse the two selected units (cached by selection key)."""
    key = (ss.os_sig, ss.os_ref.isced_code, ss.os_ref.title,
           ss.cu_sig, ss.cu_ref.isced_code, ss.cu_ref.title)
    if ss.extracted and ss.extracted_key == key and ss.os_unit is not None:
        return True
    try:
        with st.spinner("Extracting the selected units..."):
            runlog.log(f"Extracting OS unit '{ss.os_ref.title}' + "
                       f"Curriculum unit '{ss.cu_ref.title}'")
            with runlog.timed("Extract selected units"):
                ou = os_parser.parse_os_unit(ss.os_pages, ss.os_ref)
                cu = cp.parse_curriculum_unit(ss.cu_pages, ss.cu_ref)
    except Exception as e:  # noqa: BLE001
        runlog.error(f"Extraction failed: {e}")
        st.error(f"Could not extract the selected units: {e}")
        return False
    ss.os_unit = ou
    ss.curr_unit = cu
    ss.display_code = (cu.curriculum_code or ou.os_code
                       or ou.isced_code or cu.isced_code)
    ss.extracted = True
    ss.extracted_key = key
    ss.gen_key = None          # units changed -> force a fresh generation
    return True


def _render_stepper() -> None:
    """The Ant-Design stepper. Future steps are locked; click a past step to go back."""
    items = [sac.StepsItem(title=t, description=d, disabled=(i > ss.step - 1))
             for i, (t, d) in enumerate(STEP_DEFS)]
    clicked = sac.steps(items, index=ss.step - 1, return_index=True)
    if isinstance(clicked, int) and clicked < ss.step - 1:
        _goto(clicked + 1)     # backward jump to a completed step


def _render_nav() -> None:
    st.divider()
    c1, _, c3 = st.columns([1, 6, 1])
    if ss.step > 1 and c1.button("◀ Back", use_container_width=True):
        _goto(ss.step - 1)
    if ss.step < N_STEPS and c3.button("Next ▶", type="primary",
                                       use_container_width=True):
        err = _validate_step(ss.step)
        if err:
            st.warning(err)
        elif _leave_step(ss.step):
            _goto(ss.step + 1)


# --------------------------------------------------------------------------- #
# Step 1 - Occupational Standard
# --------------------------------------------------------------------------- #
def render_step1_os() -> None:
    st.subheader("Upload the Occupational Standard")
    os_file = st.file_uploader("Occupational Standard (PDF/DOCX)",
                               type=["pdf", "docx"], key="os_file")

    if os_file is not None:
        sig = (os_file.name, os_file.size)
        if sig != ss.os_sig:
            ss.os_sig = sig
            ss.os_path = _save_upload(os_file)
            _reset_os_downstream()
            with st.spinner("Reading the Occupational Standard and extracting units..."):
                try:
                    runlog.log("Loading Occupational Standard")
                    with runlog.timed("Load Occupational Standard"):
                        ss.os_pages = load_document(ss.os_path)
                    with runlog.timed("Index OS units"):
                        ss.os_refs = os_parser.index_os_units(ss.os_pages)
                    runlog.log(f"Indexed {len(ss.os_refs)} OS units")
                except Exception as e:  # noqa: BLE001
                    ss.os_refs = []
                    ss.os_pages = None
                    runlog.error(f"Failed to read Occupational Standard: {e}")
                    st.error(f"Failed to read the Occupational Standard: {e}")

    if ss.os_refs:
        st.success(f"Extracted **{len(ss.os_refs)}** units from the Occupational Standard.")
        os_idx = st.selectbox(
            "Select the OS unit of competency",
            range(len(ss.os_refs)), index=None,
            placeholder="- select an OS unit -",
            format_func=lambda i: _ref_label(ss.os_refs[i]),
            key=f"os_pick::{ss.os_sig}")
        new_ref = ss.os_refs[os_idx] if os_idx is not None else None
        if new_ref is not ss.os_ref:
            ss.os_ref = new_ref
            # a different OS unit invalidates the prior extraction + generation
            ss.extracted_key = None
            ss.gen_key = None
    elif ss.os_pages is not None:
        st.error(_no_units_message(ss.os_pages, "Occupational Standard"))
    else:
        st.info("Upload the Occupational Standard to begin.")


# --------------------------------------------------------------------------- #
# Step 2 - Curriculum
# --------------------------------------------------------------------------- #
def render_step2_curriculum() -> None:
    st.subheader("Upload the Curriculum")
    cu_file = st.file_uploader("Curriculum (PDF/DOCX)",
                               type=["pdf", "docx"], key=f"cu_file::{ss.os_sig}")

    if cu_file is not None:
        sig = (cu_file.name, cu_file.size)
        if sig != ss.cu_sig:
            ss.cu_sig = sig
            ss.cu_path = _save_upload(cu_file)
            _reset_cu_downstream()
            with st.spinner("Reading the Curriculum and extracting units..."):
                try:
                    runlog.log("Loading Curriculum")
                    with runlog.timed("Load Curriculum"):
                        ss.cu_pages = load_document(ss.cu_path)
                    with runlog.timed("Index Curriculum units"):
                        ss.cu_refs = cp.index_curriculum_units(ss.cu_pages)
                    runlog.log(f"Indexed {len(ss.cu_refs)} curriculum units")
                except Exception as e:  # noqa: BLE001
                    ss.cu_refs = []
                    ss.cu_pages = None
                    runlog.error(f"Failed to read Curriculum: {e}")
                    st.error(f"Failed to read the Curriculum: {e}")

    if ss.cu_refs:
        st.success(f"Extracted **{len(ss.cu_refs)}** units from the Curriculum.")
        cu_idx = st.selectbox(
            "Select the Curriculum unit of learning",
            range(len(ss.cu_refs)), index=None,
            placeholder="- select a curriculum unit -",
            format_func=lambda i: _ref_label(ss.cu_refs[i]),
            key=f"cu_pick::{ss.cu_sig}")
        new_ref = ss.cu_refs[cu_idx] if cu_idx is not None else None
        if new_ref is not ss.cu_ref:
            ss.cu_ref = new_ref
            ss.extracted_key = None
            ss.gen_key = None

        if (ss.os_ref and ss.cu_ref and ss.os_ref.isced_code and ss.cu_ref.isced_code
                and _norm(ss.os_ref.isced_code) != _norm(ss.cu_ref.isced_code)):
            st.warning("Selected OS and Curriculum units have different ISCED codes "
                       f"({ss.os_ref.isced_code} vs {ss.cu_ref.isced_code}) - confirm "
                       "they correspond.")
    elif ss.cu_pages is not None:
        st.error(_no_units_message(ss.cu_pages, "Curriculum"))
    else:
        st.info("Upload the Curriculum, then select the matching unit.")


# --------------------------------------------------------------------------- #
# Step 3 - Plan details (unit summary + inputs form)
# --------------------------------------------------------------------------- #
def render_step3_plan_details() -> None:
    os_unit: Unit = ss.os_unit
    curr_unit: CurriculumUnit = ss.curr_unit

    st.subheader(f"Plan details — {os_unit.unit_title}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Elements", len(os_unit.elements))
    m2.metric("Performance criteria", len(os_unit.all_pcs))
    n_sub = sum(len(lo.sub_topics) for lo in curr_unit.learning_outcomes)
    m3.metric("Curriculum sub-topics", n_sub)
    m4.metric("Level", os_unit.level or "-")

    with st.expander("Performance criteria - from the Occupational Standard"):
        for el in os_unit.elements:
            st.markdown(f"**{el.number}. {el.title}**")
            for pc in el.performance_criteria:
                st.markdown(f"- {pc.number} {pc.text}")
        if os_unit.assessment_methods:
            st.caption("Evidence-Guide assessment methods: "
                       + ", ".join(os_unit.assessment_methods))

    with st.expander("Learning key points - from the Curriculum"):
        for lo in curr_unit.learning_outcomes:
            st.markdown(f"**{lo.number}. {lo.title}**")
            for stp in lo.sub_topics:
                st.markdown(f"- *{stp.number} {stp.title}* - "
                            + "; ".join(stp.key_points[:4]))

    st.divider()
    d1, d2, d3, d4 = st.columns(4)
    d1.text_input("Trainer name", key="f_trainer")
    d2.text_input("Institution", value="The Rift Valley National Polytechnic",
                  key="f_inst")
    d3.text_input("Course *", key="f_course", placeholder="e.g. ICT Technician")
    d4.text_input("Level", value=os_unit.level or "", key="f_level")

    d5, d6, d7 = st.columns(3)
    d5.text_input("Number of trainees", value="25", key="f_num")
    d6.text_input("Class code", key="f_class")
    d7.date_input("Date of preparation", _dt.date.today(), key="f_date")

    s1, s2, s3 = st.columns(3)
    term_weeks = s1.number_input("Term length (weeks)", 1, 30, 12, key="f_weeks")
    s2.number_input("Sessions per week", 1, 10, 2, key="f_spw")
    cat_count = s3.number_input("Number of CATs", 0, 10, 2, key="f_cats")

    default_cats: List[int] = []
    if cat_count:
        step = max(1, int(term_weeks) // (int(cat_count) + 1))
        default_cats = sorted({min(int(term_weeks), step * (i + 1))
                               for i in range(int(cat_count))})
        default_cats[-1] = int(term_weeks)
    st.text_input("CAT weeks (comma-separated)",
                  value=", ".join(map(str, default_cats)), key="f_catweeks")
    st.caption("Fields marked * are required.")


def _collect_inputs(os_unit: Unit) -> PlanInputs:
    """Read the step-3 widget values (persisted by key) into a PlanInputs."""
    date_prep = ss.get("f_date", _dt.date.today())
    date_str = (date_prep.strftime("%d-%m-%Y")
                if hasattr(date_prep, "strftime") else str(date_prep))
    try:
        cat_weeks = [int(x) for x in str(ss.get("f_catweeks", "")).split(",") if x.strip()]
    except ValueError:
        cat_weeks = []
    return PlanInputs(
        trainer_name=str(ss.get("f_trainer", "")),
        institution=str(ss.get("f_inst", "")),
        course=str(ss.get("f_course", "")),
        level=str(ss.get("f_level", "")) or (os_unit.level if os_unit else ""),
        num_trainees=str(ss.get("f_num", "25")),
        class_code=str(ss.get("f_class", "")),
        date_of_preparation=date_str,
        term_weeks=int(ss.get("f_weeks", 12)),
        cat_weeks=cat_weeks,
        sessions_per_week=int(ss.get("f_spw", 2)),
    )


# --------------------------------------------------------------------------- #
# Step 4 - Generate (cached by unit + inputs)
# --------------------------------------------------------------------------- #
def _gen_key(os_unit: Unit, curr_unit: CurriculumUnit, inputs: PlanInputs):
    return (ss.display_code, os_unit.unit_title, os_unit.os_code,
            getattr(curr_unit, "curriculum_code", ""),
            dataclasses.astuple(inputs))


def _do_generate(os_unit: Unit, curr_unit: CurriculumUnit,
                 inputs: PlanInputs, key) -> bool:
    runlog.log(f"Generate Learning Plan for {ss.display_code or os_unit.unit_title}")
    with runlog.timed("Plan sessions"):
        sessions = planner.plan_sessions(curr_unit, os_unit, inputs)
    runlog.log(f"Planned {len(sessions)} sessions across {inputs.term_weeks} weeks")

    progress_messages: List[str] = []
    status_placeholder = st.empty()

    def _push_progress(message: str) -> None:
        progress_messages.append(message)
        status_placeholder.code("\n".join(progress_messages[-25:]), language="text")

    # Streamlit Cloud may keep a previously-imported ai_client cached, so right
    # after a redeploy the loaded module can predate progress_cb. Adapt to the
    # live signature.
    supports_progress = "progress_cb" in inspect.signature(
        ai_client.generate_sessions).parameters
    try:
        with st.spinner("Filling generative columns..."):
            with runlog.timed("AI generate sessions"):
                if supports_progress:
                    sessions = ai_client.generate_sessions(
                        os_unit, sessions, api_key=None, progress_cb=_push_progress)
                else:
                    sessions = ai_client.generate_sessions(
                        os_unit, sessions, api_key=None)
    except ai_client.AIError as e:
        runlog.error(f"AI call failed: {e}")
        st.error(f"Generation failed: {e}")
        return False

    with runlog.timed("Build .docx"):
        ss.docx_bytes = doc_builder.document_to_bytes(
            os_unit, sessions, inputs, display_code=ss.display_code)
    ss.sessions = sessions
    ss.gen_key = key
    runlog.log("Learning Plan document ready")
    return True


def _download_and_preview() -> None:
    fname = f"Learning_Plan_{(ss.display_code or 'unit').replace('/', '_')}.docx"
    st.download_button("Download Learning Plan (.docx)", data=ss.docx_bytes,
                       file_name=fname, mime=DOCX_MIME)
    with st.expander("Preview sessions", expanded=True):
        for s in ss.sessions:
            tag = "[CAT] " if s.is_cat else ""
            st.markdown(f"**Week {s.week} · Session {s.session_no} · "
                        f"{tag}{s.session_title}**")
            a, b = st.columns(2)
            a.markdown("*Specific learning outcomes*")
            a.write("\n".join(s.learning_outcomes))
            a.markdown("*Learning key points*")
            a.write("\n".join(s.key_points))
            b.markdown("*Trainee activities*")
            b.write("\n".join(s.trainee_activities))
            b.markdown("*Resources / Learning checks & assessments*")
            b.write("\n".join(s.resources + ["-"] + s.assessments))
            st.divider()


def render_step4_generate() -> None:
    os_unit: Unit = ss.os_unit
    curr_unit: CurriculumUnit = ss.curr_unit
    inputs = _collect_inputs(os_unit)

    st.subheader("Generate Learning Plan")
    cats = ", ".join(map(str, sorted(set(inputs.cat_weeks)))) or "—"
    st.markdown(f"**{os_unit.unit_title}**  ·  `{ss.display_code}`")
    st.caption(
        f"Trainer: {inputs.trainer_name or '—'} · Course: {inputs.course or '—'} · "
        f"Class: {inputs.class_code or '—'} · Level: {inputs.level or '—'}  |  "
        f"{inputs.term_weeks} weeks · {inputs.sessions_per_week}/week · CATs at {cats}")

    key = _gen_key(os_unit, curr_unit, inputs)
    fresh = ss.gen_key == key and ss.docx_bytes is not None

    if not fresh:
        st.info("Review the summary above, then generate the plan.")
        if st.button("Generate Learning Plan", type="primary"):
            if not _do_generate(os_unit, curr_unit, inputs, key):
                return
            fresh = True

    if fresh:
        st.success("Learning Plan ready.")
        _download_and_preview()
        if st.button("Regenerate"):
            ss.gen_key = None
            ss.docx_bytes = None
            ss.sessions = None
            st.rerun()


# =========================================================================== #
# Main controller
# =========================================================================== #
st.title("Learning Plan Generator")
_render_stepper()

if ss.step == 1:
    render_step1_os()
elif ss.step == 2:
    render_step2_curriculum()
elif ss.step == 3:
    render_step3_plan_details()
else:
    render_step4_generate()

_render_nav()
