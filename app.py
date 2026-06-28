"""Streamlit UI - Learning Plan Generator (KSTVET REF KTTC/TP/LP/F07, RVNP).

Manual, sequential flow (the two source documents are chosen independently):

  1. Upload the Occupational Standard  -> units auto-extracted -> pick the OS unit
  2. Upload the Curriculum             -> units auto-extracted -> pick the CU unit
  3. Extract unit details (deterministic, no AI) -> preview
  4. Generate Learning Plan (one grounded Mistral call) -> .docx

The Mistral key + model are configured in ai_client.py (not entered in the UI).
Kenya CBET terminology throughout (trainee/trainer, assessment, CAT, competency).
"""

from __future__ import annotations

import datetime as _dt
import inspect
import os
import re
import tempfile
from typing import List

import streamlit as st

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
    # Occupational Standard side
    os_sig=None, os_path=None, os_pages=None, os_refs=[],
    # Curriculum side
    cu_sig=None, cu_path=None, cu_pages=None, cu_refs=[],
    # extraction output
    extracted=False, extracted_key=None,
    os_unit=None, curr_unit=None, display_code="",
)
for _k, _v in _DEFAULTS.items():
    ss.setdefault(_k, _v)

DOCX_MIME = ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document")


# --------------------------------------------------------------------------- #
# Helpers
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


def _invalidate_extraction() -> None:
    ss.extracted = False
    ss.extracted_key = None
    ss.os_unit = None
    ss.curr_unit = None


# =========================================================================== #
# Stage 3/4 body - preview + generate (reads ss.os_unit / ss.curr_unit)
# =========================================================================== #
def render_preview_and_generate() -> None:
    os_unit: Unit = ss.os_unit
    curr_unit: CurriculumUnit = ss.curr_unit

    st.subheader(f"Unit details - {os_unit.unit_title}")
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

    # ----- plan details ----------------------------------------------------- #
    st.subheader("Plan details")
    d1, d2, d3 = st.columns(3)
    trainer = d1.text_input("Trainer name", key="f_trainer")
    institution = d2.text_input("Institution",
                                value="The Rift Valley National Polytechnic",
                                key="f_inst")
    level = d3.text_input("Level", value=os_unit.level or "", key="f_level")

    d4, d5, d6 = st.columns(3)
    num_trainees = d4.text_input("Number of trainees", value="25", key="f_num")
    class_code = d5.text_input("Class code", key="f_class")
    date_prep = d6.date_input("Date of preparation", _dt.date.today(), key="f_date")

    s1, s2, s3 = st.columns(3)
    term_weeks = s1.number_input("Term length (weeks)", 1, 30, 12, key="f_weeks")
    spw = s2.number_input("Sessions per week", 1, 10, 2, key="f_spw")
    cat_count = s3.number_input("Number of CATs", 0, 10, 2, key="f_cats")

    default_cats: List[int] = []
    if cat_count:
        step = max(1, int(term_weeks) // (int(cat_count) + 1))
        default_cats = sorted({min(int(term_weeks), step * (i + 1))
                               for i in range(int(cat_count))})
        default_cats[-1] = int(term_weeks)
    cat_weeks_str = st.text_input("CAT weeks (comma-separated)",
                                  value=", ".join(map(str, default_cats)),
                                  key="f_catweeks")
    try:
        cat_weeks = [int(x) for x in cat_weeks_str.split(",") if x.strip()]
    except ValueError:
        cat_weeks = default_cats
        st.warning("Could not parse CAT weeks; using defaults.")

    inputs = PlanInputs(
        trainer_name=trainer, institution=institution, level=level,
        num_trainees=num_trainees, class_code=class_code,
        date_of_preparation=date_prep.strftime("%d-%m-%Y"),
        term_weeks=int(term_weeks), cat_weeks=cat_weeks, sessions_per_week=int(spw))

    # ----- generate --------------------------------------------------------- #
    st.subheader("4. Generate Learning Plan")

    if st.button("Generate Learning Plan", type="primary"):
        runlog.log(f"Generate Learning Plan for {ss.display_code or os_unit.unit_title}")
        with runlog.timed("Plan sessions"):
            sessions = planner.plan_sessions(curr_unit, os_unit, inputs)
        runlog.log(f"Planned {len(sessions)} sessions across {inputs.term_weeks} weeks")
        st.write(f"Planned **{len(sessions)}** sessions across "
                 f"**{inputs.term_weeks}** weeks "
                 f"(CATs at weeks {', '.join(map(str, sorted(set(cat_weeks))))}).")
        progress_messages: List[str] = []
        status_placeholder = st.empty()

        def _push_progress(message: str) -> None:
            progress_messages.append(message)
            status_placeholder.code("\n".join(progress_messages[-25:]), language="text")

        # Streamlit Cloud reruns this script on change but may keep a previously
        # imported ai_client cached, so right after a redeploy the loaded module
        # can predate progress_cb. Adapt to whichever signature is live.
        supports_progress = "progress_cb" in inspect.signature(
            ai_client.generate_sessions).parameters
        try:
            with st.spinner("Filling generative columns..."):
                st.caption("Logs")
                with runlog.timed("AI generate sessions"):
                    if supports_progress:
                        sessions = ai_client.generate_sessions(
                            os_unit, sessions, api_key=None,
                            progress_cb=_push_progress)
                    else:
                        sessions = ai_client.generate_sessions(
                            os_unit, sessions, api_key=None)
        except ai_client.AIError as e:
            runlog.error(f"AI call failed: {e}")
            st.error(f"Generation failed: {e}")
            return

        with runlog.timed("Build .docx"):
            docx_bytes = doc_builder.document_to_bytes(os_unit, sessions, inputs,
                                                       display_code=ss.display_code)
        runlog.log("Learning Plan document ready")
        st.success("Learning Plan ready.")
        fname = f"Learning_Plan_{(ss.display_code or 'unit').replace('/', '_')}.docx"
        st.download_button("Download Learning Plan (.docx)", data=docx_bytes,
                           file_name=fname, mime=DOCX_MIME)

        with st.expander("Preview sessions", expanded=True):
            for s in sessions:
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


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Learning Plan Generator")


# =========================================================================== #
# 1. Upload the Occupational Standard -> auto-extract units -> pick a unit
# =========================================================================== #
st.header("1. Upload the Occupational Standard")
os_file = st.file_uploader("Occupational Standard (PDF/DOCX)",
                           type=["pdf", "docx"], key="os_file")

if os_file is not None:
    sig = (os_file.name, os_file.size)
    if sig != ss.os_sig:
        ss.os_sig = sig
        ss.os_path = _save_upload(os_file)
        # a new OS invalidates the curriculum choice and any extraction
        ss.cu_sig = None
        ss.cu_path = None
        ss.cu_pages = None
        ss.cu_refs = []
        _invalidate_extraction()
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

os_ref = None
if ss.os_refs:
    st.success(f"Extracted **{len(ss.os_refs)}** units from the Occupational Standard.")
    os_idx = st.selectbox(
        "Select the OS unit of competency",
        range(len(ss.os_refs)), index=None,
        placeholder="- select an OS unit -",
        format_func=lambda i: _ref_label(ss.os_refs[i]),
        key=f"os_pick::{ss.os_sig}")
    if os_idx is not None:
        os_ref = ss.os_refs[os_idx]
elif ss.os_pages is not None:
    st.error(_no_units_message(ss.os_pages, "Occupational Standard"))
elif os_file is None:
    st.info("Upload the Occupational Standard to begin.")


# =========================================================================== #
# 2. Upload the Curriculum -> auto-extract units -> pick a unit
# =========================================================================== #
cu_ref = None
if os_ref is not None:
    st.header("2. Upload the Curriculum")
    cu_file = st.file_uploader("Curriculum (PDF/DOCX)",
                               type=["pdf", "docx"], key=f"cu_file::{ss.os_sig}")

    if cu_file is not None:
        sig = (cu_file.name, cu_file.size)
        if sig != ss.cu_sig:
            ss.cu_sig = sig
            ss.cu_path = _save_upload(cu_file)
            _invalidate_extraction()
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
        if cu_idx is not None:
            cu_ref = ss.cu_refs[cu_idx]
    elif ss.cu_pages is not None:
        st.error(_no_units_message(ss.cu_pages, "Curriculum"))
    elif cu_file is None:
        st.info("Upload the Curriculum, then select the matching unit.")


# =========================================================================== #
# 3. Extract unit details (deterministic) -> preview + generate
# =========================================================================== #
if os_ref is not None and cu_ref is not None:
    st.header("3. Extract unit details")

    if (os_ref.isced_code and cu_ref.isced_code
            and _norm(os_ref.isced_code) != _norm(cu_ref.isced_code)):
        st.warning("Selected OS and Curriculum units have different ISCED codes "
                   f"({os_ref.isced_code} vs {cu_ref.isced_code}) - confirm they "
                   "correspond.")

    cur_key = (ss.os_sig, os_ref.isced_code, os_ref.title,
               ss.cu_sig, cu_ref.isced_code, cu_ref.title)

    if st.button("Extract unit details", type="primary"):
        with st.spinner("Extracting the selected units..."):
            runlog.log(f"Extracting OS unit '{os_ref.title}' + "
                       f"Curriculum unit '{cu_ref.title}'")
            with runlog.timed("Extract selected units"):
                ou = os_parser.parse_os_unit(ss.os_pages, os_ref)
                cu = cp.parse_curriculum_unit(ss.cu_pages, cu_ref)
            ss.os_unit = ou
            ss.curr_unit = cu
            ss.display_code = (cu.curriculum_code or ou.os_code
                               or ou.isced_code or cu.isced_code)
            ss.extracted = True
            ss.extracted_key = cur_key

    if ss.extracted and ss.extracted_key == cur_key and ss.os_unit is not None:
        render_preview_and_generate()
    elif ss.extracted and ss.extracted_key != cur_key:
        st.info("Selection changed - click **Extract unit details** again.")
