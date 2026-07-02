"""Streamlit UI - Learning Plan Generator (KSTVET REF KTTC/TP/LP/F07, RVNP).

Manual, sequential flow (the two source documents are chosen independently):

  1. Upload the Occupational Standard  -> units auto-extracted -> pick the OS unit
  2. Upload the Curriculum             -> units auto-extracted -> pick the CU unit
  3. Generate Learning Plan -> the selected units are extracted automatically
     (deterministic, no AI) -> preview + plan details -> one grounded Mistral
     call -> .docx

The Mistral key + model are configured in ai_client.py (not entered in the UI).
Kenya CBET terminology throughout (trainee/trainer, assessment, CAT, competency).
"""

from __future__ import annotations

import datetime as _dt
import inspect
import os
import re
import tempfile
import time
from typing import List

import streamlit as st

import ai_client
import curriculum_parser as cp
import doc_builder
import learning_plan_parser
import os_parser
import planner
import runlog
import session_plan_builder
from models import CurriculumUnit, PlanInputs, Session, Unit
from pdf_utils import load_document

# Session Plans no longer collect per-session inputs; the minute-by-minute
# breakdown is sized to this standard TVET double session unless the source says
# otherwise (the Learning Plan doesn't carry a per-session duration).
DEFAULT_SP_DURATION = 120

# Session plans are AI-only (NO deterministic fallback): each session is one API
# call, retried until it succeeds. This bounds how long we keep retrying a single
# session before giving up so the app never hangs indefinitely.
SP_MAX_ATTEMPTS = 8

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
    # generated Learning Plan sessions (kept so Session Plans can be generated
    # afterwards, and survive Streamlit reruns)
    lp_sessions=None, lp_key=None,
    # uploaded-Learning-Plan path (generate Session Plans without OS + Curriculum)
    up_sig=None, up_unit=None, up_sessions=None, up_inputs=None,
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
    # a new selection invalidates any previously-generated plan / session plans
    ss.lp_sessions = None
    ss.lp_key = None


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
    d1, d2, d3, d4 = st.columns(4)
    trainer = d1.text_input("Trainer name", key="f_trainer")
    institution = d2.text_input("Institution",
                                value="The Rift Valley National Polytechnic",
                                key="f_inst")
    course = d3.text_input("Course", key="f_course",
                           placeholder="e.g. ICT Technician")
    level = d4.text_input("Level", value=os_unit.level or "", key="f_level")

    d5, d6, d7 = st.columns(3)
    num_trainees = d5.text_input("Number of trainees", value="25", key="f_num")
    class_code = d6.text_input("Class code", key="f_class")
    date_prep = d7.date_input("Date of preparation", _dt.date.today(), key="f_date")

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
        trainer_name=trainer, institution=institution, course=course, level=level,
        num_trainees=num_trainees, class_code=class_code,
        date_of_preparation=date_prep.strftime("%d-%m-%Y"),
        term_weeks=int(term_weeks), cat_weeks=cat_weeks, sessions_per_week=int(spw))

    # ----- generate --------------------------------------------------------- #
    st.divider()

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

        runlog.log("Learning Plan document ready")
        st.success("Learning Plan ready.")

        # Keep the sessions so the preview (with per-session regenerate) and the
        # Session Plans section survive the Streamlit reruns those widgets trigger.
        ss.lp_sessions = sessions
        ss.lp_key = ss.extracted_key

    # Persistent Learning-Plan preview (download + per-session regenerate) and the
    # Session Plans section. Rendered on every rerun (outside the generate button)
    # and gated on a Learning Plan having been generated for THIS unit selection.
    if ss.lp_sessions and ss.lp_key == ss.extracted_key:
        render_learning_plan_preview(os_unit, inputs)
        render_session_plans(os_unit, inputs, ss.lp_sessions, key_prefix="sp")


# =========================================================================== #
# Session Plans (one detailed lesson plan per chosen session)
# =========================================================================== #
def _sp_filename(sess: Session) -> str:
    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", sess.session_title)[:40].strip("_")
    return (f"Session_Plan_W{sess.week}_S{sess.session_no}_"
            f"{safe_title or 'session'}.docx")


def _make_session_plan(os_unit: Unit, sess: Session, inputs: PlanInputs,
                       progress_cb=None):
    """One AI-generated session plan - NO offline fallback; retry until it works.

    Everything is pulled from the Learning Plan; date, time and trainer number
    are left blank for the trainer to fill in. The single API call is retried
    (with backoff) up to SP_MAX_ATTEMPTS times; if every attempt fails the final
    AIError propagates so the caller can report it (we never substitute a
    deterministic plan).
    """
    last_err: ai_client.AIError | None = None
    for attempt in range(1, SP_MAX_ATTEMPTS + 1):
        try:
            return ai_client.generate_session_plan(
                os_unit, sess, inputs, display_code=ss.display_code,
                trainer_number="", session_date="", session_time="",
                duration_minutes=DEFAULT_SP_DURATION, api_key=None,
                progress_cb=progress_cb)
        except ai_client.AIError as e:
            last_err = e
            runlog.error(f"Session-plan AI attempt {attempt}/{SP_MAX_ATTEMPTS} "
                         f"failed for '{sess.session_title}': {e}")
            if progress_cb is not None:
                progress_cb(f"Attempt {attempt}/{SP_MAX_ATTEMPTS} failed; retrying...")
            if attempt < SP_MAX_ATTEMPTS:
                time.sleep(min(3 * attempt, 15))          # backoff, capped
    raise ai_client.AIError(
        f"'{sess.session_title}' failed after {SP_MAX_ATTEMPTS} attempts: {last_err}")


def _lines_md(lines: List[str]) -> str:
    """Render an activity list as a 'Trainer:' heading + bullets (matches doc)."""
    body = session_plan_builder._activity_lines(lines)
    if not body:
        return "_—_"
    return "**Trainer:**\n" + "\n".join(f"- {l}" for l in body)


def _render_plan_preview(sess: Session, plan, key_prefix: str) -> None:
    """Read-only preview of a generated session plan, mirroring the .docx blocks."""
    st.divider()
    st.subheader(f"Session Plan — Week {sess.week} · Session {sess.session_no} · "
                 f"{sess.session_title}")
    st.download_button("⬇ Download Session Plan (.docx)",
                       data=session_plan_builder.document_to_bytes(plan),
                       file_name=_sp_filename(sess), mime=DOCX_MIME,
                       key=f"{key_prefix}_dl")

    with st.container(border=True):
        st.markdown(f"**Unit:** {plan.unit_title or '—'}  ·  **Code:** "
                    f"{plan.unit_code or '—'}  ·  **Level:** {plan.level or '—'}")
        st.markdown("**Learning outcome(s):**")
        st.markdown("\n".join(f"- {l}" for l in plan.learning_outcomes) or "_—_")
        st.markdown("**Resources:**")
        st.markdown("\n".join(f"- {l}" for l in plan.resources) or "_—_")
        st.markdown("**LLN / special-needs requirements:** "
                    + (plan.lln_requirements or "—"))
        st.markdown("**Safety requirements:** " + (plan.safety_requirements or "—"))

    st.markdown("#### 1. Introduction (5 minutes)")
    with st.container(border=True):
        st.markdown(_lines_md(plan.introduction))

    st.markdown("#### 2. Session Delivery")
    with st.container(border=True):
        for stp in plan.delivery_steps:
            st.markdown(f"**{stp.step_label} — {stp.minutes} minutes**")
            a, b = st.columns(2)
            a.markdown("*Trainer activity*")
            a.markdown("\n".join(f"- {l}" for l in stp.trainer_activity) or "_—_")
            a.markdown("*Trainee activity*")
            a.markdown("\n".join(f"- {l}" for l in stp.trainee_activity) or "_—_")
            b.markdown("*Learning check / assessment*")
            b.markdown("\n".join(f"- {l}" for l in stp.learning_check) or "_—_")

    st.markdown("#### 3. Session Review (5 minutes)")
    with st.container(border=True):
        st.markdown(_lines_md(plan.review))

    st.markdown(f"**Assignment:** {plan.assignment or '—'}")
    st.markdown(f"**TOTAL TIME:** {plan.total_minutes} minutes")


def _regenerate_lp_session(os_unit: Unit, idx: int) -> None:
    """AI-regenerate ONE Learning-Plan session in place, retrying (no fallback)."""
    sessions = ss.lp_sessions
    last_err: ai_client.AIError | None = None
    for attempt in range(1, SP_MAX_ATTEMPTS + 1):
        try:
            ai_client.regenerate_learning_plan_session(
                os_unit, sessions[idx], api_key=None)
            ss.lp_sessions = sessions
            return
        except ai_client.AIError as e:
            last_err = e
            runlog.error(f"Regenerate session {sessions[idx].session_no} attempt "
                         f"{attempt}/{SP_MAX_ATTEMPTS} failed: {e}")
            if attempt < SP_MAX_ATTEMPTS:
                time.sleep(min(3 * attempt, 15))
    raise ai_client.AIError(
        f"session {sessions[idx].session_no} failed after "
        f"{SP_MAX_ATTEMPTS} attempts: {last_err}")


def render_learning_plan_preview(os_unit: Unit, inputs: PlanInputs) -> None:
    """Persistent Learning-Plan preview: download + per-session 🔄 Regenerate."""
    sessions = ss.lp_sessions
    st.divider()
    st.subheader("Learning Plan preview")
    st.caption("Download the plan, or regenerate any single session with AI where "
               "needed. Regenerating rewrites only that session's row.")

    st.download_button(
        "⬇ Download Learning Plan (.docx)",
        data=doc_builder.document_to_bytes(os_unit, sessions, inputs,
                                           display_code=ss.display_code),
        file_name=f"Learning_Plan_{(ss.display_code or 'unit').replace('/', '_')}.docx",
        mime=DOCX_MIME, key="lp_dl")

    if not ai_client.load_api_key():
        st.info("Set MISTRAL_API_KEY to enable per-session regeneration.")

    for i, s in enumerate(sessions):
        c1, c2 = st.columns([0.8, 0.2])
        tag = "[CAT] " if s.is_cat else ""
        c1.markdown(f"**Week {s.week} · Session {s.session_no} · {tag}{s.session_title}**")
        regen = c2.button("🔄 Regenerate", key=f"lp_regen_{i}",
                          help="Regenerate this session with AI",
                          use_container_width=True,
                          disabled=not ai_client.load_api_key())
        with st.container(border=True):
            a, b = st.columns(2)
            a.markdown("*Specific learning outcomes*")
            a.markdown("\n".join(f"- {x}" for x in s.learning_outcomes) or "_—_")
            a.markdown("*Learning key points*")
            a.markdown("\n".join(f"- {x}" for x in s.key_points) or "_—_")
            b.markdown("*Trainee activities*")
            b.markdown("\n".join(f"- {x}" for x in s.trainee_activities) or "_—_")
            b.markdown("*Resources / Learning checks & assessments*")
            b.markdown("\n".join(f"- {x}" for x in s.resources) or "_—_")
            b.markdown("\n".join(f"- {x}" for x in s.assessments) or "_—_")
        if regen:
            try:
                with st.spinner(f"Regenerating session {s.session_no} with AI…"):
                    _regenerate_lp_session(os_unit, i)
            except ai_client.AIError as e:
                st.error(f"Couldn't regenerate this session: {e}")
            else:
                st.rerun()


def render_session_plans(os_unit: Unit, inputs: PlanInputs,
                         sessions: List[Session], *, key_prefix: str = "sp") -> None:
    st.divider()
    st.header("Generate Session Plans")
    st.caption("Everything is pulled from the Learning Plan. Details a Learning "
               "Plan doesn't carry (date, time, trainer number) are left blank on "
               "the document for you to fill in.")

    if not ai_client.load_api_key():
        st.error("Session plans are generated with AI, but no Mistral API key is "
                 "configured. Set MISTRAL_API_KEY and reload.")
        return

    # ----- one session at a time ------------------------------------------- #
    idx = st.selectbox(
        "Session to expand", range(len(sessions)),
        format_func=lambda i: (f"Week {sessions[i].week} · Session "
                               f"{sessions[i].session_no} · {sessions[i].session_title}"),
        key=f"{key_prefix}_pick")

    if st.button("Generate Session Plan", key=f"{key_prefix}_gen"):
        sess = sessions[int(idx)]
        runlog.log(f"Generate Session Plan for '{sess.session_title}'")
        try:
            with st.spinner("Generating the session plan with AI..."):
                with runlog.timed("AI generate session plan"):
                    plan = _make_session_plan(os_unit, sess, inputs)
        except ai_client.AIError as e:
            st.error(f"AI generation failed after {SP_MAX_ATTEMPTS} attempts: {e}. "
                     "Please try again.")
        else:
            ss[f"{key_prefix}_plan"] = plan
            ss[f"{key_prefix}_plan_idx"] = int(idx)
            st.success(f"Session plan ready ({plan.total_minutes} minutes, "
                       f"{len(plan.delivery_steps)} delivery steps).")

    # persistent read-only preview of the last-generated session plan
    stored = ss.get(f"{key_prefix}_plan")
    if stored is not None:
        stored_idx = min(int(ss.get(f"{key_prefix}_plan_idx", 0)), len(sessions) - 1)
        _render_plan_preview(sessions[stored_idx], stored, key_prefix)

    # ----- batch: all sessions -> one .zip --------------------------------- #
    st.divider()
    st.markdown("**Batch**")
    st.caption(f"Generate an AI session plan for every one of the {len(sessions)} "
               "sessions - one API call each, retried until it succeeds - then "
               "download them together as a .zip.")
    if st.button(f"Generate ALL {len(sessions)} Session Plans (.zip)",
                 key=f"{key_prefix}_batch"):
        runlog.log(f"Batch-generating {len(sessions)} session plans (AI-only)")
        progress = st.progress(0.0, text="Starting...")
        named_plans, failures = [], []
        with st.spinner("Generating every session plan with AI..."):
            for k, sess in enumerate(sessions, start=1):
                progress.progress(
                    k / len(sessions),
                    text=f"({k}/{len(sessions)}) {sess.session_title[:60]}")
                try:
                    with runlog.timed(f"Session plan {k}/{len(sessions)}"):
                        plan = _make_session_plan(os_unit, sess, inputs)
                    named_plans.append((_sp_filename(sess), plan))
                except ai_client.AIError as e:
                    runlog.error(f"Batch: giving up on '{sess.session_title}': {e}")
                    failures.append(sess.session_title)

        if named_plans:
            with runlog.timed("Zip session plans"):
                zip_bytes = session_plan_builder.plans_to_zip(named_plans)
            st.success(f"Generated **{len(named_plans)}/{len(sessions)}** session "
                       "plans with AI.")
            zip_name = f"Session_Plans_{(ss.display_code or 'unit').replace('/', '_')}.zip"
            st.download_button("Download all Session Plans (.zip)", data=zip_bytes,
                               file_name=zip_name, mime="application/zip",
                               key=f"{key_prefix}_zip_dl")
        if failures:
            st.error("These sessions failed after "
                     f"{SP_MAX_ATTEMPTS} attempts each and are NOT in the zip - "
                     "click again to retry them:\n"
                     + "\n".join(f"- {t}" for t in failures))


# =========================================================================== #
# Flow B - upload an existing Learning Plan and go straight to Session Plans
# =========================================================================== #
def render_upload_flow() -> None:
    st.header("Upload your Learning Plan")
    st.caption("Upload a Learning Plan (.docx) this app generated; its sessions "
               "are read back so you can generate Session Plans without the "
               "Occupational Standard and Curriculum.")
    lp_file = st.file_uploader("Learning Plan (.docx)", type=["docx"],
                               key="lp_upload")

    if lp_file is not None:
        sig = (lp_file.name, lp_file.size)
        if sig != ss.up_sig:
            ss.up_sig = sig
            path = _save_upload(lp_file)
            try:
                with st.spinner("Reading the Learning Plan..."):
                    with runlog.timed("Parse uploaded Learning Plan"):
                        unit, sessions, inputs = \
                            learning_plan_parser.parse_learning_plan(path)
                ss.up_unit, ss.up_sessions, ss.up_inputs = unit, sessions, inputs
                ss.display_code = unit.os_code or unit.isced_code
                runlog.log(f"Parsed uploaded Learning Plan: {len(sessions)} sessions")
            except Exception as e:  # noqa: BLE001
                ss.up_unit = ss.up_sessions = ss.up_inputs = None
                runlog.error(f"Failed to read uploaded Learning Plan: {e}")
                st.error(f"Couldn't read that Learning Plan: {e}")

    if ss.up_sessions:
        st.success(f"Loaded **{len(ss.up_sessions)}** sessions from "
                   f"**{ss.up_unit.unit_title or 'your Learning Plan'}**.")
        with st.expander("Sessions found in the Learning Plan"):
            for s in ss.up_sessions:
                tag = "[CAT] " if s.is_cat else ""
                st.markdown(f"- Week {s.week} · Session {s.session_no} · "
                            f"{tag}{s.session_title}")
        render_session_plans(ss.up_unit, ss.up_inputs, ss.up_sessions,
                             key_prefix="up")
    elif lp_file is None:
        st.info("Upload a Learning Plan (.docx) that this app generated.")


# =========================================================================== #
# Flow A - build a Learning Plan from the Occupational Standard + Curriculum
# =========================================================================== #
def render_create_flow() -> None:
    # ----- 1. Upload the Occupational Standard ----------------------------- #
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

    # ----- 2. Upload the Curriculum ---------------------------------------- #
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

    # ----- 3. Generate Learning Plan --------------------------------------- #
    if os_ref is not None and cu_ref is not None:
        st.header("3. Generate Learning Plan")

        if (os_ref.isced_code and cu_ref.isced_code
                and _norm(os_ref.isced_code) != _norm(cu_ref.isced_code)):
            st.warning("Selected OS and Curriculum units have different ISCED codes "
                       f"({os_ref.isced_code} vs {cu_ref.isced_code}) - confirm they "
                       "correspond.")

        cur_key = (ss.os_sig, os_ref.isced_code, os_ref.title,
                   ss.cu_sig, cu_ref.isced_code, cu_ref.title)

        # Extract the selected units automatically (only when the selection is new,
        # so editing the plan-details form never re-parses).
        if not (ss.extracted and ss.extracted_key == cur_key and ss.os_unit is not None):
            with st.spinner("Extracting the selected units..."):
                try:
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
                except Exception as e:  # noqa: BLE001
                    runlog.error(f"Extraction failed: {e}")
                    st.error(f"Could not extract the selected units: {e}")

        if ss.extracted and ss.extracted_key == cur_key and ss.os_unit is not None:
            render_preview_and_generate()


# --------------------------------------------------------------------------- #
# Header + mode dispatch
# --------------------------------------------------------------------------- #
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0.5rem;'>Learning Plan Generator</h1>",
    unsafe_allow_html=True)

_MODE_CREATE = "Create a Learning Plan (from OS + Curriculum)"
_MODE_UPLOAD = "Upload a Learning Plan → Session Plans"
_mode = st.radio("What would you like to do?", [_MODE_CREATE, _MODE_UPLOAD],
                 horizontal=True, key="mode")
st.divider()

if _mode == _MODE_UPLOAD:
    render_upload_flow()
else:
    render_create_flow()


# =========================================================================== #
# Footer (centered, fixed to the bottom of the screen)
# =========================================================================== #
st.markdown(
    "<style>"
    "  .block-container { padding-bottom: 4.5rem; }"   # keep content clear of footer
    "  .app-footer {"
    "    position: fixed; left: 0; bottom: 0; width: 100%;"
    "    text-align: center; color: gray; font-size: 0.85rem; padding: 8px 0;"
    "    border-top: 1px solid rgba(128,128,128,0.25);"
    "    background-color: rgba(127,127,127,0.06);"
    "    -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);"
    "    z-index: 1000;"
    "  }"
    "</style>"
    "<div class='app-footer'>Made with &#10084;&#65039; by Musumbi &#128081;</div>",
    unsafe_allow_html=True)
