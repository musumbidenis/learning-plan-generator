"""Session Plan: deterministic builder, AI safety nets, and .docx rendering.

ZERO API calls - the AI merge is tested by feeding canned dicts straight into the
coercion/merge helpers, and the offline builder is fully deterministic.
"""

from docx.oxml.ns import qn

import ai_client
import session_plan_builder
from models import PlanInputs, Session, SessionPlan, Unit


def _unit() -> Unit:
    return Unit(unit_title="Apply Market Research", os_code="HOS/OS/03", level="5",
                assessment_methods=["Oral questioning", "Observation",
                                    "Portfolio of evidence"])


def _session() -> Session:
    return Session(
        week=2, session_no="1", is_cat=False,
        session_title="Market Research Tools and Techniques",
        pcs=["1.1 Market research tools are identified"],
        key_points=["IDENTIFICATION OF TOOLS", "- Questionnaires"],
        learning_outcomes=["By the end of the session, the trainee should be able to:",
                           "a. Define market research tools.",
                           "b. Identify at least four tools."],
        trainee_activities=["- Group Discussion: explore tools in groups.",
                            "- Think-Pair-Share: compare tools.",
                            "- Demonstration with Participation: build a questionnaire.",
                            "Follow up Activity:",
                            "1. Develop two tools for an establishment. (15 Marks)",
                            "Due date:"],
        resources=["- Kotler, P. (2021). Marketing Management. Pearson.",
                   "- PowerPoint on market research"],
        assessments=["Knowledge Checks:", "1. Oral questioning", "2. Written assessment",
                     "Skills:", "1. Observation of developed tools",
                     "Attitudes:", "1. Attention to detail"])


def _inputs() -> PlanInputs:
    return PlanInputs(trainer_name="Jane Doe", institution="RVNP", level="5",
                      num_trainees="25", class_code="HM-2A")


# --------------------------------------------------------------------------- #
# Offline (deterministic) builder
# --------------------------------------------------------------------------- #
def test_offline_plan_is_complete_and_grounded():
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), display_code="HOS/OS/03",
        trainer_number="TR/9", session_date="23/8/2026", duration_minutes=120)

    assert plan.unit_code == "HOS/OS/03"
    assert plan.session_title == "Market Research Tools and Techniques"
    assert plan.trainer_name == "Jane Doe"
    assert plan.introduction and plan.review
    assert plan.delivery_steps
    # carried over verbatim from the Learning-Plan session
    assert plan.learning_outcomes[0].startswith("By the end")
    assert any("Kotler" in r for r in plan.resources)
    # assignment is taken from the session's follow-up activity
    assert "Develop two tools" in plan.assignment
    assert plan.lln_requirements and plan.safety_requirements


def test_offline_step_minutes_sum_to_total():
    duration = 120
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), duration_minutes=duration)
    assert sum(s.minutes for s in plan.delivery_steps) == duration - 10
    assert plan.total_minutes == duration          # 5' intro + steps + 5' review


def test_offline_practice_step_includes_skills_check():
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), duration_minutes=120)
    last = plan.delivery_steps[-1].learning_check
    assert any(line.strip().lower() == "skills" for line in last)


# --------------------------------------------------------------------------- #
# Assessment grouping
# --------------------------------------------------------------------------- #
def test_group_assessments_splits_by_heading():
    groups = ai_client._group_assessments(_session().assessments)
    assert groups["Knowledge"] == ["1. Oral questioning", "2. Written assessment"]
    assert groups["Skills"] == ["1. Observation of developed tools"]
    assert groups["Attitudes"] == ["1. Attention to detail"]


# --------------------------------------------------------------------------- #
# AI merge safety nets
# --------------------------------------------------------------------------- #
def test_merge_empty_ai_falls_back_to_defaults():
    body = ai_client._merge_session_plan_body({}, _unit(), _session(), 120)
    assert body["delivery_steps"]                  # never empty
    assert body["introduction"] and body["review"]
    assert body["assignment"]


def test_merge_keeps_valid_ai_steps():
    ai_rows = {
        "introduction": ["Trainer:", "Takes roll call."],
        "delivery_steps": [
            {"step_label": "Step 1", "minutes": 40,
             "trainer_activity": ["Trainer:", "Leads a Group Discussion."],
             "trainee_activity": ["Trainee(s):", "Discuss tools."],
             "learning_check": ["Knowledge", "1. Oral questioning"]},
        ],
        "review": ["Trainer:", "Summarizes."],
        "assignment": "Build a questionnaire.",
        "lln_requirements": "Provide simplified handouts.",
        "safety_requirements": "Observe lab SOPs.",
    }
    body = ai_client._merge_session_plan_body(ai_rows, _unit(), _session(), 120)
    assert len(body["delivery_steps"]) == 1
    assert body["delivery_steps"][0].minutes == 40
    assert body["assignment"] == "Build a questionnaire."
    assert body["lln_requirements"] == "Provide simplified handouts."


def test_merge_drops_blank_steps_then_falls_back():
    # a step with no trainer/trainee content is dropped; with none left we fall back
    ai_rows = {"delivery_steps": [{"step_label": "Step 1", "minutes": 10,
                                   "trainer_activity": [], "trainee_activity": [],
                                   "learning_check": []}]}
    body = ai_client._merge_session_plan_body(ai_rows, _unit(), _session(), 120)
    assert body["delivery_steps"]                  # fallback kicked in


def test_merge_coerces_string_to_single_line_no_charsplit():
    # a bare string must not be iterated character-by-character
    ai_rows = {"introduction": "Trainer takes roll call and reviews the session."}
    body = ai_client._merge_session_plan_body(ai_rows, _unit(), _session(), 120)
    assert body["introduction"] == ["Trainer takes roll call and reviews the session."]


# --------------------------------------------------------------------------- #
# .docx rendering
# --------------------------------------------------------------------------- #
def _shaded_fill(cell):
    shd = cell._tc.get_or_add_tcPr().find(qn("w:shd"))
    return shd.get(qn("w:fill")) if shd is not None else None


def test_docx_has_title_grid_and_banners():
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), duration_minutes=120)
    doc = session_plan_builder.build_session_plan_document(plan)

    # title
    assert doc.paragraphs[0].text == "SESSION PLAN"
    assert doc.paragraphs[0].runs[0].bold is True

    # portrait Letter page
    sec = doc.sections[0]
    assert round(sec.page_width.cm, 1) == 21.6
    assert round(sec.page_height.cm, 1) == 27.9

    table = doc.tables[0]
    assert table.style.name == "Table Grid"
    grid = [round(int(g.get(qn("w:w"))) / 20 / 28.3465, 2)
            for g in table._element.find(qn("w:tblGrid")).findall(qn("w:gridCol"))]
    assert grid == session_plan_builder.GRID_WIDTHS

    # the four grey section banners are present and shaded
    banner_texts = {row.cells[0].paragraphs[0].text for row in table.rows
                    if _shaded_fill(row.cells[0]) == "D9D9D9"}
    assert "Session Presentation" in banner_texts
    assert "2. Session Delivery" in banner_texts
    assert any(t.startswith("3.") for t in banner_texts)


def test_docx_delivery_rows_match_steps():
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), duration_minutes=120)
    doc = session_plan_builder.build_session_plan_document(plan)
    table = doc.tables[0]
    # step rows carry 5 <w:tc> cells (label | minutes | trainer[span2] | trainee | check)
    step_rows = [r for r in table.rows
                 if len(r._tr.findall(qn("w:tc"))) == 5
                 and r.cells[0].paragraphs[0].text.startswith("Step")]
    assert len(step_rows) == len(plan.delivery_steps)


def test_docx_multiword_resource_not_charsplit():
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), duration_minutes=120)
    doc = session_plan_builder.build_session_plan_document(plan)
    text = "\n".join(p.text for t in doc.tables for row in t.rows
                     for cell in row.cells for p in cell.paragraphs)
    assert "Kotler" in text and "Marketing Management" in text


def test_docx_has_signature_block():
    plan = ai_client.build_session_plan_offline(
        _unit(), _session(), _inputs(), duration_minutes=120)
    doc = session_plan_builder.build_session_plan_document(plan)
    roles = [p.text for p in doc.paragraphs if p.text.strip()]
    assert any(t.startswith("PREPARED BY") for t in roles)
    assert any(t.startswith("VERIFIED BY") for t in roles)
    assert any(t.startswith("APPROVED BY") for t in roles)
