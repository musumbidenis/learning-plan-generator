"""Module B - deterministic planner tests + AI safety-net tests. ZERO API calls."""

import ai_client
import planner
from models import PlanInputs, Session, Unit


def test_one_session_per_subtopic(prog_curr_unit, prog_os_unit):
    sessions = planner.build_content_sessions(prog_curr_unit, prog_os_unit)
    assert len(sessions) == 23
    # PC mapping: sub-topic number aligns 1:1 with the OS PC number
    assert sessions[0].pcs[0].startswith("1.1 ")
    assert sessions[0].session_title == "Identification of Programming Languages"
    assert sessions[-1].pcs[0].startswith("3.8 ")


def test_schedule_spans_exact_term_weeks(prog_curr_unit, prog_os_unit):
    inp = PlanInputs(term_weeks=13, cat_weeks=[4, 8, 13], sessions_per_week=2)
    sessions = planner.plan_sessions(prog_curr_unit, prog_os_unit, inp)
    weeks = sorted({s.week for s in sessions})
    assert weeks[0] == 1 and weeks[-1] == 13
    assert all(s.session_no == "1&2" for s in sessions)


def test_cats_placed_at_configured_weeks(prog_curr_unit, prog_os_unit):
    inp = PlanInputs(term_weeks=13, cat_weeks=[4, 8, 13], sessions_per_week=2)
    sessions = planner.plan_sessions(prog_curr_unit, prog_os_unit, inp)
    cat_weeks = sorted(s.week for s in sessions if s.is_cat)
    assert cat_weeks == [4, 8, 13]
    # final-week CAT must be the LAST row of its week (the final exam)
    final = [s for s in sessions if s.week == 13]
    assert final[-1].is_cat
    # all content sessions preserved (23) plus 3 CATs
    assert sum(1 for s in sessions if not s.is_cat) == 23
    assert sum(1 for s in sessions if s.is_cat) == 3


def test_no_content_dropped_on_tight_term(prog_curr_unit, prog_os_unit):
    # Fewer weeks than needed: overflow must NOT drop any content session.
    inp = PlanInputs(term_weeks=8, cat_weeks=[8], sessions_per_week=2)
    sessions = planner.plan_sessions(prog_curr_unit, prog_os_unit, inp)
    assert sum(1 for s in sessions if not s.is_cat) == 23


# --------------------------------------------------------------------------- #
# AI safety nets (the char-splitting + coercion bugs)
# --------------------------------------------------------------------------- #
def test_as_list_string_not_char_split():
    # A bare string must become a ONE-element list, never iterated per character.
    out = ai_client._as_list("Group Discussion")
    assert out == ["Group Discussion"]
    assert "G" not in out


def test_as_list_multiline_string():
    out = ai_client._as_list("line one\nline two\n")
    assert out == ["line one", "line two"]


def test_as_list_dict_and_nested():
    out = ai_client._as_list({"Knowledge": ["a", "b"]})
    assert out == ["Knowledge: a; b"]
    assert ai_client._as_list(["x", ["y", "z"]]) == ["x", "y", "z"]


def test_merge_backfills_every_cell(prog_curr_unit, prog_os_unit):
    inp = PlanInputs(term_weeks=13, cat_weeks=[4, 8, 13])
    sessions = planner.plan_sessions(prog_curr_unit, prog_os_unit, inp)
    merged = ai_client.merge_ai_into_sessions(sessions, [], prog_os_unit)  # empty AI
    for s in merged:
        assert s.learning_outcomes
        assert s.key_points
        assert len(s.trainee_activities) >= 3
        assert len(s.resources) >= 2
        assert s.assessments


def test_merge_does_not_override_schedule(prog_curr_unit, prog_os_unit):
    inp = PlanInputs(term_weeks=13, cat_weeks=[4, 8, 13])
    sessions = planner.plan_sessions(prog_curr_unit, prog_os_unit, inp)
    before = [(s.week, s.session_no, s.is_cat) for s in sessions]
    # AI tries to change week/is_cat - must be ignored
    rogue = [{"week": 99, "is_cat": True, "session_no": "X",
              "learning_outcomes": ["a."], "key_points": ["K"],
              "trainee_activities": ["- a", "- b", "- c"],
              "resources": ["- r1", "- r2"], "assessments": ["Knowledge Checks:"]}
             for _ in sessions]
    merged = ai_client.merge_ai_into_sessions(sessions, rogue, prog_os_unit)
    after = [(s.week, s.session_no, s.is_cat) for s in merged]
    assert before == after
