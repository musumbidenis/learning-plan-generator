"""Module A2 - deterministic curriculum parser tests. ZERO API calls."""

import curriculum_parser as cp


def test_units_parse(curr_units):
    assert len(curr_units) >= 10
    for u in curr_units:
        assert u.unit_title
        assert u.learning_outcomes


def test_match_by_isced(curr_units, prog_curr_unit):
    assert prog_curr_unit.unit_title == "COMPUTER PROGRAMMING PRINCIPLES"
    assert prog_curr_unit.curriculum_code == "IT/CU/ICTA/CC/02/5/MA"


def test_match_by_loose_os_code(curr_units):
    # OS code (IT/OS/...) must resolve to the curriculum unit (IT/CU/...)
    u = cp.find_unit(curr_units, os_code="IT/OS/ICTA/CC/02/5/MA")
    assert u is not None
    assert u.curriculum_code == "IT/CU/ICTA/CC/02/5/MA"


def test_learning_outcomes_and_subtopics(prog_curr_unit):
    los = prog_curr_unit.learning_outcomes
    assert [lo.number for lo in los] == ["1", "2", "3"]
    counts = [len(lo.sub_topics) for lo in los]
    assert counts == [5, 10, 8]               # 23 sessions worth of sub-topics
    assert los[0].title == "Apply computer programming skills"


def test_subtopic_titles_clean(prog_curr_unit):
    st = prog_curr_unit.learning_outcomes[0].sub_topics[0]
    assert st.number == "1.1"
    assert st.title == "Identification of Programming Languages"


def test_key_points_no_footer_bleed(prog_curr_unit):
    # The '© TVET CDACC, 2025' footer year must not contaminate key points.
    for lo in prog_curr_unit.learning_outcomes:
        for st in lo.sub_topics:
            for kp in st.key_points:
                assert "2025" not in kp
                assert "•" not in kp


def test_key_points_content(prog_curr_unit):
    st11 = prog_curr_unit.learning_outcomes[0].sub_topics[0]
    joined = " ".join(st11.key_points).lower()
    assert "overview of programming language categories" in joined
    assert "criteria for selecting languages" in joined


def test_suggested_methods_merged(prog_curr_unit):
    methods = prog_curr_unit.learning_outcomes[0].suggested_methods
    assert "Portfolio of Evidence" in methods   # wrapped 2-line entry merged
    assert "Practical Activities" in methods
    assert "Methods" not in methods             # column header dropped
