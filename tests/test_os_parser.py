"""Module A1 - deterministic OS parser tests. ZERO API calls."""

import os_parser
from pdf_utils import detect_column_split


def test_all_units_parse(os_units):
    # The sample OS has 17 units of competency; every one must yield elements+PCs.
    assert len(os_units) == 17
    for u in os_units:
        assert u.unit_title and u.unit_title.isupper()
        assert u.elements, f"no elements for {u.unit_title}"
        assert u.all_pcs, f"no PCs for {u.unit_title}"


def test_programming_unit_identity(prog_os_unit):
    u = prog_os_unit
    assert u.unit_title == "APPLY COMPUTER PROGRAMMING PRINCIPLES"
    assert u.os_code == "IT/OS/ICTA/CC/02/5/MA"
    assert u.isced_code.startswith("0613 451")
    assert u.level == "6"                     # from KNQF cover, not the unit code


def test_programming_elements_and_pcs(prog_os_unit):
    u = prog_os_unit
    assert [e.number for e in u.elements] == ["1", "2", "3"]
    counts = [len(e.performance_criteria) for e in u.elements]
    assert counts == [5, 10, 8]               # 23 PCs total
    # PC numbering preserved, no cross-column bleed on the last PC
    assert u.all_pcs[0].number == "1.1"
    assert u.all_pcs[-1].number == "3.8"
    assert "Polymorphism" in u.all_pcs[-1].text
    # RANGE-section text must NOT have leaked into the final PC
    assert "environment" not in u.all_pcs[-1].text.lower()


def test_no_two_column_bleed(prog_os_unit):
    # Element titles must be clean (no PC text glued on)
    for el in prog_os_unit.elements:
        assert "as per" not in el.title.lower()
        assert len(el.title) < 70


def test_hyphenation_repair(prog_os_unit):
    assert prog_os_unit.elements[2].title == "Demonstrate object-oriented programming skills"


def test_assessment_methods(prog_os_unit):
    methods = prog_os_unit.assessment_methods
    assert "Observation" in methods
    assert any("Written" in m for m in methods)
    assert all(2 < len(m) < 60 for m in methods)


def test_skill_task_from_action_verb(prog_os_unit):
    assert prog_os_unit.skill_task.lower().startswith("apply")


def test_required_knowledge_extracted(prog_os_unit):
    # Underpinning theory list - used to enrich the OS-only fallback + AI prompt.
    rk = prog_os_unit.required_knowledge
    assert rk, "required_knowledge should not be empty"
    assert "programming" in " ".join(rk).lower()
    assert all(len(x) > 2 for x in rk)


def test_column_split_detection():
    # detect_column_split should land between the element (~73-160) and PC (~230+)
    words = [{"x0": 79, "top": 1}, {"x0": 92, "top": 1}, {"x0": 230, "top": 1},
             {"x0": 248, "top": 1}]
    split = detect_column_split(words)
    assert 160 <= split <= 240
