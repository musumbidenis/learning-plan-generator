"""Module B0 - the PC weighting tool built from the OS. ZERO API calls.

The worked examples are the ones the method is taught with (CDACC's 2022
slides and the Understanding CBET series), so a change that breaks them has
stopped being CDACC's method.
"""

import io

from docx import Document

import assessment_docs as docs
import assessment_pc_tool as t
import assessment_weighting as weighting_parser
from models import Element, PerformanceCriterion, Unit
from pdf_utils import Page

PROCESS = "Specimens are handled as per procedure"


def _unit(pc_texts, titles=None, level="6"):
    """A unit whose element i has the PCs in pc_texts[i]."""
    elements = []
    for i, texts in enumerate(pc_texts, 1):
        elements.append(Element(
            number=str(i), title=(titles or {}).get(i, f"Element {i}"),
            performance_criteria=[PerformanceCriterion(f"{i}.{j}", text)
                                  for j, text in enumerate(texts, 1)]))
    return Unit(unit_title="TEST UNIT", os_code="X/OS/1/6/A", level=level,
                elements=elements)


def _aspects_on(pcs):
    return [t.CriticalAspect(number=str(i), text=f"aspect {i}", pcs=[pc])
            for i, pc in enumerate(pcs, 1)]


def _build(unit, aspects=(), ratio=(1, 1), factor=None):
    return t.build(unit, list(aspects), t.read_facets(unit), ratio, factor)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_tie_breaks_follow_the_video_walk_through():
    """Part 3's six-element example, rank for rank.

    Elements 1 and 5 have 3 critical aspects each, both process-only, no
    creativity: 1 has 5 PCs and 5 has 4, so they take 6 and 5. Elements 2, 3
    and 4 tie on 2: element 3 produces a product ('results') and takes 4;
    element 2 (6 PCs) then ranks above element 4 (4 PCs). Element 6 is last.
    """
    unit = _unit([
        [PROCESS] * 5,
        [PROCESS] * 6,
        [PROCESS, PROCESS, "Test results are recorded", PROCESS],
        [PROCESS] * 4,
        [PROCESS] * 4,
        [PROCESS] * 3,
    ])
    aspects = _aspects_on(["1.1", "1.2", "1.3", "5.1", "5.2", "5.3",
                           "2.1", "2.2", "3.1", "3.2", "4.1", "4.2", "6.1"])
    tool = _build(unit, aspects)
    assert [e.rank for e in tool.elements] == [6, 3, 4, 2, 5, 1]
    assert "number of PCs" in tool.elements[1].reason
    assert "product over process" in tool.elements[2].reason


def test_full_ties_share_the_ideal_rank():
    """Part 4: 5, 6, 4, 4, 4, 3 PCs and nothing else to tell them apart -
    elements 3, 4 and 5 share (4 + 3 + 2) / 3 = 3, and the ranks total 21."""
    unit = _unit([[PROCESS] * n for n in (5, 6, 4, 4, 4, 3)])
    tool = _build(unit)
    assert [e.rank for e in tool.elements] == [5, 6, 3, 3, 3, 1]
    assert tool.rank_total == 21
    assert "averaged" in tool.elements[2].reason


def test_weights_total_100_and_follow_rank_over_total():
    """weight % = rank / sum of ranks x 100, rounded to total exactly 100."""
    unit = _unit([[PROCESS] * n for n in (5, 6, 4, 4, 4, 3)])
    tool = _build(unit)
    assert sum(e.weight for e in tool.elements) == 100
    # 6 / 21 x 100 = 28.57
    assert tool.elements[1].weight in (28, 29)


def test_ranks_always_total_n_n_plus_1_over_2():
    for n in range(1, 13):
        unit = _unit([[PROCESS] * (1 + i % 3) for i in range(n)])
        assert _build(unit).rank_total == n * (n + 1) / 2


# --------------------------------------------------------------------------- #
# Factor, ratio and the share-out
# --------------------------------------------------------------------------- #
def test_factor_is_the_lowest_elements_number_of_pcs():
    """Part 4: the lowest weight (~4-5) cannot be split over its 3 PCs at
    1:1, so the factor is 3 - 300 marks, three CATs of 100."""
    unit = _unit([[PROCESS] * n for n in (5, 6, 4, 4, 4, 3)])
    tool = _build(unit)
    assert tool.factor == 3
    assert sum(e.marks for e in tool.elements) == 300
    assert "element 6" in tool.factor_reason


def test_no_factor_when_the_lowest_weight_already_shares_out():
    unit = _unit([[PROCESS] * 2, [PROCESS] * 2])
    tool = _build(unit)
    assert tool.factor == 1
    assert tool.factor_reason.startswith("No factor")


def test_a_set_factor_overrides_the_rule():
    unit = _unit([[PROCESS] * n for n in (5, 6, 4, 4, 4, 3)])
    assert _build(unit, factor=2).factor == 2


def test_every_total_adds_up():
    unit = _unit([[PROCESS] * n for n in (5, 6, 4, 4, 4, 3)])
    tool = _build(unit, ratio=(2, 3))
    for el in tool.elements:
        pcs = tool.pcs_of(el.number)
        assert el.theory + el.practical == el.marks
        assert sum(p.theory for p in pcs) == el.theory
        assert sum(p.practical for p in pcs) == el.practical
    grand = 100 * tool.factor
    assert sum(p.theory for p in tool.pcs) == grand * 2 // 5
    assert sum(p.practical for p in tool.pcs) == grand * 3 // 5


def test_knowledge_pcs_take_no_practical_and_performance_no_theory():
    """As CDACC's 2026 ICT tool does: 'Hardware devices are disassembled'
    0 theory; 'Ergonomics risk factors observed' 0 practical."""
    unit = _unit([["Hardware devices are disassembled",
                   "Ergonomic risk factors are identified",
                   "Request form is received and interpreted"]])
    tool = _build(unit)
    by = {p.number: p for p in tool.pcs}
    assert by["1.1"].nature == t.PERFORMANCE_PC and by["1.1"].theory == 0
    assert by["1.2"].nature == t.KNOWLEDGE_PC and by["1.2"].practical == 0
    assert by["1.3"].nature == t.BOTH_PC
    assert by["1.3"].theory and by["1.3"].practical


def test_critical_pcs_take_more():
    unit = _unit([[PROCESS] * 4])
    tool = _build(unit, _aspects_on(["1.2", "1.2"]))
    by = {p.number: p for p in tool.pcs}
    assert by["1.2"].theory > by["1.1"].theory
    assert by["1.2"].practical > by["1.1"].practical
    assert "critical" in by["1.2"].reason.lower()


def test_ratio_defaults_to_cdaccs_2022_table():
    assert t.default_ratio("6") == (1, 1)
    assert t.default_ratio("5") == (2, 3)
    assert t.default_ratio("4") == (3, 7)
    assert t.default_ratio("3") == (1, 4)


# --------------------------------------------------------------------------- #
# Critical aspects
# --------------------------------------------------------------------------- #
def test_aspects_map_onto_the_pc_they_paraphrase():
    unit = _unit([["Sets are represented as per workplace requirements",
                   "Set operations are applied as per workplace requirements"],
                  ["Matrices are identified as per workplace requirements",
                   "Matrix operations are applied as per workplace "
                   "requirements"]],
                 titles={1: "Apply set theory", 2: "Apply matrices"})
    mapped = t.map_aspects(unit, ["Applied set operations as per workplace "
                                  "requirements.",
                                  "Performed matrix operations."])
    assert mapped[0].pcs == ["1.2"]
    assert mapped[1].pcs == ["2.2"]


def test_an_aspect_sharing_nothing_is_left_unmapped():
    unit = _unit([["Sets are represented as per workplace requirements"]])
    assert t.map_aspects(unit, ["Operated a forklift"])[0].pcs == []


def test_extracts_critical_aspects_from_the_evidence_guide():
    import os_parser
    text = ("EVIDENCE GUIDE\n1. Critical aspects Assessment requires evidence "
            "that the candidate:\nof Competency 1.1 Applied set operations as "
            "per\nworkplace requirements.\n© 2025, TVET CDACC 26\n"
            "1.2 Performed matrix operations. © 2025, TVET CDACC 27\n"
            "2. Resource The following resources should be provided:\n"
            "Implications 2.1 Access to a workplace\n")
    got = os_parser._extract_critical_aspects([Page(index=0, text=text, words=[])])
    assert got == ["Applied set operations as per workplace requirements.",
                   "Performed matrix operations."]


# --------------------------------------------------------------------------- #
# Hand-off to the rest of the assessment path
# --------------------------------------------------------------------------- #
def test_converts_to_a_weighting_that_validates():
    unit = _unit([[PROCESS] * n for n in (5, 6, 4, 4, 4, 3)])
    weighting = _build(unit, ratio=(2, 3)).to_weighting()
    assert not [p for p in weighting_parser.validate(weighting) if p.blocking]
    assert weighting.ratio == (2, 3)
    assert len(weighting.pcs) == 26


def test_a_badly_read_os_is_refused():
    unit = _unit([[PROCESS, PROCESS], []])
    unit.elements[0].performance_criteria[1].number = "1.1"
    blocking = [p for p in t.check(unit) if p.blocking]
    assert any("no performance criteria" in p.message for p in blocking)
    assert any("read 2 times" in p.message for p in blocking)


def test_the_tool_document_builds():
    unit = _unit([["Test results are recorded", PROCESS], [PROCESS] * 3])
    tool = _build(unit, _aspects_on(["1.1"]))
    doc = Document(io.BytesIO(docs.build_weighting_tool(tool)))
    text = "\n".join(c.text for tb in doc.tables for r in tb.rows
                     for c in r.cells)
    assert "GRAND TOTAL" in text and "1.1 Test results are recorded" in text
