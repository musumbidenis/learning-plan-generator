"""The ten checks, in both directions, and the bounded repair loop.

ZERO real API calls: the repair pass's `_chat_json` is monkeypatched and the
HTTP layer under it is stubbed as well, so a missed patch fails loudly rather
than dialling out.

Each check is exercised twice - once on a paper that breaks it and once on the
clean fixture, which must stay silent. A validator that never goes quiet is as
useless as one that never fires.
"""

import copy

import pytest

import ai_client
import assessment_validators as av
from ai_client import AIError
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               MAX_CHECKLIST_ITEMS, MAX_REPAIR_PASSES,
                               MIN_CHECKLIST_ITEMS)
from assessment_models import (ANALYSING, APPLYING, CAT_1, CREATING,
                               EVALUATING, KNOWLEDGE, PRACTICAL, THEORY,
                               UNDERSTANDING, Allocation, AssessmentTool,
                               CatDefinition, ChecklistItem, ContentTopic,
                               ElementContent, Item, MarkingPoint,
                               OralQuestion, Problem, Scenario, TaskBrief)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Nothing here may reach the API, repair pass included."""
    def refuse(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(ai_client, "_post", refuse)
    monkeypatch.setattr(ai_client, "list_chat_models", refuse)


# --------------------------------------------------------------------------- #
# A paper that passes every check
# --------------------------------------------------------------------------- #
PCS = [
    ("1", "Prepare for servicing", "1.1",
     "Tools and equipment are identified according to workplace procedures",
     KNOWLEDGE, 4),
    ("1", "Prepare for servicing", "1.2",
     "Personal protective equipment is selected as per safety standards",
     UNDERSTANDING, 4),
    ("2", "Service the vehicle", "2.1",
     "Work area is prepared according to job requirements", APPLYING, 4),
    ("2", "Service the vehicle", "2.2",
     "Faults are diagnosed based on standard operating procedures",
     ANALYSING, 4),
    ("2", "Service the vehicle", "2.3",
     "Work outcomes are assessed against specifications", EVALUATING, 2),
    ("2", "Service the vehicle", "2.4",
     "Maintenance schedule is developed as per manufacturer manual",
     CREATING, 2),
]

ITEMS = [
    ("State FOUR hand tools issued at Mwea Garage for the routine service.",
     ["Spanner set", "Torque wrench", "Trolley jack", "Feeler gauge"]),
    ("Explain FOUR reasons a technician at Mwea Garage wears the gear issued "
     "before grinding.",
     ["Prevents metal sparks reaching the eyes",
      "Shields the hands from hot fragments",
      "Stops loose clothing catching in the rotating wheel",
      "Guards the feet against dropped components"]),
    ("Demonstrate the sequence followed before the saloon car is driven into "
     "bay two.",
     ["Sweeps the floor and removes oil spills",
      "Positions the chocks and the axle stands",
      "Lays out a clean bench cover for parts",
      "Sets the fire extinguisher within reach"]),
    ("Analyse the likely causes of the knocking heard by the driver at idle.",
     ["Traces the sound to worn big end bearings",
      "Rules out a loose exhaust bracket",
      "Checks the engine mounts for play",
      "Links the symptom to low oil pressure"]),
    ("Justify the decision to replace rather than resurface the brake discs.",
     ["Wear exceeds the minimum thickness stamped on the disc",
      "Resurfacing would leave less than the limit allowed"]),
    ("Design a weekly servicing routine for the two courtesy saloon cars.",
     ["Lists the daily checks against each day of the week",
      "Names who signs each check and when"]),
]


def _written() -> AssessmentTool:
    tool = AssessmentTool(
        unit_title="Perform Basic Vehicle Servicing",
        cdacc_code="ENG/OS/AUT/CR/03/5",
        knqf_level=CONSTRUCTED_RESPONSE_ONLY_LEVELS[0],
        cat=CatDefinition(cat_id=CAT_1, assessment_type=THEORY,
                          total_marks=20, duration_minutes=90),
        allocations=[Allocation(element_number=e, element_title=t,
                                pc_number=pc, pc_text=text, weight=marks,
                                marks=marks, bloom=bloom)
                     for e, t, pc, text, bloom, marks in PCS],
        scenarios=[Scenario(id="S1", title="Mwea Garage",
                            text="A saloon car is booked in for a service.")],
    )
    for n, ((stem, points), alloc) in enumerate(zip(ITEMS, tool.allocations),
                                                start=1):
        per = alloc.marks // len(points)
        tool.items.append(Item(
            number=n, element_number=alloc.element_number,
            pc_number=alloc.pc_number, bloom=alloc.bloom, stem=stem,
            marks=alloc.marks,
            marking_scheme=[MarkingPoint(text=p, marks=per) for p in points],
            item_format="short_response",
            scenario_id="S1"))
    return tool


OBSERVATION = [
    ("Positions the wheel chocks behind the rear wheels before jacking", "1.1", 3),
    ("Selects the 13 mm spanner and the torque wrench from the issue board",
     "1.1", 3),
    ("Wears the face shield before striking the seized sump plug", "1.2", 3),
    ("Keeps the drain pan under the sump for the full drain", "1.2", 2),
    ("Wipes each spill from the bay floor as it happens", "1.2", 2),
    ("Torques the sump plug in the order given on the job card", "2.1", 3),
    ("Returns every hand tool to the shadow board at the close", "2.1", 2),
]

PRODUCT = [
    ("The sump plug shows no weep after five minutes at idle", "1.1", 1),
    ("The oil level sits between the two marks on the dipstick", "2.1", 1),
    ("The job card carries the reading taken and the fitter's signature",
     "2.1", 1),
]


def _practical() -> AssessmentTool:
    wanted = {"1.1": 7, "1.2": 7, "2.1": 7}
    tool = AssessmentTool(
        unit_title="Perform Basic Vehicle Servicing",
        knqf_level=CONSTRUCTED_RESPONSE_ONLY_LEVELS[0],
        cat=CatDefinition(cat_id=CAT_1, assessment_type=PRACTICAL,
                          total_marks=21, duration_minutes=180),
        allocations=[Allocation(element_number=pc.split(".")[0],
                                element_title="Service the vehicle",
                                pc_number=pc,
                                pc_text=next(p[3] for p in PCS if p[2] == pc),
                                weight=marks, marks=marks)
                     for pc, marks in wanted.items()],
        task_brief=TaskBrief(task="Service the saloon car in bay two.",
                             conditions="Working alone in the workshop.",
                             tools_equipment_materials=["Spanner set"],
                             safety_requirements=["Wear overalls and boots"],
                             time_allowed="180 minutes"),
        oral_questions=[OralQuestion(number=1, pc_numbers=["1.2"],
                                     question="Why is the oil drained warm?",
                                     response_indicators=["It carries the "
                                                          "debris out"])],
    )
    n = 1
    for text, pc, marks in OBSERVATION:
        tool.observation_checklist.append(
            ChecklistItem(number=n, text=text, pc_numbers=[pc], marks=marks))
        n += 1
    for text, pc, marks in PRODUCT:
        tool.product_checklist.append(
            ChecklistItem(number=n, text=text, pc_numbers=[pc], marks=marks))
        n += 1
    return tool


def _checks(problems):
    return {p.check for p in problems}


# --------------------------------------------------------------------------- #
# Silence on a clean paper
# --------------------------------------------------------------------------- #
def test_a_written_paper_that_follows_the_rules_reports_nothing():
    assert av.validate(_written()) == []


def test_a_practical_tool_that_follows_the_rules_reports_nothing():
    assert av.validate(_practical()) == []


# --------------------------------------------------------------------------- #
# 1 PC coverage
# --------------------------------------------------------------------------- #
def test_an_allocated_pc_that_no_item_assesses_is_caught():
    tool = _written()
    tool.items[3].pc_number = "1.1"
    problems = av.validate(tool)
    assert av.PC_COVERAGE in _checks(problems)
    assert any("2.2" in p.message for p in problems if p.check == av.PC_COVERAGE)


def test_a_practical_pc_reaches_coverage_through_an_oral_question():
    tool = _practical()
    tool.allocations.append(Allocation(element_number="2", element_title="x",
                                       pc_number="1.2", pc_text="y",
                                       weight=0, marks=0))
    assert av.PC_COVERAGE not in _checks(av.validate(tool))


def test_pc_coverage_cannot_be_repaired_by_rewording():
    tool = _written()
    tool.items[3].pc_number = "1.1"
    covered = [p for p in av.validate(tool) if p.check == av.PC_COVERAGE]
    assert covered and not any(p.repairable for p in covered)
    assert all(p.blocking for p in covered)


# --------------------------------------------------------------------------- #
# 2 Mark fidelity
# --------------------------------------------------------------------------- #
def test_an_item_written_to_the_wrong_marks_is_caught():
    tool = _written()
    tool.items[0].marks = 6
    assert av.MARK_FIDELITY in _checks(av.validate(tool))


def test_an_item_tagged_to_the_wrong_level_is_caught():
    tool = _written()
    tool.items[5].bloom = KNOWLEDGE
    assert av.MARK_FIDELITY in _checks(av.validate(tool))


def test_a_missing_item_is_reported_as_unmatchable_rather_than_guessed():
    tool = _written()
    tool.items.pop()
    problems = [p for p in av.validate(tool) if p.check == av.MARK_FIDELITY]
    assert problems and "cannot be matched" in problems[0].message


def test_practical_marks_are_checked_against_each_pcs_total():
    tool = _practical()
    tool.observation_checklist[0].marks = 5
    problems = [p for p in av.validate(tool) if p.check == av.MARK_FIDELITY]
    assert problems and problems[0].where == "1.1"


def test_mark_fidelity_is_not_repairable():
    tool = _written()
    tool.items[0].marks = 6
    assert not any(p.repairable for p in av.validate(tool)
                   if p.check == av.MARK_FIDELITY)


# --------------------------------------------------------------------------- #
# 3 Total reconciliation
# --------------------------------------------------------------------------- #
def test_a_paper_that_does_not_total_to_the_cat_is_caught():
    tool = _written()
    tool.cat.total_marks = 25
    problems = [p for p in av.validate(tool)
                if p.check == av.TOTAL_RECONCILIATION]
    assert problems and "out of 25" in problems[0].message


def test_a_marking_scheme_that_does_not_sum_to_its_item_is_caught():
    tool = _written()
    tool.items[0].marking_scheme[0].marks = 2
    problems = [p for p in av.validate(tool)
                if p.check == av.TOTAL_RECONCILIATION]
    assert problems and problems[0].item_number == 1


def test_a_practical_tool_that_does_not_total_is_caught():
    tool = _practical()
    tool.cat.total_marks = 30
    assert av.TOTAL_RECONCILIATION in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# 4 Bloom conformance
# --------------------------------------------------------------------------- #
def test_a_lead_verb_from_another_level_is_caught():
    tool = _written()
    tool.items[0].stem = "Analyse the hand tools issued at Mwea Garage."
    problems = [p for p in av.validate(tool)
                if p.check == av.BLOOM_CONFORMANCE]
    assert problems and "analysing verb" in problems[0].message
    assert problems[0].repairable and not problems[0].blocking


def test_a_verb_in_no_bank_at_all_is_caught():
    tool = _written()
    tool.items[0].stem = "Ponder the hand tools issued at Mwea Garage."
    problems = [p for p in av.validate(tool)
                if p.check == av.BLOOM_CONFORMANCE]
    assert problems and "in no verb bank" in problems[0].message


def test_a_two_word_verb_from_the_bank_passes():
    tool = _written()
    tool.items[2].stem = ("Carry out the sequence followed before the saloon "
                          "car is driven into bay two.")
    assert av.BLOOM_CONFORMANCE not in _checks(av.validate(tool))


def test_an_item_the_model_numbered_itself_still_finds_its_verb():
    tool = _written()
    tool.items[0].stem = "1. " + tool.items[0].stem
    assert av.BLOOM_CONFORMANCE not in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# 5 Bloom completeness
# --------------------------------------------------------------------------- #
def test_a_paper_that_misses_a_level_is_caught():
    tool = _written()
    tool.items[5].bloom = EVALUATING
    problems = [p for p in av.validate(tool)
                if p.check == av.BLOOM_COMPLETENESS]
    assert problems and CREATING in problems[0].message
    # an allocation decision, so not something a rewrite can put right
    assert not problems[0].repairable and problems[0].blocking


def test_a_practical_tool_is_not_asked_to_span_the_levels():
    assert av.BLOOM_COMPLETENESS not in _checks(av.validate(_practical()))


# --------------------------------------------------------------------------- #
# 6 Item independence
# --------------------------------------------------------------------------- #
def test_a_stem_carrying_another_items_marking_key_is_caught():
    tool = _written()
    tool.items[2].stem = ("Demonstrate the use of the spanner set, torque "
                          "wrench, trolley jack and feeler gauge in the bay.")
    problems = [p for p in av.validate(tool)
                if p.check == av.ITEM_INDEPENDENCE]
    assert problems
    assert problems[0].item_number == 3 and "item 1" in problems[0].message
    assert problems[0].repairable


def test_sharing_the_units_own_vocabulary_is_not_a_leak():
    # Every item on a unit talks about its tools, equipment and procedures.
    tool = _written()
    tool.items[2].stem = ("Demonstrate how tools and equipment are identified "
                          "according to workplace procedures in bay two.")
    assert av.ITEM_INDEPENDENCE not in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# 7 Stem clue detection
# --------------------------------------------------------------------------- #
def test_a_stem_that_gives_away_its_own_answer_is_caught():
    tool = _written()
    tool.items[0].stem = ("State FOUR hand tools issued: the spanner set, the "
                          "torque wrench, the trolley jack and the feeler "
                          "gauge.")
    problems = [p for p in av.validate(tool) if p.check == av.STEM_CLUE]
    assert problems and problems[0].item_number == 1
    assert problems[0].repairable and not problems[0].blocking


def test_an_item_with_no_marking_key_is_not_accused_of_telegraphing():
    tool = _written()
    tool.items[0].marking_scheme = []
    assert av.STEM_CLUE not in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# 8 Practical item count
# --------------------------------------------------------------------------- #
def test_too_few_items_of_evaluation_is_caught():
    tool = _practical()
    tool.observation_checklist = tool.observation_checklist[:2]
    problems = [p for p in av.validate(tool)
                if p.check == av.PRACTICAL_ITEM_COUNT]
    assert problems and "only 5" in problems[0].message
    assert not problems[0].repairable


def test_too_many_items_of_evaluation_is_caught():
    tool = _practical()
    extra = MAX_CHECKLIST_ITEMS + 1 - len(tool.observation_checklist) \
        - len(tool.product_checklist)
    for i in range(extra):
        tool.product_checklist.append(
            ChecklistItem(number=100 + i, text=f"Checks something {i}",
                          pc_numbers=["2.1"], marks=0))
    assert av.PRACTICAL_ITEM_COUNT in _checks(av.validate(tool))


def test_sub_parts_do_not_count_towards_the_item_total():
    tool = _practical()
    assert len(tool.observation_checklist) + len(tool.product_checklist) \
        >= MIN_CHECKLIST_ITEMS
    for c in tool.observation_checklist:
        c.sub_parts = ["stage one", "stage two", "stage three"]
    assert av.PRACTICAL_ITEM_COUNT not in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# 9 Item of evaluation != performance criterion
# --------------------------------------------------------------------------- #
def test_an_item_of_evaluation_that_restates_its_pc_is_caught():
    tool = _practical()
    tool.observation_checklist[0].text = (
        "Tools and equipment are identified according to workplace procedures")
    problems = [p for p in av.validate(tool) if p.check == av.ITEM_NOT_PC]
    assert problems and problems[0].item_number == 1
    assert problems[0].repairable


def test_a_concrete_item_of_evaluation_on_the_same_pc_passes():
    assert av.ITEM_NOT_PC not in _checks(av.validate(_practical()))


# --------------------------------------------------------------------------- #
# 10 Format compliance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stem", [
    "Select the correct spanner from the following list of hand tools.",
    "State whether the following statement is true or false.",
    "Match the tools in column A with their uses in column B.",
    "State the tool used to loosen a sump plug: ________",
])
def test_a_selected_response_item_at_a_constrained_level_is_caught(stem):
    tool = _written()
    tool.items[0].stem = stem
    assert av.FORMAT_COMPLIANCE in _checks(av.validate(tool))


def test_an_item_declaring_a_selected_response_format_is_caught():
    tool = _written()
    tool.items[0].item_format = "multiple_choice"
    problems = [p for p in av.validate(tool)
                if p.check == av.FORMAT_COMPLIANCE]
    assert problems and problems[0].repairable


def test_the_format_rule_applies_only_at_the_constrained_levels():
    tool = _written()
    tool.knqf_level = "3"
    assert "3" not in CONSTRUCTED_RESPONSE_ONLY_LEVELS
    tool.items[0].item_format = "multiple_choice"
    assert av.FORMAT_COMPLIANCE not in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# Housekeeping the caller relies on
# --------------------------------------------------------------------------- #
def test_every_problem_is_a_problem_the_rest_of_the_app_understands():
    tool = _written()
    tool.items[0].marks = 6
    tool.items[0].stem = "Ponder the tools."
    problems = av.validate(tool)
    assert all(isinstance(p, Problem) for p in problems)
    assert all(p.message for p in problems)


def test_blocking_follows_repairability():
    tool = _written()
    tool.items[0].marks = 6
    tool.items[0].stem = "Ponder the tools."
    for p in av.validate(tool):
        assert p.blocking is not p.repairable


def test_a_check_that_throws_does_not_silence_the_others(monkeypatch):
    def explode(tool):
        raise ValueError("boom")
    monkeypatch.setattr(av, "_CHECKS", (explode, av._check_totals))
    tool = _written()
    tool.cat.total_marks = 25
    assert av.TOTAL_RECONCILIATION in _checks(av.validate(tool))


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #
class Repairer:
    """Stands in for `_chat_json` on the repair path."""

    def __init__(self, fix=True):
        self.fix = fix
        self.prompts = []

    def __call__(self, prompt, api_key, model, schema, schema_name,
                 progress_cb=None, temperature=0.2, system=""):
        self.prompts.append(prompt)
        if not self.fix:
            # A model that hands back the same faulty item, pass after pass.
            return {"items": [{"number": 1,
                               "stem": "Ponder the hand tools issued.",
                               "item_format": "short_response",
                               "marking_scheme": []}]}
        return {"items": [{"number": 1,
                           "stem": "List FOUR hand tools issued at Mwea "
                                   "Garage for the routine service.",
                           "item_format": "short_response",
                           "marking_scheme": []}]}


def _broken() -> AssessmentTool:
    tool = _written()
    tool.items[0].stem = "Ponder the hand tools issued at Mwea Garage."
    return tool


def test_only_the_failing_item_is_sent_back_and_the_rest_are_frozen(monkeypatch):
    rep = Repairer()
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _broken()
    av.repair(tool, av.validate(tool), api_key="k", model="m")
    prompt = rep.prompts[0]
    assert "ITEM 1 - element 1" in prompt and "WHAT IS WRONG" in prompt
    assert "ITEM 2 (frozen" in prompt and "ITEM 6 (frozen" in prompt
    assert "Return the 1 corrected item(s) now" in prompt


def test_the_violation_text_travels_with_the_item(monkeypatch):
    rep = Repairer()
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _broken()
    av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert "in no verb bank" in rep.prompts[0]


def test_a_corrected_item_lands_and_the_frozen_ones_do_not_move(monkeypatch):
    monkeypatch.setattr(av, "_chat_json", Repairer())
    tool = _broken()
    before = [i.stem for i in tool.items[1:]]
    tool = av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert tool.items[0].stem.startswith("List FOUR")
    assert [i.stem for i in tool.items[1:]] == before


def test_a_cleared_fault_stops_the_loop_after_one_pass(monkeypatch):
    rep = Repairer()
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _broken()
    av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert len(rep.prompts) == 1
    assert av.BLOOM_CONFORMANCE not in _checks(av.validate(tool))


def test_the_loop_gives_up_after_max_repair_passes(monkeypatch):
    rep = Repairer(fix=False)
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _broken()
    tool = av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert len(rep.prompts) == MAX_REPAIR_PASSES
    # and what is left comes back for the user to edit by hand
    assert av.BLOOM_CONFORMANCE in _checks(av.validate(tool))


def test_an_item_that_was_not_asked_for_is_ignored(monkeypatch):
    def meddle(prompt, api_key, model, schema, schema_name, progress_cb=None,
               temperature=0.2, system=""):
        return {"items": [{"number": 4, "stem": "Rewritten without being asked",
                           "item_format": "short_response",
                           "marking_scheme": []}]}
    monkeypatch.setattr(av, "_chat_json", meddle)
    tool = _broken()
    fourth = tool.items[3].stem
    tool = av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert tool.items[3].stem == fourth


def test_a_failed_repair_keeps_the_paper_we_already_have(monkeypatch):
    def fail(*a, **k):
        raise AIError("no model answered")
    monkeypatch.setattr(av, "_chat_json", fail)
    tool = _broken()
    tool = av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert tool.items[0].stem.startswith("Ponder")


@pytest.mark.parametrize("payload", [None, [], "sorry", {"items": "one"},
                                     {"items": [None, 3]}])
def test_a_malformed_repair_answer_does_not_take_the_run_down(monkeypatch,
                                                              payload):
    monkeypatch.setattr(av, "_chat_json",
                        lambda *a, **k: copy.deepcopy(payload))
    tool = _broken()
    tool = av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert len(tool.items) == 6


def test_a_clean_paper_costs_no_repair_call(monkeypatch):
    rep = Repairer()
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _written()
    av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert rep.prompts == []


def test_only_repairable_faults_are_sent_to_the_model(monkeypatch):
    rep = Repairer()
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _written()
    tool.items[0].marks = 6              # not repairable: no call for it alone
    av.repair(tool, av.validate(tool), api_key="k", model="m")
    assert rep.prompts == []


def test_a_practical_tools_items_of_evaluation_are_repaired_too(monkeypatch):
    """An item that restates its PC is the commonest thing wrong with a CDACC
    checklist, and the whole reason the check exists. Reporting it for hand
    editing would leave the tool failing the rule it was written to enforce."""
    rep = Repairer()
    monkeypatch.setattr(av, "_chat_json", rep)
    tool = _practical()
    tool.observation_checklist[0].text = (
        "Tools and equipment are identified according to workplace procedures")
    problems = av.validate(tool)
    assert av.ITEM_NOT_PC in _checks(problems)

    av.repair(tool, problems, api_key="k", model="m")

    assert rep.prompts, "the failing item was never sent back"
    assert "WHAT IS WRONG" in rep.prompts[0]


def test_a_repaired_item_of_evaluation_keeps_its_marks_and_its_pcs(monkeypatch):
    """The rewrite may change what the assessor READS, never what they are
    told to inspect or what it is worth."""
    tool = _practical()
    tool.observation_checklist[0].text = (
        "Tools and equipment are identified according to workplace procedures")
    marks = tool.observation_checklist[0].marks
    pcs = list(tool.observation_checklist[0].pc_numbers)
    number = tool.observation_checklist[0].number

    landed = av._apply_practical_repair(
        tool, {"items": [{"number": number, "item": "Selects the spanner set "
                                                    "and torque wrench before "
                                                    "starting"}]},
        {number: ["repeats its PC"]})

    assert landed == 1
    assert tool.observation_checklist[0].marks == marks
    assert tool.observation_checklist[0].pc_numbers == pcs
    assert "spanner" in tool.observation_checklist[0].text


def test_an_item_of_evaluation_nobody_asked_about_is_not_rewritten():
    """A model that decides to improve a frozen item cannot."""
    tool = _practical()
    frozen = tool.product_checklist[0].text
    number = tool.product_checklist[0].number

    av._apply_practical_repair(
        tool, {"items": [{"number": number, "item": "something else"}]}, {})

    assert tool.product_checklist[0].text == frozen


def test_the_helpers_split_the_problems_the_way_the_ui_needs():
    tool = _written()
    tool.items[0].marks = 6
    tool.items[0].stem = "Ponder the tools."
    problems = av.validate(tool)
    assert av.repairable(problems) and av.blocking(problems)
    assert set(av.repairable(problems)).isdisjoint(av.blocking(problems))


# --------------------------------------------------------------------------- #
# What a repair is allowed to know
# --------------------------------------------------------------------------- #
def test_a_repair_is_shown_the_taught_content():
    """Otherwise the rewrite is made from the model's own knowledge of the
    trade - the exact fault the content was introduced to prevent, at the one
    point where nobody reads the wording again afterwards."""
    tool = _written()
    tool.content = [ElementContent(
        element_number="1", element_title="Prepare for servicing",
        topics=[ContentTopic(number="1.1", title="Servicing tools",
                             key_points=["Torque wrench calibration"])])]

    prompt = av.build_repair_prompt(tool, {1: ["wrong verb"]})

    assert "CONTENT TAUGHT" in prompt
    assert "Torque wrench calibration" in prompt


def test_a_repair_without_any_content_still_builds_a_prompt():
    prompt = av.build_repair_prompt(_written(), {1: ["wrong verb"]})

    assert "CONTENT TAUGHT" not in prompt
    assert "ITEMS TO CORRECT" in prompt


def test_a_repair_answer_speaks_the_same_field_names_as_the_paper():
    """`response_type`, not `item_format`: the repair schema and the written
    schema describe the same item to the same model."""
    tool = _written()

    av._apply_repair(tool, {"items": [{"number": 1, "stem": "State FOUR tools.",
                                       "response_type": "extended_response"}]},
                     {1: ["wrong format"]})

    assert tool.items[0].item_format == "extended_response"
