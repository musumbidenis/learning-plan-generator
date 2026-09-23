"""The assessment module's single AI call. ZERO real API calls - the HTTP layer
and `_chat_json` are monkeypatched, and model discovery is stubbed out too.

What is worth pinning down here is not that the call happens but what is sent
and what is done with the answer: one request per tool and never a batch, the
standing instructions in the system half and only data in the user half, the
raw generation on disk before anything is made of it, and a junk response
producing an empty paper rather than a traceback.
"""

import json
import os

import pytest

import ai_client
import assessment_ai
from ai_client import AIError
from assessment_config import (CONSTRUCTED_RESPONSE_ONLY_LEVELS,
                               MAX_CHECKLIST_ITEMS, MIN_CHECKLIST_ITEMS,
                               TEMPERATURE, VERB_BANK)
from assessment_models import (ANALYSING, APPLYING, CAT_1, CREATING,
                               EVALUATING, KNOWLEDGE, PRACTICAL, THEORY,
                               UNDERSTANDING, Allocation, AssessmentTool,
                               CatDefinition, ContentTopic, ElementContent)


class FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Nothing in this module may reach the network, model discovery included.

    `generate` resolves a model before it writes anything, which would
    otherwise ask Groq what this account may call.
    """
    monkeypatch.setattr(
        ai_client, "_post",
        lambda *a, **k: FakeResp(200, {"choices": [{"message":
                                                    {"content": "{}"}}]}))
    monkeypatch.setattr(ai_client, "list_chat_models",
                        lambda key: ["openai/gpt-oss-120b"])
    monkeypatch.setattr(assessment_ai, "resolve_model",
                        lambda key, model=None, progress_cb=None:
                            model or "openai/gpt-oss-120b")
    # Raw generations go to a throwaway directory, never the project's.
    monkeypatch.setattr(assessment_ai, "RAW_DIR", str(tmp_path / "raw"))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
ALLOCATIONS = [
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


def _tool(assessment_type=THEORY, total=20):
    return AssessmentTool(
        unit_title="Perform Basic Vehicle Servicing",
        cdacc_code="ENG/OS/AUT/CR/03/5",
        isced_code="0716 452 05A",
        knqf_level=CONSTRUCTED_RESPONSE_ONLY_LEVELS[0],
        programme="Automotive Engineering Level 5",
        cat=CatDefinition(cat_id=CAT_1, assessment_type=assessment_type,
                          total_marks=total, duration_minutes=90,
                          selected_pcs=[a[2] for a in ALLOCATIONS]),
        allocations=[Allocation(element_number=e, element_title=t,
                                pc_number=pc, pc_text=text, weight=marks,
                                marks=marks, bloom=bloom)
                     for e, t, pc, text, bloom, marks in ALLOCATIONS],
    )


WRITTEN_PAYLOAD = {
    "unit_of_competency": "Perform Basic Vehicle Servicing",
    "items": [
        {"item_number": 1,
         "element_number": "1", "pc_number": "1.1", "bloom_level": KNOWLEDGE,
         "response_type": "short_response",
         "stem": "State FOUR hand tools issued for the routine service.",
         "marks": 4,
         "marking_scheme": [{"text": "Spanner set", "marks": 1},
                            {"text": "Torque wrench", "marks": 1},
                            {"text": "Trolley jack", "marks": 1},
                            {"text": "Feeler gauge", "marks": 1}]},
        {"item_number": 2,
         "element_number": "1", "pc_number": "1.2",
         "bloom_level": UNDERSTANDING, "response_type": "short_response",
         "stem": "Explain FOUR reasons a technician wears the gear issued "
                 "before grinding.",
         "marks": 4,
         "marking_scheme": [{"text": "Prevents metal sparks reaching the eyes",
                             "marks": 1},
                            {"text": "Shields the hands from hot fragments",
                             "marks": 1},
                            {"text": "Stops loose clothing catching in the "
                                     "wheel", "marks": 1},
                            {"text": "Guards the feet against dropped "
                                     "components", "marks": 1}]},
    ],
}


PRACTICAL_PAYLOAD = {
    "task_brief": {
        "task": "Service the saloon car booked into bay two.",
        "conditions": "Working alone, in the workshop, within the time given.",
        "tools_equipment_materials": ["Spanner set", "Trolley jack"],
        "safety_requirements": ["Wear overalls and safety boots"],
        "time_allowed": "90 minutes",
    },
    "observation_checklist": [
        {"text": "Positions the wheel chocks before raising the vehicle",
         "pc_numbers": ["1.1"], "marks": 3, "sub_parts": ["front", "rear"]},
        {"text": "Selects the 13 mm spanner from the issue board",
         "pc_numbers": ["1.1"], "marks": 3, "sub_parts": []},
    ],
    "product_checklist": [
        {"text": "The sump plug is torqued to 25 Nm and shows no weep",
         "pc_numbers": ["2.1"], "marks": 4, "sub_parts": []},
    ],
    "oral_questions": [
        {"pc_numbers": ["2.2"], "question": "Why is the oil drained warm?",
         "response_indicators": ["It carries the suspended debris out"]},
    ],
}


class Recorder:
    """Stands in for `_chat_json`, remembering every call made through it."""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {}
        self.calls = []

    def __call__(self, prompt, api_key, model, schema, schema_name,
                 progress_cb=None, temperature=0.2, system=""):
        self.calls.append({"prompt": prompt, "schema": schema,
                           "schema_name": schema_name, "system": system,
                           "temperature": temperature, "model": model})
        return self.payload


def _run(monkeypatch, tool, payload):
    rec = Recorder(payload)
    monkeypatch.setattr(assessment_ai, "_chat_json", rec)
    return assessment_ai.generate(tool, api_key="k", model="m"), rec


# --------------------------------------------------------------------------- #
# One call per tool
# --------------------------------------------------------------------------- #
def test_one_assessment_tool_costs_exactly_one_call(monkeypatch):
    _, rec = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    assert len(rec.calls) == 1


def test_two_cats_are_never_written_in_one_call(monkeypatch):
    # Batching would make item independence uncheckable across the boundary,
    # so each tool goes out on its own request.
    rec = Recorder(WRITTEN_PAYLOAD)
    monkeypatch.setattr(assessment_ai, "_chat_json", rec)
    assessment_ai.generate(_tool(), api_key="k", model="m")
    assessment_ai.generate(_tool(), api_key="k", model="m")
    assert len(rec.calls) == 2
    assert all(len(c["prompt"].split("MARK ALLOCATION TABLE")) == 2
               for c in rec.calls)


def test_the_paper_is_written_cold(monkeypatch):
    _, rec = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    assert rec.calls[0]["temperature"] == TEMPERATURE


# --------------------------------------------------------------------------- #
# The system / user split
# --------------------------------------------------------------------------- #
def test_the_standing_instructions_travel_as_a_system_message(monkeypatch):
    _, rec = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    system = rec.calls[0]["system"]
    assert system == assessment_ai.AS_WRITTEN_SYSTEM
    for rule in ("marking scheme", "multiple-choice", "independently"):
        assert rule.lower() in system.lower()


def test_the_data_half_carries_data_and_no_standing_instructions(monkeypatch):
    _, rec = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    prompt = rec.calls[0]["prompt"]
    assert "Perform Basic Vehicle Servicing" in prompt
    assert "pc_number: 2.4" in prompt and "marks: 2" in prompt
    # The rules are in the system half; the data half must not repeat them.
    for rule in ("Never invent", "MARKING SCHEME", "TERMINOLOGY"):
        assert rule not in prompt


def test_the_allowed_verbs_come_from_the_verb_bank(monkeypatch):
    _, rec = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    prompt = rec.calls[0]["prompt"]
    for verb in VERB_BANK[CREATING]:
        assert verb in prompt
    # and each row is offered only its own level's bank
    creating_row = [ln for ln in prompt.splitlines()
                    if "pc_number: 2.4" in ln][0]
    assert VERB_BANK[KNOWLEDGE][0] not in creating_row


def test_a_constrained_level_is_told_it_is_constructed_response_only(monkeypatch):
    tool = _tool()
    tool.knqf_level = CONSTRUCTED_RESPONSE_ONLY_LEVELS[-1]
    _, rec = _run(monkeypatch, tool, WRITTEN_PAYLOAD)
    assert "constructed response" in rec.calls[0]["prompt"]


def test_the_practical_brief_is_told_the_checklist_bounds(monkeypatch):
    assert str(MIN_CHECKLIST_ITEMS) in assessment_ai.AS_PRACTICAL_SYSTEM
    assert str(MAX_CHECKLIST_ITEMS) in assessment_ai.AS_PRACTICAL_SYSTEM
    assert "NEVER REVEALS HOW IT IS MARKED" in assessment_ai.AS_PRACTICAL_SYSTEM


def test_the_practical_path_uses_its_own_schema_and_instructions(monkeypatch):
    _, rec = _run(monkeypatch, _tool(PRACTICAL, total=10), PRACTICAL_PAYLOAD)
    call = rec.calls[0]
    assert call["schema_name"] == "assessment_practical"
    assert call["system"] == assessment_ai.AS_PRACTICAL_SYSTEM
    assert set(call["schema"]["properties"]) == {
        "task_brief", "observation_checklist", "product_checklist",
        "oral_questions"}


def test_the_written_schema_is_the_one_the_instructions_describe(monkeypatch):
    """Section 14 of the standing instructions and the strict schema are two
    statements of one contract. Drift between them tells the model to return
    one shape and decodes it as another."""
    _, rec = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    call = rec.calls[0]
    item = call["schema"]["properties"]["items"]["items"]["properties"]

    assert call["schema_name"] == "assessment_written"
    assert set(call["schema"]["properties"]) == {"unit_of_competency", "items"}
    assert set(item) == {"item_number", "element_number", "pc_number",
                         "bloom_level", "marks", "response_type", "stem",
                         "marking_scheme"}
    # strict decoding: every declared property required, no extras
    assert call["schema"]["additionalProperties"] is False


# --------------------------------------------------------------------------- #
# Reading the answer
# --------------------------------------------------------------------------- #
def test_a_written_generation_becomes_numbered_items(monkeypatch):
    tool, _ = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    assert [i.number for i in tool.items] == [1, 2]
    assert tool.items[0].marks == 4
    assert tool.items[0].bloom == KNOWLEDGE
    assert tool.items[0].item_format == "short_response"
    assert [p.text for p in tool.items[0].marking_scheme][0] == "Spanner set"


def test_the_written_paper_carries_no_shared_scenario(monkeypatch):
    """Section 9: no one common scenario across the CAT. Where a situation is
    needed to carry a higher Bloom level it goes inside that item's own stem,
    so there is nothing for the paper to collect at the top."""
    tool, _ = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)

    assert tool.scenarios == []


def test_a_practical_generation_becomes_a_brief_and_two_checklists(monkeypatch):
    tool, _ = _run(monkeypatch, _tool(PRACTICAL, total=10), PRACTICAL_PAYLOAD)
    assert tool.task_brief.time_allowed == "90 minutes"
    assert [c.number for c in tool.observation_checklist] == [1, 2]
    # the product checklist numbers on from the observation one
    assert [c.number for c in tool.product_checklist] == [3]
    assert tool.observation_checklist[0].sub_parts == ["front", "rear"]
    assert tool.oral_questions[0].number == 1


def test_a_mark_the_model_changed_is_kept_not_quietly_corrected(monkeypatch):
    # Stamping the allocation's figure over it would leave a question written
    # to four marks sitting under a total that says six. The validators catch
    # it instead.
    payload = json.loads(json.dumps(WRITTEN_PAYLOAD))
    payload["items"][0]["marks"] = 6
    tool, _ = _run(monkeypatch, _tool(), payload)
    assert tool.items[0].marks == 6
    assert tool.allocations[0].marks == 4


# --------------------------------------------------------------------------- #
# A bad generation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [
    [],                                        # an array where an object was asked for
    "the paper could not be written",          # prose
    {"items": "four questions"},               # the right key, the wrong type
    {"items": [None, 7, "State something"]},   # rows that are not objects
    {"scenarios": {"id": "S1"}, "items": []},  # an object where an array belongs
])
def test_a_malformed_generation_does_not_take_the_run_down(monkeypatch, payload):
    tool, _ = _run(monkeypatch, _tool(), payload)
    assert tool.items == []


def test_an_item_missing_its_tags_falls_back_to_its_allocation(monkeypatch):
    tool, _ = _run(monkeypatch, _tool(),
                   {"scenarios": [], "items": [{"stem": "State FOUR tools."}]})
    item = tool.items[0]
    assert (item.pc_number, item.element_number, item.bloom) == \
        ("1.1", "1", KNOWLEDGE)
    assert item.marks == -1        # no mark arrived; not invented as the allocation's


def test_a_malformed_practical_generation_leaves_an_empty_tool(monkeypatch):
    tool, _ = _run(monkeypatch, _tool(PRACTICAL), ["nonsense"])
    assert tool.task_brief is None
    assert tool.observation_checklist == [] and tool.product_checklist == []


# --------------------------------------------------------------------------- #
# The raw generation on disk
# --------------------------------------------------------------------------- #
def test_the_raw_generation_is_kept_before_it_is_parsed(monkeypatch):
    _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    files = [f for f in os.listdir(assessment_ai.RAW_DIR) if f.endswith(".json")]
    assert len(files) == 1
    with open(os.path.join(assessment_ai.RAW_DIR, files[0]), encoding="utf-8") as fh:
        assert json.load(fh) == WRITTEN_PAYLOAD


def test_the_raw_directory_keeps_itself_out_of_git(monkeypatch):
    _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    with open(os.path.join(assessment_ai.RAW_DIR, ".gitignore"),
              encoding="utf-8") as fh:
        assert fh.read().strip() == "*"


def test_a_generation_that_could_not_be_parsed_is_still_on_disk(monkeypatch):
    # The whole point: the file is what makes a bad paper debuggable.
    _run(monkeypatch, _tool(), {"items": "four questions"})
    files = [f for f in os.listdir(assessment_ai.RAW_DIR) if f.endswith(".json")]
    assert len(files) == 1


def test_a_directory_that_cannot_be_written_does_not_fail_the_paper(monkeypatch):
    monkeypatch.setattr(assessment_ai.os, "makedirs",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    tool, _ = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)
    assert len(tool.items) == 2


def test_the_oldest_generations_are_trimmed(monkeypatch, tmp_path):
    monkeypatch.setattr(assessment_ai, "MAX_RAW_FILES", 3)
    rec = Recorder(WRITTEN_PAYLOAD)
    monkeypatch.setattr(assessment_ai, "_chat_json", rec)
    for _ in range(5):
        assessment_ai.generate(_tool(), api_key="k", model="m")
    files = [f for f in os.listdir(assessment_ai.RAW_DIR) if f.endswith(".json")]
    assert len(files) == 3


# --------------------------------------------------------------------------- #
# Refusing to start
# --------------------------------------------------------------------------- #
def test_a_cat_with_no_allocations_is_refused_before_any_call(monkeypatch):
    rec = Recorder(WRITTEN_PAYLOAD)
    monkeypatch.setattr(assessment_ai, "_chat_json", rec)
    tool = _tool()
    tool.allocations = []
    with pytest.raises(AIError):
        assessment_ai.generate(tool, api_key="k", model="m")
    assert rec.calls == []


def test_no_key_says_so_rather_than_writing_half_a_paper(monkeypatch):
    monkeypatch.setattr(assessment_ai, "load_api_key", lambda: "")
    with pytest.raises(AIError) as ei:
        assessment_ai.generate(_tool(), api_key="")
    assert "GROQ_API_KEY" in str(ei.value)


# --------------------------------------------------------------------------- #
# The curriculum's content reaches the model
# --------------------------------------------------------------------------- #
CONTENT = [
    ElementContent(
        element_number="1", element_title="Prepare for servicing",
        outcome_number="1", outcome_title="Prepare for servicing",
        duration_hours=40,
        suggested_methods=["Practical", "Written tests"],
        topics=[ContentTopic(
            number="1.1", title="Identification of servicing tools",
            key_points=["Torque wrench and its calibration",
                        "Trolley jack rated load"])]),
]


def _tool_with_content(assessment_type=THEORY):
    tool = _tool(assessment_type)
    tool.content = CONTENT
    return tool


def test_the_written_prompt_carries_the_taught_topics():
    """A PC is one general line. Without the topics behind it the model sets
    the paper from its own knowledge of the trade, and the questions land on
    things nobody taught."""
    prompt = assessment_ai.build_written_prompt(_tool_with_content())

    assert "TOPICS TAUGHT" in prompt
    assert "1.1 Identification of servicing tools" in prompt
    assert "Torque wrench and its calibration" in prompt


def test_the_practical_prompt_carries_it_too():
    prompt = assessment_ai.build_practical_prompt(
        _tool_with_content(PRACTICAL))

    assert "Trolley jack rated load" in prompt


def test_the_practical_prompt_passes_on_what_the_curriculum_suggests():
    prompt = assessment_ai.build_practical_prompt(
        _tool_with_content(PRACTICAL))

    assert "ASSESSMENT METHODS THE CURRICULUM SUGGESTS" in prompt
    assert "- Written tests" in prompt


def test_a_unit_with_no_curriculum_read_still_gets_a_prompt():
    """The section is left out entirely rather than printed empty: both
    standing instructions say what to do when it is absent."""
    prompt = assessment_ai.build_written_prompt(_tool())

    assert "CONTENT TAUGHT" not in prompt
    assert "MARK ALLOCATION TABLE" in prompt


def test_the_topics_bound_the_paper_and_the_notes_fill_it():
    """The two halves of the change: the curriculum says WHICH topics, the
    notes say what is in them."""
    written = assessment_ai.AS_WRITTEN_SYSTEM

    assert "WHERE THE CONTENT OF A QUESTION COMES FROM" in written
    assert "Do not assess a topic that is not in the TOPICS TAUGHT" in written
    assert "this is what the questions are made of" in written.lower()
    assert "Do not hand a line back" in written


def test_a_topic_nobody_covered_is_still_out_of_bounds():
    """The boundary that survives the notes becoming the content: they go
    deeper into what was taught and never add a topic to it."""
    written = assessment_ai.AS_WRITTEN_SYSTEM

    assert "never add one" in assessment_ai._knowledge_block(
        _tool_with_notes())
    assert "Do not assess a topic that is not in the TOPICS TAUGHT" in written
    assert "they never add new ones" in written
    assert "never add one" in assessment_ai.AS_PRACTICAL_SYSTEM


def test_the_paper_is_told_to_name_real_tools_and_standards():
    written = assessment_ai.AS_WRITTEN_SYSTEM

    assert "the standards and tools used" in written
    assert "the real tools, standards" in assessment_ai.AS_PRACTICAL_SYSTEM


def test_a_performance_criterion_is_a_link_not_a_source():
    """The distinction the instructions turn on: a PC says which competency an
    item belongs to; it does not licence assessing its wording."""
    assert "PERFORMANCE CRITERIA - these are NOT content" in \
        assessment_ai.AS_WRITTEN_SYSTEM


def test_the_paper_is_not_told_to_invent_a_common_scenario():
    assert "Do not create one common scenario for the whole CAT" in \
        assessment_ai.AS_WRITTEN_SYSTEM
    assert "scenario" not in assessment_ai.build_written_prompt(_tool())


# --------------------------------------------------------------------------- #
# Numbers the model echoed a label onto
# --------------------------------------------------------------------------- #
def test_a_pc_number_echoed_with_its_label_is_read_as_a_number(monkeypatch):
    """Live fault: every pc_number came back as "PC 1.1" because the
    allocation row said "PC 1.1: ...". Six items read as assessing nothing and
    three performance criteria read as never assessed."""
    payload = json.loads(json.dumps(WRITTEN_PAYLOAD))
    payload["items"][0]["pc_number"] = "PC 1.1"
    payload["items"][1]["element_number"] = "element 1"
    tool, _ = _run(monkeypatch, _tool(), payload)

    assert tool.items[0].pc_number == "1.1"
    assert tool.items[1].element_number == "1"


def test_a_number_that_was_never_labelled_is_untouched(monkeypatch):
    tool, _ = _run(monkeypatch, _tool(), WRITTEN_PAYLOAD)

    assert [i.pc_number for i in tool.items] == ["1.1", "1.2"]


def test_the_row_labels_match_the_field_names_it_asks_to_be_echoed():
    """Section 3 says echo these back verbatim, so the row labels each value
    with the JSON field it belongs in - otherwise "verbatim" includes the
    label."""
    prompt = assessment_ai.build_written_prompt(_tool())

    for field in ("element_number:", "pc_number:", "bloom_level:", "marks:"):
        assert field in prompt


def test_each_row_says_how_many_responses_to_ask_for():
    """7 marks at understanding is THREE things, not seven - the arithmetic is
    done here rather than left to the model, which got it wrong live."""
    tool = _tool()
    tool.allocations[0].bloom = UNDERSTANDING
    tool.allocations[0].marks = 7
    row = [ln for ln in assessment_ai.build_written_prompt(tool).splitlines()
           if "pc_number: 1.1" in ln][0]

    assert "ask for: 3" in row


def test_a_checklist_pc_number_echoed_with_its_label_is_read(monkeypatch):
    """The same fault as the written path, found on the practical one a run
    later: one generation returned "1.1" and the next "PC 1.1" from the same
    prompt. Every PC read as unassessed and carrying nought marks."""
    payload = json.loads(json.dumps(PRACTICAL_PAYLOAD))
    payload["observation_checklist"][0]["pc_numbers"] = ["PC 1.1"]
    tool, _ = _run(monkeypatch, _tool(PRACTICAL, total=10), payload)

    assert tool.observation_checklist[0].pc_numbers == ["1.1"]


def test_the_practical_row_labels_match_the_fields_too():
    prompt = assessment_ai.build_practical_prompt(_tool(PRACTICAL))

    assert "pc_number:" in prompt and "element_number:" in prompt


# --------------------------------------------------------------------------- #
# The practical path does not ask the model for arithmetic
# --------------------------------------------------------------------------- #
def _practical_payload(groups):
    """groups: {pc_number: [proposed marks, ...]}"""
    obs = [{"text": f"observes {pc} step {i}", "pc_numbers": [pc], "marks": m}
           for pc, marks in groups.items() for i, m in enumerate(marks)]
    return {"task_brief": {"task": "Harden the workstation."},
            "observation_checklist": obs, "product_checklist": [],
            "oral_questions": []}


def _funded(groups, allocations, total):
    tool = AssessmentTool(
        cat=CatDefinition(assessment_type=PRACTICAL, total_marks=total),
        allocations=[Allocation("1", "E", pc, "criterion", m, m)
                     for pc, m in allocations.items()])
    tool = assessment_ai.apply_practical(tool, _practical_payload(groups))
    return tool.observation_checklist + tool.product_checklist


def test_each_criterions_items_sum_to_its_allocated_marks():
    """Three live runs wrote 72, 57 and 50 marks against an allocation of 40.
    None of it was a wording fault, so no repair pass could clear it - the
    trainer simply lost the generation. The arithmetic is ours now."""
    items = _funded({"1.1": [5, 5, 5], "1.2": [7, 7], "2.1": [9, 9, 9, 9]},
                    {"1.1": 13, "1.2": 9, "2.1": 18}, 40)

    per_pc = {}
    for c in items:
        per_pc[c.pc_numbers[0]] = per_pc.get(c.pc_numbers[0], 0) + c.marks

    assert per_pc == {"1.1": 13, "1.2": 9, "2.1": 18}
    assert sum(c.marks for c in items) == 40


def test_no_item_of_evaluation_comes_out_worth_nothing():
    items = _funded({"1.1": [1, 1, 1, 1, 1]}, {"1.1": 7}, 7)

    assert [c.marks for c in items] == [2, 2, 1, 1, 1]
    assert sum(c.marks for c in items) == 7


def test_the_model_still_says_which_items_matter_most():
    """Its figures are read as relative worth, not as marks - so a step it
    called twice as important still gets more."""
    items = _funded({"1.1": [8, 2, 2]}, {"1.1": 24}, 24)

    assert items[0].marks > items[1].marks
    assert sum(c.marks for c in items) == 24


def test_proposals_of_nothing_become_an_even_split():
    items = _funded({"1.1": [0, 0, 0]}, {"1.1": 9}, 9)

    assert [c.marks for c in items] == [3, 3, 3]


def test_more_items_than_marks_leaves_some_unfunded_to_be_reported():
    """Not enough marks to go round. The rows that come out at nought are
    what `_check_unfunded_items` is for."""
    items = _funded({"1.1": [1, 1, 1, 1, 1]}, {"1.1": 3}, 3)

    assert sum(c.marks for c in items) == 3
    assert any(c.marks == 0 for c in items)


def test_an_item_tracing_to_two_criteria_keeps_the_first():
    """Every check counts an item's marks in full against each PC it names
    while the total is a plain sum, so two PCs on one item makes the two views
    of the same paper disagree by that item's marks."""
    tool = AssessmentTool(
        cat=CatDefinition(assessment_type=PRACTICAL, total_marks=10),
        allocations=[Allocation("1", "E", "1.1", "c", 10, 10)])
    tool = assessment_ai.apply_practical(tool, {
        "task_brief": {"task": "t"},
        "observation_checklist": [{"text": "does the thing",
                                   "pc_numbers": ["1.1", "2.1"], "marks": 5}],
        "product_checklist": [], "oral_questions": []})

    assert tool.observation_checklist[0].pc_numbers == ["1.1"]


def test_apportionment_never_loses_or_invents_a_mark():
    import random
    random.seed(13)
    for _ in range(400):
        weights = [random.randint(0, 9) for _ in range(random.randint(1, 8))]
        total = random.randint(0, 40)
        shares = assessment_ai._apportion(weights, total)
        assert len(shares) == len(weights)
        assert sum(shares) == max(0, total)
        assert all(x >= 0 for x in shares)


# --------------------------------------------------------------------------- #
# Reference notes, and the size of the request
# --------------------------------------------------------------------------- #
def _notes(count=6, summary_chars=300):
    from assessment_models import KnowledgeNote
    return [KnowledgeNote(
        key_point=f"Key point {n}", topic_number="1.1", element_number="1",
        summary="S" * summary_chars,
        facts=["Detection: a signature scan compares a file to known patterns",
               "Prevention: patching and least privilege"],
        covers=["Propagation", "Detection"], named=["NIST", "CVE"],
        source_title="Article") for n in range(count)]


def _tool_with_notes(assessment_type=THEORY):
    tool = _tool_with_content(assessment_type)
    tool.knowledge = _notes(2, 80)
    return tool


def test_the_written_prompt_carries_the_reference_notes():
    tool = _tool_with_content()
    tool.knowledge = _notes(2, 80)

    prompt = assessment_ai.build_written_prompt(tool)

    assert "TEACHING NOTES" in prompt
    assert "THIS IS WHAT THE QUESTIONS ARE MADE OF" in prompt
    assert "Detection: a signature scan compares a file to known patterns" \
        in prompt


def test_the_notes_are_the_content_and_the_topics_are_the_scope():
    """The whole point of the change. The notes say what a question is about;
    the topics say which topics are allowed."""
    prompt = assessment_ai.build_written_prompt(_tool_with_notes())

    assert "THIS IS WHAT THE QUESTIONS ARE MADE OF" in prompt
    assert "never add one" in prompt
    assert "TOPICS TAUGHT" in prompt


def test_a_bare_list_of_names_never_reaches_the_model():
    """It is raw material for invention. A note built from an article with no
    sections fell back to its name list and the paper asked for FOUR
    vulnerability scanning tools, marking "OSS (Open Source Scanner)" and
    "CIS scanner" - neither of which is a tool."""
    from assessment_models import KnowledgeNote
    tool = _tool_with_content()
    tool.knowledge = [KnowledgeNote(
        key_point="Vulnerability scanning tools", topic_number="1.1",
        element_number="1", summary="A scanner assesses systems for known "
        "weaknesses, and is run authenticated or unauthenticated.",
        facts=[], covers=["Overview", "Strengths"],
        named=["OSS", "CIS", "Critical Security Controls"],
        source_title="Vulnerability scanner")]

    prompt = assessment_ai.build_written_prompt(tool)

    assert "A scanner assesses systems" in prompt
    assert "Critical Security Controls" not in prompt
    assert "OSS" not in prompt


def test_each_row_is_pointed_at_the_notes_it_is_written_from():
    tool = _tool_with_notes()

    prompt = assessment_ai.build_written_prompt(tool)

    assert "write it from: N" in prompt


def test_rows_on_one_criterion_do_not_all_get_the_same_note():
    """Three rows all pointed at the same material came back as the same
    question three times."""
    tool = _tool_with_content()
    tool.knowledge = _notes(3, 60)
    for note in tool.knowledge:
        note.element_number = tool.allocations[0].element_number
    for alloc in tool.allocations:
        alloc.element_number = tool.allocations[0].element_number

    served = assessment_ai._notes_per_row(tool, 0)

    assert len({tuple(v) for v in served.values()}) > 1


def test_a_unit_with_no_notes_gets_no_notes_section():
    prompt = assessment_ai.build_written_prompt(_tool_with_content())

    assert "REFERENCE NOTES" not in prompt


def test_an_oversized_prompt_is_trimmed_to_fit():
    """Groq's tier refuses a request over its limit outright - HTTP 413, no
    paper at all - so the optional material has to be able to give way."""
    tool = _tool_with_content()
    tool.knowledge = _notes(30, 900)

    prompt = assessment_ai.build_written_prompt(tool)

    assert (len(assessment_ai.AS_WRITTEN_SYSTEM) + len(prompt)
            <= assessment_ai.PROMPT_CHAR_CEILING)
    assert "MARK ALLOCATION TABLE" in prompt      # never the part that goes


def test_trimming_never_gives_up_the_taught_topics():
    tool = _tool_with_content()
    tool.knowledge = _notes(60, 2000)

    prompt = assessment_ai.build_written_prompt(tool)

    assert "TOPICS TAUGHT" in prompt
    assert "Torque wrench and its calibration" in prompt


def test_the_budgets_can_be_asked_for_explicitly():
    tool = _tool_with_content()
    tool.knowledge = _notes(4, 200)

    none_at_all = assessment_ai.build_written_prompt(tool, -1, -1)

    assert "REFERENCE NOTES" not in none_at_all
    assert "MARK ALLOCATION TABLE" in none_at_all


def test_a_request_refused_as_too_large_is_tried_again_smaller(monkeypatch):
    """The character ceiling is a guess: the provider counts tokens, and the
    schema and its own framing alongside them. The refusal is the measurement
    that actually counts."""
    tool = _tool_with_content()
    tool.knowledge = _notes(6, 300)
    sizes = []

    def fake_chat(prompt, *a, **kw):
        sizes.append(len(prompt))
        if len(sizes) < 3:
            raise AIError("openai/gpt-oss-120b: HTTP 413 Request too large")
        return {"items": []}

    monkeypatch.setattr(assessment_ai, "_chat_json", fake_chat)

    assessment_ai._send(tool, False, "k", "m", {}, "n", "sys", None)

    assert len(sizes) == 3
    assert sizes[1] < sizes[0] and sizes[2] < sizes[1]


def test_an_error_that_is_not_about_size_is_not_retried(monkeypatch):
    calls = []

    def fake_chat(prompt, *a, **kw):
        calls.append(1)
        raise AIError("openai/gpt-oss-120b: HTTP 401 invalid api key")

    monkeypatch.setattr(assessment_ai, "_chat_json", fake_chat)

    with pytest.raises(AIError, match="401"):
        assessment_ai._send(_tool_with_content(), False, "k", "m", {}, "n",
                            "sys", None)

    assert len(calls) == 1
