"""The trainer's own material: reading it, choosing from it, failing on it.

Nothing here touches the network. A resource reader that needs the internet to
be tested is a resource reader nobody can trust offline.
"""
import io

import pytest

import assessment_resources as R
from assessment_models import ContentTopic, ElementContent

NOTES = """# ICT security threats

A virus needs a host file and runs when that file is opened. A worm needs no
host and copies itself across the network on its own.

# Assessing vulnerabilities

We use OpenVAS in the lab and Nessus on the college network. Always run the
scan with credentials where you have them.
"""


def _content():
    return [ElementContent(
        element_number="1", element_title="Identify threats", topics=[
            ContentTopic("1.1", "Threats", ["Types of malware: virus, worm"]),
            ContentTopic("1.2", "Vulnerabilities",
                         ["Vulnerability scanning tools"])])]


# --------------------------------------------------------------------------- #
# Words have to meet before anything else can work
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word,stem", [
    ("scanning", "scan"), ("scanners", "scan"), ("scan", "scan"),
    ("vulnerabilities", "vulnerability"), ("threats", "threat"),
    ("planned", "plan"), ("malware", "malware"),
])
def test_a_words_forms_are_reduced_until_they_meet(word, stem):
    """The first version compared whole words and quietly failed: a topic
    saying "Vulnerability SCANNING tools" scored zero against a page saying
    "the SCANNERS used in the workshop", and the one resource that answered
    the question was dropped from the prompt."""
    assert R._stem(word) == stem


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_markdown_is_cut_at_its_own_headings():
    resource = R.read("notes.md", NOTES.encode("utf-8"))

    assert resource.ok
    assert [c.heading for c in resource.chunks] == [
        "ICT security threats", "Assessing vulnerabilities"]


def test_a_slide_deck_is_one_chunk_per_slide_with_its_notes():
    """A deck is already chunked, and by its author. Speaker notes are taken
    too - they are usually the half the trainer actually says out loud."""
    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Types of malware"
    slide.placeholders[1].text = "Virus\nWorm"
    slide.notes_slide.notes_text_frame.text = (
        "A virus needs a host file; a worm spreads across the network by "
        "itself without needing one at all.")
    buffer = io.BytesIO()
    deck.save(buffer)

    resource = R.read("lecture.pptx", buffer.getvalue())

    assert len(resource.chunks) == 1
    assert resource.chunks[0].where == "slide 1"
    assert resource.chunks[0].heading == "Types of malware"
    assert "needs a host file" in resource.chunks[0].text     # the notes
    assert resource.kind == "slides"


def test_a_long_passage_is_cut_at_a_sentence():
    body = ("This sentence is about earthing and bonding in an installation. "
            * 40).encode("utf-8")

    resource = R.read("long.txt", body)

    assert len(resource.chunks) > 1
    for chunk in resource.chunks:
        assert len(chunk.text) <= R.CHUNK_CHARS + 2


# --------------------------------------------------------------------------- #
# Failing is ordinary, and it says why
# --------------------------------------------------------------------------- #
def test_a_format_that_cannot_be_read_says_so_by_name():
    """The trainer must see WHICH resource was not used, not wonder why the
    paper ignored it."""
    resource = R.read("diagram.png", b"\x89PNG")

    assert not resource.ok
    assert "not a format this reads" in resource.note


def test_a_corrupt_file_costs_that_file_and_not_the_paper():
    resource = R.read("broken.pdf", b"not a pdf at all")

    assert not resource.ok
    assert resource.note                      # the reason is kept
    assert R.select([resource], _content()) == []


def test_an_oversized_recording_is_refused_with_advice(monkeypatch):
    monkeypatch.setattr(R, "TRANSCRIBE_LIMIT_MB", 1)

    resource = R.read("lecture.mp4", b"x" * (2 * 1024 * 1024))

    assert not resource.ok
    assert "MB" in resource.note


# --------------------------------------------------------------------------- #
# Choosing what the prompt carries
# --------------------------------------------------------------------------- #
def test_every_assessed_topic_gets_material_before_any_gets_seconds():
    """Taking the best pieces overall would give the whole budget to whichever
    topic the notes dwell on, and every other item would be written from
    nothing."""
    resource = R.read("notes.md", NOTES.encode("utf-8"))

    chosen = R.select([resource], _content())

    assert len(chosen) == 2
    assert any("virus" in c.text.lower() for c in chosen)
    assert any("OpenVAS" in c.text for c in chosen)


def test_the_budget_is_respected():
    resource = R.read("notes.md", (NOTES * 6).encode("utf-8"))

    chosen = R.select([resource], _content(), budget=500)

    assert chosen
    assert sum(len(c.text) for c in chosen) <= 500 + R.CHUNK_CHARS


def test_material_that_matches_nothing_is_still_used():
    """A paper written from off-topic notes is visible to the trainer; one
    written from nothing is not."""
    resource = R.read("other.md", b"# Baking\n\nProving dough needs warmth "
                                  b"and time, and a covered bowl helps it.\n")

    chosen = R.select([resource], _content())

    assert chosen


def test_with_no_curriculum_the_trainers_own_order_is_kept():
    resource = R.read("notes.md", NOTES.encode("utf-8"))

    chosen = R.select([resource], [])

    assert [c.heading for c in chosen] == [c.heading for c in resource.chunks]


def test_nothing_attached_is_nothing_chosen():
    assert R.select([], _content()) == []
    assert R.render([]) == ""
    assert R.summarise([], []) == "no resources attached"


# --------------------------------------------------------------------------- #
# Which gaps are left for the lookup
# --------------------------------------------------------------------------- #
def test_a_topic_the_trainer_covers_needs_no_lookup():
    resource = R.read("notes.md", NOTES.encode("utf-8"))

    assert R.covered(resource.chunks, "Vulnerability scanning tools")
    assert R.covered(resource.chunks, "Types of malware: virus, worm")


def test_a_topic_the_trainer_misses_is_a_gap():
    resource = R.read("notes.md", NOTES.encode("utf-8"))

    assert not R.covered(resource.chunks, "Fire extinguisher classes")
    assert not R.covered(resource.chunks, "")


# --------------------------------------------------------------------------- #
# What the model reads
# --------------------------------------------------------------------------- #
def test_the_render_says_where_each_piece_came_from():
    """A trainer reading a question must be able to find the page it was
    written from."""
    resource = R.read("Class notes.md", NOTES.encode("utf-8"))

    rendered = R.render(R.select([resource], _content()))

    assert "FROM: Class notes.md" in rendered
    assert "ICT security threats" in rendered


def test_fit_trims_again_without_reordering():
    resource = R.read("notes.md", (NOTES * 4).encode("utf-8"))
    chosen = R.select([resource], _content())

    smaller = R.fit(chosen, 400)

    assert smaller == chosen[:len(smaller)]
    assert len(smaller) < len(chosen)
