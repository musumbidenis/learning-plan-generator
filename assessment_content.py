"""What the trainees were taught, matched onto what is being assessed.

A performance criterion is one general line - "Programming languages are
identified as per the task requirements". Ask a model to set a question from
that alone and it writes from its own knowledge of the trade: it picks the
languages, the tools and the standards it happens to know, and the paper drifts
away from the syllabus the trainees actually sat through. Every unusable item
this module exists to prevent has the same shape - plausible, well written, and
about something nobody taught.

The curriculum has the missing half. Under each learning outcome it lists the
sub-topics and, under those, the key points: the real body of knowledge the
unit covers. Handing that to the model alongside the PCs is what makes a paper
deep instead of generic, and it is also what makes hallucination unnecessary -
there is no gap left for the model to fill from memory.

THE MATCH
The occupational standard and the curriculum are two documents written by
different committees, so an element and its learning outcome are joined by
their titles, not by their numbering. They are usually word-for-word identical,
which is why a similarity match is reliable here; where the titles have drifted
too far apart to trust, the element falls back to the outcome with the same
number, and an element that matches nothing simply carries no content rather
than borrowing somebody else's.

Nothing here computes or judges anything. It gathers text for the prompt.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence

import runlog
from assessment_config import OUTCOME_MATCH_SIMILARITY
from assessment_models import ContentTopic, ElementContent

_RE_NOISE = re.compile(r"[^a-z0-9 ]+")
_RE_SPACE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Titles as they compare: lower case, letters and digits, single spaces."""
    return _RE_SPACE.sub(" ", _RE_NOISE.sub(" ", (text or "").lower())).strip()


def similarity(left: str, right: str) -> float:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------- #
# Element -> learning outcome
# --------------------------------------------------------------------------- #
def match_outcomes(elements: Sequence, outcomes: Sequence) -> Dict[str, object]:
    """{element number: the learning outcome that teaches it}.

    Best titles first, each outcome claimed once: matching element by element
    in order would let element 1 take an outcome that is a near-perfect match
    for element 3, and everything after it shifts by one. Sorting the whole
    field of candidates by how good the match is and letting the strongest
    pairs settle first is what stops a single weak title dragging the rest of
    the unit out of alignment.

    Elements left over fall back to the outcome numbered the same, if that
    outcome is still free. An element that matches nothing is absent from the
    result: no content is better than another element's content, which would
    be a question about the wrong half of the unit.
    """
    pairs = []
    for element in elements:
        for outcome in outcomes:
            score = similarity(getattr(element, "title", ""),
                               getattr(outcome, "title", ""))
            if score >= OUTCOME_MATCH_SIMILARITY:
                pairs.append((score, str(getattr(element, "number", "")),
                              outcome))
    pairs.sort(key=lambda p: (-p[0], p[1]))

    matched: Dict[str, object] = {}
    claimed = set()
    for _score, number, outcome in pairs:
        if number in matched or id(outcome) in claimed:
            continue
        matched[number] = outcome
        claimed.add(id(outcome))

    by_number = {str(getattr(o, "number", "")): o for o in outcomes}
    for element in elements:
        number = str(getattr(element, "number", ""))
        if number in matched:
            continue
        outcome = by_number.get(number)
        if outcome is not None and id(outcome) not in claimed:
            matched[number] = outcome
            claimed.add(id(outcome))
    return matched


def _topics(outcome) -> List[ContentTopic]:
    out: List[ContentTopic] = []
    for sub in getattr(outcome, "sub_topics", []) or []:
        title = str(getattr(sub, "title", "") or "").strip()
        points = [str(p).strip() for p in getattr(sub, "key_points", []) or []
                  if str(p).strip()]
        if not title and not points:
            continue
        out.append(ContentTopic(number=str(getattr(sub, "number", "") or ""),
                                title=title, key_points=points))
    return out


def content_for(curr_unit, elements: Sequence,
                wanted: Optional[Iterable[str]] = None) -> List[ElementContent]:
    """The taught content behind each element, in the unit's own order.

    `elements` is every element of the unit, not only the ones being assessed:
    the match is made across the whole unit and filtered afterwards, because
    matching a three-element subset against eight learning outcomes is how an
    element ends up holding the content of an outcome it has nothing to do
    with. `wanted` then keeps only the elements this CAT covers.
    """
    if curr_unit is None:
        return []
    outcomes = list(getattr(curr_unit, "learning_outcomes", []) or [])
    if not outcomes:
        return []

    keep = None if wanted is None else {str(w) for w in wanted}
    matched = match_outcomes(list(elements), outcomes)
    out: List[ElementContent] = []
    for element in elements:
        number = str(getattr(element, "number", ""))
        if keep is not None and number not in keep:
            continue
        outcome = matched.get(number)
        if outcome is None:
            runlog.warn(f"Assessment: element {number} matched no learning "
                        f"outcome in the curriculum; it is assessed from its "
                        f"performance criteria alone")
            continue
        topics = _topics(outcome)
        if not topics:
            continue
        out.append(ElementContent(
            element_number=number,
            element_title=str(getattr(element, "title", "") or ""),
            outcome_number=str(getattr(outcome, "number", "") or ""),
            outcome_title=str(getattr(outcome, "title", "") or ""),
            duration_hours=int(getattr(outcome, "duration_hours", 0) or 0),
            topics=topics,
            suggested_methods=[str(m).strip() for m
                               in getattr(outcome, "suggested_methods", []) or []
                               if str(m).strip()],
        ))
    return out


# --------------------------------------------------------------------------- #
# The prompt block
# --------------------------------------------------------------------------- #
def render(content: Sequence[ElementContent]) -> str:
    """The taught content as the model reads it, or '' when there is none.

    Numbered the way the curriculum numbers it, so that when a trainer reads a
    question and asks where it came from, the answer is a sub-topic number they
    can point at in their own document.
    """
    if not content:
        return ""
    blocks: List[str] = []
    for block in content:
        head = f"ELEMENT {block.element_number}"
        if block.element_title:
            head += f" - {block.element_title}"
        if block.duration_hours:
            head += f"   [{block.duration_hours} hours taught]"
        lines = [head]
        for topic in block.topics:
            label = " ".join(x for x in (topic.number, topic.title) if x)
            # One line per sub-topic rather than one per key point. This is a
            # list of topic names and the model reads it as one either way,
            # but the indented form cost about a third more of a prompt that
            # has to fit inside 8000 tokens with the teaching notes - and the
            # notes are what the questions are made of, so they get the room.
            if topic.key_points:
                lines.append(f"  {label}: " + "; ".join(topic.key_points))
            else:
                lines.append(f"  {label}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def methods(content: Sequence[ElementContent]) -> List[str]:
    """Every assessment method the curriculum suggests, once each, in order."""
    out: List[str] = []
    for block in content:
        for method in block.suggested_methods:
            if method not in out:
                out.append(method)
    return out


def summarise(content: Sequence[ElementContent]) -> str:
    """One line for the progress log: how much depth the paper is drawing on."""
    if not content:
        return "no curriculum content matched; writing from the PCs alone"
    topics = sum(len(b.topics) for b in content)
    points = sum(b.key_point_count for b in content)
    return (f"{len(content)} element(s), {topics} sub-topic(s) and "
            f"{points} key point(s) of taught content")
