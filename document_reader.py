"""Read a source document whichever way recovers the most of it.

A Word document can be read two ways, and neither wins everywhere.

`pdf_utils.load_word_pages` reads the file directly. It flattens every block
onto three synthesised x positions before the parsers re-split them into
columns, so a unit's structure has to be inferred all over again from those
positions. On some programmes that yields nothing usable at all: across the
four Word-only programmes first sampled, not one of eight units produced any
content.

`word_to_pdf` renders the document through Word or LibreOffice and the result
is read like every other PDF, by its ruling lines. On Refrigeration and Air
Conditioning that turns a curriculum yielding zero learning outcomes into one
yielding seven per unit with 48-64 sub-topics, and it is the only reading that
produces durations at all - across the Word documents in the library, 0 hours
against 354.

But measured unit by unit it is not uniformly better. On Applied Biology and
Criminal Justice some units come back with less: COMMUNICATION SKILLS went
from 9 outcomes and 167 key points to 1 and 95, and one converted unit
reported 46 learning outcomes and 3818 assessment methods, which is not a
reading of anything. Converting unconditionally would trade one set of broken
documents for another.

So the document is read BOTH ways and the better reading is kept, which is
what `curriculum_parser._choose_outcomes` already does one level down for the
two ways of parsing a unit. The upside is kept whole and there is no downside
case left to find. The cost is parsing a Word document twice, once, the first
time it is opened: the conversion and the verdict are both cached.
"""

from __future__ import annotations

import io
import os
from typing import List

import runlog
from pdf_utils import Page, load_document, load_pdf_pages, load_word_pages

OS, CU = "OS", "CU"


def _key(number: str, text: str) -> str:
    """One piece of content, in a form two readings of it agree on."""
    return (number or "").strip() + "|" + " ".join((text or "").lower().split())


def _score(pages: List[Page], role: str) -> int:
    """How much of the document this reading actually recovered.

    Counted in the units a plan is built from - performance criteria on the
    standard's side, sub-topics and their key points on the curriculum's - and
    not in learning outcomes or elements, which is where a bad reading
    inflates. A flattened Word reading routinely reports twenty-odd "learning
    outcomes" for a unit that has five, each with a single sub-topic; counting
    outcomes would reward exactly that.

    DISTINCT items, not a total. A reading that invents units - "MODULE II"
    over a page range that spans the real units inside it - harvests the same
    content once for the module and again for each unit within, and a sum
    rewards it for the duplication. On the Refrigeration and Air Conditioning
    curriculum that put the Word reading ahead 368 to 306, when read as Word
    its first four units are MODULE II and MODULE III with no content at all.
    Counting each piece of content once removes the whole effect.
    """
    import unit_index

    try:
        refs = unit_index.index_units(pages, role).refs
    except Exception:                       # noqa: BLE001 - scoring never raises
        return 0

    found = set()
    for ref in refs:
        try:
            if role == OS:
                import os_parser
                unit = os_parser.parse_os_unit(pages, ref)
                for element in unit.elements:
                    found.update(_key(pc.number, pc.text)
                                 for pc in element.performance_criteria)
            else:
                import curriculum_parser
                unit = curriculum_parser.parse_curriculum_unit(pages, ref)
                if unit is None:
                    continue
                for outcome in unit.learning_outcomes:
                    for sub in outcome.sub_topics:
                        found.add(_key(sub.number, sub.title))
                        found.update(_key(sub.number, point)
                                     for point in sub.key_points)
        except Exception:                   # noqa: BLE001
            continue
    return len(found)


def _verdict_path(converted_pdf: str) -> str:
    """Where the decision for this document is remembered."""
    return os.path.splitext(converted_pdf)[0] + ".winner"


def _remembered(converted_pdf: str, role: str) -> str:
    try:
        with io.open(_verdict_path(converted_pdf), encoding="utf-8") as fh:
            for line in fh:
                kept, _, which = line.strip().partition(" ")
                if kept == role:
                    return which
    except OSError:
        pass
    return ""


def _remember(converted_pdf: str, role: str, which: str) -> None:
    """Append this role's verdict, keeping any the other role already left.

    The two sides of a programme are separate documents, but the same file can
    be opened as either when a document is mislabelled, and one verdict must
    not erase the other.
    """
    lines = []
    try:
        with io.open(_verdict_path(converted_pdf), encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh
                     if ln.strip() and not ln.startswith(role + " ")]
    except OSError:
        pass
    lines.append(f"{role} {which}")
    try:
        with io.open(_verdict_path(converted_pdf), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass                                # a lost verdict only costs time


def read(path: str, role: str = CU) -> List[Page]:
    """The best available reading of `path`, as pages.

    `role` is 'OS' or 'CU' - which document this is - because the two are
    scored on different things. A PDF is simply loaded; only a Word document
    has a choice to make.
    """
    import word_reader
    import word_to_pdf

    if word_reader.sniff_format(path) == word_reader.FMT_PDF:
        return load_pdf_pages(path)
    if word_reader.sniff_format(path) not in word_reader._READERS:
        return load_document(path)          # raises, with the message to show

    converted = word_to_pdf.convert(path)
    if not converted:
        return load_word_pages(path)

    name = os.path.basename(path)
    remembered = _remembered(converted, role)
    if remembered == "word":
        return load_word_pages(path)
    if remembered == "pdf":
        return load_pdf_pages(converted)

    direct = load_word_pages(path)
    rendered = load_pdf_pages(converted)
    direct_score = _score(direct, role)
    rendered_score = _score(rendered, role)

    # The tie goes to the conversion: it is the only reading that carries the
    # durations, so when the two recover the same content it recovers more.
    keep_rendered = rendered_score >= direct_score
    _remember(converted, role, "pdf" if keep_rendered else "word")
    runlog.log(f"{name}: read as {'PDF' if keep_rendered else 'Word'} "
               f"(converted recovered {rendered_score}, "
               f"the Word reader {direct_score})")
    return rendered if keep_rendered else direct
