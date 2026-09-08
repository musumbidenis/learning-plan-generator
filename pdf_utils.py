"""Shared low-level helpers for coordinate-based PDF parsing and DOCX fallback.

The whole point of this module: turn a page of a two/three-column Kenya CDACC
document into clean, per-column text lines using WORD X-COORDINATES, never the
naive `extract_text()` line order (which interleaves columns -> the classic
"two-column bleed" bug).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pdfplumber

import runlog


# --------------------------------------------------------------------------- #
# Noise filtering
# --------------------------------------------------------------------------- #
# PDF extraction mangles the copyright glyph (c) into U+FFFD; page-number lines
# and council footers must also be stripped before parsing.
_NOISE_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),                       # bare page number / stray year
    re.compile(r"TVET\s+CDACC\s*,?\s*20\d{2}", re.I),     # ©TVET CDACC 2025
    re.compile(r"^\s*[�©]\s*TVET", re.I),       # mangled-copyright footer
    # any '(c)<BODY> <year>' footer - '©QAI 2025' was being read as part of the
    # learning-outcome title above it
    re.compile(r"^\s*[�©]\s*[A-Za-z]{2,12}\s*,?\s*20\d{2}\s*$"),
    re.compile(r"^\s*[ivxlcdm]+\s*$", re.I),              # roman-numeral page nums
]


def is_noise_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    for pat in _NOISE_PATTERNS:
        if pat.search(t):
            return True
    return False


def clean_text(text: str) -> str:
    """Normalise whitespace and repair the mangled copyright glyph."""
    if not text:
        return ""
    text = text.replace("�", "")          # dropped non-ASCII (often ©)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    # Re-join words hyphenated across a line wrap: 'object- oriented' -> 'object-oriented'.
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "-", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Word / line model
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    """A reconstructed text line, kept separately per column."""
    top: float
    text: str
    x0: float                       # x0 of the first (left-most) word on the line


def _group_words_into_lines(words: List[dict], y_tol: float = 3.0) -> List[List[dict]]:
    """Cluster words that share (approximately) the same baseline into lines."""
    rows: List[Tuple[float, List[dict]]] = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        placed = False
        for row in rows:
            if abs(row[0] - w["top"]) <= y_tol:
                row[1].append(w)
                placed = True
                break
        if not placed:
            rows.append((w["top"], [w]))
    rows.sort(key=lambda r: r[0])
    return [sorted(r[1], key=lambda w: w["x0"]) for r in rows]


def _line_from_words(line_words: List[dict]) -> Optional[Line]:
    if not line_words:
        return None
    text = clean_text(" ".join(w["text"] for w in line_words))
    if not text:
        return None
    return Line(top=line_words[0]["top"], text=text, x0=line_words[0]["x0"])


def column_lines(words: List[dict], x_min: float, x_max: float,
                 y_tol: float = 3.0) -> List[Line]:
    """Reconstruct text lines using only words whose x0 falls in [x_min, x_max).

    This is the core anti-bleed primitive: filter by column FIRST, then group
    into lines, so a continuation line in one column never absorbs text from a
    neighbouring column on the same physical row.
    """
    col_words = [w for w in words if x_min <= w["x0"] < x_max]
    lines: List[Line] = []
    for lw in _group_words_into_lines(col_words, y_tol=y_tol):
        ln = _line_from_words(lw)
        if ln and not is_noise_line(ln.text):
            lines.append(ln)
    return lines


def detect_column_split(words: List[dict], search_lo: float = 150.0,
                        search_hi: float = 260.0, default: float = 200.0) -> float:
    """Find the two-column boundary as the largest x-gap inside a search band.

    Kenya OS element/PC tables put the Element column near x0~73-160 and the PC
    column near x0~230+. We look for the widest empty horizontal strip between
    those clusters and split at its midpoint. Falls back to `default`.
    """
    xs = sorted({round(w["x0"], 1) for w in words
                 if search_lo - 60 <= w["x0"] <= search_hi + 60})
    best_gap, best_mid = 0.0, default
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2.0
        gap = b - a
        if search_lo <= mid <= search_hi and gap > best_gap:
            best_gap, best_mid = gap, mid
    # Require a meaningful gap; otherwise trust the default.
    return best_mid if best_gap >= 25 else default


# --------------------------------------------------------------------------- #
# Document abstraction (PDF + DOCX)
# --------------------------------------------------------------------------- #
@dataclass
class Page:
    index: int
    text: str                       # naive extract_text (used for region detection)
    words: List[dict]               # extract_words (used for column splitting)


def _text_from_words(words: List[dict]) -> str:
    """Reconstruct page text from extracted words in reading order.

    This lets the fast loader skip a second full layout pass (pdfplumber runs
    one for extract_words() and another for extract_text()). The result is only
    used for region detection and the header-block regexes, which the column-
    aware parsers do not rely on for content accuracy.
    """
    return "\n".join(
        " ".join(w["text"] for w in grp)
        for grp in _group_words_into_lines(words)
    )


def _load_pdf_pages_fitz(path: str) -> List[Page]:
    """Fast path: PyMuPDF word extraction (~50x faster than pdfplumber here).

    PyMuPDF's word tuples are (x0, y0, x1, y1, word, block, line, word_no) in a
    top-left origin coordinate space, so y0 maps directly onto pdfplumber's
    `top`, and x0/x1 are the same physical PDF coordinates the column parsers
    already expect - no re-tuning needed.
    """
    import fitz  # PyMuPDF - already a dependency (also used by splitter.py)

    pages: List[Page] = []
    total_words = 0
    doc = fitz.open(path)
    try:
        for pno in range(len(doc)):
            raw = doc[pno].get_text("words")
            words = [{"text": w[4], "x0": w[0], "x1": w[2],
                      "top": w[1], "bottom": w[3]} for w in raw]
            total_words += len(words)
            pages.append(Page(index=pno, text=_text_from_words(words), words=words))
    finally:
        doc.close()
    if total_words == 0:
        # Likely a scanned / image-only PDF; let the caller fall back.
        raise ValueError("PyMuPDF extracted no words (scanned or image-only PDF?)")
    return pages


def _load_pdf_pages_pdfplumber(path: str) -> List[Page]:
    """Robust fallback path used only if the fast PyMuPDF path fails."""
    pages: List[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            pages.append(Page(
                index=i,
                text=p.extract_text() or "",
                words=p.extract_words() or [],
            ))
    return pages


def load_pdf_pages(path: str) -> List[Page]:
    start = time.perf_counter()
    runlog.log(f"Loading PDF: {path}")
    try:
        pages = _load_pdf_pages_fitz(path)
        backend = "PyMuPDF"
    except Exception as exc:  # noqa: BLE001
        runlog.warn(f"PyMuPDF load failed ({exc}); falling back to pdfplumber")
        pages = _load_pdf_pages_pdfplumber(path)
        backend = "pdfplumber"
    runlog.log(f"Loaded {len(pages)} pages via {backend} in "
               f"{time.perf_counter() - start:.2f}s", level="TIMING")
    return pages


def _docx_page_break_count(para) -> int:
    """How many explicit page breaks this paragraph carries."""
    from docx.oxml.ns import qn
    return sum(1 for br in para._element.findall(".//" + qn("w:br"))
               if br.get(qn("w:type")) == "page")


def _split_blocks_into_pages(blocks: List[Tuple[bool, List[Tuple[str, float]]]]
                             ) -> List[List[Tuple[str, float]]]:
    """Group DOCX blocks into pseudo-pages.

    A .docx has no pages, but the parsers work per page: a unit's span is
    [start_page, end_page). Returning one page for the whole file - as this
    loader used to - therefore capped a multi-unit document at a single unit,
    and usually found none at all, because one page carrying every unit's code
    fails the 'exactly one code' test.

    Pages break at explicit page breaks; failing that (or where a break-free
    document still holds several units) they break before each unit heading -
    the line preceding a code line - which is exactly the boundary the parsers
    want.
    """
    pages: List[List[Tuple[str, float]]] = []
    current: List[Tuple[str, float]] = []
    for is_break, lines in blocks:
        if is_break and current:
            pages.append(current)
            current = []
        current.extend(lines)
    if current:
        pages.append(current)
    return [_split_page_on_unit_headers(pg) for pg in pages]


# Lines of lead-in carried onto a new pseudo-page, so the unit title is on the
# same page as the code that identifies it.
_UNIT_HEADER_LOOKBACK = 4


def _split_page_on_unit_headers(lines: List[Tuple[str, float]]):
    """Split one pseudo-page again wherever a second unit header begins."""
    code_rows = [i for i, (text, x) in enumerate(lines)
                 if x == 80.0 and (RE_ISCED_CODE_SHAPE.search(text)
                                   or RE_TVET_CODE_SHAPE.search(text))]
    if len(code_rows) < 2:
        return [lines]
    cuts = []
    for i in code_rows[1:]:
        # Start the new page a few lines above the code, not just one. The unit
        # heading does not always sit immediately above it, and cutting tight
        # left the title resolver with a single candidate - which, when that one
        # line was trailing prose from the previous unit, became the unit title.
        cut = max(0, i - _UNIT_HEADER_LOOKBACK)
        if not cuts or cut > cuts[-1]:
            cuts.append(cut)
    out, prev = [], 0
    for cut in cuts:
        if cut > prev:
            out.append(lines[prev:cut])
            prev = cut
    out.append(lines[prev:])
    return [chunk for chunk in out if chunk]


def load_word_pages(path: str) -> List[Page]:
    """Load ANY supported word-processor document into the uniform Page list.

    Word-processor formats have no x-coordinates, so we synthesise pseudo-columns
    from table cells: each row contributes its cells as words at fixed pseudo-x
    positions (col 0 -> x0=80, col 1 -> x0=235, col 2 -> x0=440), letting the
    same column-aware parsers work. Plain paragraphs become single left-column
    words.

    They have no pages either, so we synthesise those too - see
    `_split_blocks_into_pages`.

    The reading itself lives in `word_reader`, which covers .docx/.docm/.dotx/
    .dotm, .odt/.ott, .rtf and the legacy .doc binary behind one interface.
    """
    import word_reader

    PSEUDO_X = [80.0, 235.0, 440.0]

    # (starts_a_new_page, [(line_text, pseudo_x), ...]) in document order
    blocks: List[Tuple[bool, List[Tuple[str, float]]]] = []
    for block in word_reader.read_blocks(path):
        if block.kind == word_reader.BREAK:
            blocks.append((True, []))
            continue
        lines = []
        for ci, cell in enumerate(block.cells[:3]):
            # A cell's own paragraphs each become a line in that cell's column.
            # Collapsing them into one line merged a whole row of performance
            # criteria - or of curriculum key points - into a single item.
            for part in cell.split("\n"):
                text = clean_text(part)
                if text:
                    lines.append((text, PSEUDO_X[min(ci, 2)]))
        blocks.append((False, lines))

    pages: List[Page] = []
    for group in _split_blocks_into_pages(blocks):
        for chunk in group:
            words: List[dict] = []
            top = 0.0
            last_x = None
            for text, x0 in chunk:
                # a new left-column line starts a new baseline; cells of one
                # table row share theirs, which is what the column parsers expect
                if last_x is not None and x0 <= last_x:
                    top += 14
                last_x = x0
                # Every token of a cell shares the cell's x0. Advancing x per
                # token instead lets a long first cell run past the second
                # column's start, and the column filters then read both cells as
                # one - which is how element/PC and outcome/content pairs were
                # being merged into a single column.
                for tok in text.split(" "):
                    if not tok:
                        continue
                    words.append({"text": tok, "x0": x0, "x1": x0 + 5 * len(tok),
                                  "top": top, "bottom": top + 10})
            if not words:
                continue
            full_text = "\n".join(
                " ".join(w["text"] for w in grp)
                for grp in _group_words_into_lines(words))
            pages.append(Page(index=len(pages), text=full_text, words=words))

    return pages or [Page(index=0, text="", words=[])]


# Kept under its old name: callers and tests still ask for a DOCX by name.
load_docx_pages = load_word_pages


def load_document(path: str) -> List[Page]:
    """Load a document into the uniform Page list, whatever format it is in.

    The format is sniffed from the file's magic bytes rather than trusted from
    its extension - a renamed file is a common reason a document "won't open",
    and there is no good reason for it to fail here.
    """
    import word_reader

    fmt = word_reader.sniff_format(path)
    if fmt == word_reader.FMT_PDF:
        return load_pdf_pages(path)
    if fmt in word_reader._READERS:
        return load_word_pages(path)
    raise ValueError(
        f"Unsupported document type: {os.path.basename(path)}. Supported: PDF, "
        "and Word documents in .docx, .docm, .dotx, .dotm, .doc, .rtf, .odt or "
        ".ott form.")


# --------------------------------------------------------------------------- #
# Regexes reused across parsers
# --------------------------------------------------------------------------- #
RE_ISCED = re.compile(r"ISCED\s+UNIT\s+CODE\s*:?\s*([0-9][0-9A-Za-z ]+?)(?:\s{2,}|$)", re.I)
RE_OS_CODE = re.compile(r"TVET\s+CDACC\s+(?:UNIT\s+)?CODE\s*:?\s*([A-Z0-9/]+)", re.I)
RE_LEVEL = re.compile(r"(?:KNQF\s+)?LEVEL\s*[:\-]?\s*(\d+)", re.I)


# --------------------------------------------------------------------------- #
# Unit identity by CODE SHAPE (label-independent)
#
# Every CDACC unit carries two codes with distinctive shapes, regardless of how
# the surrounding label is spelt ("ISCED UNIT CODE", "UNIT CODE", a typo'd
# "ISCDE", or no prefix at all). Anchoring on the shape - not the label - lets a
# single code path locate units across documents whose labels differ in wording.
#   * ISCED code:      '0612 351 03A'         -> 4 digits, 3 digits, 2 digits, letter
#   * TVET CDACC code: 'IT/OS/ICTA/CR/01/4/MA' -> slash-delimited upper/numeric tokens
# --------------------------------------------------------------------------- #
RE_ISCED_CODE_SHAPE = re.compile(r"(\d{4}\s*\d{3}\s*\d{2}\s*[A-Za-z])")
RE_TVET_CODE_SHAPE = re.compile(r"([A-Z]{2,}(?:/[A-Z0-9]+){4,})")


def find_isced_code(text: str) -> str:
    """First ISCED-shaped code in *text* (cleaned), or '' - label-independent."""
    m = RE_ISCED_CODE_SHAPE.search(text or "")
    return clean_text(m.group(1)) if m else ""


def find_tvet_code(text: str) -> str:
    """First TVET-CDACC-shaped code in *text*, or '' - label-independent."""
    m = RE_TVET_CODE_SHAPE.search(text or "")
    return m.group(1) if m else ""


# A code line carries either of the two code shapes, or is a bare 'UNIT CODE:'
# label whose code wrapped onto the next line. The unit name is never one of
# these, so title resolution must walk past them.
RE_CODE_LABEL = re.compile(r"\bCODE\s*:?\s*$", re.I)


def _is_code_line(text: str) -> bool:
    return bool(RE_TVET_CODE_SHAPE.search(text)
                or RE_ISCED_CODE_SHAPE.search(text)
                or RE_CODE_LABEL.search(text.strip()))


def unit_title_above_code(page: "Page") -> str:
    """The unit name = the nearest non-noise, non-code line ABOVE the line that
    bears the ISCED code. Anchoring on the code shape (not a label) makes this
    work whatever the document calls its code field.

    Skipping code lines is essential: some documents order the header
    title -> TVET code -> ISCED code, so the line directly above the ISCED
    anchor is the TVET CDACC code, not the title. Walking past any code line
    reaches the real name regardless of how the two codes are ordered."""
    lines = column_lines(page.words, 0, 10_000)
    for idx, ln in enumerate(lines):
        if RE_ISCED_CODE_SHAPE.search(ln.text):
            for back in range(idx - 1, -1, -1):
                cand = lines[back].text.strip()
                if cand and not is_noise_line(cand) and not _is_code_line(cand):
                    return clean_text(cand)
    return ""


def page_starts_unit(page: "Page") -> bool:
    """True when *page* is a unit's FIRST page, detected purely by code shape.

    Two label-free guards reject the summary / MODULE / CATEGORY tables that also
    contain codes (both verified against the shipped CDACC PDFs):
      1. exactly ONE distinct ISCED code on the page - real unit pages carry one,
         those tables list several (the dropped pages had 2-14);
      2. a non-empty title resolved above that code.
    The code must also occur in a header context - its line carries a 'CODE'
    token, or a TVET code co-occurs on the page - which matches the real header
    block and rejects incidental code mentions.
    """
    lines = column_lines(page.words, 0, 10_000)
    code_line = next((ln for ln in lines if RE_ISCED_CODE_SHAPE.search(ln.text)), None)
    if code_line is None:
        return False
    if "CODE" not in code_line.text.upper() and not RE_TVET_CODE_SHAPE.search(page.text):
        return False
    distinct = {re.sub(r"\s+", "", c) for c in RE_ISCED_CODE_SHAPE.findall(page.text)}
    if len(distinct) != 1:
        return False
    return bool(unit_title_above_code(page))


def unit_start_pages(pages: List["Page"]) -> List[int]:
    """Indexes of pages that begin a unit, by ISCED code SHAPE (label-independent).

    Shared by both the OS and curriculum parsers so unit-boundary detection has a
    single source of truth (see `page_starts_unit`)."""
    return [p.index for p in pages if page_starts_unit(p)]
