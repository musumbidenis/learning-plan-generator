"""Shared low-level helpers for coordinate-based PDF parsing and DOCX fallback.

The whole point of this module: turn a page of a two/three-column Kenya CDACC
document into clean, per-column text lines using WORD X-COORDINATES, never the
naive `extract_text()` line order (which interleaves columns -> the classic
"two-column bleed" bug).
"""

from __future__ import annotations

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


def load_docx_pages(path: str) -> List[Page]:
    """Best-effort DOCX loader.

    DOCX has no x-coordinates, so we synthesise pseudo-columns from table cells:
    each table row contributes its cells as words at fixed pseudo-x positions
    (col 0 -> x0=80, col 1 -> x0=235, col 2 -> x0=440), letting the same
    column-aware parsers work. Plain paragraphs become single left-column words.
    """
    from docx import Document  # local import so PDF-only installs still work

    doc = Document(path)
    words: List[dict] = []
    top = 0.0
    PSEUDO_X = [80.0, 235.0, 440.0]

    def add_cell(text: str, x0: float, ytop: float) -> None:
        text = clean_text(text)
        if not text:
            return
        # split a cell into words sharing the same baseline & x0
        for tok in text.split(" "):
            words.append({"text": tok, "x0": x0, "x1": x0 + 5 * len(tok),
                          "top": ytop, "bottom": ytop + 10})

    # Iterate body elements in document order (paragraphs + tables).
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)
            if para.text.strip():
                add_cell(para.text, PSEUDO_X[0], top)
                top += 14
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            for row in table.rows:
                cells = row.cells
                for ci, cell in enumerate(cells[:3]):
                    add_cell(cell.text, PSEUDO_X[min(ci, 2)], top)
                top += 14

    # Re-derive page text from the words for region detection.
    full_text = "\n".join(
        " ".join(w["text"] for w in grp)
        for grp in _group_words_into_lines(words)
    )
    return [Page(index=0, text=full_text, words=words)]


def load_document(path: str) -> List[Page]:
    """Load a PDF or DOCX into the uniform Page list."""
    low = path.lower()
    if low.endswith(".pdf"):
        return load_pdf_pages(path)
    if low.endswith(".docx"):
        return load_docx_pages(path)
    raise ValueError(f"Unsupported document type: {path}")


# --------------------------------------------------------------------------- #
# Regexes reused across parsers
# --------------------------------------------------------------------------- #
RE_ISCED = re.compile(r"ISCED\s+UNIT\s+CODE\s*:?\s*([0-9][0-9A-Za-z ]+?)(?:\s{2,}|$)", re.I)
RE_OS_CODE = re.compile(r"TVET\s+CDACC\s+(?:UNIT\s+)?CODE\s*:?\s*([A-Z0-9/]+)", re.I)
RE_LEVEL = re.compile(r"(?:KNQF\s+)?LEVEL\s*[:\-]?\s*(\d+)", re.I)
