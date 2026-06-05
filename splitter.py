"""Split a single unit's pages out of a full OS / Curriculum PDF.

The 'PDF Unit Extractor' stage uses this: after the user picks a unit, we carve
out just that unit's page span (from its UnitRef) into a fresh, downloadable PDF
- the per-unit Occupational Standard and/or per-unit Curriculum.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import fitz  # PyMuPDF

from models import UnitRef


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (s or "unit").strip())
    return s.strip("_") or "unit"


def split_unit_pdf(src_path: str, ref: UnitRef) -> bytes:
    """Return a new PDF (bytes) containing only `ref`'s pages from `src_path`.

    `ref.start_page` is inclusive, `ref.end_page` exclusive (0-based), matching
    the indexes produced by the parsers.
    """
    if not src_path.lower().endswith(".pdf"):
        raise ValueError("Per-unit splitting is only supported for PDF sources.")
    src = fitz.open(src_path)
    try:
        start = max(0, ref.start_page)
        end = min(len(src), ref.end_page if ref.end_page else len(src))
        if end <= start:
            end = min(len(src), start + 1)
        out = fitz.open()
        out.insert_pdf(src, from_page=start, to_page=end - 1)
        data = out.tobytes()
        out.close()
        return data
    finally:
        src.close()


def unit_filename(ref: UnitRef, kind: str) -> str:
    """kind = 'OS' or 'Curriculum'."""
    code = _safe_name(ref.code or ref.isced_code or ref.title)
    return f"{kind}_{_safe_name(ref.title)}_{code}.pdf"
