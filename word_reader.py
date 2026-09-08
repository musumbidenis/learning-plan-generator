"""Read a word-processor document of ANY common format into neutral blocks.

The app used to accept `.docx` alone, which quietly excluded most of what a
training institution actually holds: legacy `.doc` files, macro-enabled and
template variants, LibreOffice `.odt`, and `.rtf` exports. This module reads all
of them, and deliberately does so in PURE PYTHON - the same reason OCR was left
out, no trainer should need LibreOffice or Word installed for the app to open a
file they already have.

    .docx .docm .dotx .dotm   WordprocessingML - the OOXML package
    .odt  .ott                OpenDocument text
    .rtf                      Rich Text Format
    .doc  .dot                the legacy Word 97-2003 binary (best effort)

Everything is reduced to the same neutral `Block` list - paragraphs, table rows
and page breaks - which `pdf_utils` then lays out on synthesised coordinates so
the existing column-aware parsers work unchanged.

The format is decided by SNIFFING the file's magic bytes, never by its
extension. Documents in this library are routinely renamed, and a `.doc` that is
really a `.docx` (or the reverse) is one of the most common ways a file "won't
open" - here it simply works.
"""

from __future__ import annotations

import os
import re
import struct
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional
from xml.etree import ElementTree as ET

import runlog

# --------------------------------------------------------------------------- #
# Neutral document model
# --------------------------------------------------------------------------- #
PARA = "para"
ROW = "row"
BREAK = "break"


@dataclass
class Block:
    """One paragraph, one table row, or an explicit page break."""
    kind: str = PARA
    cells: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(c for c in self.cells if c)


class UnsupportedDocument(Exception):
    """The file isn't a word-processor document we can read."""


# --------------------------------------------------------------------------- #
# Format sniffing
# --------------------------------------------------------------------------- #
FMT_PDF = "pdf"
FMT_OOXML = "ooxml"          # .docx .docm .dotx .dotm
FMT_ODT = "odt"              # .odt .ott
FMT_RTF = "rtf"
FMT_DOC = "doc"              # Word 97-2003 binary
FMT_UNKNOWN = "unknown"

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK\x03\x04"

# Extensions we accept in the UI and in the Drive library.
WORD_EXTENSIONS = (".docx", ".docm", ".dotx", ".dotm", ".odt", ".ott",
                   ".rtf", ".doc", ".dot")


def sniff_format(path: str) -> str:
    """Identify a document by its content, falling back to its extension.

    Magic bytes come first on purpose: a mislabelled file is common and should
    still open.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        head = b""

    if head.startswith(b"%PDF-"):
        return FMT_PDF
    if head.startswith(_OLE_MAGIC):
        return FMT_DOC
    if head.lstrip()[:5].startswith(b"{\\rtf"):
        return FMT_RTF
    if head.startswith(_ZIP_MAGIC):
        return _sniff_zip(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return FMT_PDF
    if ext in (".docx", ".docm", ".dotx", ".dotm"):
        return FMT_OOXML
    if ext in (".odt", ".ott"):
        return FMT_ODT
    if ext == ".rtf":
        return FMT_RTF
    if ext in (".doc", ".dot"):
        return FMT_DOC
    return FMT_UNKNOWN


def _sniff_zip(path: str) -> str:
    """A zip is either an OOXML package or an OpenDocument one."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "word/document.xml" in names:
                return FMT_OOXML
            if "content.xml" in names:
                return FMT_ODT
    except (zipfile.BadZipFile, OSError):
        pass
    return FMT_UNKNOWN


# --------------------------------------------------------------------------- #
# WordprocessingML (.docx .docm .dotx .dotm)
# --------------------------------------------------------------------------- #
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _ooxml_paragraph_text(p_el) -> str:
    """All visible text in a w:p, honouring tabs and line breaks."""
    parts: List[str] = []
    for node in p_el.iter():
        tag = node.tag
        if tag == _W + "t":
            parts.append(node.text or "")
        elif tag == _W + "tab":
            parts.append(" ")
        elif tag == _W + "br" and node.get(_W + "type") != "page":
            parts.append(" ")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _ooxml_has_page_break(p_el) -> bool:
    for node in p_el.iter():
        if node.tag == _W + "br" and node.get(_W + "type") == "page":
            return True
        if node.tag == _W + "lastRenderedPageBreak":
            return True
    return False


def _blocks_ooxml(path: str) -> List[Block]:
    """Read word/document.xml directly.

    Going to the XML rather than through python-docx means the macro-enabled
    and template variants (.docm/.dotx/.dotm) read through the very same code -
    python-docx rejects several of them on their content type alone, which is
    why they were unreadable before.
    """
    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("word/document.xml")
        except KeyError as e:
            raise UnsupportedDocument(
                "This looks like an Office package but has no Word document "
                "inside it.") from e
    root = ET.fromstring(xml)
    body = root.find(_W + "body")
    if body is None:
        return []

    blocks: List[Block] = []
    for child in body:
        if child.tag == _W + "p":
            if _ooxml_has_page_break(child):
                blocks.append(Block(kind=BREAK))
            text = _ooxml_paragraph_text(child)
            if text:
                blocks.append(Block(kind=PARA, cells=[text]))
        elif child.tag == _W + "tbl":
            for row in child.findall(_W + "tr"):
                cells = []
                for cell in row.findall(_W + "tc"):
                    texts = [_ooxml_paragraph_text(p)
                             for p in cell.iter(_W + "p")]
                    cells.append(" ".join(t for t in texts if t).strip())
                if any(cells):
                    blocks.append(Block(kind=ROW, cells=cells))
    return blocks


# --------------------------------------------------------------------------- #
# OpenDocument text (.odt .ott)
# --------------------------------------------------------------------------- #
_TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
_TABLE_NS = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
_OFFICE_NS = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"


def _odt_text(el) -> str:
    """Flatten an ODT element's text, expanding the repeated-space element."""
    parts: List[str] = []

    def walk(node):
        if node.tag == _TEXT_NS + "s":
            parts.append(" " * int(node.get(_TEXT_NS + "c", "1")))
        elif node.tag in (_TEXT_NS + "tab", _TEXT_NS + "line-break"):
            parts.append(" ")
        else:
            if node.text:
                parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _blocks_odt(path: str) -> List[Block]:
    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("content.xml")
        except KeyError as e:
            raise UnsupportedDocument(
                "This OpenDocument file has no content.xml.") from e
    root = ET.fromstring(xml)
    body = root.find(_OFFICE_NS + "body")
    text_root = body.find(_OFFICE_NS + "text") if body is not None else None
    if text_root is None:
        return []

    blocks: List[Block] = []

    def emit(container):
        for child in container:
            tag = child.tag
            if tag in (_TEXT_NS + "p", _TEXT_NS + "h"):
                if child.find(_TEXT_NS + "soft-page-break") is not None:
                    blocks.append(Block(kind=BREAK))
                text = _odt_text(child)
                if text:
                    blocks.append(Block(kind=PARA, cells=[text]))
            elif tag == _TABLE_NS + "table":
                for row in child.iter(_TABLE_NS + "table-row"):
                    cells = [_odt_text(c)
                             for c in row.findall(_TABLE_NS + "table-cell")]
                    if any(cells):
                        blocks.append(Block(kind=ROW, cells=cells))
            elif tag in (_TEXT_NS + "section", _TEXT_NS + "list"):
                emit(child)
            elif tag == _TEXT_NS + "list-item":
                emit(child)

    emit(text_root)
    return blocks


# --------------------------------------------------------------------------- #
# Rich Text Format (.rtf)
# --------------------------------------------------------------------------- #
# Destinations whose contents are metadata, not document text.
_RTF_SKIP_DESTINATIONS = {
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "themedata",
    "colorschememapping", "latentstyles", "datastore", "listtable",
    "listoverridetable", "rsidtbl", "generator", "xmlnstbl", "filetbl",
    "header", "footer", "headerl", "headerr", "footerl", "footerr",
    "footnote", "annotation", "mmathPr", "operator", "upr",
}


def _blocks_rtf(path: str) -> List[Block]:
    """Tokenise RTF into paragraphs and table rows.

    RTF is plain text markup, so this needs no third-party reader: `\\par` ends
    a paragraph, `\\cell` ends a table cell and `\\row` ends a table row, which
    is enough to keep the tables the parsers rely on.
    """
    with open(path, "rb") as fh:
        raw = fh.read().decode("latin-1", errors="replace")

    blocks: List[Block] = []
    buf: List[str] = []
    row: List[str] = []
    # (skip_this_group, unicode_skip_count) per nesting level
    stack: List[bool] = [False]
    skipping = False
    i, n = 0, len(raw)

    def flush_para():
        text = re.sub(r"\s+", " ", "".join(buf)).strip()
        buf.clear()
        return text

    while i < n:
        ch = raw[i]

        if ch == "{":
            stack.append(skipping)
            i += 1
            continue
        if ch == "}":
            if len(stack) > 1:
                skipping = stack.pop()
            i += 1
            continue
        if ch == "\\":
            m = re.match(r"\\([a-zA-Z]+)(-?\d+)? ?", raw[i:])
            if not m:
                # escaped literal: \\ \{ \} or \'hh
                if raw[i:i + 2] == "\\'" and i + 4 <= n:
                    try:
                        buf.append(bytes([int(raw[i + 2:i + 4], 16)])
                                   .decode("cp1252", errors="replace"))
                    except ValueError:
                        pass
                    i += 4
                    continue
                if i + 1 < n and raw[i + 1] in "\\{}":
                    buf.append(raw[i + 1])
                    i += 2
                    continue
                if i + 1 < n and raw[i + 1] == "*":
                    skipping = True        # \* marks an ignorable destination
                    i += 2
                    continue
                i += 1
                continue

            word, arg = m.group(1), m.group(2)
            i += m.end()

            if word == "u" and arg is not None:
                code = int(arg)
                buf.append(chr(code if code >= 0 else code + 65536))
                continue
            if word in _RTF_SKIP_DESTINATIONS:
                # only THIS group is skipped; the stack holds the value to
                # restore when '}' closes it and must not be overwritten
                skipping = True
                continue
            if skipping:
                continue
            if word == "par" or word == "line":
                text = flush_para()
                if text:
                    blocks.append(Block(kind=PARA, cells=[text]))
            elif word == "cell":
                row.append(flush_para())
            elif word == "row":
                if buf:
                    row.append(flush_para())
                if any(c.strip() for c in row):
                    blocks.append(Block(kind=ROW, cells=list(row)))
                row.clear()
            elif word == "page":
                blocks.append(Block(kind=BREAK))
            elif word == "tab":
                buf.append(" ")
            continue

        if ch in "\r\n":
            i += 1
            continue
        if not skipping:
            buf.append(ch)
        i += 1

    text = flush_para()
    if text:
        blocks.append(Block(kind=PARA, cells=[text]))
    return blocks


# --------------------------------------------------------------------------- #
# Legacy Word binary (.doc, .dot)
# --------------------------------------------------------------------------- #
# Word marks structure inline: a cell ends with 0x07, a paragraph with 0x0D.
_DOC_CELL_END = "\x07"
_DOC_PARA_END = "\r"
_DOC_PAGE_BREAK = "\x0c"


def _read_ole_streams(path: str):
    """The WordDocument and table streams of an OLE2 compound file."""
    try:
        import olefile
    except ImportError as e:
        raise UnsupportedDocument(
            "Reading legacy .doc files needs the 'olefile' package "
            "(pip install -r requirements.txt). Alternatively open the file in "
            "Word or LibreOffice and save it as .docx.") from e

    if not olefile.isOleFile(path):
        raise UnsupportedDocument("Not a Word 97-2003 document.")
    try:
        ole = olefile.OleFileIO(path)
    except Exception as e:  # noqa: BLE001 - a damaged container reads as anything
        raise UnsupportedDocument(
            "This .doc file's internal structure couldn't be read - it may be "
            "damaged or incomplete. Open it in Word or LibreOffice and save it "
            f"as .docx, then try again. ({type(e).__name__})") from e
    try:
        if not ole.exists("WordDocument"):
            raise UnsupportedDocument(
                "This is an Office 97-2003 file but not a Word document.")
        doc = ole.openstream("WordDocument").read()
        flags = struct.unpack_from("<H", doc, 0x0A)[0]
        table_name = "1Table" if (flags & 0x0200) else "0Table"
        if not ole.exists(table_name):
            table_name = "0Table" if table_name == "1Table" else "1Table"
        table = ole.openstream(table_name).read() if ole.exists(table_name) else b""
        return doc, table
    finally:
        ole.close()


def _doc_piece_text(doc: bytes, table: bytes) -> str:
    """Rebuild the document text from the piece table.

    Word 97 stores text in pieces that are each either CP-1252 or UTF-16, listed
    in a piece table in the table stream. Reading it (rather than scraping the
    stream for printable runs) is what keeps the text in document order and free
    of deleted fragments.
    """
    fc_clx, lcb_clx = struct.unpack_from("<II", doc, 0x01A2)
    clx = table[fc_clx:fc_clx + lcb_clx]

    # Walk past any Prc entries to the Pcdt (0x02) that holds the piece table.
    pos, plc = 0, b""
    while pos < len(clx):
        kind = clx[pos]
        if kind == 0x01:                       # Prc: 2-byte length, then data
            size = struct.unpack_from("<H", clx, pos + 1)[0]
            pos += 3 + size
        elif kind == 0x02:                     # Pcdt: 4-byte length, then PlcPcd
            size = struct.unpack_from("<I", clx, pos + 1)[0]
            plc = clx[pos + 5:pos + 5 + size]
            break
        else:
            break
    if not plc:
        raise UnsupportedDocument(
            "Couldn't read the text layout of this .doc file. Open it in Word "
            "or LibreOffice and save it as .docx.")

    count = (len(plc) - 4) // 12
    cps = [struct.unpack_from("<I", plc, i * 4)[0] for i in range(count + 1)]
    out: List[str] = []
    for i in range(count):
        pcd = plc[(count + 1) * 4 + i * 8:(count + 1) * 4 + i * 8 + 8]
        fc = struct.unpack_from("<I", pcd, 2)[0]
        compressed = bool(fc & 0x40000000)
        fc &= 0x3FFFFFFF
        length = cps[i + 1] - cps[i]
        if compressed:
            chunk = doc[fc // 2:fc // 2 + length]
            out.append(chunk.decode("cp1252", errors="replace"))
        else:
            chunk = doc[fc:fc + length * 2]
            out.append(chunk.decode("utf-16-le", errors="replace"))
    return "".join(out)


def _blocks_doc(path: str) -> List[Block]:
    """Best-effort read of a Word 97-2003 binary document.

    Table fidelity is approximate - the binary format marks cells inline rather
    than nesting them - but paragraphs, cells and rows all survive, which is
    what unit detection needs.
    """
    doc, table = _read_ole_streams(path)
    try:
        text = _doc_piece_text(doc, table)
    except UnsupportedDocument:
        raise
    except Exception as e:  # noqa: BLE001 - struct/index errors on a damaged file
        # The binary format offers no way to validate before reading, so a
        # truncated or unusual file surfaces as a low-level error. Say something
        # the trainer can act on instead.
        raise UnsupportedDocument(
            "This .doc file couldn't be read - it may be damaged or use an "
            "older Word format. Open it in Word or LibreOffice and save it as "
            f".docx, then try again. ({type(e).__name__})") from e

    blocks: List[Block] = []
    row: List[str] = []
    buf: List[str] = []

    def clean(s: str) -> str:
        # drop Word's field and formatting control characters
        s = re.sub(r"[\x00-\x06\x08\x0b\x0e-\x1f]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    for ch in text:
        if ch == _DOC_CELL_END:
            row.append(clean("".join(buf)))
            buf.clear()
        elif ch in (_DOC_PARA_END, "\n"):
            if row:
                # a paragraph mark closes the row the cell marks opened
                if any(c for c in row):
                    blocks.append(Block(kind=ROW, cells=list(row)))
                row.clear()
                buf.clear()
                continue
            piece = clean("".join(buf))
            buf.clear()
            if piece:
                blocks.append(Block(kind=PARA, cells=[piece]))
        elif ch == _DOC_PAGE_BREAK:
            blocks.append(Block(kind=BREAK))
        else:
            buf.append(ch)

    trailing = clean("".join(buf))
    if trailing:
        blocks.append(Block(kind=PARA, cells=[trailing]))
    return blocks


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
_READERS = {
    FMT_OOXML: _blocks_ooxml,
    FMT_ODT: _blocks_odt,
    FMT_RTF: _blocks_rtf,
    FMT_DOC: _blocks_doc,
}

_FORMAT_NAMES = {
    FMT_OOXML: "Word (OOXML)",
    FMT_ODT: "OpenDocument text",
    FMT_RTF: "Rich Text Format",
    FMT_DOC: "Word 97-2003",
}


def read_blocks(path: str, fmt: Optional[str] = None) -> List[Block]:
    """Read any supported word-processor document into neutral blocks."""
    fmt = fmt or sniff_format(path)
    reader = _READERS.get(fmt)
    if reader is None:
        raise UnsupportedDocument(
            f"'{os.path.basename(path)}' isn't a document this app can read. "
            "Supported: PDF, and Word documents in .docx, .docm, .dotx, .dotm, "
            ".doc, .rtf, .odt or .ott form.")
    blocks = reader(path)
    runlog.log(f"Read {len(blocks)} blocks from a "
               f"{_FORMAT_NAMES.get(fmt, fmt)} document")
    return blocks
