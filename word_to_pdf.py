"""Render a Word document as a PDF, so it can be read like every other one.

The extraction that matters reads RULING LINES. `table_reader` asks pdfplumber
for a page's table grid and gets back real cells, which is how the OS's
`ELEMENT | PERFORMANCE CRITERIA` table and the curriculum's `Learning Outcome |
Content | Suggested Assessment Methods` table come out whole. A Word document
has no ruling lines to find, and the reader that does exist for it
(`pdf_utils.load_word_pages`) flattens every block onto three synthesised x
positions before the parsers re-split them into columns, so the row boundaries
are gone by the time anything looks for them. Across the four Word-only
programmes in the library, none of the eight units sampled yielded any content
at all.

Converting first removes the whole problem: after this the document is a PDF
like the other 541 and takes exactly the same path through the code.

The conversion is done by whatever the machine actually has - Microsoft Word
through COM, or LibreOffice headless - because both lay the tables out with
real borders. Neither is a dependency: with no converter installed the caller
falls back to reading the Word file directly, exactly as before.

Word is driven from a SUBPROCESS rather than in-process. COM wants its thread
initialised, Streamlit runs the script on a worker thread, and a document that
makes Word raise a dialog blocks forever. A subprocess gives the conversion a
timeout and keeps a hung Word out of the app.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Optional

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".converted")

# Word opens a 150-page curriculum in a minute or two; much beyond that means
# a dialog nobody can see. The child enforces the deadline itself and the
# parent allows a little longer, because only the child can shut Word down
# tidily - see `_watchdog`.
TIMEOUT_SECONDS = 240
_PARENT_GRACE = 30

# One at a time. Nothing in the app converts two documents at once, but a
# script surveying the library did, and three Word instances at once turned
# four conversions into four timeouts - each leaving its Word behind.
_ONE_AT_A_TIME = threading.Lock()

WORD = "Microsoft Word"
LIBREOFFICE = "LibreOffice"

_SOFFICE_PATHS = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/bin/libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)


# --------------------------------------------------------------------------- #
# What this machine can do
# --------------------------------------------------------------------------- #
def _soffice() -> str:
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    return next((p for p in _SOFFICE_PATHS if os.path.isfile(p)), "")


def _word_registered() -> bool:
    """Whether Word is installed, asked WITHOUT starting it.

    Creating the COM object to find out launches Word, which is slow and
    leaves a process behind on a machine that only wanted the answer.
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg
    except ImportError:
        return False
    try:
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                                       "Word.Application"))
    except OSError:
        return False
    try:
        import win32com.client                          # noqa: F401
    except ImportError:
        return False
    return True


def converter_name() -> str:
    """The converter that would be used, or '' when there is none."""
    if _word_registered():
        return WORD
    if _soffice():
        return LIBREOFFICE
    return ""


# --------------------------------------------------------------------------- #
# The conversions themselves
# --------------------------------------------------------------------------- #
def _convert_with_word(src: str, dst: str) -> str:
    """Drive Word in a child process. Returns '' on success, else the reason."""
    with _ONE_AT_A_TIME:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), src, dst],
            capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS + _PARENT_GRACE)
    if proc.returncode == 0 and os.path.getsize(dst or os.devnull) > 0:
        return ""
    return (proc.stderr or proc.stdout or "").strip()[-300:] or "Word failed"


def _convert_with_libreoffice(src: str, dst: str) -> str:
    """soffice names the output itself, so it writes into a scratch directory
    and the one PDF that appears is moved into place."""
    soffice = _soffice()
    if not soffice:
        return "LibreOffice is not installed"
    with tempfile.TemporaryDirectory() as scratch:
        proc = subprocess.run(
            [soffice, "--headless", "--norestore", "--convert-to", "pdf",
             "--outdir", scratch, src],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        made = [f for f in os.listdir(scratch) if f.lower().endswith(".pdf")]
        if not made:
            return (proc.stderr or proc.stdout or "").strip()[-300:] \
                or "LibreOffice produced no PDF"
        shutil.move(os.path.join(scratch, made[0]), dst)
    return ""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cache_path(path: str) -> str:
    """Where this document's PDF lives, keyed so an edited file reconverts."""
    full = os.path.abspath(path)
    try:
        stat = os.stat(full)
        stamp = f"{stat.st_mtime_ns}-{stat.st_size}"
    except OSError:
        stamp = "0-0"
    digest = hashlib.sha1(f"{full}|{stamp}".encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(full))[0][:48]
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    return os.path.join(CACHE_DIR, f"{safe.strip() or 'document'}-{digest}.pdf")


def convert(path: str) -> Optional[str]:
    """This Word document as a PDF, converted once and cached, or None.

    None means no converter is installed or every one of them failed; the
    caller reads the Word file directly rather than showing an error, because
    a partial reading is better than none.
    """
    import runlog

    dst = cache_path(path)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst

    name = converter_name()
    if not name:
        runlog.log(f"No PDF converter installed, so {os.path.basename(path)} "
                   "is read as Word - its tables will not come out whole. "
                   "Install Microsoft Word or LibreOffice.", level="WARN")
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    order = ([_convert_with_word, _convert_with_libreoffice] if name == WORD
             else [_convert_with_libreoffice, _convert_with_word])
    for convert_with in order:
        try:
            problem = convert_with(os.path.abspath(path), dst)
        except subprocess.TimeoutExpired:
            problem = f"gave up after {TIMEOUT_SECONDS}s"
        except Exception as e:                  # noqa: BLE001 - never block a load
            problem = f"{type(e).__name__}: {e}"
        if not problem:
            runlog.log(f"Converted {os.path.basename(path)} to PDF")
            return dst
        runlog.log(f"Converting {os.path.basename(path)} to PDF failed "
                   f"({problem})", level="WARN")
        if os.path.isfile(dst) and os.path.getsize(dst) == 0:
            os.remove(dst)
    return None


# --------------------------------------------------------------------------- #
# The Word half, run as a child process - see the module docstring
# --------------------------------------------------------------------------- #
_WD_FORMAT_PDF = 17


def _watchdog(get_word, dst):
    """Shut Word down and leave, rather than be killed still holding it.

    Being killed from outside is what leaves an invisible WINWORD running for
    good: the `finally` below never runs, so nothing calls Quit. A document
    that makes Word raise a dialog does exactly that. Ending it from inside is
    the only way the process comes down with us, and the half-written PDF goes
    with it so it is never mistaken for a finished conversion.
    """
    def give_up():
        try:
            word = get_word()
            if word is not None:
                word.Quit(0)                    # wdDoNotSaveChanges
        except Exception:                       # noqa: BLE001
            pass
        try:
            if os.path.isfile(dst):
                os.remove(dst)
        except OSError:
            pass
        print(f"Word did not finish within {TIMEOUT_SECONDS}s",
              file=sys.stderr, flush=True)
        os._exit(3)

    timer = threading.Timer(TIMEOUT_SECONDS, give_up)
    timer.daemon = True
    timer.start()
    return timer


def _main(src: str, dst: str) -> int:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    state = {"word": None}
    timer = _watchdog(lambda: state["word"], dst)
    document = None
    try:
        # DispatchEx asks for Word's own process, so converting here never
        # disturbs a document the person has open.
        word = state["word"] = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(src, ReadOnly=True,
                                       AddToRecentFiles=False, Visible=False)
        document.SaveAs(dst, FileFormat=_WD_FORMAT_PDF)
        return 0
    finally:
        timer.cancel()
        try:
            if document is not None:
                document.Close(False)
        finally:
            if state["word"] is not None:
                state["word"].Quit()
            pythoncom.CoUninitialize()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: word_to_pdf.py SOURCE DESTINATION.pdf", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(_main(sys.argv[1], sys.argv[2]))
