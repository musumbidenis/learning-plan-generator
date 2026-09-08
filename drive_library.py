"""The OS + Curriculum document library, backed by a shared Google Drive folder.

The library root holds one folder per programme, each with that programme's two
documents:

    Occupational Standards and CBET Curriculum/   <- the library root
      Accountancy Level 6/                        <- a programme
        Occupational Standards.pdf
        Curriculum.pdf
      ICT Technician Level 6/
      ...

Some libraries group programmes under a further level (RVNP, CDACC CYCLE 03,
...). Rather than make that a setting the trainer has to get right, the shape is
worked out from the folder itself: if the root's subfolders hold documents they
are programmes, and if they hold only folders they are collections of them. See
`detect_layout`.

Read-only by design: new programmes are created in Drive itself and show up on
the next refresh. Nothing here writes to Drive.

Downloads are cached under `.drive_cache/`, keyed on the file's Drive id AND its
modifiedTime - so a document replaced in Drive is fetched again automatically
while an unchanged one is only ever downloaded once.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

import config
import runlog
import word_reader
from drive_client import DriveError, DriveFile, download, list_children

# "Occupational Standards and CBET Curriculum", shared 'anyone with the link ->
# Viewer'. Overridable so the app can be pointed at a different library without
# a code change (e.g. a personal copy of the folder).
DEFAULT_LIBRARY_FOLDER_ID = "11gAs0Y3zLmPj4mykyKzjBWy-0TxT3mbU"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".drive_cache")

# Only these are worth handing to the parsers; anything else in a programme
# folder (spreadsheets, images, notes) is ignored. The word-processor formats
# come from `word_reader`, so the library and the uploader accept the same set.
READABLE_EXTENSIONS = (".pdf",) + word_reader.WORD_EXTENSIONS

# Files named like documents that are not documents: macOS AppleDouble stubs
# (a 4 KB '._Curriculum.pdf' sitting beside the real one), Finder metadata and
# Word lock files. Offering one of these as a choice of document is never right.
_RE_JUNK_NAME = re.compile(r"^(?:\._|~\$|\.DS_Store$)")

# Role labels used across this module and the UI.
ROLE_OS = "OS"
ROLE_CU = "CU"

# How the root folder is organised.
LAYOUT_FLAT = "flat"        # root -> programme -> documents
LAYOUT_NESTED = "nested"    # root -> collection -> programme -> documents

# Folders sampled when working the layout out - enough to see past an empty
# folder or two without turning the first page load into a crawl.
_LAYOUT_PROBE = 4


def library_root_id() -> str:
    return config.get("DRIVE_LIBRARY_FOLDER_ID") or DEFAULT_LIBRARY_FOLDER_ID


# --------------------------------------------------------------------------- #
# Browsing
# --------------------------------------------------------------------------- #
def is_readable(f: DriveFile) -> bool:
    """Whether `f` is a document one of the parsers could actually open."""
    if f.is_folder or _RE_JUNK_NAME.search(f.name or ""):
        return False
    return f.is_google_doc or (f.name or "").lower().endswith(READABLE_EXTENSIONS)


def _folders(items: List[DriveFile]) -> List[DriveFile]:
    return [f for f in items if f.is_folder]


def top_level_folders() -> List[DriveFile]:
    """The root's subfolders - programmes or collections, per `detect_layout`."""
    return _folders(list_children(library_root_id()))


def detect_layout(folders: List[DriveFile]) -> str:
    """Whether `folders` are programmes themselves, or collections of them.

    A programme folder is recognised by what it holds: its documents. Only the
    first few are sampled, so a single empty programme folder can't decide it.
    """
    for f in folders[:_LAYOUT_PROBE]:
        try:
            children = list_children(f.id)
        except DriveError:
            continue                    # a folder we can't read proves nothing
        if any(is_readable(c) for c in children):
            return LAYOUT_FLAT
    return LAYOUT_NESTED


def list_collections() -> List[DriveFile]:
    """The top-level groupings, for a library organised with them."""
    return top_level_folders()


def list_programmes(collection_id: str) -> List[DriveFile]:
    """The programme folders inside one collection."""
    return _folders(list_children(collection_id))


def programme_files(programme_id: str) -> List[DriveFile]:
    """The readable documents inside one programme folder.

    Google Docs are included (they are exported to .docx on download); folders,
    unreadable formats and filesystem litter are filtered out.
    """
    return [f for f in list_children(programme_id) if is_readable(f)]


# --------------------------------------------------------------------------- #
# Which file is the Occupational Standard, and which the Curriculum?
# --------------------------------------------------------------------------- #
# Filenames in the library follow no convention, so this is a guess the UI lets
# the user override. Curriculum is tested FIRST because a curriculum filename
# occasionally carries a unit code containing '/OS/' while the reverse - an
# Occupational Standard whose name says 'curriculum' - does not happen.
_RE_CURRICULUM = re.compile(r"curricul", re.I)
_RE_OS_WORDS = re.compile(r"occupational\s*standard", re.I)
# a standalone 'OS' token: 'ICT OS Level 5', 'OS-ICT.pdf', 'IT/OS/ICTA/...'
_RE_OS_TOKEN = re.compile(r"(?:^|[\s_\-./(])OS(?:$|[\s_\-./)0-9])", re.I)


def guess_role(name: str) -> Optional[str]:
    """ROLE_CU, ROLE_OS, or None when the filename doesn't say."""
    if _RE_CURRICULUM.search(name or ""):
        return ROLE_CU
    if _RE_OS_WORDS.search(name or "") or _RE_OS_TOKEN.search(name or ""):
        return ROLE_OS
    return None


def _format_rank(f: DriveFile) -> int:
    """Which copy to prefer when a folder holds the same document twice.

    Dozens of programmes carry both 'Curriculum.docx' and 'Curriculum.pdf'. The
    PDF is the better source: it keeps the page geometry the parsers split
    columns on, and Word's automatic list numbering - which the .docx text
    drops entirely - is baked into it as ordinary text.
    """
    ext = os.path.splitext(f.name or "")[1].lower()
    if ext == ".pdf":
        return 0
    if f.is_google_doc or ext in word_reader.WORD_EXTENSIONS:
        return 1
    return 2


def _best(candidates: List[DriveFile]) -> Optional[DriveFile]:
    if not candidates:
        return None
    return min(candidates, key=_format_rank)    # ties keep the folder's order


def classify(files: List[DriveFile]) -> Tuple[Optional[DriveFile],
                                              Optional[DriveFile],
                                              List[DriveFile]]:
    """Split a programme folder into (occupational_standard, curriculum, extras).

    `extras` holds everything not chosen - files whose name says nothing, and
    the other copies of a document already picked. The UI surfaces them so the
    trainer can assign them by hand; nothing is silently discarded.
    """
    by_role: Dict[Optional[str], List[DriveFile]] = {ROLE_OS: [], ROLE_CU: [],
                                                     None: []}
    for f in files:
        by_role[guess_role(f.name)].append(f)

    os_file = _best(by_role[ROLE_OS])
    cu_file = _best(by_role[ROLE_CU])

    # A folder holding exactly two files, one recognised and one the filename
    # says nothing about: the silent one is almost certainly the missing half.
    # A second file of a role we already filled is NOT a candidate - two
    # curricula stay two curricula.
    if (len(files) == 2 and len(by_role[None]) == 1
            and (os_file is None) != (cu_file is None)):
        if os_file is None:
            os_file = by_role[None][0]
        else:
            cu_file = by_role[None][0]

    chosen = {id(os_file), id(cu_file)}
    extras = [f for f in files if id(f) not in chosen]
    return os_file, cu_file, extras


# --------------------------------------------------------------------------- #
# Local cache
# --------------------------------------------------------------------------- #
def cache_path(file: DriveFile) -> str:
    """Where `file` lives on disk once fetched.

    The modifiedTime is folded into the name, so replacing a document in Drive
    produces a different path and the stale copy is simply never read again.
    """
    stamp = hashlib.sha1(
        (file.modified_time or "").encode("utf-8")).hexdigest()[:10]
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", file.id or "unknown")
    return os.path.join(CACHE_DIR, f"{safe_id}__{stamp}{file.extension}")


def ensure_local(file: DriveFile) -> str:
    """Return a local path to `file`, downloading it only if not already cached."""
    path = cache_path(file)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        runlog.log(f"Drive: '{file.name}' served from cache")
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    with runlog.timed(f"Download '{file.name}' from Drive"):
        return download(file, path)


__all__ = [
    "DriveError", "DriveFile", "ROLE_CU", "ROLE_OS", "CACHE_DIR",
    "DEFAULT_LIBRARY_FOLDER_ID", "LAYOUT_FLAT", "LAYOUT_NESTED",
    "library_root_id", "top_level_folders", "detect_layout", "list_collections",
    "list_programmes", "programme_files", "is_readable", "guess_role",
    "classify", "cache_path", "ensure_local",
]
