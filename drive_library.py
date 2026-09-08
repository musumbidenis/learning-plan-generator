"""The OS + Curriculum document library, backed by a shared Google Drive folder.

Layout (three levels, exactly as the folder is organised in Drive):

    Training Tools/                       <- the library root
      CDACC CYCLE 03/                     <- a collection
        ICT Technician Level 6/           <- a programme
          <Occupational Standard>.pdf
          <Curriculum>.pdf
      CDACC CYCLE 04 .../
      RVNP/                               <- appears here as soon as it exists

Read-only by design: new programmes and collections are created in Drive itself
and show up on the next refresh. Nothing here writes to Drive.

Downloads are cached under `.drive_cache/`, keyed on the file's Drive id AND its
modifiedTime - so a document replaced in Drive is fetched again automatically
while an unchanged one is only ever downloaded once.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config
import runlog
from drive_client import DriveError, DriveFile, download, list_children

# The "Training Tools" folder, shared 'anyone with the link -> Viewer'.
# Overridable so the app can be pointed at a different library without a code
# change (e.g. a personal copy of the folder).
DEFAULT_LIBRARY_FOLDER_ID = "1dGIPeezcayb-xHSYZsPa0MGv3clkRMGi"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".drive_cache")

# Only these are worth handing to the parsers; anything else in a programme
# folder (spreadsheets, images, notes) is ignored.
READABLE_EXTENSIONS = (".pdf", ".docx")

# Role labels used across this module and the UI.
ROLE_OS = "OS"
ROLE_CU = "CU"


def library_root_id() -> str:
    return config.get("DRIVE_LIBRARY_FOLDER_ID") or DEFAULT_LIBRARY_FOLDER_ID


# --------------------------------------------------------------------------- #
# Browsing
# --------------------------------------------------------------------------- #
def list_collections() -> List[DriveFile]:
    """The top-level groupings: RVNP, CDACC CYCLE 03, CDACC CYCLE 04, ..."""
    return [f for f in list_children(library_root_id()) if f.is_folder]


def list_programmes(collection_id: str) -> List[DriveFile]:
    """The programme folders inside one collection."""
    return [f for f in list_children(collection_id) if f.is_folder]


def programme_files(programme_id: str) -> List[DriveFile]:
    """The readable documents inside one programme folder.

    Google Docs are included (they are exported to .docx on download); folders
    and unreadable formats are filtered out.
    """
    out = []
    for f in list_children(programme_id):
        if f.is_folder:
            continue
        if f.is_google_doc or f.name.lower().endswith(READABLE_EXTENSIONS):
            out.append(f)
    return out


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


def classify(files: List[DriveFile]) -> Tuple[Optional[DriveFile],
                                              Optional[DriveFile],
                                              List[DriveFile]]:
    """Split a programme folder into (occupational_standard, curriculum, extras).

    `extras` holds everything not confidently assigned - files whose name says
    nothing, and any duplicate beyond the first of each role. The UI surfaces
    them so the trainer can assign them by hand; nothing is silently discarded.
    """
    os_file: Optional[DriveFile] = None
    cu_file: Optional[DriveFile] = None
    extras: List[DriveFile] = []

    for f in files:
        role = guess_role(f.name)
        if role == ROLE_OS and os_file is None:
            os_file = f
        elif role == ROLE_CU and cu_file is None:
            cu_file = f
        else:
            extras.append(f)

    # A folder holding exactly two files, one recognised and one the filename
    # says nothing about: the silent one is almost certainly the missing half.
    # A second file of a role we already filled is NOT a candidate - two
    # curricula stay two curricula.
    if (len(files) == 2 and len(extras) == 1
            and guess_role(extras[0].name) is None
            and (os_file is None) != (cu_file is None)):
        if os_file is None:
            os_file, extras = extras[0], []
        else:
            cu_file, extras = extras[0], []

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
    "DEFAULT_LIBRARY_FOLDER_ID", "library_root_id", "list_collections",
    "list_programmes", "programme_files", "guess_role", "classify",
    "cache_path", "ensure_local",
]
