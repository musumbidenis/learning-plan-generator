"""Minimal Google Drive REST v3 client - read-only, API-key authenticated.

Why an API key and not OAuth: the library folder is already shared "anyone with
the link -> Viewer", so no user consent is needed to read it. An API key is a
single string in `.env` - no consent screen, no service account, no token file
to refresh. What a share link alone can NOT do is list a folder's children:
Google exposes no unauthenticated listing endpoint, which is the one thing this
module exists to provide.

Deliberately Streamlit-free so it stays unit-testable; `app.py` owns the
caching of these calls.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

import config
import runlog

DRIVE_FILES_ENDPOINT = "https://www.googleapis.com/drive/v3/files"

# Google-native formats have no bytes to download; they must be exported. Only
# Docs matter here (a Curriculum occasionally lives as a Google Doc), and .docx
# is what python-docx reads.
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
DOCX_MIME = ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document")
FOLDER_MIME = "application/vnd.google-apps.folder"

# Suffixes a downloaded file may legitimately keep (see word_reader).
_KNOWN_EXTENSIONS = (".pdf", ".docx", ".docm", ".dotx", ".dotm", ".doc",
                     ".dot", ".rtf", ".odt", ".ott")

LIST_TIMEOUT = 30          # seconds
DOWNLOAD_TIMEOUT = 180     # a curriculum PDF can be several MB
_PAGE_SIZE = 200


class DriveError(Exception):
    """Any failure talking to Drive, already phrased for the user."""


@dataclass
class DriveFile:
    """One Drive item - a folder or a document."""
    id: str = ""
    name: str = ""
    mime_type: str = ""
    modified_time: str = ""       # RFC 3339; part of the local cache key
    size: int = 0

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER_MIME

    @property
    def is_google_doc(self) -> bool:
        return self.mime_type == GOOGLE_DOC_MIME

    @property
    def extension(self) -> str:
        """The suffix the downloaded bytes will actually have."""
        if self.is_google_doc:
            return ".docx"                 # we export Docs as .docx
        ext = os.path.splitext(self.name)[1].lower()
        # keep the real suffix for every format the readers understand; the
        # loader sniffs the content anyway, so an odd one still opens
        return ext if ext in _KNOWN_EXTENSIONS else ".pdf"


def load_api_key() -> str:
    """The Google API key, from GOOGLE_API_KEY (env / .env, then st.secrets)."""
    return config.get("GOOGLE_API_KEY")


def is_configured() -> bool:
    return bool(load_api_key())


def _require_key() -> str:
    key = load_api_key()
    if not key:
        raise DriveError(
            "No Google API key configured. Set GOOGLE_API_KEY in .env "
            "(Google Cloud console -> enable the Drive API -> create an API key).")
    return key


def _explain(status: int, detail: str) -> str:
    """Turn an HTTP status into something a trainer can act on."""
    # A malformed or revoked key comes back as 400, not 401 - say so plainly
    # rather than leaving the reader with a bare status code.
    if status == 400 and "api key not valid" in detail.lower():
        return ("Google rejected the API key as invalid. Check the GOOGLE_API_KEY "
                "value in .env against the one in the Cloud console.")
    # Two different 403s, with two different fixes - saying "check your
    # restrictions" for the first one sends the reader to the wrong screen.
    low = detail.lower()
    if status == 403 and "has not been used in project" in low:
        return ("The Google Drive API isn't enabled on the project this key "
                "belongs to. Open the link in the message below, press Enable, "
                "and give it a minute to propagate. " + detail)
    if status == 403 and "are blocked" in low:
        return ("This API key is restricted to a subset of Drive methods, and "
                "the one the app needs isn't in it. In the Cloud console open "
                "Credentials -> your key -> API restrictions, and either pick "
                "'Don't restrict key' or select the Google Drive API without "
                "narrowing it to individual methods. " + detail)
    if status in (401, 403):
        return ("Google rejected the API key (HTTP %d). Check that the Drive API "
                "is enabled on the project and that the key's API restrictions "
                "allow it. %s" % (status, detail))
    if status == 404:
        return ("That folder or file wasn't found (HTTP 404) - it may have been "
                "moved, or it is no longer shared with 'anyone with the link'. "
                + detail)
    if status == 429:
        return "Google is rate-limiting the requests (HTTP 429). Try again shortly."
    return f"Drive request failed (HTTP {status}). {detail}"


def _scrub(text: str) -> str:
    """Remove the API key from anything on its way to a user or a log.

    requests puts the full request URL into its exception messages, so a plain
    DNS failure would otherwise print the key into the Streamlit error box and
    the run log. The key travels in a header rather than the query string, but
    this stays as a second line of defence for redirects and echoed input.
    """
    key = load_api_key()
    text = str(text)
    if key:
        text = text.replace(key, "<GOOGLE_API_KEY>")
    # any key-shaped token, ours or not, never belongs in output
    return re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "<GOOGLE_API_KEY>", text)


def _get(url: str, params: dict, *, stream: bool = False,
         timeout: int = LIST_TIMEOUT) -> requests.Response:
    """GET a Drive endpoint, authenticating with the key as a header.

    The key goes in X-goog-api-key, NOT in the query string, so it can't leak
    through an exception message, a proxy log or a browser referrer.
    """
    headers = {"X-goog-api-key": _require_key()}
    try:
        resp = requests.get(url, params=params, headers=headers, stream=stream,
                            timeout=timeout)
    except requests.RequestException as e:
        raise DriveError(f"Couldn't reach Google Drive: {_scrub(e)}") from None
    if resp.status_code != 200:
        detail = ""
        try:
            detail = str(resp.json().get("error", {}).get("message", ""))[:300]
        except Exception:                  # noqa: BLE001 - body may not be JSON
            detail = resp.text[:200]
        raise DriveError(_scrub(_explain(resp.status_code, detail)))
    return resp


def list_children(folder_id: str) -> List[DriveFile]:
    """Every non-trashed direct child of `folder_id`, folders first then by name."""
    _require_key()
    out: List[DriveFile] = []
    page_token: Optional[str] = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,size)",
            "orderBy": "folder,name",
            "pageSize": _PAGE_SIZE,
            # public link-shared content lives outside any shared drive, but
            # these keep the query valid if the folder is ever moved into one
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        payload = _get(DRIVE_FILES_ENDPOINT, params).json()
        for f in payload.get("files", []):
            out.append(DriveFile(
                id=f.get("id", ""),
                name=f.get("name", ""),
                mime_type=f.get("mimeType", ""),
                modified_time=f.get("modifiedTime", ""),
                size=int(f.get("size") or 0),
            ))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    runlog.log(f"Drive: listed {len(out)} item(s) in folder {folder_id}")
    return out


def download(file: DriveFile, dest_path: str) -> str:
    """Stream `file` to `dest_path`, exporting Google Docs to .docx.

    Writes to a temporary sibling first so an interrupted download can never
    leave a truncated file behind for the cache to trust.
    """
    _require_key()
    if file.is_google_doc:
        url = f"{DRIVE_FILES_ENDPOINT}/{file.id}/export"
        params = {"mimeType": DOCX_MIME}
    else:
        url = f"{DRIVE_FILES_ENDPOINT}/{file.id}"
        params = {"alt": "media", "supportsAllDrives": "true"}

    resp = _get(url, params, stream=True, timeout=DOWNLOAD_TIMEOUT)
    partial = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    try:
        with open(partial, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                if chunk:
                    fh.write(chunk)
        os.replace(partial, dest_path)
    except OSError as e:
        raise DriveError(f"Couldn't save '{file.name}' locally: {e}") from e
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
    runlog.log(f"Drive: downloaded '{file.name}' "
               f"({os.path.getsize(dest_path)} bytes)")
    return dest_path
