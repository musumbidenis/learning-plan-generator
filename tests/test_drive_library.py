"""Filename role-guessing and the local download cache - no network."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import drive_library as dl
from drive_client import DriveFile

# Names taken from the library itself: nearly every one of its ~290 programme
# folders holds 'Curriculum.pdf' + 'Occupational Standard(s).pdf', and the ones
# that don't are the interesting cases below.
CURRICULUM_REAL = "Moduralized Curriculum ICT Operator Level 4.V1 18 MARCH.pdf"

WORD_MIME = ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document")


def f(name, file_id=None, modified="2026-05-14T08:31:17Z"):
    return DriveFile(id=file_id or name, name=name, mime_type="application/pdf",
                     modified_time=modified, size=1234)


def word(name, file_id=None):
    return DriveFile(id=file_id or name, name=name, mime_type=WORD_MIME,
                     modified_time="2026-05-14T08:31:17Z", size=1234)


def folder(name, file_id=None):
    return DriveFile(id=file_id or name, name=name,
                     mime_type="application/vnd.google-apps.folder")


# --------------------------------------------------------------------------- #
# guess_role
# --------------------------------------------------------------------------- #
def test_guesses_curriculum_from_the_real_library_filename():
    assert dl.guess_role(CURRICULUM_REAL) == dl.ROLE_CU


def test_guesses_occupational_standard_from_words_and_from_a_bare_token():
    assert dl.guess_role("ICT Technician Level 5 Occupational Standard.pdf") == dl.ROLE_OS
    assert dl.guess_role("OS-ICT-Technician-L5.pdf") == dl.ROLE_OS
    assert dl.guess_role("ICT OS Level 6.docx") == dl.ROLE_OS


def test_curriculum_wins_when_a_filename_carries_an_os_unit_code():
    # a curriculum whose name embeds an IT/OS/... code must not read as an OS
    assert dl.guess_role("Curriculum IT/OS/ICTA/CC/02/5/MA.pdf") == dl.ROLE_CU


def test_unrecognisable_filename_yields_no_role():
    assert dl.guess_role("Level 6 final draft v2.pdf") is None
    assert dl.guess_role("") is None


def test_os_token_is_not_matched_inside_another_word():
    assert dl.guess_role("Composite Materials Level 5.pdf") is None


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
def test_classifies_a_complete_programme_folder():
    files = [f("ICT Technician L5 Occupational Standard.pdf"),
             f("ICT Technician L5 Curriculum.pdf")]
    os_file, cu_file, extras = dl.classify(files)
    assert os_file is files[0]
    assert cu_file is files[1]
    assert extras == []


def test_curriculum_only_folder_leaves_the_os_side_empty():
    files = [f(CURRICULUM_REAL)]
    os_file, cu_file, extras = dl.classify(files)
    assert os_file is None
    assert cu_file is files[0]
    assert extras == []


def test_empty_folder_classifies_to_nothing():
    assert dl.classify([]) == (None, None, [])


def test_two_files_with_one_recognised_assumes_the_other_is_the_missing_half():
    files = [f("ICT Technician L5 Curriculum.pdf"), f("ICTT L5 final.pdf")]
    os_file, cu_file, extras = dl.classify(files)
    assert cu_file is files[0]
    assert os_file is files[1]
    assert extras == []


def test_a_stray_third_file_is_surfaced_rather_than_guessed_at():
    files = [f("OS ICT L5.pdf"), f("Curriculum ICT L5.pdf"), f("Notes.docx")]
    os_file, cu_file, extras = dl.classify(files)
    assert os_file is files[0]
    assert cu_file is files[1]
    assert extras == [files[2]]


def test_a_duplicate_role_goes_to_extras_and_is_never_dropped():
    files = [f("Curriculum v1.pdf"), f("Curriculum v2.pdf")]
    os_file, cu_file, extras = dl.classify(files)
    assert os_file is None
    assert cu_file is files[0]
    assert extras == [files[1]]


def test_every_input_file_is_accounted_for():
    files = [f("OS ICT L5.pdf"), f("Curriculum ICT L5.pdf"),
             f("Notes.docx"), f("Curriculum old.pdf")]
    os_file, cu_file, extras = dl.classify(files)
    seen = [x for x in (os_file, cu_file) if x is not None] + extras
    assert sorted(id(x) for x in seen) == sorted(id(x) for x in files)


def test_the_pdf_copy_wins_over_the_word_copy_of_the_same_document():
    """Dozens of programme folders hold both; the PDF is the better source."""
    files = [word("Curriculum.docx"), f("Curriculum.pdf"),
             word("Occupational Standards.docx"), f("Occupational Standards.pdf")]
    os_file, cu_file, extras = dl.classify(files)
    assert os_file.name == "Occupational Standards.pdf"
    assert cu_file.name == "Curriculum.pdf"
    assert [e.name for e in extras] == ["Curriculum.docx",
                                        "Occupational Standards.docx"]


def test_a_word_copy_is_used_when_it_is_the_only_one():
    files = [word("Curriculum.docx"), word("Occupational Standard.docx")]
    os_file, cu_file, _ = dl.classify(files)
    assert os_file is files[1] and cu_file is files[0]


# --------------------------------------------------------------------------- #
# is_readable - what belongs in a programme folder's file list
# --------------------------------------------------------------------------- #
def test_macos_appledouble_stubs_are_not_offered_as_documents():
    """A real folder holds nine 4 KB '._' stubs beside the two real PDFs."""
    junk = f("._Wildlife management curriculum level 5.pdf")
    assert not dl.is_readable(junk)
    assert dl.is_readable(f("curriculum.pdf"))


def test_finder_and_word_litter_is_ignored():
    assert not dl.is_readable(f(".DS_Store"))
    assert not dl.is_readable(word("~$Curriculum.docx"))


def test_a_google_doc_without_an_extension_is_readable():
    gdoc = DriveFile(id="1", name="Occupational Standard",
                     mime_type="application/vnd.google-apps.document")
    assert dl.is_readable(gdoc)


def test_folders_and_unreadable_formats_are_not_documents():
    assert not dl.is_readable(folder("Assessment Guides"))
    assert not dl.is_readable(f("Trainee list.xlsx"))


# --------------------------------------------------------------------------- #
# detect_layout - programmes at the root, or grouped under collections
# --------------------------------------------------------------------------- #
def _fake_children(mapping, monkeypatch):
    monkeypatch.setattr(dl, "list_children", lambda fid: mapping.get(fid, []))


def test_folders_holding_documents_are_programmes(monkeypatch):
    top = [folder("Accountancy Level 6", "a"), folder("Rigging Level 4", "b")]
    _fake_children({"a": [f("Curriculum.pdf"), f("Occupational Standards.pdf")]},
                   monkeypatch)
    assert dl.detect_layout(top) == dl.LAYOUT_FLAT


def test_folders_holding_only_folders_are_collections(monkeypatch):
    top = [folder("CDACC CYCLE 03", "a"), folder("CDACC CYCLE 04", "b")]
    _fake_children({"a": [folder("ICT Technician Level 6", "a1")],
                    "b": [folder("ICT Level 4", "b1")]}, monkeypatch)
    assert dl.detect_layout(top) == dl.LAYOUT_NESTED


def test_an_empty_first_programme_folder_does_not_decide_the_layout(monkeypatch):
    top = [folder("Rigging Level 4", "a"), folder("Accountancy Level 6", "b")]
    _fake_children({"a": [], "b": [f("Curriculum.pdf")]}, monkeypatch)
    assert dl.detect_layout(top) == dl.LAYOUT_FLAT


def test_a_folder_that_cannot_be_read_does_not_decide_the_layout(monkeypatch):
    top = [folder("Private", "a"), folder("Accountancy Level 6", "b")]

    def children(fid):
        if fid == "a":
            raise dl.DriveError("no permission")
        return [f("Curriculum.pdf")]

    monkeypatch.setattr(dl, "list_children", children)
    assert dl.detect_layout(top) == dl.LAYOUT_FLAT


def test_a_library_with_no_folders_at_all_is_not_an_error(monkeypatch):
    _fake_children({}, monkeypatch)
    assert dl.detect_layout([]) == dl.LAYOUT_NESTED


# --------------------------------------------------------------------------- #
# cache_path
# --------------------------------------------------------------------------- #
def test_cache_path_changes_when_the_document_is_replaced_in_drive():
    before = f("Curriculum.pdf", file_id="abc123", modified="2026-05-14T08:31:17Z")
    after = f("Curriculum.pdf", file_id="abc123", modified="2026-09-01T10:00:00Z")
    assert dl.cache_path(before) != dl.cache_path(after)


def test_cache_path_is_stable_for_an_unchanged_document():
    a = f("Curriculum.pdf", file_id="abc123")
    b = f("Curriculum.pdf", file_id="abc123")
    assert dl.cache_path(a) == dl.cache_path(b)


def test_cache_path_keeps_files_of_different_ids_apart():
    a = f("Curriculum.pdf", file_id="abc123")
    b = f("Curriculum.pdf", file_id="def456")
    assert dl.cache_path(a) != dl.cache_path(b)


def test_cache_path_uses_the_downloadable_extension():
    pdf = DriveFile(id="1", name="Curriculum.pdf", mime_type="application/pdf",
                    modified_time="x")
    gdoc = DriveFile(id="2", name="Curriculum",
                     mime_type="application/vnd.google-apps.document",
                     modified_time="x")
    assert dl.cache_path(pdf).endswith(".pdf")
    # Google Docs are exported as .docx, so that is what lands on disk
    assert dl.cache_path(gdoc).endswith(".docx")


def test_cache_path_never_escapes_the_cache_directory():
    hostile = DriveFile(id="../../etc/passwd", name="x.pdf", modified_time="x")
    assert os.path.dirname(os.path.abspath(dl.cache_path(hostile))) == \
        os.path.abspath(dl.CACHE_DIR)


# --------------------------------------------------------------------------- #
# The API key must never reach a user-facing message or the run log
# --------------------------------------------------------------------------- #
import drive_client  # noqa: E402


def test_the_api_key_is_scrubbed_from_error_text(monkeypatch):
    key = "AIzaSyBExampleKeyValue1234567890abcdef"
    monkeypatch.setattr(drive_client, "load_api_key", lambda: key)
    leaked = f"Max retries exceeded with url: /drive/v3/files?key={key}&q=x"
    scrubbed = drive_client._scrub(leaked)
    assert key not in scrubbed
    assert "<GOOGLE_API_KEY>" in scrubbed


def test_any_key_shaped_token_is_scrubbed_even_when_none_is_configured(monkeypatch):
    monkeypatch.setattr(drive_client, "load_api_key", lambda: "")
    other = "AIzaSyDifferentKey0987654321zyxwvu"
    assert other not in drive_client._scrub(f"failed with key={other}")


def test_requests_are_authenticated_by_header_not_by_query_string(monkeypatch):
    """The key in a URL would land in exception text, proxy logs and referrers."""
    key = "AIzaSyBExampleKeyValue1234567890abcdef"
    monkeypatch.setattr(drive_client, "load_api_key", lambda: key)
    seen = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"files": []}

    def fake_get(url, params=None, headers=None, **kw):
        seen["params"] = params or {}
        seen["headers"] = headers or {}
        return FakeResponse()

    monkeypatch.setattr(drive_client.requests, "get", fake_get)
    drive_client.list_children("some-folder-id")

    assert seen["headers"].get("X-goog-api-key") == key
    assert not any(key in str(v) for v in seen["params"].values())
    assert "key" not in seen["params"]
