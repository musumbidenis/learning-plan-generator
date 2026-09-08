"""Filename role-guessing and the local download cache - no network."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import drive_library as dl
from drive_client import DriveFile

# The one document actually in the library today, plus the shapes we expect to
# meet as the folders fill up.
CURRICULUM_REAL = "Moduralized Curriculum ICT Operator Level 4.V1 18 MARCH.pdf"


def f(name, file_id=None, modified="2026-05-14T08:31:17Z"):
    return DriveFile(id=file_id or name, name=name, mime_type="application/pdf",
                     modified_time=modified, size=1234)


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
