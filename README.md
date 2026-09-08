# Learning Plan Generator

Generates a Kenya **KSTVET Learning Plan** (REF `KTTC/TP/LP/F07`, RVNP format) as a
downloadable Word (`.docx`) from an **Occupational Standard** + a **Curriculum**.

Core design principle: **extraction is deterministic (no AI); only the generative
session content uses AI, in grounded, schema-constrained API calls.**

## Pipeline

```
[OS file] + [Curriculum file]   ← from the Drive library, or uploaded
   │
   ├─(A) DETERMINISTIC PARSERS  ─ os_parser.py / curriculum_parser.py   [no AI]
   │      └ word-coordinate column splitting (anti-bleed); regex structure
   │
   ├─(B) DETERMINISTIC PLANNER  ─ planner.py                            [no AI]
   │      └ one session per curriculum sub-topic, PCs mapped 1:1, CATs placed
   │
   ├─(C) GROUNDED MISTRAL CALLS ─ ai_client.py                          [the only AI]
   │      └ fills learning_outcomes / activities / resources / assessments,
   │        grounded in parsed data; JSON schema mode forces valid JSON.
   │        Sessions go in batches of LP_SESSION_CHUNK (8); a Session Plan and
   │        a single-session regenerate are one call each.
   │
   └─(D) DOC BUILDER            ─ doc_builder.py                        [no AI]
          └ A4, Times New Roman, Table Grid, 9-column RVNP session table
```

Stages A, B and D are pure Python and unit-tested with **zero API calls**.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your MISTRAL_API_KEY
streamlit run app.py
```

The API key is sent as a Bearer token to Mistral's chat completions API. Default
model: `mistral-small-latest` (override with `MISTRAL_MODEL`); 401/403 surfaces a
clear "key invalid/expired" message and 429 is reported as a Mistral API error.

## The Drive document library (optional)

Instead of uploading the two PDFs every time, the app can read them from the
shared Google Drive folder they already live in:

```text
Training Tools/                      ← the library root
  CDACC CYCLE 03/                    ← a collection
    ICT Technician Level 6/          ← a programme
      <Occupational Standard>.pdf
      <Curriculum>.pdf
  CDACC CYCLE 04 .../
  RVNP/
```

Pick a **collection** then a **programme**, and both documents are fetched and
indexed for you. Filenames follow no convention, so which file is the
Occupational Standard and which the Curriculum is *guessed* from the name and
shown in two dropdowns you can correct. A programme folder holding only one of
the two still works: the missing side gets a file uploader.

**Read-only, on purpose.** The app never writes to Drive. To add a programme —
or a whole new collection such as `RVNP` — create the folder in Drive, drop the
files in, and press **🔄 Refresh library**.

Setup (~3 minutes, no OAuth consent screen and no service account):

1. [Google Cloud console](https://console.cloud.google.com) → create a project.
2. **APIs & Services → Library → Google Drive API → Enable**.
3. **APIs & Services → Credentials → Create credentials → API key**.
4. Restrict the key → **API restrictions → Google Drive API**.
5. Put `GOOGLE_API_KEY=...` in `.env` (and `DRIVE_LIBRARY_FOLDER_ID` if you are
   pointing at a different folder).

The library folder must be shared **Anyone with the link → Viewer**; that is
what lets an API key alone read it. A share link on its own is *not* enough —
Google offers no unauthenticated way to list a folder's contents, which is
exactly what the key provides. Leave `GOOGLE_API_KEY` unset and the app simply
falls back to the upload flow.

Fetched documents are cached under `.drive_cache/`, keyed on the Drive file id
*and* its `modifiedTime` — so a document replaced in Drive is re-fetched
automatically, and an unchanged one is downloaded only once.

## Modules

| File | Module | Role |
|------|--------|------|
| `models.py` | – | dataclasses flowing through the pipeline |
| `config.py` | – | one place resolving settings: `.env` → environment → `st.secrets` |
| `drive_client.py` | – | read-only Google Drive REST v3 calls (API key, no OAuth) |
| `drive_library.py` | – | collection → programme → OS/Curriculum, plus the local download cache |
| `unit_index.py` | A0 | finds the units in a document: reads its preliminary units table, then locates each one |
| `unit_match.py` | – | pairs OS units with Curriculum units for the matched selection table |
| `pdf_utils.py` | – | word-coordinate column splitting, PDF/DOCX loading, noise filtering |
| `os_parser.py` | A1 | parse OS units: title, codes, level, description, elements + PCs, evidence-guide methods |
| `curriculum_parser.py` | A2 | parse curriculum LOs → sub-topics (sessions) + key points + suggested methods |
| `planner.py` | B | build the session skeleton, map PCs, place CATs, stamp the schedule |
| `ai_client.py` | C | the single grounded Mistral call + coercion/re-stamp/backfill safety nets |
| `doc_builder.py` | D | render the `.docx` (header table + 9-column session table) |
| `app.py` | – | Streamlit UI wiring all stages |

## Key robustness measures (lessons baked in)

- **Two-column bleed** → columns split by word x-coordinates (`detect_column_split`),
  never by `extract_text()` line order. Region bounded by y-coordinate so RANGE
  text never leaks into the last PC.
- **Char-splitting bug** → `_as_list()` normalises any string/dict/None into a real
  list before rendering; cells are written one paragraph per line.
- **Source typos / format drift** → tolerant markers (`PERFORMANCE CRETIRIA`),
  PC numbering `1.1` *and* `1.1.text`, per-page bullet-column detection.
- **Footer noise** → bare numbers and `©TVET CDACC 2025` filtered; mangled `©` repaired.
- **AI calls** use JSON schema mode + `max_tokens 14000`, batched so long plans
  never truncate, with a salvage pass that recovers the complete leading objects
  of a cut-off array; the deterministic schedule is re-stamped afterwards so the
  AI can't override it.
- **Documents the shape detector couldn't read** → `unit_index.py` tries four
  strategies and keeps whichever finds the most units, so nothing that parses
  today parses worse. The best of them reads the **preliminary units table**
  ("Summary of Units of Learning", "UNIT CATEGORY | UNIT CODE | UNITS NAME"),
  which names every unit up front, then goes and locates each one — turning
  detection from a guess into a search for known targets. Three shapes that
  previously yielded **zero** units now read correctly: documents carrying only
  TVET codes and no ISCED code, multi-unit `.docx` files, and headers that print
  the code above the title. Units the roster names but that can't be located are
  reported in the UI rather than silently dropped.
- **Mismatched OS/Curriculum units** → the two unit lists are paired into one
  table (`unit_match.py`), cascading ISCED code → TVET code ignoring the `OS`/`CU`
  segment → title → fuzzy title. A unit found in only one document still gets a
  row rather than vanishing, and title-only pairings are flagged as unconfirmed.

## Tests

```bash
python -m pytest tests/ -q
```

Covers both parsers, a PC-coverage regression (every OS unit with elements must
yield performance criteria), the planner, the AI safety nets, the doc builder,
the Drive library's filename classification and cache keying, and unit pairing —
all with **zero API calls and no network**. The parser tests that need the sample
PDFs skip when those files aren't present.
