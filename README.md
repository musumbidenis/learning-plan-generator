# Learning Plan Generator

Generates a Kenya **KSTVET Learning Plan** (REF `KTTC/TP/LP/F07`, RVNP format) as a
downloadable Word (`.docx`) from an **Occupational Standard** + a **Curriculum**.

Core design principle: **extraction is deterministic (no AI); only the generative
session content uses AI, in grounded, schema-constrained API calls.**

## Pipeline

```
[OS file] + [Curriculum file]   ← PDF or any Word format; Drive library or upload
   │
   ├─(A) DETERMINISTIC PARSERS  ─ os_parser.py / curriculum_parser.py   [no AI]
   │      └ word-coordinate column splitting (anti-bleed); regex structure
   │
   ├─(B) DETERMINISTIC PLANNER  ─ planner.py                            [no AI]
   │      └ one session per curriculum sub-topic, PCs mapped 1:1, CATs placed
   │
   ├─(C) GROUNDED GROQ CALLS    ─ ai_client.py                          [the only AI]
   │      └ fills learning_outcomes / activities / resources / assessments,
   │        grounded in parsed data; JSON schema mode forces valid JSON.
   │        Sessions go in batches of LP_SESSION_CHUNK (4); a Session Plan and
   │        a single-session regenerate are one call each.
   │
   └─(D) DOC BUILDER            ─ doc_builder.py                        [no AI]
          └ A4, Times New Roman, Table Grid, 9-column RVNP session table
```

Stages A, B and D are pure Python and unit-tested with **zero API calls**.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your GROQ_API_KEY
streamlit run app.py
```

The API key is sent as a Bearer token to Groq's OpenAI-compatible chat
completions API. `GROQ_MODEL` (default `openai/gpt-oss-120b`) says which model to
*prefer* — which model actually generates is settled at run time.

**Why Groq.** The free tier allows **1,000 requests a day per model** with no card, and
`openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` honour
`response_format: json_schema` with `strict: true` by *constrained decoding* — the
guarantee everything downstream is built on. What was measured before choosing it:

| | Free tier | Per day | Per minute | Strict JSON schema |
|---|---|---|---|---|
| **Groq** | Yes, no card | **1,000** | 30 | **Yes** — constrained decoding |
| OpenRouter | Yes | **50** (1,000 after a one-off $10) | 20 | Varies by routed endpoint |
| Google Gemini | Yes | shown in AI Studio | ~15 | Yes, but a different request shape |
| Cerebras | $5 trial credit only | — | — | — |
| Mistral (was here) | Yes | — | 0 for the capable models on a free plan | Yes |

**How the model is chosen.** Not every model an account can see will take a strict
schema, and a model's daily allowance can run out — so rather than carry a list of second
choices, the app asks `/v1/models` what this key can see, orders the candidates
most-capable-first, and tries them until one answers a real request. The preferred model
leads; after that it is Groq's own list, so a model released tomorrow is tried on its
merits. Speech, safety and embedding models are skipped — they would answer, they just
write poor lesson prose. The choice is settled once per process and named in the run log.

A model that answers a probe and then can't complete a real batch is not a model that
works: two timeouts running, or three dropped connections, and the next candidate takes
over mid-generation rather than the plan failing. One timeout is usually the API being
busy, so it is retried on the same model first. A 429 is read against its reset: a busy
minute is waited out (3s, 8s, 15s, 30s, honouring `Retry-After`), while a reset hours away
is the day's allowance and moves to a model with its own. 401, and any 403, still surface
as a clear "key invalid/expired" message.

Sessions go out in batches of **four**, capped at 6,000 output tokens. The binding limit is
not the 1,000 requests a day but **8,000 tokens a minute** on the gpt-oss models — eight
sessions would spend a whole minute's allowance in one request.

## The Drive document library (optional)

Instead of uploading the two PDFs every time, the app can read them from the
shared Google Drive folder they already live in:

```text
Occupational Standards and CBET Curriculum/   ← the library root
  Accountancy Level 6/                        ← a programme
    Occupational Standards.pdf
    Curriculum.pdf
  ICT Technician Level 6/
  ...                                         ← ~290 programmes
```

Pick a **programme** - that is the whole instruction (the box is type-to-search).
Which file is the Occupational Standard and which the Curriculum is worked out
from the filenames and both are fetched automatically; a correction control
appears only when the folder is ambiguous (an unrecognised file, or more than
two of them). Where a programme carries the same document twice, once as `.pdf`
and once as `.docx`, the PDF is preferred — it keeps the page geometry the
parsers read columns from, and Word's automatic list numbering survives in it as
text. A folder holding only one of the two documents still works: the missing
side gets a file uploader.

A library that groups programmes one level deeper —

```text
Training Tools/
  CDACC CYCLE 03/                    ← a collection
    ICT Technician Level 6/          ← a programme
```

— also works: the app looks at what the root's subfolders contain and adds a
**Collection** box only when it needs one. Nothing to configure.

**Read-only, on purpose.** The app never writes to Drive. To add a programme,
create its folder in Drive, drop the files in, and press **🔄 Refresh library**.

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
| `drive_library.py` | – | programme → OS/Curriculum, the folder-shape detection, plus the local download cache |
| `word_reader.py` | A0 | read any word-processor format (.docx/.docm/.dotx/.dotm, .doc, .rtf, .odt/.ott) into neutral blocks |
| `unit_index.py` | A0 | finds the units in a document: reads its preliminary units table, then locates each one |
| `unit_match.py` | – | pairs OS units with Curriculum units for the matched selection table |
| `pdf_utils.py` | – | word-coordinate column splitting, document loading, noise filtering |
| `os_parser.py` | A1 | parse OS units: title, codes, level, description, elements + PCs, evidence-guide methods |
| `curriculum_parser.py` | A2 | parse curriculum LOs → sub-topics (sessions) + key points + suggested methods |
| `planner.py` | B | build the session skeleton, map PCs, place CATs, stamp the schedule |
| `ai_client.py` | C | the grounded Groq calls + model discovery + coercion/re-stamp/backfill safety nets |
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
- **Word documents that aren't `.docx`** → `word_reader.py` reads the whole
  family in pure Python — no LibreOffice or Word install required, for the same
  reason OCR was left out. `.docx`/`.docm`/`.dotx`/`.dotm` come from
  `word/document.xml` directly (python-docx rejects several of them on content
  type alone), `.odt`/`.ott` from `content.xml`, `.rtf` from a tokeniser that
  keeps `\cell`/`\row` tables, and legacy `.doc` from the Word 97 piece table
  via `olefile`. The format is decided by **sniffing magic bytes, never the
  extension**, so a `.docx` someone renamed to `.doc` still opens. Verified on
  real CDACC files: three `.doc` documents that previously could not be opened
  at all now yield 15, 14 and 10 units with their tables intact.
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
- **Mismatched OS/Curriculum units** → `unit_match.py` pairs the two unit lists
  automatically, cascading ISCED code → TVET code ignoring the `OS`/`CU` segment
  → title → fuzzy title, and the matched pairs are what step 2 leads with. What
  it *cannot* place is the point: those units are listed separately, per
  document, to be paired by hand. The two files word the same unit very
  differently ("Perform Secure Computer Operations" against "Computer
  Operations") and a unit often exists in only one of them, so nothing is
  dropped for failing to match. On the real Cyber Security pair, 14 of 14 OS
  units match on the ISCED code and one curriculum unit is left to pair by hand.

## Tests

```bash
python -m pytest tests/ -q
```

Covers both parsers, a PC-coverage regression (every OS unit with elements must
yield performance criteria), the planner, the AI safety nets, the doc builder,
the Drive library's filename classification and cache keying, and unit pairing —
all with **zero API calls and no network**. The parser tests that need the sample
PDFs skip when those files aren't present.
