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
   │        Standing rules live in LP_SYSTEM; only data varies per call.
   │        CAT rows are rebuilt deterministically and never sent.
   │        Sessions go in batches of LP_SESSION_CHUNK (8), LP_MAX_PARALLEL (3)
   │        in flight at once; a Session Plan and a single-session regenerate
   │        are one call each.
   │
   ├─(C2) VERIFIED RESOURCES    ─ resource_finder.py        [searched, not recalled]
   │      └ videos searched on YouTube, pages proposed then FETCHED to prove
   │        they exist; the model picks from the pool by number, never types
   │        a URL. Cached per unit under .resource_cache/.
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

Sessions go out in batches of **eight**, capped at 8,000 output tokens. The binding limit is
not the 1,000 requests a day but **8,000 tokens a minute** on the gpt-oss models.

### Speed

Two things follow from tokens-a-minute being the limit rather than requests-a-day.

**Batches go out together, not in a row** (`LP_MAX_PARALLEL`, three at a time). The
allowance refills continuously and is counted per *model*, so batches queued behind each
other leave it unspent while the connection sits idle. An 11-session plan took **142s** one
after another and **36s** three at a time.

**Reasoning models are told to think briefly** (`REASONING_EFFORT = "low"`). The gpt-oss and
qwen3 models bill their thinking as completion tokens, and on this task it is dead weight:
over one real batch, `low` returned *more* prose (1,032 words vs 930) in **28% fewer tokens**
(2,568 vs 3,577) and a quarter less time. Fewer tokens is a speed-up twice over — each call
is quicker, and more calls fit inside a minute's allowance. The field is only sent to model
families that accept it; others would answer a 400.

**Batches are as large as the answer allows** (`LP_SESSION_CHUNK = 8`). Measured against the
live API, the cost of a batch is mostly its *instructions*: the fixed block of the prompt is
**~1,336 tokens** and each session adds only **~94**. Every extra batch therefore resends
1,336 tokens for nothing. Eight only became possible once the model stopped thinking at
length — that halved output per session to ~456 — and a batch that still overflows is split
and retried as before.

**A refused generation is recovered rather than discarded.** Groq validates the *finished*
answer against the schema instead of constraining every token, so a model can write a
complete, correct plan and still be refused over one invented field — and it hands the whole
generation back in `failed_generation`. This used to cost the batch *and* the model, because
a 400 otherwise rules a model out. The rows are now recovered and stray keys put back where
they belong. The underlying cause is fixed too: the "Follow up Activity:" instruction now
says plainly that the line is another string in `trainee_activities`, not a field of its own.

**CAT sessions are never sent.** Their rows are rebuilt deterministically from the sessions
they assess (`_apply_cat_content`), so the model's CAT output was always discarded - about a
fifth of a term's output tokens, generated and binned. Only content sessions are batched.

### What this measures

| | 11-session plan |
|---|---|
| Sequential batches of 4 | 142s |
| Parallel batches of 4 | 36.4s |
| Parallel batches of 8 | 11.9s |
| **v2 prompts, 8 of 11 sessions generated** | **13.2s** |

That 11.9s assumes a rested token allowance. An 11-session plan costs about **8,700 tokens**,
which is roughly one minute's worth, so generating two plans back to back throttles the
second (measured 20.7s) and a third waits longer still. That is the free tier's real
throughput — about a plan a minute — and no amount of batching changes it; the fixes above
lower the bill (from ~11,300 tokens to ~8,700) rather than raise the ceiling.

## The prompts

Both prompts are split in two: the standing instructions (`LP_SYSTEM`, `SP_SYSTEM`) go in a
system message, and only the unit's data goes in the user message. Nothing that varies may
appear in the static half - one interpolated value at the top would defeat the whole point.

The original reason was Groq's prefix cache, since [cached tokens do not count against rate
limits](https://console.groq.com/docs/prompt-caching) and tokens-per-minute is what governs
speed here. **Measured on this account, it is worth much less than that promises**: the cache
hit roughly one warm call in five and topped out near 768 tokens, so expect ~150 tokens a
call, not the ~1,450 the static half contains. The split is kept because instructions and
data are different things and reading them apart is easier - not for the cache.

### Structured output, formatted here

The model returns *structure*, not formatted strings:

```
learning_outcomes   {stem, items[]}
key_points          [{heading, points[]}]
assessments         {knowledge_checks[], skills[], attitudes[]}
```

`_flatten_*` turns these into the lines the document renders. That matters because
`doc_builder` decides what to embolden by inspecting the text - `_keypoint_bold` bolds a line
only if it is at least 85% uppercase. Asking the model to capitalise its headings meant a
heading it under-capitalised rendered silently unbolded. Now the `upper()` is ours and cannot
drift; the same goes for numbering the assessment items, which the model left out on *every*
row when it was asked for.

### The self-check is code, not a request

Generation runs at `reasoning_effort: "low"` on purpose, so asking the model to verify its own
work asks for something it is configured not to do. `validate_rows` checks the rules instead -
three `a./b./c.` outcomes, no unassessable opener verbs, 2-3 key-point blocks of 2-4 points,
exactly five activity lines with `Follow up Activity:` fourth, a named active-learning method,
2-4 resources, 1-3 assessments a group, word caps, banned terminology, placeholders, and
method variety across the whole plan.

Whatever fails goes back to the model **once**, naming each fault, and whatever is still
imperfect is logged rather than retried again: a second round costs another batch's tokens for
diminishing returns, and the deterministic backfill means an imperfect row is still usable.

Building the checks revealed two rules that were wrong rather than two models that were: every
row "failed" assessment numbering (now applied here), and `Think-Pair-Share` written with a
non-breaking hyphen was reported as naming no method (now matched through the typography).

### Resources are found on the internet, and proved to exist

`resource_finder.py`. The Resources column used to be whatever the model
remembered, and a model's memory of a URL is a guess that reads like a fact.
Measured against the live API, `openai/gpt-oss-120b` was asked for ten resources
for one unit: **three resolved**, and of the three YouTube links it gave,
**none existed** - the video ids were well-formed inventions.

Two rules replace that.

**Candidates come from an index where one exists.** Videos are searched on
YouTube's own results page, so a real video id is never guessed to begin with.
No key is needed: the page embeds its data as JSON. Pages are still proposed by
the model, but asked for stable landing pages rather than deep links, because a
deep link is the shape that usually turns out to be invented.

**Nothing is returned until it has been fetched.** Every URL is requested before
it can reach a document. YouTube gets a stronger check than a fetch: a deleted
or invented watch page still answers `200` with an "unavailable" notice, so the
oEmbed endpoint is used instead - it 404s for an id that does not exist, and
hands back the video's **real title and channel**, which is how a resource ends
up named accurately rather than as the model described it.

**The model never writes a URL.** The verified pool is listed in the user
message and the model chooses *by number*; `_flatten_resources` renders the line
from the pool. It cannot mistype or invent what it does not type, and
`validate_rows` rejects anything that is not a number on the list.

One pool per unit, not per session: a unit's sessions cover one subject, a
search each would multiply the slowest step by eight, and a trainer reading the
plan gets a coherent set of references. Pools are cached under
`.resource_cache/` for a fortnight, so only the first plan for a unit pays for
the search (measured: 37s uncached, 0s after).

If nothing can be verified - offline, or a search that fails - the plan is still
produced, with resources named in words and no links, and the run log says so.
A plan is never lost to this.

### Rows carry their own identity

Each session has a `session_id` (`W3-S1`) which the model echoes back, and rows are matched to
sessions by it. A short, reordered or duplicated answer can no longer slide content onto the
wrong session - the failure that positional matching needed padding to survive. A model that
drops the field falls back to pairing in order.

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
- **AI calls** use JSON schema mode + `max_tokens 8000`, batched so long plans
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
