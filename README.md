# Learning Plan Generator

Generates a Kenya **KSTVET Learning Plan** (REF `KTTC/TP/LP/F07`, RVNP format) as a
downloadable Word (`.docx`) from an **Occupational Standard** + a **Curriculum**.

Core design principle: **extraction is deterministic (no AI); only the generative
session content uses AI, in exactly ONE grounded API call.**

## Pipeline

```
[OS file] + [Curriculum file]
   │
   ├─(A) DETERMINISTIC PARSERS  ─ os_parser.py / curriculum_parser.py   [no AI]
   │      └ word-coordinate column splitting (anti-bleed); regex structure
   │
   ├─(B) DETERMINISTIC PLANNER  ─ planner.py                            [no AI]
   │      └ one session per curriculum sub-topic, PCs mapped 1:1, CATs placed
   │
   ├─(C) ONE MISTRAL CALL       ─ ai_client.py                          [the only AI]
   │      └ fills learning_outcomes / activities / resources / assessments,
   │        grounded in parsed data; JSON schema mode forces valid JSON
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
model chain: `mistral-large-latest`, then `mistral-medium-latest`, then
`mistral-small-latest`; HTTP 429 advances to the next model, while 401/403
surfaces a clear "key invalid/expired" message.

**Offline mode**: leave the key blank (or tick *Offline mode*) and the app fills
the generative columns with grounded deterministic defaults — you still get a
complete, correctly-shaped document.

## Modules

| File | Module | Role |
|------|--------|------|
| `models.py` | – | dataclasses flowing through the pipeline |
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
- **Single AI call** uses JSON schema mode + `max_tokens 32000` so long plans
  never truncate; deterministic schedule is re-stamped so the AI can't override it.

## Tests

```bash
python -m pytest tests/ -q
```

36 tests covering both parsers (against the bundled sample PDFs), a PC-coverage
regression (every OS unit with elements must yield performance criteria), the
planner, the AI safety nets, and the doc builder — all with **zero API calls**.
