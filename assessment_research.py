"""Real CDACC questions, fetched from public past-paper repositories.

The generator writes good questions from the taught content, but it writes
them the way a language model writes questions. What it cannot get from a
curriculum is how a TVET CDACC paper actually SOUNDS - how much situation goes
in front of the verb, how many marks a "differentiate" question carries, how a
real setter phrases the ask. That is not in any syllabus. It is in the papers.

So the papers are fetched. Kenyan institutions publish theirs in public DSpace
repositories, which carry a real search API - and that matters, because
general web search is not available here: DuckDuckGo answers a bot challenge
and Bing serves fallback junk (a query about malware came back about book
genres). A repository that indexes itself is worth more than a search engine
that will not talk to us.

WHAT AN EXEMPLAR IS FOR, AND WHAT IT IS NOT FOR
An exemplar is a STYLE reference. It shows the shape of a question, never the
subject of one. A Digital Literacy paper must never make an ICT Security
assessment ask about spreadsheets, so exemplars are scored for relevance
before they are shown, capped, and handed over under instructions that say
plainly they are patterns to follow and not content to assess. The CONTENT
TAUGHT remains the only thing that decides what a paper is about.

Everything here degrades to nothing. A repository that is down, slow, or has
never heard of the unit costs the paper its exemplars and not its generation -
the same rule the rest of this project follows.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import requests

import runlog
from assessment_models import Exemplar

USER_AGENT = "LearningPlanGenerator/1.0 (Kenyan TVET assessment tooling)"
_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.9"}

# DSpace repositories that publish Kenyan exam papers. Each is tried in turn
# and any that fails is skipped; one source answering is enough.
REPOSITORIES = (
    ("amref", "https://pastpapers.amref.ac.ke"),
)

SEARCH_PATH = "/server/api/discover/search/objects"
BUNDLES_PATH = "/server/api/core/items/%s/bundles"

SEARCH_TIMEOUT = 25
DOWNLOAD_TIMEOUT = 60

# How many papers to open for one unit, and how many questions to keep. Both
# deliberately small: the point is to show a pattern, not to supply a bank.
MAX_PAPERS = 3
MAX_EXEMPLARS = 10

# A question shorter than this is a fragment the extraction mis-split; longer
# than this is usually a whole section that lost its numbering.
MIN_QUESTION_CHARS = 30
MAX_QUESTION_CHARS = 320

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".assessment_research")

# Re-scouted after this long. Past papers are published in batches once or
# twice a year, so anything shorter is just traffic.
CACHE_DAYS = 30


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get(url: str, params: Optional[dict] = None, timeout: int = SEARCH_TIMEOUT):
    """A GET that tolerates these hosts' certificates.

    Two of the repositories serve an incomplete chain, so verification fails
    against a clean trust store even though the site is the real one. It is
    retried unverified rather than abandoned: what is being read is a public
    exam paper, nothing is sent, and a broken chain on a university's static
    hosting is a configuration fault rather than an attack. The fallback is
    logged so it never becomes invisible.
    """
    try:
        return requests.get(url, params=params, headers=_HEADERS,
                            timeout=timeout)
    except requests.exceptions.SSLError:
        runlog.warn(f"Research: {url.split('/')[2]} has an incomplete "
                    f"certificate chain; reading it unverified")
        return requests.get(url, params=params, headers=_HEADERS,
                            timeout=timeout, verify=False)


# --------------------------------------------------------------------------- #
# The repositories
# --------------------------------------------------------------------------- #
def search(query: str, size: int = 8) -> List[Dict[str, str]]:
    """Papers whose titles match `query`, across every repository.

    Returns [{repository, base, uuid, title}]. Never raises.
    """
    found: List[Dict[str, str]] = []
    for name, base in REPOSITORIES:
        try:
            resp = _get(base + SEARCH_PATH, {"query": query, "size": size})
            if resp.status_code != 200:
                runlog.warn(f"Research: {name} answered {resp.status_code} "
                            f"searching for '{query}'")
                continue
            objects = (resp.json()["_embedded"]["searchResult"]["_embedded"]
                       ["objects"])
        except (requests.RequestException, ValueError, KeyError) as e:
            runlog.warn(f"Research: {name} could not be searched: {e}")
            continue
        for obj in objects:
            item = obj.get("_embedded", {}).get("indexableObject", {})
            if item.get("type") != "item" or not item.get("uuid"):
                continue
            found.append({"repository": name, "base": base,
                          "uuid": item["uuid"],
                          "title": str(item.get("name") or "").strip()})
    return found


def _bitstream_urls(base: str, uuid: str) -> List[str]:
    """The downloadable files on one repository item, originals only."""
    try:
        bundles = _get(base + BUNDLES_PATH % uuid).json()["_embedded"]["bundles"]
    except (requests.RequestException, ValueError, KeyError):
        return []
    urls: List[str] = []
    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue                      # thumbnails and licences
        try:
            href = bundle["_links"]["bitstreams"]["href"]
            streams = _get(href).json()["_embedded"]["bitstreams"]
        except (requests.RequestException, ValueError, KeyError):
            continue
        urls.extend(s["_links"]["content"]["href"] for s in streams
                    if "_links" in s and "content" in s["_links"])
    return urls


def paper_text(base: str, uuid: str) -> str:
    """One paper's text, or '' if it cannot be read.

    PDFs only. A repository holds Word documents and scans too; a scan carries
    no text layer and a Word paper is rare enough not to be worth a converter
    here, where a missing paper costs nothing.
    """
    for url in _bitstream_urls(base, uuid):
        try:
            resp = _get(url, timeout=DOWNLOAD_TIMEOUT)
        except requests.RequestException as e:
            runlog.warn(f"Research: a paper could not be downloaded: {e}")
            continue
        if resp.status_code != 200 or resp.content[:4] != b"%PDF":
            continue
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(resp.content)) as doc:
                return "\n".join(p.extract_text() or "" for p in doc.pages)
        except Exception as e:                    # noqa: BLE001
            runlog.warn(f"Research: a paper could not be read: {e}")
    return ""


# --------------------------------------------------------------------------- #
# Pulling the questions out
# --------------------------------------------------------------------------- #
# A numbered question ending in its marks, which is how every CDACC paper
# prints one: "3. Outline THREE measures for ... (3 marks)".
_RE_QUESTION = re.compile(
    r"^[ \t]*(\d{1,2})\s*[.)]\s*(.+?)[\(\[]\s*(\d{1,2})\s*[Mm]arks?\s*[\)\]]",
    re.M | re.S)

# Dotted answer lines, page furniture and the scoring grid.
_RE_DOTS = re.compile(r"[.…]{4,}|_{4,}")
_RE_FURNITURE = re.compile(
    r"page \d+ of \d+|for official use only|scoring grid|"
    r"this is the last printed page|turn over", re.I)


def questions(text: str) -> List[Exemplar]:
    """Every numbered, mark-bearing question in a paper's text."""
    out: List[Exemplar] = []
    for _number, body, marks in _RE_QUESTION.findall(text or ""):
        body = _RE_DOTS.sub(" ", body)
        body = _RE_FURNITURE.sub(" ", body)
        body = " ".join(body.split()).strip(" .-")
        if not MIN_QUESTION_CHARS <= len(body) <= MAX_QUESTION_CHARS:
            continue
        try:
            worth = int(marks)
        except ValueError:
            continue
        out.append(Exemplar(text=body, marks=worth))
    return out


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
_RE_WORD = re.compile(r"[a-z]{3,}")
_STOP = frozenset("""
the and for with that this from are was were have has had you your they them
their which what when where who how why all any one two three four five six
state list give name explain describe identify outline define discuss analyse
analyze following each other than into out marks mark question candidate
trainee assessor unit level paper section answer
""".split())


def _tokens(text: str) -> set:
    return {w for w in _RE_WORD.findall((text or "").lower()) if w not in _STOP}


def rank(exemplars: Sequence[Exemplar], topics: Sequence[str]) -> List[Exemplar]:
    """Exemplars most like the unit's own subject matter, best first.

    An exemplar is a style reference, but a wildly off-subject one still drags
    a paper sideways - a model shown a spreadsheet question while writing on
    ICT security will reach for spreadsheets. Ranking by shared vocabulary
    keeps the examples in the same world as the unit without pretending they
    are about it.
    """
    wanted = _tokens(" ".join(topics))
    if not wanted:
        return list(exemplars)
    scored = []
    for ex in exemplars:
        mine = _tokens(ex.text)
        overlap = len(mine & wanted) / len(mine) if mine else 0.0
        scored.append((overlap, ex))
    scored.sort(key=lambda p: -p[0])
    return [ex for _score, ex in scored]


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    return (re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_") or "unit")[:80]


def cache_path(unit_title: str) -> str:
    return os.path.join(CACHE_DIR, _safe(unit_title) + ".json")


def _cached(unit_title: str) -> Optional[List[Exemplar]]:
    path = cache_path(unit_title)
    try:
        with io.open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    if time.time() - raw.get("fetched", 0) > CACHE_DAYS * 86400:
        return None
    rows = raw.get("exemplars")
    if not isinstance(rows, list):
        return None
    return [Exemplar(text=str(r.get("text", "")), marks=int(r.get("marks", 0)),
                     source=str(r.get("source", "")),
                     repository=str(r.get("repository", "")))
            for r in rows if isinstance(r, dict) and r.get("text")]


def _remember(unit_title: str, exemplars: Sequence[Exemplar]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        marker = os.path.join(CACHE_DIR, ".gitignore")
        if not os.path.exists(marker):
            with io.open(marker, "w", encoding="utf-8") as fh:
                fh.write("*\n")
        with io.open(cache_path(unit_title), "w", encoding="utf-8") as fh:
            json.dump({"unit_title": unit_title, "fetched": time.time(),
                       "exemplars": [asdict(e) for e in exemplars]}, fh,
                      indent=1, ensure_ascii=False)
    except OSError as e:
        runlog.warn(f"Research: the exemplars could not be cached: {e}")


# --------------------------------------------------------------------------- #
# What the rest of the module calls
# --------------------------------------------------------------------------- #
def exemplars_for(unit_title: str, topics: Sequence[str] = (),
                  limit: int = MAX_EXEMPLARS,
                  progress_cb=None) -> List[Exemplar]:
    """Real questions from real papers on this unit, or as near as exists.

    Cached per unit: the first CAT on a unit pays for the scouting and every
    later one reads it off disk. Returns [] rather than raising when nothing
    can be reached, and the caller simply generates without exemplars.
    """
    if not (unit_title or "").strip():
        return []
    remembered = _cached(unit_title)
    if remembered is not None:
        runlog.log(f"Research: {len(remembered)} exemplar(s) for "
                   f"'{unit_title}' read from the cache")
        return remembered[:limit]

    if progress_cb:
        progress_cb(f"Research: looking for real papers on '{unit_title}'")
    hits = search(unit_title)
    if not hits:
        # The full title rarely matches; its distinctive words often do.
        words = [w for w in _tokens(unit_title)]
        if words:
            hits = search(" ".join(sorted(words)[:4]))
    if not hits:
        runlog.log(f"Research: no past papers found for '{unit_title}'")
        _remember(unit_title, [])
        return []

    collected: List[Exemplar] = []
    for hit in hits[:MAX_PAPERS]:
        if progress_cb:
            progress_cb(f"Research: reading '{hit['title'][:60]}'")
        found = questions(paper_text(hit["base"], hit["uuid"]))
        for ex in found:
            ex.source, ex.repository = hit["title"], hit["repository"]
        collected.extend(found)
        runlog.log(f"Research: {len(found)} question(s) from "
                   f"'{hit['title']}'")

    best = rank(collected, list(topics) + [unit_title])[:limit]
    _remember(unit_title, best)
    if progress_cb:
        progress_cb(f"Research: {len(best)} exemplar question(s) kept")
    return best


def render(exemplars: Sequence[Exemplar]) -> str:
    """The exemplars as the model reads them, or '' when there are none."""
    if not exemplars:
        return ""
    lines = []
    for ex in exemplars:
        marks = f" ({ex.marks} marks)" if ex.marks else ""
        lines.append(f"- \"{ex.text}\"{marks}")
    sources = sorted({e.source for e in exemplars if e.source})
    tail = ("\n  [from: " + "; ".join(s[:60] for s in sources[:4]) + "]"
            if sources else "")
    return "\n".join(lines) + tail
