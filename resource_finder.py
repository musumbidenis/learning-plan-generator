"""Module C2 - REAL learning resources, found on the internet and proved to exist.

The Learning Plan's Resources column used to be whatever the model remembered,
and a model's memory of a URL is a guess that reads like a fact. Measured
against the live API, `openai/gpt-oss-120b` proposed ten resources for one unit
and three of them resolved; of the three YouTube links it offered, **none**
existed - the video ids were well-formed inventions. That is the problem this
module exists to remove.

Two rules:

1. **Candidates come from an index wherever one is available.** Videos are
   searched on YouTube, so a real video id is never guessed in the first place.
2. **Nothing is returned until it has been fetched.** Every URL is requested
   before it can reach a document, so an invented one cannot survive, whatever
   proposed it.

YouTube gets a stronger check than a plain fetch: a dead watch page still
answers 200 with an "unavailable" notice, so fetching it proves nothing. The
oEmbed endpoint 404s for an id that does not exist, and hands back the video's
real title and channel - which is also how a resource ends up named accurately
rather than as the model imagined it.

No API key is needed for the video half: YouTube's results page and its oEmbed
endpoint are both public. The page half reuses the Groq key the plan already
generates with, and works without it - there are simply no pages in the pool.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import requests

import runlog

# Presented to every site we touch, so an operator reading their logs can see
# what this is. Browser-shaped because some hosts refuse an unknown agent.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.9"}

YOUTUBE_SEARCH = "https://www.youtube.com/results"
YOUTUBE_OEMBED = "https://www.youtube.com/oembed"
YOUTUBE_WATCH = "https://www.youtube.com/watch?v=%s"

# One verification should not hold up a generation; a slow host is a dead host
# as far as a trainer waiting for a plan is concerned.
VERIFY_TIMEOUT = 12
SEARCH_TIMEOUT = 20
VERIFY_WORKERS = 10

# How many of each kind to aim for across a whole unit. Sessions draw 2-4 each
# from the shared pool, so this is a library, not a per-session quota.
WANT_VIDEOS = 6
WANT_PAGES = 10

# Candidate video ids to pull off a results page before verifying. Some are
# playlists, shorts or channel links that oEmbed will reject, so ask for more
# than are wanted.
_VIDEO_CANDIDATES = 12

# Session topics to run a video search for, and how many to keep from each.
# Three covers the range of a unit without turning this into eight searches.
_VIDEO_QUERIES = 3
_PER_QUERY_VIDEOS = 3

_RE_VIDEO_ID = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')
_RE_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_RE_TAG = re.compile(r"<[^>]+>")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".resource_cache")
# A unit's resources do not change between one generation and the next, and a
# search is the slowest thing in the pipeline. Re-planning the same unit should
# not pay for it twice.
CACHE_TTL_SECONDS = 14 * 24 * 60 * 60


@dataclass
class Resource:
    """One resource that has been confirmed to exist at `url`."""
    title: str
    url: str
    kind: str = "documentation"        # video | documentation | standard | course
    channel: str = ""                  # for a video, who published it

    def as_line(self) -> str:
        """The single line the Resources column shows."""
        if self.kind == "video":
            who = f", {self.channel}" if self.channel else ""
            return f"- Video: {self.title}{who} - {self.url}"
        return f"- {self.title} - {self.url}"


@dataclass
class ResourcePool:
    """Everything verified for one unit, plus what happened while finding it."""
    resources: List[Resource] = field(default_factory=list)
    searched: bool = False             # did any index actually answer
    note: str = ""                     # why it is thin, when it is

    @property
    def videos(self) -> List[Resource]:
        return [r for r in self.resources if r.kind == "video"]

    def allow_list(self) -> str:
        """The pool as the prompt presents it, numbered for easy reference."""
        return "\n".join(f"{i}. [{r.kind}] {r.title}"
                         + (f" ({r.channel})" if r.channel else "")
                         + f" - {r.url}"
                         for i, r in enumerate(self.resources, start=1))

    def urls(self) -> set:
        return {r.url for r in self.resources}


# --------------------------------------------------------------------------- #
# Verification - the part that makes the rest trustworthy
# --------------------------------------------------------------------------- #
def _is_youtube(url: str) -> bool:
    return "youtube.com/watch" in url or "youtu.be/" in url


def verify_video(url: str) -> Optional[Resource]:
    """A Resource carrying YouTube's OWN title and channel, or None.

    oEmbed is used rather than a fetch of the watch page because a deleted or
    invented video still answers 200 there. It also means the title we print is
    the video's real one instead of the title a model claimed for it.
    """
    try:
        resp = requests.get(YOUTUBE_OEMBED, params={"url": url, "format": "json"},
                            timeout=VERIFY_TIMEOUT, headers=_HEADERS)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    title = (data.get("title") or "").strip()
    if not title:
        return None
    return Resource(title=title, url=url, kind="video",
                    channel=(data.get("author_name") or "").strip())


def verify_page(url: str) -> bool:
    """Whether `url` answers. HEAD first; some hosts only honour GET."""
    for method in (requests.head, requests.get):
        try:
            resp = method(url, timeout=VERIFY_TIMEOUT, allow_redirects=True,
                          headers=_HEADERS, stream=(method is requests.get))
            if resp.status_code < 400:
                return True
            if method is requests.get:
                return False
        except requests.RequestException:
            if method is requests.get:
                return False
    return False


def verify(resource: Resource) -> Optional[Resource]:
    """The verified resource, or None if it does not exist."""
    url = (resource.url or "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    if _is_youtube(url):
        return verify_video(url)
    return resource if verify_page(url) else None


def verify_all(candidates: List[Resource]) -> List[Resource]:
    """Verify in parallel, keeping order and dropping duplicates."""
    if not candidates:
        return []
    with ThreadPoolExecutor(min(VERIFY_WORKERS, len(candidates))) as pool:
        checked = list(pool.map(verify, candidates))
    out, seen = [], set()
    for resource in checked:
        if resource and resource.url not in seen:
            seen.add(resource.url)
            out.append(resource)
    return out


# --------------------------------------------------------------------------- #
# Videos - searched, never guessed
# --------------------------------------------------------------------------- #
def search_videos(query: str, want: int = WANT_VIDEOS) -> List[Resource]:
    """Real videos for `query`, titled by YouTube itself.

    The results page embeds its data as JSON, so the ids can be read straight
    out of it without a key. Each is then confirmed through oEmbed, which is
    what turns a scraped id into a resource we are willing to print.
    """
    try:
        resp = requests.get(YOUTUBE_SEARCH,
                            params={"search_query": query, "hl": "en", "gl": "KE"},
                            timeout=SEARCH_TIMEOUT, headers=_HEADERS)
    except requests.RequestException as e:
        runlog.log(f"Resources: YouTube search failed ({type(e).__name__})",
                   level="WARN")
        return []
    if resp.status_code != 200:
        runlog.log(f"Resources: YouTube search returned {resp.status_code}",
                   level="WARN")
        return []

    ids, seen = [], set()
    for match in _RE_VIDEO_ID.finditer(resp.text):
        vid = match.group(1)
        if vid not in seen:
            seen.add(vid)
            ids.append(vid)
        if len(ids) >= _VIDEO_CANDIDATES:
            break

    found = verify_all([Resource(title="", url=YOUTUBE_WATCH % vid, kind="video")
                        for vid in ids])
    return found[:want]


# --------------------------------------------------------------------------- #
# Pages - proposed, then made to prove it
# --------------------------------------------------------------------------- #
def _page_schema() -> dict:
    """Candidate pages, in the strict-object shape Groq requires.

    Without this `_post` falls back to the Learning-Plan session schema and the
    model dutifully answers with sessions - which is exactly what happened the
    first time this ran.
    """
    import ai_client
    return ai_client._strict({
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "kind": {"type": "string"},
                    },
                },
            },
        },
    })


_PAGE_PROMPT = """List learning resources for teaching this topic on a Kenyan TVET \
Level 6 course:

{topics}

Rules that decide whether a resource is usable:
- Give the site's STABLE landing page, not a deep link into it. A deep link to a
  particular PDF or article is far more likely to have moved; a landing page is not.
- Only well-known organisations whose pages persist: standards bodies, government
  agencies, universities, established documentation and course sites.
- Include Kenyan and African sources where they are relevant.
- Never invent a URL. If you are unsure a page exists, leave it out.
- No YouTube or video links: those are searched separately.

Give {want} items. Return ONLY:
{{"resources":[{{"title":"","url":"","kind":"documentation|standard|course"}}]}}"""


def propose_pages(topics: str, api_key: str, model: str,
                  want: int = WANT_PAGES) -> List[Resource]:
    """Candidate pages from the model - unverified, and treated as such.

    Every one of these is a guess. `verify_all` is what decides which are real;
    the instructions above exist only to raise the hit rate, because a deep
    link is the shape that usually turns out to be invented.
    """
    import ai_client                        # late: ai_client imports this module

    prompt = _PAGE_PROMPT.format(topics=topics, want=want)
    try:
        resp = ai_client._post(model, api_key, prompt, timeout=60,
                               max_tokens=1500,
                               schema=_page_schema(),
                               schema_name="candidate_resources")
    except requests.RequestException as e:
        runlog.log(f"Resources: page search failed ({type(e).__name__})",
                   level="WARN")
        return []
    if getattr(resp, "status_code", 0) != 200:
        runlog.log(f"Resources: page search returned "
                   f"{getattr(resp, 'status_code', '?')}", level="WARN")
        return []
    try:
        text = ai_client._extract_text(resp.json())
    except Exception:                       # noqa: BLE001 - any shape, same answer
        return []
    return _parse_resources(text)


def _parse_resources(text: str) -> List[Resource]:
    """Pull a resource list out of whatever the model wrapped it in."""
    match = _RE_JSON_OBJECT.search(text or "")
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    rows = data.get("resources") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = _RE_TAG.sub("", str(row.get("title") or "")).strip()
        url = str(row.get("url") or "").strip()
        kind = str(row.get("kind") or "documentation").strip().lower()
        if title and url:
            out.append(Resource(title=title, url=url,
                                kind=kind if kind != "video" else "documentation"))
    return out


# --------------------------------------------------------------------------- #
# The pool for one unit
# --------------------------------------------------------------------------- #
def _cache_path(unit_code: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", unit_code or "unit").strip("_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _read_cache(unit_code: str) -> Optional[ResourcePool]:
    path = _cache_path(unit_code)
    try:
        if time.time() - os.path.getmtime(path) > CACHE_TTL_SECONDS:
            return None
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return ResourcePool(
        resources=[Resource(**r) for r in data.get("resources", [])],
        searched=bool(data.get("searched")), note=str(data.get("note") or ""))


def _write_cache(unit_code: str, pool: ResourcePool) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(unit_code), "w", encoding="utf-8") as handle:
            json.dump({"resources": [asdict(r) for r in pool.resources],
                       "searched": pool.searched, "note": pool.note},
                      handle, ensure_ascii=False, indent=1)
    except OSError as e:
        runlog.log(f"Resources: could not cache ({e})", level="WARN")


def find_for_unit(unit, sessions, api_key: str = "", model: str = "",
                  progress_cb=None, use_cache: bool = True) -> ResourcePool:
    """Verified resources for a whole unit, videos included.

    One pool per unit rather than per session: the sessions of a unit cover one
    subject, a search per session would multiply the slowest step in the
    pipeline by eight, and a trainer reading the plan sees a coherent set of
    references rather than eight disconnected ones.
    """
    unit_code = getattr(unit, "os_code", "") or getattr(unit, "unit_title", "")
    if use_cache:
        cached = _read_cache(unit_code)
        if cached is not None:
            runlog.log(f"Resources: {len(cached.resources)} from cache "
                       f"for {unit_code}")
            _emit(progress_cb, f"Resources: {len(cached.resources)} known for "
                               f"this unit ({len(cached.videos)} videos)")
            return cached

    title = getattr(unit, "unit_title", "") or "the unit"
    titles = list(dict.fromkeys(
        s.session_title for s in sessions if getattr(s, "session_title", "")))
    topics = "; ".join(titles)[:600]

    # A single query spanning every session title returns something generic -
    # for this unit it found a UK BTEC exam walk-through. Querying a few
    # individual topics returns material that is actually about them.
    queries = [f"{t} tutorial"[:120] for t in titles[:_VIDEO_QUERIES]]
    if not queries:
        queries = [f"{title} tutorial"[:120]]

    _emit(progress_cb, "Resources: searching for real material, videos included")
    started = time.time()

    with ThreadPoolExecutor(len(queries) + 1) as pool:
        video_tasks = [pool.submit(search_videos, q, _PER_QUERY_VIDEOS)
                       for q in queries]
        pages_task = pool.submit(
            propose_pages, f"{title}\nSession topics: {topics}", api_key, model) \
            if api_key and model else None
        videos, seen = [], set()
        for task in video_tasks:
            for resource in task.result():
                if resource.url not in seen:
                    seen.add(resource.url)
                    videos.append(resource)
        videos = videos[:WANT_VIDEOS]
        proposed = pages_task.result() if pages_task else []

    pages = verify_all(proposed)
    dropped = len(proposed) - len(pages)

    found = ResourcePool(resources=videos + pages,
                         searched=bool(videos or proposed))
    if not found.resources:
        found.note = "no resource could be verified"
    elif not videos:
        found.note = "no video could be verified"

    took = time.time() - started
    runlog.log(f"Resources: {len(videos)} videos + {len(pages)} pages verified "
               f"for {unit_code} in {took:.1f}s "
               f"({dropped} proposed page(s) did not exist)")
    _emit(progress_cb,
          f"Resources: {len(found.resources)} verified "
          f"({len(videos)} videos); {dropped} proposed link(s) did not exist")

    if use_cache and found.resources:
        _write_cache(unit_code, found)
    return found


def _emit(progress_cb, message: str) -> None:
    if progress_cb is not None:
        try:
            progress_cb(message)
        except Exception:                   # noqa: BLE001 - never fail on the UI
            pass
