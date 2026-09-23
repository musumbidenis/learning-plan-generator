"""Real substance on the taught topics, so a question can go deeper than a syllabus line.

The curriculum's key points are headings, not knowledge. "Types of malware:
virus, worm, trojan" is four words and a colon. A model given only that, and
told to write a deep question, has two ways out and both are bad: it hands the
line back ("List THREE types of malware") or it invents the depth from its own
memory of the trade, which is where the plausible-but-untaught question comes
from.

This module supplies the missing third way. For each taught key point it goes
and reads about that key point, and brings back the part a question can be
built on: what the thing actually is, the dimensions along which it is
normally broken down, and the names a practitioner would use - the tools, the
standards, the classifications, the real examples. The trainees were taught
those names in class; the curriculum simply does not write them down.

WHY WIKIPEDIA, AND NOT A SEARCH ENGINE
Because a search engine will not talk to us. DuckDuckGo and Mojeek both answer
a bot challenge, the public SearXNG instances answer 403 or 429, and Bing's
fallback is junk - a query about malware came back about book genres. The
MediaWiki API, by contrast, is a documented public API that answers politely
and returns plain text. It is one source and it is not the whole internet, but
it is a real one that works, and a reference that answers is worth more than
six that refuse.

HOW A TOPIC IS LOOKED UP, AND WHY IT IS DONE THIS WAY
Three things were learnt the hard way and each is now a rule here:

1  Search the key point ALONE. Adding the unit's subject to the query to
   disambiguate it makes things worse, not better: MediaWiki's full-text
   search ranks over whole article bodies, so "Risk rating matrix ICT
   security" returned an article on the Israeli occupation of the West Bank,
   and "Social engineering attacks ICT security" returned the general
   Computer security article rather than the specific one.

2  Pool BOTH searches. Title search finds "Earthing" and "Role-based access
   control" - phrases that are already article titles - and finds nothing at
   all for a descriptive phrase like "malware virus worm trojan". Full-text
   search is the opposite. Together they cover both.

3  Verify with the taught vocabulary, not with the query. The wrong sense of a
   phrase reads perfectly well on its own: "Types of wiring systems" resolves
   happily to Writing system, and nothing about that article looks broken
   until you notice it is about alphabets. What gives it away is that none of
   the unit's own words - cable, circuit, conduit, socket, earthing - appear
   anywhere in it. So every candidate is scored on how much of the unit's
   vocabulary it actually contains, and a candidate that shares nothing with
   the unit is dropped however well it matches the phrase.

WHAT A NOTE IS FOR, AND WHAT IT IS NOT FOR
A note is DEPTH on a topic the curriculum already lists. It is never a new
topic. The content taught still decides the scope of the paper, exactly as
before; this only decides how far inside that scope a question may reach. A
note attaches to the key point it was fetched for and travels with it, so the
model can never treat one as a licence to assess something nobody covered.

Everything here degrades to nothing. Wikipedia unreachable, rate limiting, a
topic no encyclopaedia has heard of - each costs the paper its notes and not
its generation, the same rule the rest of the project follows.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import requests

import runlog
from assessment_models import ElementContent, KnowledgeNote
from assessment_research import CACHE_DIR

API = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/wiki/%s"

# MediaWiki asks for a descriptive agent that says who is calling and why.
USER_AGENT = ("LearningPlanGenerator/1.0 (Kenyan TVET assessment tooling; "
              "reference lookup for curriculum topics)")

TIMEOUT = 25

# At least this long between calls. The API is free and public and there is no
# hurry: a paper is generated once, and the whole lookup is then cached for a
# month and a half. Hammering it earns an empty response, which is how three
# of the first probe's nine topics came back with nothing.
MIN_GAP_SECONDS = 0.5
RETRIES = 4
BACKOFF_SECONDS = 1.5

# How many taught key points to look up for one paper. A note costs several
# API calls and a few hundred characters of a prompt that has little room to
# spare, and a paper of six items cannot draw on thirty topics anyway. Spread
# across the elements rather than taken in order, so the last element is not
# the one that goes without.
MAX_NOTES = 8

# Candidates considered per key point, pooled from the two searches.
MAX_CANDIDATES = 6

# The share of the unit's own vocabulary a page must contain to be believed.
# Measured: the right page scores 0.16 to 0.46, the wrong sense of the same
# phrase scores 0.00 to 0.06. Anything at or below this is a different subject
# wearing the same words.
MIN_COVERAGE = 0.10

# The same gate for an article whose TITLE is the topic. Vulnerability scanner
# is two and a half thousand words on exactly the right subject and scored
# 0.09, because a short article cannot hold much of anything; it was thrown
# out for being brief. A title match is its own evidence, so it is asked only
# to show that it is not about something else entirely - which the wrong sense
# of a phrase fails outright, at 0.00.
MIN_COVERAGE_TITLED = 0.05
TITLE_MATCH_RATIO = 0.75

# Shorter than this and it is a stub or a signpost, not an article. The real
# ones run from two and a half thousand characters upwards.
MIN_ARTICLE_CHARS = 1000

# How many of the ranked candidates are checked against their full article
# before the key point is given up on. The intro ranking is a guess, and a
# rejected first guess is usually rejected for being a disambiguation page -
# with the real article sitting directly behind it.
CANDIDATES_VERIFIED = 3

# An article whose title IS the phrase is the article, whatever it scores.
# Without this, "Role based access control" loses to Relationship-based access
# control, which shares more of a security vocabulary by being longer.
EXACT_TITLE_BONUS = 0.5
TITLE_WEIGHT = 0.15

# How much of an article survives into the prompt, and how much of the prompt
# the notes may take in total.
#
# These notes are what the questions are made of, so they get real room. The
# budget is a backstop rather than a squeeze: a measured full prompt for an
# ordinary unit comes to about 5,900 tokens against the tier's 8,000, and the
# refusals that once looked like a size limit were the rolling per-minute
# window, which is now waited out rather than trimmed around.
#
# Within a note the room goes to the facts. A 120B model does not need three
# sentences explaining what malware is - it needs to know WHICH sense of the
# topic is meant, and then what each part of it actually says, which is the
# half it does not have.
SUMMARY_SENTENCES = 2
SUMMARY_CHARS = 230
MAX_COVERS = 6
MAX_NAMED = 6

# The teaching body of a note: one line per section of the article. This is
# the part a question is actually made from, so it gets the room.
MAX_FACTS = 4
FACT_CHARS = 210
MIN_FACT_CHARS = 60

NOTE_BLOCK_CHARS = 5200

CACHE_DAYS = 45

_RE_WORD = re.compile(r"[a-z]{4,}")
_RE_HEADING = re.compile(r"^==+ *(.+?) *==+$", re.M)
_RE_SENTENCE = re.compile(r"(?<=[.!?]) +")

# A proper noun or an acronym. Whether the capital is real or just the start of
# a sentence is decided by looking at what precedes the match, in `_named`.
_RE_NAMED = re.compile(r"\b([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})*|[A-Z]{2,6})\b")

# The scaffolding a curriculum writer puts in front of a topic. "Types of
# malware" is not a subject; malware is.
_RE_SCAFFOLD = re.compile(
    r"^(types?|kinds?|forms?|classes?|categor\w+|methods?|techniques?|"
    r"principles?|importance|definitions?|meaning|list|uses?|purposes?|"
    r"identification|assessment|evaluation|configuration|selection|"
    r"preparation|application|introduction|overview|concepts?|"
    r"characteristics|features|advantages|disadvantages|factors)"
    r"\s+(of|in|for)\s+", re.I)

# Section headings that are apparatus rather than subject matter.
_SKIP_HEADINGS = frozenset((
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "gallery", "citations", "footnotes",
    "literature", "in popular culture", "etymology", "terminology",
    "explanatory notes", "works cited", "general references",
))

# Sections about a subject's past rather than its practice. A trainee is
# assessed on the work, not on when it was invented.
_BACKSTORY_HEADINGS = frozenset((
    "history", "background", "origins", "reception", "criticism",
    "controversy", "timeline", "in fiction", "cultural references",
    "notable examples", "legal issues", "legislation and regulation",
))

# Words that open a sentence often enough to be mistaken for names, and the
# months, which are dates rather than the names of anything assessable.
_NOT_NAMES = frozenset("""
according despite since another although however therefore moreover
furthermore traditionally generally typically usually often sometimes
because while when where which what this that these those there their
history types purpose purposes overview introduction examples example
january february march april may june july august september october november
december
""".split())


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #
_session: Optional[requests.Session] = None
_last_call = [0.0]


def _api(params: dict):
    """One MediaWiki call, paced and retried. Returns {} or [] on failure.

    Never raises: a reference lookup is an improvement to a paper, not a
    precondition for one.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT,
                                 "Accept-Language": "en-GB,en;q=0.9"})
    gap = MIN_GAP_SECONDS - (time.time() - _last_call[0])
    if gap > 0:
        time.sleep(gap)
    for attempt in range(RETRIES):
        try:
            resp = _session.get(API, params=dict(params, format="json"),
                                timeout=TIMEOUT)
            _last_call[0] = time.time()
            if resp.status_code == 200:
                return resp.json()
            # 429 and 5xx are both "come back later", and a public API under
            # load answers one of them rather than failing outright.
            runlog.log(f"Knowledge: Wikipedia answered {resp.status_code}; "
                       f"backing off")
        except (ValueError, requests.RequestException):
            _last_call[0] = time.time()
        time.sleep(BACKOFF_SECONDS * (attempt + 1))
    return {}


def _phrasings(phrase: str) -> List[str]:
    """The phrase, and the shorter phrase inside it, for the title search.

    A curriculum writes "Password policy settings" and "Vulnerability scanning
    tools"; the encyclopaedia has "Password policy" and "Vulnerability
    scanning". The trailing word is the curriculum's, not the subject's, and
    dropping it turns a near miss into an exact title match - which then wins
    outright on the exact-title bonus. Without this, those two topics drew
    Group Policy and Dynamic application security testing: both adjacent,
    neither the topic that was taught.

    Only the title search gets the shortened form. Full-text search is already
    happy with a long descriptive phrase and gains nothing from a shorter one.
    """
    words = phrase.split()
    out = [phrase]
    if len(words) >= 3:
        out.append(" ".join(words[:-1]))
    return out


def candidates(phrase: str, limit: int = MAX_CANDIDATES) -> List[str]:
    """Article titles worth considering for `phrase`, best guesses first.

    Both searches are asked, because they fail in opposite directions - see
    the module docstring. Order is kept and duplicates dropped, so a title the
    two searches agree on stays where the title search put it.
    """
    per_source = max(2, limit // 3)
    lists: List[List[str]] = []
    for search_phrase in _phrasings(phrase):
        opened = _api({"action": "opensearch", "search": search_phrase,
                       "limit": per_source, "namespace": 0})
        if isinstance(opened, list) and len(opened) > 1:
            lists.append([str(t) for t in opened[1]][:per_source])
    full = _api({"action": "query", "list": "search",
                 "srsearch": phrase, "srlimit": per_source})
    if isinstance(full, dict):
        lists.append([str(hit.get("title", "")) for hit
                      in full.get("query", {}).get("search", [])][:per_source])

    # Each search gets its own places rather than first-come-first-served.
    # Taking the pool in order let one search fill it: "Risk rating" returned
    # Risk-taking, Risk ratio and two Risk of Rain games, and Risk matrix -
    # which the full-text search had found - never got in.
    out: List[str] = []
    for rank_index in range(per_source):
        for titles in lists:
            if rank_index < len(titles) and titles[rank_index] not in out:
                out.append(titles[rank_index])
    return [t for t in out if t][:limit]


def _extracts(titles: Sequence[str], intro_only: bool) -> Dict[str, str]:
    """{title: plain text}, several titles in one call where possible."""
    if not titles:
        return {}
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "titles": "|".join(titles), "redirects": 1,
              "exlimit": "max" if intro_only else 1}
    if intro_only:
        params["exintro"] = 1
    data = _api(params)
    if not isinstance(data, dict):
        return {}
    pages = data.get("query", {}).get("pages", {})
    return {str(p.get("title", "")): str(p.get("extract", "") or "")
            for p in pages.values() if isinstance(p, dict)}


# --------------------------------------------------------------------------- #
# Choosing the right article
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def vocabulary(unit_title: str, content: Sequence[ElementContent]) -> set:
    """Every distinctive word the unit itself uses.

    Built from the WHOLE taught content and not from the one key point being
    looked up, because one key point is three words and three words cannot
    tell Writing system from Electrical wiring. The unit as a whole can:
    an electrical unit says cable, circuit, conduit, socket and earthing
    somewhere, and an article about alphabets says none of them.
    """
    words = [unit_title or ""]
    for block in content:
        words.append(block.element_title)
        for topic in block.topics:
            words.append(topic.title)
            words.extend(topic.key_points)
    return set(_RE_WORD.findall(" ".join(words).lower()))


def coverage(text: str, vocab: set) -> float:
    """The share of the unit's vocabulary that appears in this article."""
    if not vocab:
        return 0.0
    return len(set(_RE_WORD.findall((text or "").lower())) & vocab) / len(vocab)


def _score(title: str, text: str, phrase: str, vocab: set) -> Tuple[float, float]:
    """(ranking score, raw coverage) for one candidate article.

    The exact-title bonus is tested against every phrasing, not just the full
    one. "Password policy settings" is what a curriculum writes and "Password
    policy" is what the article is called, and on coverage alone the shorter
    article loses simply for being short - its intro is 546 characters against
    Group Policy's 1166, so it holds less of the unit's vocabulary while being
    far more on the subject. An exact title is worth more than word counting.
    """
    cover = coverage(text, vocab)
    near = max(SequenceMatcher(None, _norm(title), _norm(p)).ratio()
               for p in _phrasings(phrase))
    score = cover + TITLE_WEIGHT * near
    if any(_norm(title) == _norm(p) for p in _phrasings(phrase)):
        score += EXACT_TITLE_BONUS
    return score, cover


_RE_SIGNPOST = re.compile(r"\b(may|can|could|might) (also )?refer to\b", re.I)


def _is_signpost(text: str) -> bool:
    """True for a disambiguation page or a stub with nothing in it.

    These win the ranking outright and then teach nothing. "Social
    engineering" is a four-hundred-character page whose entire content is a
    list of other pages, and because its title matches the taught key point
    exactly it beat the real article - twice, before this was noticed. A page
    that only points elsewhere is not a page to question a candidate on.
    """
    body = (text or "").strip()
    return len(body) < MIN_ARTICLE_CHARS or bool(_RE_SIGNPOST.search(body[:400]))


def _gate_for(title: str, phrase: str) -> float:
    """How much of the unit this article has to contain to be believed."""
    near = max(SequenceMatcher(None, _norm(title), _norm(p)).ratio()
               for p in _phrasings(phrase))
    return MIN_COVERAGE_TITLED if near >= TITLE_MATCH_RATIO else MIN_COVERAGE


def subject_of(key_point: str) -> str:
    """The searchable subject inside a curriculum key point.

    "Types of malware: virus, worm, trojan" is a heading with a subject buried
    in it. The heading words are stripped and the punctuation flattened,
    leaving "malware virus worm trojan" - which the full-text search resolves
    to Malware.
    """
    text = _RE_SCAFFOLD.sub("", key_point or "")
    text = text.replace(":", " ").replace(",", " ").replace(";", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    return " ".join(text.split())[:120]


# What is left when a key point was nothing but scaffolding. "Definition of
# terms" strips down to "terms", which is not a subject any encyclopaedia can
# help with - it fetched Terms of Endearment and Terms for Syriac Christians.
# A key point like this is genuine curriculum housekeeping and is skipped.
def _is_housekeeping(phrase: str) -> bool:
    """True when what is left of a key point is not a subject at all.

    Tested on the FIRST word rather than the whole phrase, because a
    curriculum writes "Definition of terms: threat, vulnerability, risk,
    asset" and the scaffold strip leaves "terms threat vulnerability risk
    asset" - a list of words with "terms" at the front, which searched to the
    Common Vulnerability Scoring System. CVSS is real and is about
    vulnerabilities and has nothing to do with defining four words, and a note
    like that invites a question on something nobody taught.

    A key point that opens with "terms", "definitions" or "overview" is
    telling the trainer to define what follows. There is nothing to read up.
    """
    words = [w for w in _norm(phrase).split() if w not in ("the", "a", "an")]
    return bool(words) and words[0] in _NOT_SUBJECTS


_NOT_SUBJECTS = frozenset((
    "terms", "term", "terminology", "definition", "definitions", "meaning",
    "meanings", "introduction", "overview", "concepts", "concept",
    "principles", "principle", "background", "general", "others", "other",
    "importance", "objectives", "scope", "content", "contents", "topics",
    "topic", "revision", "summary", "outline",
))


# --------------------------------------------------------------------------- #
# Turning an article into a note
# --------------------------------------------------------------------------- #
def _summary(text: str) -> str:
    lead = (text or "").split("\n==")[0]
    lead = " ".join(lead.split())
    said = " ".join(_RE_SENTENCE.split(lead)[:SUMMARY_SENTENCES])
    if len(said) > SUMMARY_CHARS:
        said = said[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."
    return said


def _covers(text: str) -> List[str]:
    """The article's section headings.

    The most useful few hundred characters in the whole page, and the least
    obvious. A heading list is a map of the dimensions a topic is normally
    broken down along - Propagation, Detection, Prevention, Legal issues - and
    those are exactly the dimensions an assessor questions along. It is what
    turns "name three types" into "explain how each spreads".
    """
    out: List[str] = []
    for head in _RE_HEADING.findall(text or ""):
        clean = " ".join(head.split())
        if not clean or clean.lower() in _SKIP_HEADINGS:
            continue
        if clean.lower() not in {o.lower() for o in out}:
            out.append(clean)
    return out[:MAX_COVERS]


def _facts(text: str) -> List[str]:
    """"Heading: what that section actually says" - the body of the note.

    The headings alone say what a topic is divided into; they do not say
    anything a question can be marked against. "Detection" is a dimension;
    "signature scanning compares a file against known patterns, heuristics
    look at behaviour" is something a candidate can be asked for and an
    assessor can mark.

    So each section gives up its opening sentence. That is where an
    encyclopaedia puts the claim and the rest of the section supports it,
    which makes the first sentence the densest line in the section and the
    only one worth the prompt's very limited room.

    History and reception sections are skipped. They are about the subject's
    past rather than its practice, and a trainee is being assessed on the
    work.
    """
    out: List[str] = []
    pieces = _RE_HEADING.split(text or "")
    # split() with one group gives [lead, heading, body, heading, body, ...]
    for index in range(1, len(pieces) - 1, 2):
        heading = " ".join(pieces[index].split())
        low = heading.lower()
        if low in _SKIP_HEADINGS or low in _BACKSTORY_HEADINGS:
            continue
        body = " ".join(pieces[index + 1].split())
        if len(body) < MIN_FACT_CHARS:
            continue                       # a heading with subheadings under it
        said = _RE_SENTENCE.split(body)[0]
        if len(said) > FACT_CHARS:
            said = said[:FACT_CHARS].rsplit(" ", 1)[0] + "..."
        out.append(f"{heading}: {said}")
        if len(out) >= MAX_FACTS:
            break
    return out


def _before_apparatus(text: str) -> str:
    """The article up to its references and reading lists.

    A bibliography is a dense field of capitalised surnames, and reading names
    out of one produced Bickel, Bratvold and Thomas as things a candidate
    might be asked about. Nothing after the first apparatus heading is about
    the subject.
    """
    for match in _RE_HEADING.finditer(text or ""):
        if " ".join(match.group(1).split()).lower() in _SKIP_HEADINGS:
            return text[:match.start()]
    return text or ""


def _named(text: str) -> List[str]:
    """Tools, standards and real examples the article names.

    The test that does the work is whether the same word also appears in LOWER
    CASE somewhere in the article. A real name does not: Symantec, NASA and
    CVSS are never written "symantec", while design, subject, types and
    problems are written in lower case constantly, and those are exactly the
    words that survive a naive capital-letter match.

    A sentence-opening rule was tried first and then removed. It did catch
    According and Despite, but it also threw away every name that happened to
    open a sentence - "NIST publishes guidance" lost NIST - and the lower-case
    test catches the same openers anyway, since articles write "according to"
    and "despite this" in the ordinary way further down. The handful that
    never appear in lower case are listed in _NOT_NAMES.

    Acronyms skip the lower-case test. CVE and DAC are written that way
    everywhere, and an article that also writes "cve" in a URL should not lose
    the term because of it.
    """
    body = _before_apparatus(text or "")
    lower_words = set(re.findall(r"(?<![A-Za-z])([a-z]{3,})(?![A-Za-z])", body))
    counts: Dict[str, int] = {}
    first: Dict[str, str] = {}
    for match in _RE_NAMED.finditer(body):
        word = match.group(1)
        key = word.lower()
        if key in _NOT_NAMES:
            continue
        is_acronym = word.isupper()
        if not is_acronym and " " not in word and key in lower_words:
            continue                       # an ordinary word, capitalised
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, word)

    # By how often the article says it. A tool, a standard or a body the
    # subject actually turns on is named repeatedly - NIST, RBAC, CVSS - while
    # the fragments that survive the tests above ("Employee Salaries", a
    # researcher's surname) are said once and never again.
    ranked = sorted(counts, key=lambda k: (-counts[k], k))
    return [first[k] for k in ranked[:MAX_NAMED]]


def note_for(key_point: str, vocab: set,
             element_number: str = "", topic_number: str = "") -> Optional[KnowledgeNote]:
    """One reference note for one taught key point, or None.

    None is an ordinary outcome: no article, nothing that shares the unit's
    vocabulary, or the API not answering. The key point is then simply taught
    content with no note attached, which is where every key point started.
    """
    phrase = subject_of(key_point)
    if not phrase or len(phrase) < 4 or _is_housekeeping(phrase):
        return None
    titles = candidates(phrase)
    if not titles:
        return None
    intros = _extracts(titles, intro_only=True)
    if not intros:
        return None

    ranked = sorted(((_score(t, txt, phrase, vocab)[0], t)
                     for t, txt in intros.items()), key=lambda r: -r[0])

    # The intro decides the ORDER; the whole article decides whether to
    # believe it. An intro is a paragraph, and a paragraph is too small a
    # sample to tell a topic apart from its neighbour - Password policy's
    # holds less of the unit's vocabulary than Group Policy's purely by being
    # shorter. The full text does not have that problem, so the gate is
    # applied there, on the two best candidates rather than only the first.
    for _score_value, title in ranked[:CANDIDATES_VERIFIED]:
        body = _extracts([title], intro_only=False).get(title, "")
        if not body:
            runlog.log(f"Knowledge: '{title}' could not be read in full")
            body = intros.get(title, "")
        if _is_signpost(body):
            runlog.log(f"Knowledge: '{title}' is a disambiguation page, "
                       f"not an article")
            continue
        if coverage(body, vocab) < _gate_for(title, phrase):
            continue
        summary = _summary(body)
        if not summary:
            continue
        return KnowledgeNote(
            key_point=key_point, element_number=element_number,
            topic_number=topic_number, summary=summary, facts=_facts(body),
            covers=_covers(body), named=_named(body), source_title=title,
            source_url=PAGE_URL % title.replace(" ", "_"))

    runlog.log(f"Knowledge: nothing trustworthy for '{key_point}' "
               f"(considered {', '.join(t for _s, t in ranked[:3])})")
    return None


# --------------------------------------------------------------------------- #
# Which key points to look up
# --------------------------------------------------------------------------- #
def key_points(content: Sequence[ElementContent],
               limit: int = MAX_NOTES) -> List[Tuple[str, str, str]]:
    """(element number, topic number, key point), spread across the sub-topics.

    Round-robin rather than in order, and across SUB-TOPICS rather than across
    elements. Spreading by element looked right and was not: an element with
    two sub-topics of four key points each spent its whole share on the first
    sub-topic, and on a real unit that left "Assessment of vulnerabilities"
    with no note while "Identification of threats" had four. The item on the
    risk rating matrix then had nothing to be written from - which is the
    exact fault this module exists to remove, moved one level down.

    Every sub-topic gets a note before any sub-topic gets a second.
    """
    queues: List[List[Tuple[str, str, str]]] = []
    for block in content:
        for topic in block.topics:
            rows = [(block.element_number, topic.number, point)
                    for point in topic.key_points]
            if not rows and topic.title:
                rows = [(block.element_number, topic.number, topic.title)]
            if rows:
                queues.append(rows)

    out: List[Tuple[str, str, str]] = []
    depth = 0
    while queues and len(out) < limit:
        progressed = False
        for rows in queues:
            if depth < len(rows) and len(out) < limit:
                out.append(rows[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return out


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    return (re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_") or "unit")[:80]


def cache_path(unit_title: str) -> str:
    return os.path.join(CACHE_DIR, _safe(unit_title) + "__knowledge.json")


def _cached(unit_title: str) -> Optional[List[KnowledgeNote]]:
    try:
        with io.open(cache_path(unit_title), encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    if time.time() - raw.get("fetched", 0) > CACHE_DAYS * 86400:
        return None
    rows = raw.get("notes")
    if not isinstance(rows, list):
        return None
    out = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("key_point"):
            continue
        out.append(KnowledgeNote(
            key_point=str(row.get("key_point", "")),
            element_number=str(row.get("element_number", "")),
            topic_number=str(row.get("topic_number", "")),
            summary=str(row.get("summary", "")),
            facts=[str(x) for x in row.get("facts", [])],
            covers=[str(x) for x in row.get("covers", [])],
            named=[str(x) for x in row.get("named", [])],
            source_title=str(row.get("source_title", "")),
            source_url=str(row.get("source_url", ""))))
    return out


def _remember(unit_title: str, notes: Sequence[KnowledgeNote]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        marker = os.path.join(CACHE_DIR, ".gitignore")
        if not os.path.exists(marker):
            with io.open(marker, "w", encoding="utf-8") as fh:
                fh.write("*\n")
        with io.open(cache_path(unit_title), "w", encoding="utf-8") as fh:
            json.dump({"unit_title": unit_title, "fetched": time.time(),
                       "notes": [asdict(n) for n in notes]}, fh,
                      indent=1, ensure_ascii=False)
    except OSError as e:
        runlog.warn(f"Knowledge: the notes could not be cached: {e}")


# --------------------------------------------------------------------------- #
# What the rest of the module calls
# --------------------------------------------------------------------------- #
def notes_for(unit_title: str, content: Sequence[ElementContent],
              limit: int = MAX_NOTES, progress_cb=None) -> List[KnowledgeNote]:
    """Reference notes on this unit's taught key points. Never raises.

    Cached per unit: the first CAT on a unit pays for the reading and every
    later one takes it off disk.
    """
    if not content:
        return []
    remembered = _cached(unit_title)
    if remembered is not None:
        runlog.log(f"Knowledge: {len(remembered)} note(s) for '{unit_title}' "
                   f"read from the cache")
        return remembered[:limit]

    vocab = vocabulary(unit_title, content)
    wanted = key_points(content, limit)
    if progress_cb:
        progress_cb(f"Reference: reading up on {len(wanted)} taught topic(s)")

    notes: List[KnowledgeNote] = []
    for element, topic, point in wanted:
        try:
            note = note_for(point, vocab, element, topic)
        except Exception as e:                            # noqa: BLE001
            runlog.warn(f"Knowledge: '{point}' could not be looked up: {e}")
            note = None
        if note is not None:
            notes.append(note)
            if progress_cb:
                progress_cb(f"Reference: {note.source_title} -> {point[:48]}")

    runlog.log(f"Knowledge: {len(notes)} of {len(wanted)} taught topic(s) "
               f"found a reference for '{unit_title}'")
    _remember(unit_title, notes)
    return notes


def label_of(note: KnowledgeNote, index: int) -> str:
    """How one note is headed, and how a row cites it.

    The number is what makes a row's pointer cheap. A row that says "write it
    from N1, N2" costs a dozen characters; one that repeats the key points in
    full costs two hundred, on every row, in a prompt with nothing to spare.
    """
    topic = " ".join(x for x in (note.topic_number, note.key_point) if x)
    return f"N{index}. {topic}"


def fitting(notes: Sequence[KnowledgeNote],
            budget: int = NOTE_BLOCK_CHARS) -> List[KnowledgeNote]:
    """The notes that fit the budget, in order.

    Separate from `render` so the allocation rows can cite exactly the notes
    the model was actually shown. Pointing a row at a note that the budget
    dropped is worse than not pointing at all.
    """
    return [note for note, _text in _blocks(notes, budget)]


def render(notes: Sequence[KnowledgeNote],
           budget: int = NOTE_BLOCK_CHARS) -> str:
    """The notes as the model reads them, within the prompt's character budget.

    Notes are rendered in the order they were gathered, which is round-robin
    across the elements, so a budget that runs out part-way still leaves every
    element with something. Running out is not an error and is not announced
    to the model: a paper written from four notes is a paper written from four
    notes.
    """
    return "\n\n".join(text for _note, text in _blocks(notes, budget))


def _blocks(notes: Sequence[KnowledgeNote],
            budget: int) -> List[Tuple[KnowledgeNote, str]]:
    """[(note, how it is written out)] for the notes that fit."""
    if not notes:
        return []
    blocks: List[Tuple[KnowledgeNote, str]] = []
    spent = 0
    for index, note in enumerate(notes, start=1):
        lines = [f"- {label_of(note, index)}"]
        if note.summary:
            lines.append(f"    {note.summary}")
        # Only the summary and the facts reach the model. `covers` and `named`
        # are kept on the note for the trainer to read in the UI and are
        # deliberately NOT sent.
        #
        # A bare list of names is not knowledge, it is raw material for
        # invention. The Vulnerability scanner article has no sections, so a
        # note built from it fell back to its name list - OSS, CIS, Critical
        # Security Controls, Effective Cyber Defense - and the paper came back
        # asking for FOUR vulnerability scanning tools and marking "OSS (Open
        # Source Scanner)", "CIS scanner", "Critical Security Controls
        # scanner". None of those is a tool. The real answers are Nessus,
        # OpenVAS and Nmap, and a trainee giving them would have been marked
        # wrong against that scheme.
        #
        # In a fact the same names arrive inside a sentence that says what
        # they are - "the NIST/ANSI/INCITS RBAC standard recognises three
        # levels" - and a sentence cannot be misread into a product. So a note
        # with no facts is sent thin rather than padded, and a thin note is
        # better than a confident wrong one.
        lines.extend(f"    {fact}" for fact in note.facts)
        block = "\n".join(lines)
        if spent + len(block) > budget and blocks:
            break
        blocks.append((note, block))
        spent += len(block) + 2
    return blocks


def summarise(notes: Sequence[KnowledgeNote]) -> str:
    """One line for the progress log."""
    if not notes:
        return "no reference notes; writing from the taught content alone"
    sources = len({n.source_title for n in notes})
    return (f"{len(notes)} taught topic(s) backed by {sources} reference "
            f"article(s)")
