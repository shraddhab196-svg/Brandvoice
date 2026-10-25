"""Best-effort brand research: X posts via ScrapeBadger, one official web page via Firecrawl.

Every public function here is fail-soft: on missing keys, unreliable matches, HTTP errors or
timeouts it returns what it has (possibly nothing) and generation continues on user input alone.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.schemas import ResearchReport, ResearchSource, SourceInsight, TweetRequest

logger = logging.getLogger(__name__)

SCRAPEBADGER_BASE = "https://scrapebadger.com/v1/twitter"
FIRECRAWL_BASE = "https://api.firecrawl.dev/v2"

MAX_POSTS = 8
MAX_POST_AGE_DAYS = 3 * 365  # older posts aren't the brand's current voice
MIN_POST_CHARS = 15  # letters/digits (excluding links, hashtags, mentions) needed to show writing style
MIN_ORIGINAL_POSTS = 3  # below this, the brand's own replies are used too
LEADING_MENTIONS_RE = re.compile(r"^(?:@\w+\s*)+")
HASHTAG_OR_MENTION_RE = re.compile(r"[#@]\w+")
# Website lines that quote customers/third parties ("“Love it!” — Sam") aren't the brand's own words.
TESTIMONIAL_RE = re.compile(r"^\s*(?:[\"“‘'«★⭐]|>\s)|[”\"]\s*[—–-]{1,2}\s*\w")
MIN_PAGE_TERMS = 5  # below this a scraped page has no readable content to judge
MAX_POST_CHARS = 300
MAX_WEBSITE_CHARS = 2500
FIRECRAWL_SCRAPE_TIMEOUT_MS = 15000
# Account-match thresholds: an exact name/handle match needs some audience; a prefix match
# (e.g. "notionhq" for "Notion") needs a larger one or a business verification.
MIN_FOLLOWERS_EXACT = 500
MIN_FOLLOWERS_PREFIX = 5000

# Never treated as a brand's own website.
NON_BRAND_DOMAINS = {
    "x.com", "twitter.com", "t.co", "facebook.com", "instagram.com", "linkedin.com", "youtube.com",
    "tiktok.com", "reddit.com", "wikipedia.org", "amazon.com", "amazon.in", "crunchbase.com",
    "glassdoor.com", "yelp.com", "pinterest.com", "medium.com", "github.com", "bloomberg.com",
}

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
BARE_DOMAIN_RE = re.compile(r"\b(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|io|ai|co|app|dev|org|net|in|so|xyz|shop|store)\b", re.IGNORECASE)
X_PROFILE_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})\b", re.IGNORECASE)
X_RESERVED_PATHS = {"home", "search", "explore", "intent", "share", "i", "hashtag", "settings"}


MAX_SOURCE_EXCERPTS = 3
LIMIT_MESSAGE = "Brand research limit reached. Continuing with the information provided."


@dataclass
class Source:
    """A piece of research actually retrieved this run. `text` is what the analysis may quote.

    `segments` are the units a quote must sit inside exactly (one post, or one page line), so every
    accepted quote can be shown next to the excerpt it came from.
    """
    id: str
    kind: str
    name: str
    url: str | None
    text: str
    excerpts: list[str]
    segments: list[str] = field(default_factory=list)
    # Identity, not campaign topic: "strong" = deterministically identified as the brand's own account/site;
    # "uncertain" = plausible, only usable if the analysis step confirms it's the official source.
    tier: str = "strong"
    user_supplied: bool = False


# Source roles: X posts show how the brand WRITES on social; the website is product/company context only.
SOURCE_ROLES = {
    "x_posts": "VOICE EVIDENCE: the brand's own social posts",
    "website": "PRODUCT AND COMPANY CONTEXT ONLY: not evidence of how the brand writes on social",
}


@dataclass
class Review:
    """The analysis step's verdict on retrieved sources (per request; never stored in the shared cache)."""
    accepted: set[str]
    rejected: dict[str, str] = field(default_factory=dict)  # source id -> reason
    insights: dict[str, list[SourceInsight]] = field(default_factory=dict)
    quotes: dict[str, list[str]] = field(default_factory=dict)  # verified quotes per source, to show their excerpts


@dataclass
class ResearchResult:
    x_handle: str | None = None
    posts: list[str] = field(default_factory=list)
    website_url: str | None = None
    website_text: str | None = None
    had_error: bool = False
    limit_reached: bool = False  # a per-run provider call limit stopped a research step
    skipped_reason: str | None = None  # "no_brand_name" | "not_configured" when research never ran
    providers: list[str] = field(default_factory=list)  # providers that ran this time
    excluded: list[str] = field(default_factory=list)  # retrieved content rejected as not matching the brand
    x_tier: str = ""  # set only when posts are accepted ("strong" | "uncertain")
    website_tier: str = ""
    x_user_supplied: bool = False
    website_user_supplied: bool = False
    # Diagnostics: X posts fetched / kept / rejected by reason, testimonial lines removed from the website.
    post_stats: dict[str, int] = field(default_factory=dict)
    testimonials_removed: int = 0

    def sources(self) -> list[Source]:
        """Content the backend retrieved and did not rule out; IDs are stable within a response.
        Sources still need the analysis step's review (see Review) before they count as evidence."""
        out: list[Source] = []
        if self.posts and self.x_handle:
            out.append(Source(
                id=f"S{len(out) + 1}", kind="x_posts", name=f"X posts from @{self.x_handle}",
                url=f"https://x.com/{self.x_handle}", text="\n".join(self.posts), excerpts=self.posts[:MAX_SOURCE_EXCERPTS],
                segments=list(self.posts), tier=self.x_tier, user_supplied=self.x_user_supplied,
            ))
        if self.website_text:
            out.append(Source(
                id=f"S{len(out) + 1}", kind="website", name=f"Website: {_host(self.website_url or '') or 'brand site'}",
                url=self.website_url, text=self.website_text, excerpts=_website_excerpts(self.website_text),
                segments=[line for line in self.website_text.splitlines() if line.strip()],
                tier=self.website_tier, user_supplied=self.website_user_supplied,
            ))
        return out

    def to_prompt(self) -> str | None:
        blocks = [f"[{s.id}] {s.name} ({SOURCE_ROLES[s.kind]})\n{_sanitize(s.text)}" for s in self.sources()]
        return "\n\n".join(blocks) or None

    def report(self, review: Review | None = None) -> ResearchReport:
        """Accepted sources with their findings; rejected ones listed under `excluded` with the reason.
        Without a review (research only, no analysis), every retrieved source is listed as-is."""
        sources = self.sources()
        accepted = [s for s in sources if review is None or s.id in review.accepted]
        public = []
        for s in accepted:
            excerpts = list(s.excerpts)
            for quote in (review.quotes if review else {}).get(s.id, []):
                seg = next((g for g in s.segments if quote in canonical_text(g)), None)
                if seg and seg not in excerpts:
                    excerpts.append(seg)
            public.append(ResearchSource(id=s.id, kind=s.kind, name=s.name, url=s.url, user_supplied=s.user_supplied,
                                         excerpts=excerpts, insights=(review.insights if review else {}).get(s.id, [])))
        excluded = list(self.excluded)
        for s in sources:
            if review is not None and s.id not in review.accepted:
                reason = review.rejected.get(s.id) or "it couldn't be confirmed as this brand's content"
                excluded.append(f"{s.name}: not used because {reason}.")
        status, message = self._status(len(accepted), excluded, has_voice=any(s.kind == "x_posts" for s in accepted))
        return ResearchReport(status=status, message=message, sources=public, excluded=excluded)

    def _status(self, n_sources: int, excluded: list[str] | None = None, has_voice: bool = True) -> tuple[str, str]:
        excluded = self.excluded if excluded is None else excluded
        if n_sources and not has_voice and not self.skipped_reason:
            # Website-only research is product context: it must not be presented as evidence of the social voice.
            return "partial", ("No social posts could be used, so the voice profile is inferred from your brief. "
                               "The website was used only as product and company context."
                               + (f" {LIMIT_MESSAGE}" if self.limit_reached else ""))
        if self.skipped_reason == "no_brand_name":
            return "skipped", "Research skipped: add a brand name to look up public brand sources. The voice profile is inferred from your brief."
        if self.skipped_reason:
            return "skipped", "Brand research isn't enabled. The voice profile is inferred from your brief."
        missing = ("scrapebadger" in self.providers and not self.posts) or ("firecrawl" in self.providers and not self.website_text)
        if n_sources == 0:
            if excluded:
                return "failed", ("Content was found, but none of it could be used as this brand's voice (see \"Not used\" "
                                  "for reasons). The voice profile is inferred from your brief.")
            if self.had_error:  # report the root cause: a limit may only have blocked a fallback after a failure
                return "failed", "Brand research couldn't be completed. The voice profile is inferred from your brief."
            if self.limit_reached:
                return "failed", LIMIT_MESSAGE
            return "completed", "No reliable public sources matched this brand. The voice profile is inferred from your brief."
        noun = "source" if n_sources == 1 else "sources"
        if self.limit_reached:
            return "partial", f"Used {n_sources} public {noun}. {LIMIT_MESSAGE}"
        if excluded:
            return "partial", f"Used {n_sources} verified {noun}; other retrieved content didn't match this brand and was excluded."
        if self.had_error or missing:
            return "partial", f"Used {n_sources} public {noun}; other brand sources were unavailable."
        return "completed", f"Research completed using {n_sources} public {noun}."


@dataclass
class _Session:
    """Per-request research state shared by all HTTP calls."""
    client: httpx.Client
    result: ResearchResult
    deadline: float
    default_timeout: float
    limits: dict[str, int] = field(default_factory=dict)  # provider -> max requests this run
    calls: dict[str, int] = field(default_factory=dict)  # provider -> requests attempted this run
    blocked: set[str] = field(default_factory=set)  # providers that returned 401/402

    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def calls_left(self, provider: str) -> int:
        return self.limits.get(provider, 0) - self.calls.get(provider, 0)

    def note_limit(self, provider: str, skipped: str) -> None:
        self.result.limit_reached = True
        logger.warning("Research limit reached: %s allows %d request(s) per generation; skipping %s",
                       provider, self.limits.get(provider, 0), skipped)


MIN_REQUEST_SECONDS = 1.0  # don't start a call that can't realistically finish

_cache: dict[tuple, tuple[float, ResearchResult]] = {}


def get_http_client(settings: Settings) -> httpx.Client:
    return httpx.Client(timeout=settings.research_timeout_seconds)


def _cache_key(req: TweetRequest) -> tuple:
    # Every input that influences research decisions is part of the key (the brand context, links and handles all
    # come from these fields), so results gathered for one brief are never reused for another. Voice answers are
    # not part of it: they don't change what research finds.
    return tuple(canonical_text(v or "").lower()
                 for v in (req.brand_name, req.industry, req.campaign_objective, req.description))


def remember_research(req: TweetRequest, settings: Settings, result: ResearchResult) -> None:
    """Keep a no-usable-posts result (even one with errors) so the follow-up request with the user's voice answers
    reuses it instead of spending research credits again."""
    _cache[_cache_key(req)] = (time.monotonic() + settings.research_cache_ttl_seconds, result)


def gather_research(req: TweetRequest, settings: Settings) -> ResearchResult:
    """Run all research steps. Never raises."""
    if not (settings.scrapebadger_configured or settings.firecrawl_configured):
        return ResearchResult(skipped_reason="not_configured")
    if not req.brand_name:
        return ResearchResult(skipped_reason="no_brand_name")  # input rule: no brand name, no scraping at all

    key = _cache_key(req)
    cached = _cache.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    result = ResearchResult()
    try:
        with get_http_client(settings) as client:
            # One session per /generate run: every research step shares its call counters.
            session = _Session(
                client, result, time.monotonic() + settings.research_budget_seconds, settings.research_timeout_seconds,
                limits={"scrapebadger": settings.research_max_scrapebadger_calls, "firecrawl": settings.research_max_firecrawl_calls},
            )
            if settings.scrapebadger_configured:
                result.providers.append("scrapebadger")
                _research_x(session, req, settings)
            if settings.firecrawl_configured:
                result.providers.append("firecrawl")
                _research_website(session, req, settings)
    except Exception:  # last-resort guard: research must never break generation
        logger.exception("Unexpected research failure; continuing without research")
        result.had_error = True

    if not result.had_error:
        _cache[key] = (time.monotonic() + settings.research_cache_ttl_seconds, result)
    logger.info("Research: x=%s posts=%d website=%s", result.x_handle, len(result.posts), bool(result.website_text))
    return result


# ---------- ScrapeBadger (X) ----------

def _research_x(session: _Session, req: TweetRequest, settings: Settings) -> None:
    headers = {"x-api-key": settings.scrapebadger_api_key.get_secret_value()}
    result = session.result
    user = _discover_x_account(session, req, headers)
    if not user:
        return
    handle = user["username"]
    # A handle the user linked is a lead, not proof: it must still identify as this brand.
    user_supplied = (_find_x_handle_in_text(req.description) or "").lower() == handle.lower()
    label = f"X account @{handle}" + (" (from your brief)" if user_supplied else "")
    bio = user.get("description") if isinstance(user.get("description"), str) else ""
    identity_text = " ".join(filter(None, [handle, user.get("name") if isinstance(user.get("name"), str) else "", bio]))
    # Whether this is the brand's account is decided by identity signals, never by what its posts talk about: a real
    # brand account posting memes on other topics is still the best voice evidence. No vocabulary checks on X content.
    identity = _x_identity(user, req, user_supplied)
    data = _request_json(session, "scrapebadger", "GET", f"{SCRAPEBADGER_BASE}/users/{handle}/latest_tweets", headers=headers)
    if not isinstance(data, dict):
        return  # request failed (already recorded as an error)
    posts, stats, stale_years = _clean_posts(data.get("data") or [], handle)
    result.post_stats = stats
    logger.info("Research X @%s: identity=%s posts %s", handle, identity, _format_stats(stats))
    if not posts:
        if stale_years and stats.get("stale"):
            reason = (f"its posts are all more than {MAX_POST_AGE_DAYS // 365} years old (latest from {max(stale_years)}) "
                      "and may belong to an inactive account or a different brand")
        else:
            reason = f"none of its {stats.get('fetched', 0)} fetched posts were usable ({_format_rejections(stats)})"
        _exclude(result, label, req, reason=reason)
        return
    if user_supplied and not _brand_mentioned(req, "\n".join([identity_text, *posts])):
        _exclude(result, label, req, reason="nothing in it identifies it as this brand's account")
        return
    # Strongly identified accounts are used as-is. Anything else (incl. lookalikes) is "uncertain": used only if the
    # analysis step explicitly confirms it's the official account; no verdict or "not official" -> not used.
    tier = "strong" if identity == "strong" else "uncertain"
    result.x_handle, result.posts, result.x_tier, result.x_user_supplied = handle, posts, tier, user_supplied
    if user.get("url") and identity == "strong":  # don't spend the website call on a possible lookalike's link
        result.website_url = result.website_url or user["url"]


def _x_identity(user: dict, req: TweetRequest, user_supplied: bool) -> str:
    """How confidently the account is the brand's own: "strong" | "plausible" | "weak". No campaign-topic input.

    strong: profile links to the brand's own domain, OR exact handle/name match with business/legacy verification.
    plausible: exact or prefix name/handle match, the bio names the brand, or the user linked it (a lead, not proof;
               lookalikes live here and get the business-context guard + analysis confirmation)."""
    brand = _norm(req.brand_name or "")
    url = user.get("url") if isinstance(user.get("url"), str) else ""
    bio = user.get("description") if isinstance(user.get("description"), str) else ""
    exact = brand in (_norm(user.get("username", "")), _norm(user.get("name", "") or ""))
    org_verified = bool(user.get("verified")) or str(user.get("verified_type") or "").lower() in ("business", "government")
    own_domain = bool(url) and not _is_non_brand_domain(url) and _site_label(url) == brand
    names_brand = _brand_mentioned(req, bio)
    if own_domain or (exact and org_verified):
        return "strong"
    if user_supplied or exact or names_brand or _match_strength(user, brand) >= 1:
        return "plausible"
    return "weak"


def _discover_x_account(session: _Session, req: TweetRequest, headers: dict) -> dict | None:
    # 1. A handle the user linked in the description (verified afterwards like any other account).
    handle = _find_x_handle_in_text(req.description)
    if handle:
        user = _request_json(session, "scrapebadger", "GET", f"{SCRAPEBADGER_BASE}/users/{handle}/by_username", headers=headers)
        if not _is_user(user):
            session.result.excluded.append(f"X account @{handle} (from your brief): it couldn't be retrieved, so it wasn't used.")
            return None
        return user

    if not req.brand_name:
        return None  # no reliable way to identify the account
    brand = _norm(req.brand_name)
    if len(brand) < 2:
        return None

    # 2. Cheap guess (1 item ≈ 2 credits): the brand name as a handle.
    if len(brand) <= 15:
        user = _request_json(session, "scrapebadger", "GET", f"{SCRAPEBADGER_BASE}/users/{brand}/by_username", headers=headers)
        if _is_user(user) and _is_reliable_match(user, brand):
            return user

    # 3. Search (costs ~1 credit per returned user), keep the best reliable match only.
    # Only worth it if a call remains afterwards to fetch the account's posts.
    if session.calls_left("scrapebadger") < 2:
        session.note_limit("scrapebadger", "account search")
        return None
    data = _request_json(session, "scrapebadger", "GET", f"{SCRAPEBADGER_BASE}/users/search_users", headers=headers, params={"query": req.brand_name})
    users = [u for u in (data.get("data") or []) if _is_user(u)] if isinstance(data, dict) else []
    candidates = [u for u in users if _is_reliable_match(u, brand)]
    if not candidates:
        logger.info("No reliable X account match for the brand; skipping X research")
        return None
    return max(candidates, key=lambda u: (_match_strength(u, brand), _is_verified(u), u.get("followers_count") or 0))


def _match_strength(user: dict, brand: str) -> int:
    handle, name = _norm(user.get("username", "")), _norm(user.get("name", ""))
    if brand in (handle, name):
        return 2
    if len(brand) >= 4 and (handle.startswith(brand) or name.startswith(brand)):
        return 1
    return 0


def _is_verified(user: dict) -> bool:
    return bool(user.get("verified") or user.get("verified_type"))


def _is_reliable_match(user: dict, brand: str) -> bool:
    if user.get("protected"):
        return False
    followers = user.get("followers_count") or 0
    strength = _match_strength(user, brand)
    if strength == 2:
        return followers >= MIN_FOLLOWERS_EXACT or _is_verified(user)
    if strength == 1:
        return followers >= MIN_FOLLOWERS_PREFIX or bool(user.get("verified_type"))
    return False


def _post_datetime(t: dict) -> datetime | None:
    """Post time from ScrapeBadger's ISO `created_at_datetime` or X's "Thu Jan 10 18:02:11 +0000 2013" format."""
    for raw, parse in ((t.get("created_at_datetime"), lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))),
                       (t.get("created_at"), lambda s: datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y"))):
        if isinstance(raw, str):
            try:
                dt = parse(raw)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _clean_posts(tweets: list, handle: str = "") -> tuple[list[str], dict[str, int], list[int]]:
    """Filter individual posts for authenticity and quality (never for campaign topic).

    Kept: the account's own recent posts with enough words to show its writing. Original posts come first; the
    brand's own replies fill in when it has fewer than MIN_ORIGINAL_POSTS originals (reply-heavy brands still have a
    voice). Rejected, and counted by reason: retweets, posts by another author, posts older than MAX_POST_AGE_DAYS,
    too-short/link-only posts, duplicates, malformed items. Returns (posts, stats, years of stale posts)."""
    stats = {"fetched": 0, "kept": 0, "retweet": 0, "other_author": 0, "stale": 0, "too_short": 0, "duplicate": 0,
             "malformed": 0, "replies_used": 0, "over_limit": 0}
    originals, replies, stale_years, seen = [], [], [], set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_POST_AGE_DAYS)
    for t in tweets:
        stats["fetched"] += 1
        if not isinstance(t, dict):
            stats["malformed"] += 1
            continue
        if t.get("is_retweet") or t.get("retweeted_status_id"):
            stats["retweet"] += 1
            continue
        author = t.get("username")
        if handle and isinstance(author, str) and author and author.lower() != handle.lower():
            stats["other_author"] += 1  # not written by this account: not its voice
            continue
        posted = _post_datetime(t)
        if posted is not None and posted < cutoff:
            stats["stale"] += 1
            stale_years.append(posted.year)
            continue
        text = LEADING_MENTIONS_RE.sub("", URL_RE.sub("", t.get("full_text") or t.get("text") or ""))
        text = " ".join(text.split())
        if len(re.findall(r"[A-Za-z0-9]", HASHTAG_OR_MENTION_RE.sub("", text))) < MIN_POST_CHARS:
            stats["too_short"] += 1
            continue
        if _norm(text) in seen:
            stats["duplicate"] += 1
            continue
        seen.add(_norm(text))
        (replies if t.get("in_reply_to_status_id") else originals).append(text[:MAX_POST_CHARS])
    pool = originals + (replies if len(originals) < MIN_ORIGINAL_POSTS else [])
    posts = pool[:MAX_POSTS]
    stats["kept"] = len(posts)
    stats["replies_used"] = max(0, len(posts) - len(originals))
    stats["over_limit"] = len(originals) + len(replies) - len(posts)
    return posts, stats, stale_years


MIN_STYLE_POSTS = 4  # fewer posts than this are too few to describe a brand's writing mechanics
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤]")


def observed_style(posts: list[str]) -> dict | None:
    """Measurable writing mechanics of the brand's OWN posts (no interpretation, no brand-specific rules).
    Returns None when there are too few posts to say anything reliable."""
    if len(posts) < MIN_STYLE_POSTS:
        return None
    n = len(posts)
    words = sorted(len(p.split()) for p in posts)

    def share(test) -> float:
        return round(sum(1 for p in posts if test(p)) / n, 2)

    def first_letter(p: str) -> str:
        m = re.search(r"[A-Za-z]", HASHTAG_OR_MENTION_RE.sub("", p))
        return m.group(0) if m else ""

    return {
        "posts": n,
        "lowercase_start": share(lambda p: first_letter(p).islower()),
        "median_words": words[n // 2],
        "max_words": words[-1],
        "hashtags": share(lambda p: "#" in p),
        "emoji": share(lambda p: bool(EMOJI_RE.search(p))),
        "exclamation": share(lambda p: "!" in p),
        "question": share(lambda p: "?" in p),
    }


def _format_stats(stats: dict[str, int]) -> str:
    return f"fetched={stats.get('fetched', 0)} kept={stats.get('kept', 0)} rejected: {_format_rejections(stats) or 'none'}"


def _format_rejections(stats: dict[str, int]) -> str:
    labels = {"retweet": "retweets", "other_author": "by other accounts", "stale": "older than 3 years",
              "too_short": "too short", "duplicate": "duplicates", "malformed": "malformed",
              "over_limit": "beyond the post limit"}
    return ", ".join(f"{stats[k]} {v}" for k, v in labels.items() if stats.get(k))


# ---------- Firecrawl (website) ----------

def _research_website(session: _Session, req: TweetRequest, settings: Settings) -> None:
    headers = {"Authorization": f"Bearer {settings.firecrawl_api_key.get_secret_value()}"}
    # Priority: URL in the user's description > website on the matched X profile > search by brand name.
    # Each path is a single Firecrawl request, so the 1-call limit still yields page content.
    user_url = _find_website_in_text(req.description)
    url = user_url or session.result.website_url
    if url:
        data = _request_json(
            session, "firecrawl", "POST", f"{FIRECRAWL_BASE}/scrape", headers=headers,
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True, "timeout": FIRECRAWL_SCRAPE_TIMEOUT_MS},
            timeout=FIRECRAWL_SCRAPE_TIMEOUT_MS / 1000 + 5,
        )
        page = data.get("data") if isinstance(data, dict) else None
        if isinstance(page, dict):
            _apply_page(session.result, page, url, req, user_supplied=bool(user_url))
        elif user_url:
            session.result.excluded.append(f"Website {_host(user_url)} (from your brief): it couldn't be retrieved, so it wasn't used.")
    elif req.brand_name:
        _search_website(session, req, headers)


def _search_website(session: _Session, req: TweetRequest, headers: dict) -> None:
    brand = _norm(req.brand_name or "")
    if len(brand) < 3:
        return
    query = f"{req.brand_name} {req.industry or ''} official website".strip()
    # scrapeOptions returns each result's page content in the same request (no follow-up scrape).
    data = _request_json(
        session, "firecrawl", "POST", f"{FIRECRAWL_BASE}/search", headers=headers,
        json={"query": query, "limit": 3, "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}},
        timeout=FIRECRAWL_SCRAPE_TIMEOUT_MS / 1000 + 5,
    )
    web = (data.get("data") or {}).get("web") if isinstance(data, dict) else None
    candidates = [item for item in web or [] if isinstance(item, dict) and isinstance(item.get("url"), str)
                  and not _is_non_brand_domain(item["url"]) and brand in _domain_label(item["url"])]
    # Prefer the brand's main site (duolingo.com) over its subdomains (schools.duolingo.com) and lookalikes.
    candidates.sort(key=lambda item: (_site_label(item["url"]) != brand, _host(item["url"]).count(".")))
    for item in candidates:
        if _apply_page(session.result, item, item["url"], req, user_supplied=False):
            return  # otherwise try the next candidate from the same response (no extra request)
    logger.info("No official website found for the brand; skipping website research")


def _apply_page(result: ResearchResult, page: dict, url: str, req: TweetRequest, *, user_supplied: bool) -> bool:
    """Accept a page as product/company context if it belongs to the brand. Never judged by campaign topic.
    Official domain (incl. subdomains) -> strong. A lookalike domain containing the brand, or a user-supplied URL that
    names the brand, is only "uncertain" and is rejected if it describes a different business."""
    meta = page.get("metadata") or {}
    final_url = meta.get("url") or meta.get("sourceURL") or url
    label = f"Website {_host(final_url)}" + (" (from your brief)" if user_supplied else "")
    if _is_non_brand_domain(final_url):
        if user_supplied:
            _exclude(result, label, req, reason="it isn't a brand-owned site")
        return False
    text = _clean_markdown(page.get("markdown") or "")
    header = " — ".join(str(v) for v in (meta.get("title") or page.get("title"), meta.get("description") or page.get("description"))
                        if isinstance(v, str) and v)
    combined = (f"{header}\n{text}".strip() if header else text)[:MAX_WEBSITE_CHARS]
    # Customer testimonials are third-party words: never the brand's own copy.
    lines = combined.splitlines()
    kept_lines = [line for line in lines if not TESTIMONIAL_RE.search(line)]
    removed = len(lines) - len(kept_lines)
    combined = "\n".join(kept_lines)
    if len(_terms(combined)) < MIN_PAGE_TERMS:  # empty/blocked/placeholder page: say so, don't call it "unrelated"
        result.excluded.append(f"{label}: no readable content was retrieved, so it couldn't be checked.")
        return False
    brand = _norm(req.brand_name or "")
    official = bool(brand) and _site_label(final_url) == brand
    owned = official or (bool(brand) and (brand in _domain_label(final_url) or (user_supplied and _brand_mentioned(req, combined))))
    if not owned:
        _exclude(result, label, req, reason="nothing on it identifies it as this brand's site")
        return False
    if not official and _describes_other_business(req, kept_lines):
        _exclude(result, label, req, reason=f"it describes a different business than {req.brand_name}")
        return False
    result.website_url, result.website_text = final_url, combined
    result.website_tier, result.website_user_supplied = ("strong" if official else "uncertain"), user_supplied
    result.testimonials_removed = removed
    if removed:
        logger.info("Research website %s: removed %d testimonial line(s)", _host(final_url), removed)
    return True


def _clean_markdown(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)  # images
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)  # links -> text
    md = URL_RE.sub("", md)
    lines = [" ".join(line.split()) for line in md.splitlines()]
    return "\n".join(line for line in lines if len(line) > 2)


# ---------- helpers ----------

def _request_json(session: _Session, provider: str, method: str, url: str, **kwargs) -> object | None:
    result = session.result
    if provider in session.blocked:
        return None  # key already rejected (401/402) during this request
    if session.calls_left(provider) <= 0:
        # Central credit guard: no code path (retry, fallback, loop) can exceed the per-run limit.
        session.note_limit(provider, urlparse(url).path)
        return None
    remaining = session.time_left()
    if remaining < MIN_REQUEST_SECONDS:
        logger.info("Research budget exhausted; skipping %s", urlparse(url).path)
        result.had_error = True  # incomplete result: don't cache it
        return None
    kwargs["timeout"] = min(kwargs.get("timeout", session.default_timeout), remaining)
    session.calls[provider] = session.calls.get(provider, 0) + 1  # counted on attempt, even if it fails
    try:
        resp = session.client.request(method, url, **kwargs)
    except httpx.TimeoutException:
        logger.warning("Research request timed out: %s", urlparse(url).path)
        result.had_error = True
        return None
    except httpx.HTTPError as exc:
        logger.warning("Research request failed: %s (%s)", urlparse(url).path, type(exc).__name__)
        result.had_error = True
        return None
    if resp.status_code >= 400:
        # 401 bad key, 402 out of credits, 429 rate limited; 404 for a guessed handle is expected.
        logger.warning("Research request %s returned HTTP %s", urlparse(url).path, resp.status_code)
        if resp.status_code in (401, 402):
            session.blocked.add(provider)
        if resp.status_code != 404:
            result.had_error = True
        return None
    try:
        data = resp.json()
    except ValueError:
        logger.warning("Research response from %s was not JSON", urlparse(url).path)
        result.had_error = True
        return None
    if provider == "firecrawl" and isinstance(data, dict) and data.get("success") is False:
        logger.warning("Firecrawl reported failure for %s", urlparse(url).path)
        result.had_error = True  # transient failure: retry next time instead of caching
        return None
    return data


def _is_user(user: object) -> bool:
    return isinstance(user, dict) and isinstance(user.get("username"), str) and bool(user["username"])


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _host(url: str) -> str:
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _domain_label(url: str) -> str:
    return _norm("".join(_host(url).split(".")[:-1]))


def _site_label(url: str) -> str:
    """The registrable name: schools.duolingo.com -> "duolingo", shop.example.co.uk -> "example"."""
    labels = _host(url).split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in ("co", "com", "org", "net", "ac", "gov"):
        return _norm(labels[-3])
    return _norm(labels[-2]) if len(labels) >= 2 else ""


def _is_non_brand_domain(url: str) -> bool:
    host = _host(url)
    return not host or any(host == d or host.endswith("." + d) for d in NON_BRAND_DOMAINS)


def _find_website_in_text(text: str) -> str | None:
    for match in URL_RE.findall(text) + BARE_DOMAIN_RE.findall(text):
        url = match.rstrip(".,;:!?")
        if not url.lower().startswith("http"):
            url = f"https://{url}"
        if not _is_non_brand_domain(url):
            return url
    return None


def _find_x_handle_in_text(text: str) -> str | None:
    for handle in X_PROFILE_RE.findall(text):
        if handle.lower() not in X_RESERVED_PATHS:
            return handle
    return None


# ---------- relevance + exact matching ----------

# Words too generic to show that a source is about *this* brand.
STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before being below between both but by can
could did do does doing down during each few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your yours yourself
yourselves new best great get got make made help people time day days today week year years launch launching brand brands
product products service services company customer customers team world free use using used one like just really every
good better want need know love thing things way ways start started first next back look looking come coming take see
work working thank thanks happy available now official www com http https via amp us let lets keep big little much many
""".split())


def _stem(word: str) -> str:
    w = word.lower().strip("'-")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
        if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz":  # "planning" -> "plan", "running" -> "run"
            w = w[:-1]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    # Drop a silent final "e" so "summarize"/"summarized"/"summarizing" and "update"/"updated" share a stem.
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def _terms(text: str) -> set[str]:
    out = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", text or ""):
        for part in raw.split("-"):
            if part.lower() in STOPWORDS:
                continue
            stem = _stem(part)
            if len(stem) >= 3 and stem not in STOPWORDS:
                out.add(stem)
    return out


def _business_terms(req: TweetRequest) -> set[str]:
    """What business the brand is in: industry + product description. The campaign objective is deliberately
    excluded: a campaign is about one topic, while a brand's feed legitimately covers many."""
    brand_terms = _terms(req.brand_name or "")
    return _terms(" ".join(filter(None, [req.industry, req.description]))) - brand_terms


def _brand_mentioned(req: TweetRequest, text: str) -> bool:
    brand = _norm(req.brand_name or "")
    return len(brand) >= 3 and brand in _norm(text)


MIN_BRIEF_TERMS = 6  # below this the brand context is too thin to rule anything out by vocabulary
SUBSTANTIVE_TERMS = 8  # content with this much vocabulary and zero overlap is about something else


def _describes_other_business(req: TweetRequest, texts: list[str], min_terms: int = SUBSTANTIVE_TERMS) -> bool:
    """Lookalike-website guard, applied ONLY to non-official domains (e.g. notioninsurance.com for Notion).

    True when substantive page content shares no vocabulary at all with the brand's business (industry + product
    description). Never compares with the campaign objective, and never applied to X accounts: a social feed
    legitimately covers any topic, so X identity is decided by identity signals and the analysis review instead."""
    ctx = _business_terms(req)
    pooled = set().union(*(_terms(t) for t in texts if t and t.strip())) if texts else set()
    return len(ctx) >= MIN_BRIEF_TERMS and len(pooled) >= min_terms and not (ctx & pooled)


def _exclude(result: ResearchResult, what: str, req: TweetRequest, reason: str | None = None) -> None:
    why = reason or f"its content didn't match {req.brand_name} or your brief"
    result.excluded.append(f"{what}: {why}, so it wasn't used.")
    logger.info("Research: excluded %s (%s)", what, why)


def canonical_text(text: str) -> str:
    """Typographic quotes -> plain, whitespace collapsed. Case and wording are preserved (exact matching)."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split())


def _website_excerpts(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if len(line) >= 40]
    return [line[:240] for line in (lines or [text])[:MAX_SOURCE_EXCERPTS]]


def _sanitize(text: str) -> str:
    # Stop scraped content from opening/closing prompt tags like </brand_research>.
    return text.replace("<", "(").replace(">", ")")
