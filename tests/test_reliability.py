"""Regression tests for the reliability fixes: relevance tiers + model review, user-supplied links,
early metadata checks, quoted insights, claim validation, and story/name detection.
Each area has both false-negative (bad content caught) and false-positive (good content kept) cases."""

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs
from test_grounding import INSURANCE_USER, NOTION_REQ, REAL_POSTS, REAL_SITE, REAL_USER
from test_research import make_settings, post_generate, use_router

SHORT_REQ = {"brand_name": "Notion", "campaign_objective": "Get more sign-ups", "description": "A simple notes app."}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def x_routes(user, posts, handle="notion", scrape=None, search=None) -> dict:
    r = {("GET", f"/v1/twitter/users/{handle}/by_username"): (200, user),
         ("GET", f"/v1/twitter/users/{user['username']}/latest_tweets"): (200, posts)}
    if scrape is not None:
        r[("POST", "/v2/scrape")] = scrape
    if search is not None:
        r[("POST", "/v2/search")] = search
    return r


def generate(monkeypatch, routes, req=NOTION_REQ, analysis=None, generation=None, settings=None):
    router = use_router(monkeypatch, routes)
    fake = install_fake_groq(monkeypatch)
    if analysis is not None:
        fake.analysis = analysis
    if generation is not None:
        fake.generation = generation
    r = post_generate(settings or make_settings(fc=None), req)
    assert r.status_code == 200, r.text
    return r.json(), router, fake


# Insurance posts that share two brief words ("project", "meeting") by coincidence.
COINCIDENCE_POSTS = {"data": [
    {"id": "1", "text": "Our claims crew treats every project like it matters.", "created_at": "x"},
    {"id": "2", "text": "Book a meeting with a licensed agent today.", "created_at": "x"},
    {"id": "3", "text": "Protect your home and car with one policy.", "created_at": "x"},
]}
GENERIC_POSTS = {"data": [
    {"id": "1", "text": "Happy Friday! What are you building this weekend?", "created_at": "x"},
    {"id": "2", "text": "Small steps, big ideas. Have a calm week, everyone.", "created_at": "x"},
]}


# ---------- 1. relevance: tiers + explicit review ----------

def test_two_word_coincidence_is_only_uncertain_and_needs_confirmation(monkeypatch):
    use_router(monkeypatch, x_routes(INSURANCE_USER, COINCIDENCE_POSTS))
    result = research.gather_research(TweetRequest(**NOTION_REQ), make_settings(fc=None))
    assert result.x_tier == "uncertain"  # the old rule (>= 2 shared words) would have accepted it outright


@pytest.mark.parametrize("relevance", [[], [{"source_id": "S1", "relevant": False, "reason": "An insurance company."}]])
def test_unconfirmed_or_rejected_coincidental_posts_are_not_evidence(monkeypatch, relevance):
    body, _, fake = generate(monkeypatch, x_routes(INSURANCE_USER, COINCIDENCE_POSTS), req=with_prefs(NOTION_REQ),
                             analysis=analysis_payload(relevance=relevance, evidence=[
                                 {"source_id": "S1", "quote": "Book a meeting with a licensed agent today."}]),
                             generation=tweets_payload(insight_ids=["S1.1"]))
    report = body["research"]
    assert report["sources"] == [] and report["status"] == "failed"
    assert any("X posts from @notion" in e for e in report["excluded"])
    expected = "insurance company" if relevance else "couldn't be confirmed"
    assert any(expected in e for e in report["excluded"])
    assert all(t["basis"] == "inferred" for t in body["voice_profile"]["traits"])
    assert all(t["source_ids"] == [] for t in body["tweets"])


def test_model_can_reject_even_a_strong_keyword_match(monkeypatch):
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS), req=with_prefs(NOTION_REQ),
                          analysis=analysis_payload(relevance=[{"source_id": "S1", "relevant": False, "reason": "Fan account."}]))
    assert body["research"]["sources"] == [] and any("Fan account" in e for e in body["research"]["excluded"])


def test_strong_match_is_used_when_model_gives_no_verdict(monkeypatch):
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS), analysis=analysis_payload(relevance=[]))
    assert [s["id"] for s in body["research"]["sources"]] == ["S1"]


def test_generic_brand_posts_with_short_brief_are_kept_when_confirmed(monkeypatch):
    user = {**REAL_USER, "description": ""}
    body, router, _ = generate(monkeypatch, x_routes(user, GENERIC_POSTS), req=SHORT_REQ)  # default fake confirms S1
    assert [s["kind"] for s in body["research"]["sources"]] == ["x_posts"]
    assert body["research"]["status"] == "completed"


def test_short_brief_never_triggers_early_bio_rejection(monkeypatch):
    router = use_router(monkeypatch, x_routes(REAL_USER, GENERIC_POSTS))  # bio shares nothing with SHORT_REQ
    result = research.gather_research(TweetRequest(**SHORT_REQ), make_settings(fc=None))
    assert any(p.endswith("/latest_tweets") for p in router.paths())
    assert result.posts and result.excluded == []


# ---------- 3. user-supplied links are verified, not trusted ----------

def with_links(**extra) -> dict:
    return {**NOTION_REQ, "description": NOTION_REQ["description"] + " " + " ".join(extra.values())}


def test_user_supplied_handle_for_unrelated_account_is_rejected(monkeypatch):
    stranger = {**INSURANCE_USER, "username": "quillco", "name": "Quill & Co", "description": ""}
    req = with_links(handle="x.com/quillco")
    use_router(monkeypatch, x_routes(stranger, COINCIDENCE_POSTS, handle="quillco"))
    result = research.gather_research(TweetRequest(**req), make_settings(fc=None))
    assert result.posts == [] and any("@quillco (from your brief)" in e for e in result.excluded)


def test_user_supplied_lookalike_handle_is_only_a_lead(monkeypatch):
    bad = {**INSURANCE_USER, "username": "notionins", "description": "Insurance made simple: auto, home and renters coverage.",
           "url": "https://notion-insure.example"}
    use_router(monkeypatch, x_routes(bad, COINCIDENCE_POSTS, handle="notionins"))
    result = research.gather_research(TweetRequest(**with_links(h="x.com/notionins")), make_settings(fc=None))
    assert result.x_tier == "uncertain" and result.x_user_supplied  # needs the identity review to be used
    assert result.website_url is None  # its link isn't followed


def test_user_supplied_handle_that_does_not_exist_is_reported(monkeypatch):
    router = use_router(monkeypatch, {})  # every request 404s
    result = research.gather_research(TweetRequest(**with_links(h="x.com/ghostbrand")), make_settings(fc=None))
    assert any("@ghostbrand (from your brief): it couldn't be retrieved" in e for e in result.excluded)
    assert router.paths() == ["/v1/twitter/users/ghostbrand/by_username"]


def test_user_supplied_handle_for_real_account_is_used_and_marked(monkeypatch):
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS, handle="NotionHQ"), req=with_links(h="x.com/NotionHQ"))
    src = body["research"]["sources"][0]
    assert src["kind"] == "x_posts" and src["user_supplied"] is True


@pytest.mark.parametrize("page, reason", [
    ({"markdown": "Quill & Co\nHandmade leather notebooks and fountain pens.", "metadata": {"url": "https://quill.co/"}},
     "nothing on it identifies it as this brand's site"),
    ({"markdown": "Notion Insurance\nAffordable auto and home coverage with fast online claims for every family.",
      "metadata": {"url": "https://notion-cover.com/"}}, "describes a different business"),
])
def test_user_supplied_website_must_belong_to_brand_and_match(monkeypatch, page, reason):
    router = use_router(monkeypatch, {("POST", "/v2/scrape"): (200, {"success": True, "data": page})})
    result = research.gather_research(TweetRequest(**with_links(u=page["metadata"]["url"])), make_settings(sb=None))
    assert result.website_text is None and any("(from your brief)" in e and reason in e for e in result.excluded)


def test_user_supplied_website_on_other_domain_naming_the_brand_is_accepted(monkeypatch):
    page = {**REAL_SITE, "metadata": {"url": "https://try-notes.app/"}}
    use_router(monkeypatch, {("POST", "/v2/scrape"): (200, {"success": True, "data": page})})
    result = research.gather_research(TweetRequest(**with_links(u="try-notes.app")), make_settings(sb=None))
    assert result.website_text and result.website_user_supplied


def test_inaccessible_user_website_is_reported_transparently(monkeypatch):
    use_router(monkeypatch, {("POST", "/v2/scrape"): (500, {"error": "down"})})
    result = research.gather_research(TweetRequest(**with_links(u="notion.so")), make_settings(sb=None))
    report = result.report()
    assert report.status == "failed" and any("notion.so (from your brief): it couldn't be retrieved" in e for e in report.excluded)


def test_research_limits_hold_with_user_links(monkeypatch):
    router = use_router(monkeypatch, {
        **x_routes(REAL_USER, REAL_POSTS, handle="NotionHQ"),
        ("POST", "/v2/scrape"): (200, {"success": True, "data": REAL_SITE}),
    })
    research.gather_research(TweetRequest(**with_links(h="x.com/NotionHQ", u="notion.so")), make_settings())
    sb = sum(r.url.host == "scrapebadger.com" for r in router.requests)
    fc = sum(r.url.host == "api.firecrawl.dev" for r in router.requests)
    assert sb <= 2 and fc <= 1


# ---------- 5. insights: quoted findings vs inferences ----------

def test_insight_quotes_must_be_verbatim_and_relevant_to_the_insight(monkeypatch):
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS), analysis=analysis_payload(insights=[{
        "source_id": "S1", "insights": [
            {"text": "Positions AI as writing meeting notes for you.", "quote": "Meeting notes that write themselves."},
            {"text": "Leans on templates for projects.", "quote": "Meeting notes that write themselves."},  # real quote, wrong insight
            {"text": "Celebrates deep work.", "quote": "Less tab switching, more deep focus."},  # not verbatim
        ]}]))
    insights = body["research"]["sources"][0]["insights"]
    assert insights[0]["basis"] == "evidence" and insights[0]["quote"] == "Meeting notes that write themselves."
    assert insights[1]["basis"] == "inferred" and insights[1]["quote"] is None
    assert insights[2]["basis"] == "inferred" and insights[2]["quote"] is None


def test_insights_unrelated_to_their_source_are_dropped(monkeypatch):
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS), analysis=analysis_payload(insights=[{
        "source_id": "S1", "insights": [{"text": "Offers cheap car insurance.", "quote": None}]}]))
    assert body["research"]["sources"][0]["insights"] == []


def test_inferred_insights_are_not_citable(monkeypatch):
    analysis = analysis_payload(insights=[{"source_id": "S1", "insights": [{"text": "Meeting notes theme.", "quote": None}]}])
    body, _, fake = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS), analysis=analysis,
                             generation=tweets_payload(insight_ids=["S1.1"], text="Meeting notes, handled."))
    assert all(t["source_ids"] == [] for t in body["tweets"]) and "<research_insights>" not in fake.prompt("generation")


# ---------- 2 + 6. claim validation and fabricated stories ----------

BRIEF = llm._grounding_text(TweetRequest(**NOTION_REQ), [])


@pytest.mark.parametrize("text, flag", [
    ("Priya closes the ward at 3 a.m. and her notes are already done.", "Priya"),  # sentence-initial name
    ("Priya's team finally stopped losing docs.", "Priya"),
    ("One of our customers told us she got her evenings back.", "One of our customers told us"),
    ("Now syncs with Slack and Google Calendar.", "syncs with Slack"),
    ("Integrates with all your favourite tools.", "Integrates with all"),
    ("Start your free trial today.", "free trial"),
    ("Now available on iOS and Android.", "available on iOS"),
    ("Notion AI now drafts your follow-up emails.", "Notion AI now drafts"),
    ("The #1 notes app for teams.", "#1"),
])
def test_fabricated_claims_and_stories_are_flagged(text, flag):
    assert any(flag in d for d in llm.unverified_details(text, BRIEF)), llm.unverified_details(text, BRIEF)


@pytest.mark.parametrize("text", [
    "Midnight deadlines don't care about your notes. We do.",  # sentence-initial noun, not a name
    "Everyone gets a fresh start on Monday.",
    "Notion keeps the whole team's docs in one place.",  # the brand doing something grounded
    "Sign up free and give your meeting notes a home.",  # signup + free are in the brief
    "Notion AI now writes your meeting notes.",  # capability grounded in the brief
    "Imagine a week where every doc is where you left it.",  # clearly hypothetical, no specifics
])
def test_legitimate_tweets_are_not_flagged(text):
    assert llm.unverified_details(text, BRIEF) == []


def test_integration_named_in_brief_is_not_flagged():
    grounding = llm._grounding_text(TweetRequest(**{**NOTION_REQ, "description": NOTION_REQ["description"] + " Works with Slack."}), [])
    assert llm.unverified_details("Works with Slack, so your team never misses a note.", grounding) == []


def test_declared_claims_are_validated_against_the_brief_only(monkeypatch):
    analysis = analysis_payload(insights=[{"source_id": "S1", "insights": [
        {"text": "Positions AI as writing meeting notes for you.", "quote": "Meeting notes that write themselves."}]}])
    claims = [
        [{"claim": "Free for small teams", "support": "brief", "quote": "Free for teams of up to 10"}],  # ok
        [{"claim": "Free for 50 users", "support": "brief", "quote": "Free for teams of up to 50"}],  # quote not in brief
        # A verified research finding is voice evidence, not proof of a product claim (behaviour changed on purpose).
        [{"claim": "Notes write themselves", "support": "S1.1", "quote": "Meeting notes that write themselves."}],
        [{"claim": "Exports to PDF", "support": "none", "quote": ""}],
        [{"claim": "SOC 2 certified", "support": "S9.1", "quote": "certified"}],  # unknown finding
        [{"claim": "Free trial available", "support": "brief", "quote": "for notes"}],  # real but unrelated + too short
        [{"claim": "Free trial available", "support": "brief", "quote": "Free for teams of up to 10"}],  # real, unrelated
        [{"claim": "Docs and wikis together", "support": "brief", "quote": ""}],  # empty quote
    ]
    gen = tweets_payload(text="Meeting notes, handled.")
    for tweet, c in zip(gen["tweets"], claims):
        tweet["claims"] = c
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS), analysis=analysis, generation=gen)
    flags = [t["unverified_details"] for t in body["tweets"][:len(claims)]]
    assert flags[0] == []
    c = "Claim not supported by your brief or product evidence: “{}”".format
    assert flags[1] == [c("Free for 50 users")] and flags[2] == [c("Notes write themselves")]
    assert flags[3] == [c("Exports to PDF")] and flags[4] == [c("SOC 2 certified")]
    assert flags[5] == [c("Free trial available")] and flags[6] == [c("Free trial available")]
    assert flags[7] == [c("Docs and wikis together")]


def test_creative_tweet_without_citation_or_claims_is_fine(monkeypatch):
    body, _, _ = generate(monkeypatch, x_routes(REAL_USER, REAL_POSTS),
                          generation=tweets_payload(text="Imagine a Monday where every doc is where you left it."))
    t = body["tweets"][0]
    assert t["basis"] == "inferred" and t["unverified_details"] == [] and t["source_ids"] == []
