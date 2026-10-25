"""Regression tests: research relevance, exact evidence, and tweet grounding (e.g. Notion vs. unrelated insurance posts)."""

import groq
import pytest

from app import llm, research
from app.config import Settings, get_settings
from app.main import app
from app.schemas import TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs
from test_research import Router, make_settings, post_generate, use_router

NOTION_REQ = {
    "brand_name": "Notion",
    "industry": "Productivity software",
    "campaign_objective": "Drive sign-ups for our new AI meeting notes feature",
    "description": "An all-in-one workspace for notes, docs, wikis and project management. Free for teams of up to 10.",
}
SB, FC = "https://scrapebadger.com/v1/twitter", "https://api.firecrawl.dev/v2"

# A lookalike: an unverified account literally named "Notion" with a big audience, posting about insurance.
# (Verified exact-name accounts are treated as official; a verified lookalike is caught by the analysis review.)
INSURANCE_USER = {"id": "9", "username": "notion", "name": "Notion", "followers_count": 25000, "verified": False,
                  "url": "https://t.co/insure", "description": "", "created_at": "x"}
INSURANCE_POSTS = {"data": [
    {"id": "1", "text": "Get covered in minutes with Notion Insurance. Auto, home and renters policies.", "created_at": "x"},
    {"id": "2", "text": "Filing a claim shouldn't feel like a second accident. Our adjusters respond within 24 hours.", "created_at": "x"},
    {"id": "3", "text": "Hurricane season is here: review your home policy coverage today.", "created_at": "x"},
]}
INSURANCE_SITE = {"url": "https://www.notioninsurance.com/", "title": "Notion Insurance",
                  "markdown": "Notion Insurance\nAffordable auto and home coverage. File a claim online in minutes.",
                  "metadata": {"url": "https://www.notioninsurance.com/"}}

REAL_USER = {**INSURANCE_USER, "username": "NotionHQ", "url": None, "verified": True,  # the official account
             "description": "The connected workspace for your docs, wikis and projects."}
REAL_POSTS = {"data": [
    {"id": "1", "text": "Meeting notes that write themselves. Try Notion AI in your workspace.", "created_at": "x"},
    {"id": "2", "text": "Docs, wikis and projects, all connected in one calm place.", "created_at": "x"},
    {"id": "3", "text": "Your team's knowledge, finally in one workspace.", "created_at": "x"},
    {"id": "4", "text": "Templates for every project, from roadmaps to retros.", "created_at": "x"},
    {"id": "5", "text": "Less tab-switching, more deep work. That's the whole idea.", "created_at": "x"},
]}
REAL_SITE = {"url": "https://www.notion.so/", "title": "Notion",
             "markdown": "Notion – The connected workspace\nWrite, plan and organise. Notes, docs and projects in one workspace with AI built in.",
             "metadata": {"url": "https://www.notion.so/"}}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def routes(user=INSURANCE_USER, posts=INSURANCE_POSTS, web=(INSURANCE_SITE,), handle="notion") -> dict:
    return {
        ("GET", f"/v1/twitter/users/{handle}/by_username"): (200, user),
        ("GET", f"/v1/twitter/users/{user['username']}/latest_tweets"): (200, posts),
        ("POST", "/v2/search"): (200, {"success": True, "data": {"web": list(web)}}),
        ("POST", "/v2/scrape"): (200, {"success": True, "data": REAL_SITE}),
    }


def gather(monkeypatch, r, req=NOTION_REQ):
    router = use_router(monkeypatch, r)
    return research.gather_research(TweetRequest(**req), make_settings()), router


# ---------- 1. research relevance ----------

def test_notion_research_never_trusts_an_unverified_insurance_lookalike(monkeypatch):
    result, router = gather(monkeypatch, routes())

    # Identity, not topic: the lookalike isn't rejected for its subject, but it isn't trusted either.
    assert result.x_tier == "uncertain"
    assert result.website_text is None and "/v2/scrape" not in router.paths()  # its profile link was not followed
    assert any("notioninsurance.com" in e and "different business" in e for e in result.excluded)


def test_lookalike_posts_are_fetched_but_need_the_identity_review(monkeypatch):
    # Trade-off of the fix: no vocabulary pre-check on X bios any more (it also rejected real brands), so a
    # lookalike's posts are fetched (within the 2-call limit) and then left to the identity review.
    user = {**INSURANCE_USER, "description": "Insurance made simple: auto, home and renters coverage with fast claims."}
    result, router = gather(monkeypatch, routes(user=user))
    assert any(p.endswith("/latest_tweets") for p in router.paths()) and result.x_tier == "uncertain"


def test_real_brand_content_is_accepted(monkeypatch):
    result, _ = gather(monkeypatch, routes(user=REAL_USER, posts=REAL_POSTS, web=(REAL_SITE,), handle="notion"))
    assert [s.kind for s in result.sources()] == ["x_posts", "website"]
    assert result.excluded == [] and result.report().status == "completed"


def test_website_search_prefers_the_official_domain_within_the_same_request(monkeypatch):
    # Search lists the lookalike first; the brand's own domain is still chosen, without a second request.
    r = routes(user={**INSURANCE_USER, "description": "Insurance made simple: auto, home and renters coverage."},
               web=(INSURANCE_SITE, REAL_SITE))
    result, router = gather(monkeypatch, r)
    assert result.website_url == "https://www.notion.so/" and result.website_tier == "strong"
    assert sum(p.startswith("/v2/") for p in router.paths()) == 1  # still a single Firecrawl request
    assert result.excluded == [] and result.x_tier == "uncertain"  # the X lookalike awaits the identity review


def test_lookalike_site_is_rejected_when_it_is_the_only_candidate(monkeypatch):
    result, _ = gather(monkeypatch, routes(user={**INSURANCE_USER, "description": "Insurance made simple: auto, home cover."},
                                           web=(INSURANCE_SITE,)))
    assert result.website_text is None
    assert any("notioninsurance.com" in e and "different business" in e for e in result.excluded)


def test_missing_brand_name_triggers_no_scraping(monkeypatch):
    req = {**NOTION_REQ, "brand_name": None, "description": NOTION_REQ["description"] + " See notion.so and x.com/NotionHQ"}
    result, router = gather(monkeypatch, routes(), req=req)
    assert router.requests == [] and result.report().status == "skipped"


def test_generate_never_presents_insurance_content_as_evidence(monkeypatch):
    use_router(monkeypatch, routes())
    fake = install_fake_groq(monkeypatch)
    # The identity review recognises the namesake; the model also tries to cite the insurance content anyway.
    fake.analysis = analysis_payload(
        relevance=[{"source_id": "S1", "relevant": False, "reason": "Notion Insurance, a different company"}],
        evidence=[{"source_id": "S1", "quote": "Get covered in minutes with Notion Insurance."}],
        insights=[{"source_id": "S1", "insights": ["Posts promise fast insurance claims."]}])
    fake.generation = tweets_payload(insight_ids=["S1.1"])
    r = post_generate(make_settings(), with_prefs(NOTION_REQ))
    body = r.json()

    assert r.status_code == 200
    assert body["research"]["status"] == "failed" and body["research"]["sources"] == []
    assert len(body["research"]["excluded"]) == 2
    assert all(t["basis"] == "inferred" and not t["evidence"] for t in body["voice_profile"]["traits"])
    assert all(t["basis"] == "inferred" and not t["source_ids"] for t in body["tweets"])
    assert any("Notion Insurance, a different company" in e for e in body["research"]["excluded"])
    assert "insurance" not in fake.prompt("generation").lower()  # rejected content never shapes the tweets


# ---------- 2. exact evidence ----------

def accepted(monkeypatch, evidence):
    use_router(monkeypatch, routes(user=REAL_USER, posts=REAL_POSTS, web=(REAL_SITE,)))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(evidence=evidence)
    body = post_generate(make_settings(), NOTION_REQ).json()
    return body["voice_profile"]["traits"][0], body["research"]


@pytest.mark.parametrize("quote", [
    "Try Notion AI in your workspace. Docs, wikis and projects",  # end of post 1 + start of post 2
    "Meeting notes that write themselves automatically.",  # a word added
    "meeting notes that write themselves",  # case changed
    "Notes that",  # too short to be evidence
    "Get covered in minutes with Notion Insurance.",  # from content that was excluded
])
def test_fabricated_or_mismatched_quotes_are_rejected(monkeypatch, quote):
    trait, _ = accepted(monkeypatch, [{"source_id": "S1", "quote": quote}])
    assert trait["basis"] == "inferred" and trait["evidence"] == []


def test_quote_from_wrong_source_is_rejected(monkeypatch):
    trait, _ = accepted(monkeypatch, [{"source_id": "S2", "quote": "Meeting notes that write themselves."}])
    assert trait["basis"] == "inferred"


def test_exact_quote_is_accepted_and_its_excerpt_is_shown(monkeypatch):
    quote = "Less tab-switching, more deep work."  # 5th post: beyond the default 3 excerpts
    trait, report = accepted(monkeypatch, [{"source_id": "S1", "quote": quote}])
    # One post is shown as an example but isn't enough to call the trait evidence-backed.
    assert trait["basis"] == "inferred" and trait["evidence"] == [{"source_id": "S1", "quote": quote}]
    second = "Docs, wikis and projects, all connected in one calm place."
    trait, _ = accepted(monkeypatch, [{"source_id": "S1", "quote": quote}, {"source_id": "S1", "quote": second}])
    assert trait["basis"] == "evidence" and len(trait["evidence"]) == 2
    s1 = report["sources"][0]
    assert any(quote in e for e in s1["excerpts"])  # the quote can be traced to a visible excerpt


# ---------- 3. tweet grounding ----------

@pytest.mark.parametrize("text, flagged", [
    ("Teams saved 40% of meeting time with Notion AI.", "40% of meeting"),
    ("Meet Priya, a PM who now ships twice as fast.", "Priya"),
    ("Built over three years of research with real teams.", "three years of"),
    ("Tested across 1,200 workspaces before launch.", "1,200 workspaces before"),
])
def test_unsupported_specifics_are_flagged(text, flagged):
    grounding = llm._grounding_text(TweetRequest(**NOTION_REQ), [])
    assert any(flagged in d for d in llm.unverified_details(text, grounding))


@pytest.mark.parametrize("text", [
    "Free for teams of up to 10. Your meeting notes, handled.",  # number from the brief
    "2 a.m. and the notes are already written. A thread (1/)",  # time of day + thread marker
    "Try Notion AI today and see your workspace come together.",  # brand and brief words
])
def test_grounded_specifics_are_not_flagged(text):
    grounding = llm._grounding_text(TweetRequest(**NOTION_REQ), [])
    assert llm.unverified_details(text, grounding) == []


def test_unsupported_claim_is_never_shown_as_research_backed(monkeypatch):
    use_router(monkeypatch, routes(user=REAL_USER, posts=REAL_POSTS, web=(REAL_SITE,)))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(insights=[{"source_id": "S1", "insights": [
        {"text": "Posts frame AI meeting notes as effortless.", "quote": "Meeting notes that write themselves."}]}])
    gen = tweets_payload(insight_ids=["S1.1"])
    gen["tweets"][0]["text"] = "Teams saved 40% of meeting time with Notion AI."  # cites S1.1, but 40% is invented
    gen["tweets"][1]["text"] = "Meeting notes that write themselves, now for every workspace."  # grounded
    gen["tweets"][2].update(text="Fresh roadmaps every Monday.", rationale="Planning cadence.")  # S1.1 unrelated to it
    fake.generation = gen
    tweets = post_generate(make_settings(), NOTION_REQ).json()["tweets"]

    assert tweets[0]["basis"] == "inferred" and any("40% of meeting" in d for d in tweets[0]["unverified_details"])
    assert tweets[0]["source_ids"] == ["S1"]  # the influence is still disclosed, just not presented as verified
    assert tweets[1]["basis"] == "evidence" and tweets[1]["unverified_details"] == []
    assert tweets[2]["source_ids"] == [] and tweets[2]["insight"] is None and tweets[2]["basis"] == "inferred"

def test_source_ids_only_point_at_sources_in_the_report(monkeypatch):
    use_router(monkeypatch, routes(user=REAL_USER, posts=REAL_POSTS, web=(REAL_SITE,)))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(insights=[{"source_id": "S1", "insights": [
        {"text": "Posts frame AI meeting notes as effortless.", "quote": "Meeting notes that write themselves."}]}])
    fake.generation = tweets_payload(insight_ids=["S1.1", "S3.1", "S7.2"], text="Meeting notes, handled.")
    body = post_generate(make_settings(), NOTION_REQ).json()
    reported = {s["id"] for s in body["research"]["sources"]}
    assert all(set(t["source_ids"]) <= reported for t in body["tweets"])


# ---------- 4. honest failures + key routing ----------

def test_research_failures_are_reported_honestly(monkeypatch):
    failing = {k: (500, {"error": "down"}) for k in routes()}
    use_router(monkeypatch, failing)
    install_fake_groq(monkeypatch)
    body = post_generate(make_settings(), with_prefs(NOTION_REQ)).json()
    report = body["research"]
    assert report["status"] == "failed" and report["sources"] == [] and report["excluded"] == []
    assert "couldn't be completed" in report["message"] and "comes from your answers" in report["message"]
    assert all(t["basis"] == "inferred" for t in body["tweets"])


def test_two_stage_key_routing_unchanged_with_research(monkeypatch):
    use_router(monkeypatch, routes(user=REAL_USER, posts=REAL_POSTS, web=(REAL_SITE,)))
    fake = install_fake_groq(monkeypatch)
    fake_client = llm.get_client("unused", 1.0)
    keys = []
    monkeypatch.setattr(llm, "get_client", lambda api_key, timeout: keys.append(api_key) or fake_client)
    settings = Settings(_env_file=None, groq_api_key="analysis-key", groq_api_key_brand_voice="generation-key",
                        scrapebadger_api_key="sb", firecrawl_api_key="fc")
    r = post_generate(settings, NOTION_REQ)
    assert r.status_code == 200
    assert list(zip(fake.stages, keys)) == [("analysis", "analysis-key"), ("generation", "generation-key")]
