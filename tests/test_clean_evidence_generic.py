"""Chunk 2 behaviour is brand-independent: the same notice filtering, 15-post limit and homepage selection for brands
that have nothing to do with Duolingo (a coffee brand and a SaaS brand). Offline only (mocked HTTP)."""

import json
import logging

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import TweetRequest
from test_research import make_settings, use_router
from test_research_pipeline import recent

BRANDS = {
    "moonbrew": {
        "req": {"brand_name": "Moonbrew", "industry": "Coffee", "campaign_objective": "Launch our new decaf cold brew",
                "description": "Small-batch cold brew coffee for night owls."},
        "posts": ["night shift? we brewed this for you.", "3 a.m. called. it wants something smoother.",
                  "some people count sheep. we count steeping hours.", "decaf that still tastes like a decision."],
        "notices": ["We're aware of an issue with checkout and are working to fix it. Orders are safe.",
                    "Our site is back up and running. Thanks for your patience while we sorted it out!"],
        "site": "https://moonbrew.com",
    },
    "notion": {
        "req": {"brand_name": "Notion", "industry": "Productivity software",
                "campaign_objective": "Get students to plan their semester in Notion",
                "description": "Notion is a connected workspace for docs, notes and projects."},
        "posts": ["your docs and notes, connected in one calm workspace.", "plan the semester before it plans you.",
                  "templates for every project, from roadmaps to retros.", "less tab-switching, more deep work."],
        "notices": ["Some users are experiencing issues with page sync. We're investigating the issue now.",
                    "Scheduled maintenance tonight 02:00-03:00 UTC: editing may be unavailable."],
        "site": "https://www.notion.so",
    },
}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def account(handle, url):
    return {"id": "1", "username": handle, "name": handle.capitalize(), "followers_count": 900000, "verified": True,
            "verified_type": "Business", "url": url, "description": "", "created_at": "x"}


def feed(texts):
    return {"data": [{"id": str(i), "text": t, "created_at": recent()} for i, t in enumerate(texts)]}


def site(url, text):
    return {"success": True, "data": {"markdown": text, "metadata": {"url": url, "title": "Home"}}}


@pytest.mark.parametrize("brand", BRANDS)
def test_notices_are_filtered_for_any_brand(monkeypatch, caplog, brand):
    b = BRANDS[brand]
    use_router(monkeypatch, {("GET", f"/v1/twitter/users/{brand}/by_username"): (200, account(brand, None)),
                             ("GET", f"/v1/twitter/users/{brand}/latest_tweets"): (200, feed(b["posts"] + b["notices"]))})
    with caplog.at_level(logging.INFO, logger="app.research"):
        result = research.gather_research(TweetRequest(**b["req"]), make_settings(fc=None))
    assert result.posts == b["posts"] and result.x_handle == brand
    assert f"posts fetched=6 kept=4 rejected: 2 support notices" in caplog.text


@pytest.mark.parametrize("brand", BRANDS)
def test_fifteen_post_limit_applies_to_any_brand(monkeypatch, brand):
    b = BRANDS[brand]
    many = [f"{brand} post number {i}, written in our usual voice" for i in range(20)]
    use_router(monkeypatch, {("GET", f"/v1/twitter/users/{brand}/by_username"): (200, account(brand, None)),
                             ("GET", f"/v1/twitter/users/{brand}/latest_tweets"): (200, feed(many))})
    result = research.gather_research(TweetRequest(**b["req"]), make_settings(fc=None))
    assert result.posts == many[:15] and result.post_stats["over_limit"] == 5
    examples = llm.build_generation_prompt(TweetRequest(**b["req"]), llm.VoiceProfile(
        summary=["a", "b", "c"], tone="t", audience="a", themes=["x"], traits=[{"name": n, "description": "d"} for n in "abc"],
        writing_patterns=[{"pattern": "p"}, {"pattern": "q"}]), {}, result.posts)
    assert all(p in examples.split("<voice_examples>")[1] for p in many[:15]) and many[15] not in examples


@pytest.mark.parametrize("brand", BRANDS)
def test_homepage_is_derived_from_each_brands_own_profile_link(monkeypatch, brand):
    b = BRANDS[brand]
    deep_link = f"{b['site']}/app/dashboard?ref=x"
    router = use_router(monkeypatch, {
        ("GET", f"/v1/twitter/users/{brand}/by_username"): (200, account(brand, deep_link)),
        ("GET", f"/v1/twitter/users/{brand}/latest_tweets"): (200, feed(b["posts"])),
        ("POST", "/v2/scrape"): (200, site(f"{b['site']}/", f"{b['req']['brand_name']}\n{b['req']['description']}\nMore about us."))})
    result = research.gather_research(TweetRequest(**b["req"]), make_settings())
    scraped = [json.loads(r.content)["url"] for r in router.requests if r.url.path == "/v2/scrape"]
    assert scraped == [f"{b['site']}/"] and result.website_text


@pytest.mark.parametrize("brand", BRANDS)
def test_sign_in_links_and_results_are_never_evidence_for_any_brand(monkeypatch, brand):
    b = BRANDS[brand]
    # A sign-in link in the brief is swapped for that brand's homepage.
    req = {**b["req"], "description": b["req"]["description"] + f" Site: {b['site']}/login"}
    router = use_router(monkeypatch, {("POST", "/v2/scrape"): (200, site(f"{b['site']}/", b["req"]["description"] + "\nHome."))})
    research.gather_research(TweetRequest(**req), make_settings(sb=None))
    assert [json.loads(r.content)["url"] for r in router.requests] == [f"{b['site']}/"]
    # Search: sign-in results are skipped, the homepage is preferred over deeper pages.
    research._cache.clear()
    hits = {"success": True, "data": {"web": [
        {"url": f"{b['site']}/signin", "markdown": "Sign in\nEmail\nPassword\nForgot password\nCreate account",
         "metadata": {"url": f"{b['site']}/signin"}},
        {"url": f"{b['site']}/pricing/teams", "markdown": f"{b['req']['brand_name']} pricing for teams and companies",
         "metadata": {"url": f"{b['site']}/pricing/teams"}},
        {"url": f"{b['site']}/", "markdown": f"{b['req']['brand_name']}\n{b['req']['description']}",
         "metadata": {"url": f"{b['site']}/"}}]}}
    use_router(monkeypatch, {("POST", "/v2/search"): (200, hits)})
    result = research.gather_research(TweetRequest(**b["req"]), make_settings(sb=None))
    assert result.website_url == f"{b['site']}/"


def test_notice_wording_is_brand_neutral():
    # The production pattern names no brand, handle or product.
    pattern = research.NOTICE_RE.pattern.lower()
    for word in ("duolingo", "duo", "owl", "streak", "moonbrew", "notion"):
        assert word not in pattern
