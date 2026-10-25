"""Chunk 2, clean the evidence: support notices aren't voice evidence, up to 15 posts are kept and reach generation,
and website research uses the homepage, never a sign-in page. Offline only (mocked HTTP)."""

import json
import logging

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import TweetRequest
from test_research import make_settings, use_router
from test_research_pipeline import DUO, MEMES, SCHOOLS_REQ, feed, recent

NOTICES = [
    "We’re aware of login issues and are working to restore access as quickly as possible. Your streak will be protected!",
    "and we’re back! thanks for your patience as we resolved an issue affecting logins. don't worry, your streaks have been protected!",
    "some users are experiencing issues with the app right now, we're on it",
    "scheduled maintenance tonight from 2-3am UTC, lessons will be back soon",
]
LOOKALIKES = [  # brand posts that share words with notices but are not notices
    "we’re aware you skipped your lesson. the owl is aware.",
    "thanks for your patience with me while i learn spanish lol",
    "this streak is having an outage of motivation",
    "sign in daily or the owl gets ideas",
]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def tweets(texts):
    return [{"id": str(i), "text": t, "created_at": recent()} for i, t in enumerate(texts)]


# ---------- 1. support notices ----------

def test_support_notices_are_filtered_and_counted():
    posts, stats, _ = research._clean_posts(tweets(MEMES + NOTICES), "duolingo")
    assert posts == MEMES
    assert stats["notice"] == len(NOTICES) and stats["kept"] == len(MEMES)
    assert "4 support notices" in research._format_stats(stats)


@pytest.mark.parametrize("text", LOOKALIKES + MEMES)
def test_ordinary_and_joking_posts_are_kept(text):
    posts, stats, _ = research._clean_posts(tweets([text]), "duolingo")
    assert posts == [text] and stats["notice"] == 0


def test_notice_log_line_reports_the_filter(monkeypatch, caplog):
    use_router(monkeypatch, {("GET", "/v1/twitter/users/duolingo/by_username"): (200, DUO),
                             ("GET", "/v1/twitter/users/duolingo/latest_tweets"): (200, feed(MEMES + NOTICES[:2]))})
    with caplog.at_level(logging.INFO, logger="app.research"):
        research.gather_research(TweetRequest(**SCHOOLS_REQ), make_settings(fc=None))
    assert "posts fetched=7 kept=5 rejected: 2 support notices" in caplog.text


# ---------- 2. up to 15 posts, end to end ----------

def test_up_to_fifteen_posts_are_kept_and_the_rest_counted():
    many = [f"post number {i} about the owl and your streak" for i in range(20)]
    posts, stats, _ = research._clean_posts(tweets(many), "duolingo")
    assert research.MAX_POSTS == 15 and posts == many[:15]
    assert stats["kept"] == 15 and stats["over_limit"] == 5


def test_all_fifteen_posts_reach_the_generation_prompt():
    assert research.MAX_POSTS == 15 and llm.MAX_VOICE_EXAMPLES == research.MAX_POSTS + llm.MAX_EXAMPLE_TWEETS
    posts = [f"post number {i} about the owl and your streak" for i in range(15)]
    req = TweetRequest(**SCHOOLS_REQ)
    profile = llm.VoiceProfile(summary=["a", "b", "c"], tone="t", audience="a", themes=["x"],
                               traits=[{"name": n, "description": "d"} for n in "abc"],
                               writing_patterns=[{"pattern": "p"}, {"pattern": "q"}])
    prompt = llm.build_generation_prompt(req, profile, {}, posts, research.observed_style(posts))
    block = prompt.split("<voice_examples>")[1].split("</voice_examples>")[0]
    assert all(f"- {p}" in block for p in posts)
    assert json.loads(prompt.split("<observed_style>\n")[1].split("\n")[0])["posts"] == 15


# ---------- 3. homepage, never a sign-in page ----------

def scrape_urls(router):
    return [json.loads(r.content)["url"] for r in router.requests if r.url.path == "/v2/scrape"]


SITE = {"success": True, "data": {"markdown": "Duolingo\nThe free, fun and effective way to learn a language.\nBite-sized lessons.",
                                  "metadata": {"url": "https://www.duolingo.com/", "title": "Duolingo"}}}


def test_profile_deep_link_is_scraped_as_the_homepage(monkeypatch):
    user = {**DUO, "url": "https://www.duolingo.com/learn"}
    router = use_router(monkeypatch, {("GET", "/v1/twitter/users/duolingo/by_username"): (200, user),
                                      ("GET", "/v1/twitter/users/duolingo/latest_tweets"): (200, feed(MEMES)),
                                      ("POST", "/v2/scrape"): (200, SITE)})
    result = research.gather_research(TweetRequest(**SCHOOLS_REQ), make_settings())
    assert scrape_urls(router) == ["https://www.duolingo.com/"]
    assert result.website_url == "https://www.duolingo.com/" and "Bite-sized lessons" in result.website_text
    sources = {s.kind: s for s in result.sources()}  # website stays product context; X posts stay the voice evidence
    assert sources["x_posts"].segments == MEMES and research.SOURCE_ROLES["website"] != research.SOURCE_ROLES["x_posts"]


def test_linked_sign_in_page_is_replaced_by_the_homepage(monkeypatch):
    req = {**SCHOOLS_REQ, "description": SCHOOLS_REQ["description"] + " Site: https://www.duolingo.com/log-in"}
    router = use_router(monkeypatch, {("POST", "/v2/scrape"): (200, SITE)})
    research.gather_research(TweetRequest(**req), make_settings(sb=None))
    assert scrape_urls(router) == ["https://www.duolingo.com/"]


def test_page_that_redirects_to_sign_in_is_not_used(monkeypatch):
    login = {"success": True, "data": {"markdown": "Log in to Duolingo\nEmail or username\nPassword\nForgot password",
                                       "metadata": {"url": "https://www.duolingo.com/log-in", "title": "Log in"}}}
    use_router(monkeypatch, {("GET", "/v1/twitter/users/duolingo/by_username"): (200, {**DUO, "url": "https://www.duolingo.com"}),
                             ("GET", "/v1/twitter/users/duolingo/latest_tweets"): (200, feed(MEMES)),
                             ("POST", "/v2/scrape"): (200, login)})
    result = research.gather_research(TweetRequest(**SCHOOLS_REQ), make_settings())
    assert result.website_text is None and any("sign-in page" in e for e in result.excluded)


def test_search_prefers_the_homepage_and_skips_sign_in_results(monkeypatch):
    hits = {"success": True, "data": {"web": [
        {"url": "https://www.duolingo.com/log-in", "markdown": "Log in\nEmail\nPassword\nForgot password\nSign up",
         "metadata": {"url": "https://www.duolingo.com/log-in"}},
        {"url": "https://www.duolingo.com/learn/spanish", "markdown": "Learn Spanish with Duolingo, bite-sized lessons for everyone",
         "metadata": {"url": "https://www.duolingo.com/learn/spanish"}},
        {"url": "https://www.duolingo.com/", "markdown": "Duolingo\nThe free, fun and effective way to learn a language.",
         "metadata": {"url": "https://www.duolingo.com/"}}]}}
    use_router(monkeypatch, {("POST", "/v2/search"): (200, hits)})
    result = research.gather_research(TweetRequest(**SCHOOLS_REQ), make_settings(sb=None))
    assert result.website_url == "https://www.duolingo.com/"
