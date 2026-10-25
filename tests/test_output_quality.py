"""Regression tests for issues found via Swagger: tweet count/duplicates, missed and noisy claim flags,
stale/third-party research, empty website pages, and campaign constraints. Offline: fake Groq + mocked HTTP."""

from datetime import datetime, timedelta, timezone

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload
from test_research import make_settings, post_generate, use_router

AR_REQ = {
    "brand_name": "String",
    "industry": "Augmented reality",
    "campaign_objective": "Get more retail brands to try our AR product try-on for their online stores",
    "description": "String is an augmented reality try-on tool that lets shoppers see products in their own space before buying.",
}
AR_BRIEF = llm._grounding_text(TweetRequest(**AR_REQ), [])


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def generate(monkeypatch, generation, req=AR_REQ):
    use_router(monkeypatch, {})
    fake = install_fake_groq(monkeypatch)
    fake.generation = generation
    return post_generate(make_settings(sb=None, fc=None), req), fake


def flags(text: str) -> list[str]:
    return llm.unverified_details(text, AR_BRIEF, brand="String")


# ---------- 1. exactly 10 valid, distinct tweets ----------

@pytest.mark.parametrize("n", [10, 12])
def test_returns_exactly_ten_distinct_tweets(monkeypatch, n):
    gen = tweets_payload(n=n)
    for t in gen["tweets"]:
        if t["style"] == "thread opener":
            t["text"] += " 1/"  # a clean thread opener (one without a cue is flagged and loses to a clean spare)
    r, fake = generate(monkeypatch, gen)
    tweets = r.json()["tweets"]
    assert r.status_code == 200 and len(tweets) == 10
    assert len({t["text"] for t in tweets}) == 10
    assert {t["style"] for t in tweets} == {s.lower() for s in llm.TWEET_STYLES}  # one per style when available
    assert fake.stages == ["analysis", "generation"]  # no extra LLM call


@pytest.mark.parametrize("n", [0, 1, 9])
def test_fewer_than_ten_is_an_explicit_error_never_a_short_list(monkeypatch, n):
    r, fake = generate(monkeypatch, tweets_payload(n=n))
    assert r.status_code == 502
    assert r.json()["detail"] == f"The model returned only {n} usable, distinct tweets (10 required). Please try again."
    assert fake.stages == ["analysis", "generation", "generation"]  # one top-up with the generation key, never more


def test_spare_candidates_replace_invalid_ones_without_another_call(monkeypatch):
    gen = tweets_payload(n=12)
    gen["tweets"][0]["text"] = "x" * 281  # too long
    gen["tweets"][1]["rationale"] = ""  # missing reasoning
    r, fake = generate(monkeypatch, gen)
    tweets = r.json()["tweets"]
    assert r.status_code == 200 and len(tweets) == 10
    assert all(len(t["text"]) <= 280 and t["rationale"] for t in tweets)
    assert fake.stages == ["analysis", "generation"]


def test_duplicates_and_near_duplicates_are_removed_and_replaced(monkeypatch):
    gen = tweets_payload(n=12)
    gen["tweets"][3]["text"] = gen["tweets"][2]["text"]  # exact duplicate
    gen["tweets"][4]["text"] = gen["tweets"][2]["text"].upper().replace(".", "!")  # same words, different case/punctuation
    r, _ = generate(monkeypatch, gen)
    texts = [t["text"] for t in r.json()["tweets"]]
    assert r.status_code == 200 and len(texts) == 10
    assert len({t.lower().strip(".!") for t in texts}) == 10


def test_duplicates_are_never_used_to_pad_to_ten(monkeypatch):
    r, _ = generate(monkeypatch, tweets_payload(n=12, text="Try products in your own space before buying."))
    assert r.status_code == 200  # the fixture's variants are genuinely different tweets
    same = {"tweets": [dict(t, text="Try products in your own space before buying.") for t in tweets_payload(n=12)["tweets"]]}
    r, _ = generate(monkeypatch, same)
    assert r.status_code == 502 and "only 1 usable, distinct" in r.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"tweets": "not a list"},
    {"tweets": [None, 3, "text"] * 4},  # non-object entries are skipped, leaving nothing usable
    {"no_tweets": []},
])
def test_malformed_generation_output_is_a_clear_error(monkeypatch, payload):
    r, _ = generate(monkeypatch, payload)
    assert r.status_code == 502 and "10 required" in r.json()["detail"]


def test_malformed_entries_mixed_with_valid_ones_are_skipped(monkeypatch):
    gen = tweets_payload(n=10)
    gen["tweets"] = [None, "junk"] + gen["tweets"]
    r, _ = generate(monkeypatch, gen)
    assert r.status_code == 200 and len(r.json()["tweets"]) == 10


# ---------- 2. claims in the actual tweet text ----------

@pytest.mark.parametrize("text, label", [
    ("Behind the scenes: We’re always tweaking to keep AR smooth no matter the lighting.", "Company or product claim"),
    ("Our engineers rebuilt the rendering engine from scratch.", "Company or product claim"),  # no trigger keywords
    ("String now renders in real time on every phone.", "Capability not in your brief"),
    ("We'll be at the RetailTech Expo workshop next week. Come say hi!", "Event or appearance"),
    ("Join us at our launch event in Austin.", "Event or appearance"),
    ("Priya closes her boutique at 6 and lets shoppers try on scarves at home.", "Possible invented person or story"),
])
def test_unsupported_claims_in_text_are_flagged_with_explanation(text, label):
    found = flags(text)
    assert any(item.startswith(label) for item in found), found
    assert all("“" in item and ": " in item for item in found)  # each item explains the risk and quotes the wording


def test_reported_tweet_is_flagged_through_the_full_pipeline(monkeypatch):
    gen = tweets_payload(n=10)
    gen["tweets"][7]["text"] = "Behind the scenes: We’re always tweaking to keep AR smooth no matter the lighting."
    r, _ = generate(monkeypatch, gen)
    t = r.json()["tweets"][7]
    assert t["unverified_details"] == [
        "Company or product claim not in your brief: “We're always tweaking to keep AR smooth no matter the lighting.”"]
    assert t["basis"] == "inferred"


@pytest.mark.parametrize("text", [
    "See how AR can shape what's next. #ARFuture",  # hashtags are style, not claims
    "Three lessons from the floor. #ARInsights",  # list-style count
    "3 tips for better product photos at home.",
    "Product Spotlight: try products in your own space.",  # a leaked style label isn't a claim
    "Good lighting makes any AR demo shine.",  # general advice, not a company claim
    "We built String so shoppers can see products in their own space.",  # grounded in the brief
    "We can't wait to see what your store does with try-on!",  # sentiment, not a claim
    "What if shoppers could see it in their own space first?",  # question
    "String lets shoppers try products in their own space before buying.",  # grounded capability
    "Picture this: a shopper sees the sofa in their own living room.",  # clearly hypothetical
])
def test_ordinary_language_is_not_flagged(text):
    assert flags(text) == []


def test_research_only_details_are_explained_and_ignore_hashtags():
    sources = research.ResearchResult(x_handle="String", posts=[
        "New: occlusion support now keeps virtual sofas behind real tables.", "Happy Friday from the String team!"]).sources()
    found = llm.research_only_details("Occlusion support keeps sofas behind tables. #ARFuture", AR_BRIEF, sources)
    assert found and found[0].startswith("Product detail from research, not your brief: “Occlusion support")
    assert llm.research_only_details("#OcclusionSupport #Tables", AR_BRIEF, sources) == []


# ---------- 4. research relevance and honesty ----------

def x_routes(posts: list[dict]) -> dict:
    user = {"id": "1", "username": "String", "name": "String", "followers_count": 8000, "url": None, "description": "",
            "created_at": "x"}
    return {("GET", "/v1/twitter/users/string/by_username"): (200, user),
            ("GET", "/v1/twitter/users/String/latest_tweets"): (200, {"data": posts})}


def x_date(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%a %b %d %H:%M:%S %z %Y")


FISHER_PRICE = ("Great to see Fisher Price Apptivity Create & Learn hitting the Top 5 at #ces2013 with "
                "#Augmented Reality goodness:")
RECENT = "Augmented reality try-on for retail shoppers: see products in your own space with String."


def test_dormant_account_with_only_old_posts_is_excluded_honestly(monkeypatch):
    use_router(monkeypatch, x_routes([{"id": "1", "text": FISHER_PRICE, "created_at": "Thu Jan 10 18:02:11 +0000 2013"},
                                      {"id": "2", "text": RECENT, "created_at": "Tue Feb 05 10:00:00 +0000 2013"}]))
    result = research.gather_research(TweetRequest(**AR_REQ), make_settings(fc=None))
    report = result.report()
    assert result.posts == [] and report.sources == [] and report.status == "failed"
    assert any("latest from 2013" in e and "@String" in e for e in report.excluded)


def test_old_posts_are_dropped_and_recent_ones_kept(monkeypatch):
    use_router(monkeypatch, x_routes([{"id": "1", "text": FISHER_PRICE, "created_at": "Thu Jan 10 18:02:11 +0000 2013"},
                                      {"id": "2", "text": RECENT, "created_at": x_date(30)},
                                      {"id": "3", "text": "Try-on that respects your shoppers' time and space.",
                                       "created_at_datetime": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()}]))
    result = research.gather_research(TweetRequest(**AR_REQ), make_settings(fc=None))
    assert FISHER_PRICE not in result.posts and RECENT in result.posts and len(result.posts) == 2
    assert all(FISHER_PRICE not in e for s in result.report().sources for e in s.excerpts)


def test_posts_without_parseable_dates_are_not_discarded(monkeypatch):
    use_router(monkeypatch, x_routes([{"id": "1", "text": RECENT, "created_at": "x"}]))
    result = research.gather_research(TweetRequest(**AR_REQ), make_settings(fc=None))
    assert result.posts == [RECENT]


@pytest.mark.parametrize("markdown", ["", "Loading...", "<!-- -->"])
def test_empty_website_is_reported_as_unreadable_not_unrelated(monkeypatch, markdown):
    use_router(monkeypatch, {("POST", "/v2/scrape"): (200, {"success": True, "data": {
        "markdown": markdown, "metadata": {"url": "https://string.co/"}}})})
    req = {**AR_REQ, "description": AR_REQ["description"] + " See string.co"}
    result = research.gather_research(TweetRequest(**req), make_settings(sb=None))
    report = result.report()
    assert result.website_text is None and report.status == "failed"
    assert report.excluded == ["Website string.co (from your brief): no readable content was retrieved, so it couldn't be checked."]


def test_blank_brand_name_still_skips_all_scraping(monkeypatch):
    router = use_router(monkeypatch, x_routes([{"id": "1", "text": RECENT, "created_at": "x"}]))
    result = research.gather_research(TweetRequest(**{**AR_REQ, "brand_name": "   "}), make_settings())
    assert router.requests == [] and result.report().status == "skipped"


# ---------- 5. campaign constraints in the generation prompt ----------

def test_campaign_inputs_and_style_rules_reach_generation(monkeypatch):
    r, fake = generate(monkeypatch, tweets_payload(n=12))
    prompt = fake.prompt("generation")
    assert AR_REQ["campaign_objective"] in prompt and AR_REQ["description"] in prompt
    rules = llm.GENERATION_PROMPT
    assert "Exactly 12 candidates" in rules
    assert "behind-the-scenes activity" in rules and "Never invent facts" in rules
    assert "Never mention events, conferences, workshops" in rules
    assert "may come only from <brand_input> or from" in rules and "<product_evidence> (the brand\'s official website)" in rules
