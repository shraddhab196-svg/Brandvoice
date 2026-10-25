"""Regression tests for the research-pipeline fix (Duolingo case): identity over campaign topic, per-post filtering,
source-role separation (X = voice, website = product context), cache keys, and diagnostics. Offline only."""

from datetime import datetime, timedelta, timezone

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs
from test_research import FC_KEY, SB_KEY, make_settings, post_generate, use_router

# A narrow B2B brief whose vocabulary shares nothing with Duolingo's meme-style feed: the case that failed.
SCHOOLS_REQ = {
    "brand_name": "Duolingo",
    "industry": "Education",
    "campaign_objective": "Get more teachers to adopt Duolingo for Schools in their classrooms",
    "description": "Duolingo for Schools gives teachers a free dashboard to assign coursework and track student progress.",
}
DUO = {"id": "1", "username": "duolingo", "name": "Duolingo", "followers_count": 7000000, "verified": True,
       "verified_type": "Business", "url": None, "description": "Learn a language for free. Forever.", "created_at": "x"}
MEMES = [
    "me when the owl sees i skipped my lesson again",
    "duo has entered the chat. you have entered your streak era",
    "not me crying over a 400 day streak at 11:59pm",
    "POV: you said you would practice tomorrow for the sixth time",
    "the owl does not forget. the owl does not forgive.",
]
SITE_ROOT = {"url": "https://www.duolingo.com/", "title": "Duolingo",
             "markdown": "Duolingo\nThe free, fun and effective way to learn a language.\nLessons for schools, teachers and learners.",
             "metadata": {"url": "https://www.duolingo.com/"}}
SITE_SCHOOLS = {"url": "https://schools.duolingo.com/", "title": "Duolingo for Schools",
                "markdown": ("Duolingo for Schools\nFree tools for teachers to assign coursework and track student progress.\n"
                             "“My students actually ask to practice.” — a 7th-grade Spanish teacher\n"
                             "\"Best classroom tool I've used.\" - Ms. Rivera"),
                "metadata": {"url": "https://schools.duolingo.com/"}}


def recent(days=3) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%a %b %d %H:%M:%S %z %Y")


def feed(texts, **extra) -> dict:
    return {"data": [{"id": str(i), "text": t, "created_at": recent(), **extra} for i, t in enumerate(texts)]}


def routes(user=DUO, posts=None, web=(SITE_ROOT,)) -> dict:
    return {("GET", "/v1/twitter/users/duolingo/by_username"): (200, user),
            ("GET", f"/v1/twitter/users/{user['username']}/latest_tweets"): (200, posts or feed(MEMES)),
            ("POST", "/v2/search"): (200, {"success": True, "data": {"web": list(web)}})}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def gather(monkeypatch, r, req=SCHOOLS_REQ, **settings_kw):
    router = use_router(monkeypatch, r)
    return research.gather_research(TweetRequest(**req), make_settings(**settings_kw)), router


# ---------- 1. identity, not campaign topic ----------

def test_official_account_with_off_campaign_posts_is_used(monkeypatch):
    result, router = gather(monkeypatch, routes())
    assert any(p.endswith("/latest_tweets") for p in router.paths())  # bio didn't block the posts call
    assert result.x_handle == "duolingo" and result.posts == MEMES and result.x_tier == "strong"
    assert not any("@duolingo" in e for e in result.excluded)


def test_official_account_is_used_without_a_model_verdict(monkeypatch):
    use_router(monkeypatch, routes())
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(relevance=[], evidence=[{"source_id": "S1", "quote": "the owl does not forget."},
                                                               {"source_id": "S1", "quote": "duo has entered the chat."}])
    body = post_generate(make_settings(), SCHOOLS_REQ).json()
    kinds = [s["kind"] for s in body["research"]["sources"]]
    assert "x_posts" in kinds
    assert body["voice_profile"]["traits"][0]["basis"] == "evidence"  # voice evidence comes from the real posts


def test_profile_link_to_own_domain_identifies_an_unverified_account(monkeypatch):
    user = {**DUO, "verified": False, "verified_type": None, "url": "https://www.duolingo.com"}
    result, _ = gather(monkeypatch, routes(user=user))
    assert result.x_tier == "strong" and result.posts == MEMES


@pytest.mark.parametrize("verified_type", ["Blue", None])
def test_paid_or_no_verification_alone_is_not_strong_identity(monkeypatch, verified_type):
    user = {**DUO, "verified": False, "verified_type": verified_type, "is_blue_verified": True, "description": ""}
    result, _ = gather(monkeypatch, routes(user=user))
    assert result.x_tier == "uncertain"  # used only if the analysis step confirms it's official


# ---------- lookalike protection is preserved ----------

INSURANCE_BIO = "Insurance made simple: auto, home and renters coverage with fast online claims."
INSURANCE_FEED = feed(["Get covered in minutes with auto, home and renters policies today.",
                       "Filing a claim shouldn't feel like a second accident. Our adjusters respond fast.",
                       "Hurricane season is here: review your home policy coverage."])


LOOKALIKE = {**DUO, "verified": False, "verified_type": None, "description": INSURANCE_BIO,
             "url": "https://duolingo-insure.example"}


def test_unverified_lookalike_is_never_trusted_deterministically(monkeypatch):
    result, router = gather(monkeypatch, routes(user=LOOKALIKE, posts=INSURANCE_FEED), fc=None)
    assert result.x_tier == "uncertain"  # retrieved, but only usable if the identity review confirms it
    assert result.website_url is None  # a possible lookalike's profile link is not followed


@pytest.mark.parametrize("relevance, reason", [
    ([{"source_id": "S1", "relevant": False, "reason": "An insurance company with a similar name."}], "insurance company"),
    ([], "couldn't be confirmed"),  # no verdict
])
def test_unverified_lookalike_is_not_used_without_positive_confirmation(monkeypatch, relevance, reason):
    use_router(monkeypatch, routes(user=LOOKALIKE, posts=INSURANCE_FEED))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(relevance=relevance, evidence=[{"source_id": "S1", "quote": "Get covered in minutes"}])
    body = post_generate(make_settings(fc=None), with_prefs(SCHOOLS_REQ)).json()
    assert body["research"]["sources"] == [] and any(reason in e for e in body["research"]["excluded"])
    assert all(t["basis"] == "inferred" for t in body["voice_profile"]["traits"])


def test_verified_lookalike_is_rejected_by_the_identity_review(monkeypatch):
    # Deterministic checks can't tell a verified namesake apart; the analysis model's identity verdict does.
    use_router(monkeypatch, routes(posts=INSURANCE_FEED))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(relevance=[{"source_id": "S1", "relevant": False, "reason": "An insurance company."}])
    body = post_generate(make_settings(), with_prefs(SCHOOLS_REQ)).json()
    assert all(s["kind"] != "x_posts" for s in body["research"]["sources"])
    assert any("insurance company" in e for e in body["research"]["excluded"])


def test_user_linked_handle_is_a_lead_that_needs_confirmation(monkeypatch):
    fan = {**DUO, "username": "duolingofans", "name": "Duolingo Fans", "verified": False, "verified_type": None,
           "followers_count": 90000, "description": "Fan account for Duolingo learners."}
    req = {**SCHOOLS_REQ, "description": SCHOOLS_REQ["description"] + " Our X: x.com/duolingofans"}
    use_router(monkeypatch, {("GET", "/v1/twitter/users/duolingofans/by_username"): (200, fan),
                             ("GET", "/v1/twitter/users/duolingofans/latest_tweets"): (200, feed(MEMES))})
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(relevance=[])  # no verdict
    body = post_generate(make_settings(fc=None), with_prefs(req)).json()
    assert body["research"]["sources"] == []
    assert any("@duolingofans" in e and "couldn't be confirmed" in e for e in body["research"]["excluded"])


# ---------- 2. per-post filtering, with counts ----------

def test_individual_posts_are_filtered_and_counted(monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(days=4 * 365)).strftime("%a %b %d %H:%M:%S %z %Y")
    data = {"data": [
        {"id": "1", "text": MEMES[0], "created_at": recent()},
        {"id": "2", "text": "RT a partner's launch", "is_retweet": True, "created_at": recent()},
        {"id": "3", "text": "A post written by someone else entirely.", "username": "someoneelse", "created_at": recent()},
        {"id": "4", "text": "An old post from years ago about owls.", "created_at": old},
        {"id": "5", "text": "🦉🦉🦉 #duolingo https://t.co/x", "created_at": recent()},
        {"id": "6", "text": MEMES[0], "created_at": recent()},
        "not even a dict",
        {"id": "8", "text": MEMES[1], "created_at": recent()},
        {"id": "9", "text": MEMES[2], "created_at": recent()},
    ]}
    result, _ = gather(monkeypatch, routes(posts=data))
    assert result.posts == MEMES[:3]
    s = result.post_stats
    assert (s["fetched"], s["kept"], s["retweet"], s["other_author"], s["stale"], s["too_short"], s["duplicate"],
            s["malformed"]) == (9, 3, 1, 1, 1, 1, 1, 1)


def test_replies_fill_in_only_when_originals_are_scarce(monkeypatch):
    reply = {"in_reply_to_status_id": "77"}
    two_originals = {"data": feed(MEMES[:2])["data"] + feed(["@learner yes the owl saw that, do your lesson"], **reply)["data"]}
    result, _ = gather(monkeypatch, routes(posts=two_originals))
    assert result.posts == MEMES[:2] + ["yes the owl saw that, do your lesson"] and result.post_stats["replies_used"] == 1
    research._cache.clear()
    many = {"data": feed(MEMES)["data"] + feed(["@learner yes the owl saw that, do your lesson"], **reply)["data"]}
    result, _ = gather(monkeypatch, routes(posts=many))
    assert result.posts == MEMES and result.post_stats["replies_used"] == 0


def test_account_with_no_usable_posts_reports_counts_not_topic(monkeypatch):
    data = {"data": [{"id": "1", "text": "RT x", "is_retweet": True, "created_at": recent()},
                     {"id": "2", "text": "🦉 #owl", "created_at": recent()}]}
    result, _ = gather(monkeypatch, routes(posts=data))
    assert result.excluded == ["X account @duolingo: none of its 2 fetched posts were usable "
                               "(1 retweets, 1 too short), so it wasn't used."]


# ---------- 3. source roles ----------

def test_website_is_product_context_not_voice_evidence(monkeypatch):
    use_router(monkeypatch, routes(web=(SITE_SCHOOLS,)))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(
        evidence=[{"source_id": "S2", "quote": "Free tools for teachers to assign coursework and track student progress."}],
        insights=[{"source_id": "S1", "insights": [{"text": "Unhinged owl humour about streaks.", "quote": "the owl does not forget."}]},
                  {"source_id": "S2", "insights": [{"text": "Free for teachers.", "quote": "Free tools for teachers to assign coursework"}]}])
    fake.generation = tweets_payload(insight_ids=["S1.1", "S2.1"], text="the owl knows your streak. teachers, try Duolingo for Schools")
    body = post_generate(make_settings(), SCHOOLS_REQ).json()

    assert body["voice_profile"]["traits"][0]["basis"] == "inferred"  # website quote can't evidence voice
    analysis_prompt, gen_prompt = fake.prompt("analysis"), fake.prompt("generation")
    assert "(VOICE EVIDENCE: the brand's own social posts)" in analysis_prompt
    assert "(PRODUCT AND COMPANY CONTEXT ONLY: not evidence of how the brand writes on social)" in analysis_prompt
    assert "[S1.1]" in gen_prompt and "[S2.1]" not in gen_prompt  # only X findings are voice findings
    assert all(t["source_ids"] in ([], ["S1"]) for t in body["tweets"])


def test_website_only_research_is_reported_as_not_social_voice(monkeypatch):
    r = routes(web=(SITE_SCHOOLS,))
    r[("GET", "/v1/twitter/users/duolingo/latest_tweets")] = (500, {"error": "down"})
    use_router(monkeypatch, r)
    install_fake_groq(monkeypatch)
    report = post_generate(make_settings(), with_prefs(SCHOOLS_REQ)).json()["research"]
    assert [s["kind"] for s in report["sources"]] == ["website"]
    assert report["status"] == "partial"
    assert report["message"].startswith("No social posts could be used, so the voice comes from your answers, not research")
    assert "website was used only as product and company context" in report["message"]


def test_testimonials_are_removed_from_website_context(monkeypatch):
    result, _ = gather(monkeypatch, routes(web=(SITE_SCHOOLS,)), sb=None)
    assert result.testimonials_removed == 2
    assert "My students actually ask" not in result.website_text and "Ms. Rivera" not in result.website_text
    assert "Free tools for teachers" in result.website_text


def test_search_prefers_the_main_site_over_subdomains(monkeypatch):
    result, _ = gather(monkeypatch, routes(web=(SITE_SCHOOLS, SITE_ROOT)), sb=None)
    assert result.website_url == "https://www.duolingo.com/"


def test_official_subdomain_is_still_accepted_when_it_is_all_there_is(monkeypatch):
    result, _ = gather(monkeypatch, routes(web=(SITE_SCHOOLS,)), sb=None)
    assert result.website_url == "https://schools.duolingo.com/" and result.website_tier == "strong"


# ---------- 4. cache key covers the whole brief ----------

@pytest.mark.parametrize("change", [{"campaign_objective": "Promote the new chess course to adults"},
                                    {"description": "Duolingo is a free app for languages, math, music and chess."},
                                    {"industry": "Consumer apps"}])
def test_changing_any_brief_input_re_runs_research(monkeypatch, change):
    router = use_router(monkeypatch, routes())
    settings = make_settings()
    research.gather_research(TweetRequest(**SCHOOLS_REQ), settings)
    first = len(router.requests)
    research.gather_research(TweetRequest(**SCHOOLS_REQ), settings)
    assert len(router.requests) == first  # identical brief: served from cache
    research.gather_research(TweetRequest(**{**SCHOOLS_REQ, **change}), settings)
    assert len(router.requests) > first  # different brief: never reuses the other brief's decisions


# ---------- 5. diagnostics ----------

def test_post_counts_and_reasons_are_logged_without_secrets(monkeypatch, caplog):
    with caplog.at_level("INFO", logger="app.research"):
        gather(monkeypatch, routes())
    assert "Research X @duolingo: identity=strong posts fetched=5 kept=5 rejected: none" in caplog.text
    assert SB_KEY not in caplog.text and FC_KEY not in caplog.text
