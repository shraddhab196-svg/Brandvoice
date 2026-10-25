"""Regression tests for the voice-transfer fix (Duolingo live test): the brand's real posts and measured mechanics reach
generation, style quality is reviewed separately from factual grounding, official-website product evidence can
support claims, and weak evidence doesn't become confident traits or themes. Offline only; no brand is hardcoded."""

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import Tweet
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs
from test_frontend import NODE, run_review_js
from test_research import make_settings, post_generate, use_router
from test_research_pipeline import MEMES, SCHOOLS_REQ, routes

# Mechanics of a very different (capitalised, hashtag-heavy) brand: the rules must follow the posts, not a template.
FORMAL_POSTS = ["Introducing our Spring Collection. #Style #New", "Shop the Summer Edit today. #Style",
                "Our Autumn Lookbook is here. #Fashion", "Discover Winter Essentials now. #Style #Gifts"]
OUTAGE = "We’re aware of login issues and are working to restore access as quickly as possible."
CTX = llm.QualityContext(brand="Duolingo", style=research.observed_style(MEMES),
                         campaign_terms=llm._terms(SCHOOLS_REQ["campaign_objective"] + " " + SCHOOLS_REQ["description"]))


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def run(monkeypatch, generation=None, web=None, analysis=None, replies=None, req=SCHOOLS_REQ):
    use_router(monkeypatch, routes() if web is None else routes(web=web))
    fake = install_fake_groq(monkeypatch)
    if analysis is not None:
        fake.analysis = analysis
    if generation is not None:
        fake.generation = generation
    fake.generation_replies = list(replies or [])
    return post_generate(make_settings(), req), fake


def tweet(text, style="hook", **kw):
    return Tweet(style=style, text=text, rationale="Fits.", **{"content_style": "engaging", **kw})


# ---------- 1. voice transfer: real posts and measured mechanics reach generation ----------

def test_brand_posts_reach_the_generation_prompt_as_voice_examples(monkeypatch):
    r, fake = run(monkeypatch, tweets_payload(n=12))
    assert r.status_code == 200
    prompt = fake.prompt("generation")
    examples = prompt.split("<voice_examples>", 1)[1].split("</voice_examples>", 1)[0]
    assert all(post in examples for post in MEMES)
    assert "don't copy them" in examples and "don't take facts or topics from them" in examples
    assert "Imitate the observed writing mechanics of <voice_examples>" in llm.GENERATION_PROMPT


def test_observed_mechanics_become_generation_rules(monkeypatch):
    _, fake = run(monkeypatch, tweets_payload(n=12))
    style = fake.prompt("generation").split("<observed_style>", 1)[1].split("</observed_style>", 1)[0]
    assert "Start tweets in lowercase, like 4 of 5" in style  # 4/5 meme posts start lowercase ("POV:" doesn't)
    assert "Don't use hashtags: the brand used them in 0 of 5 posts." in style
    assert "median of" in style


def test_style_rules_follow_each_brands_own_posts():
    formal = llm.style_rules(research.observed_style(FORMAL_POSTS))
    assert "Use normal sentence-case capitalisation, like the brand's posts." in formal
    assert any(r.startswith("Hashtags are part of the style (4 of 4") for r in formal)
    assert not any("lowercase" in r for r in formal)
    memes = llm.style_rules(research.observed_style(MEMES))
    assert any("lowercase" in r for r in memes) and not any(r.startswith("Hashtags are part") for r in memes)


def test_too_few_posts_give_no_measured_style():
    # Three posts are not enough to claim casing/hashtag habits: no <observed_style>, no style checks.
    assert research.observed_style(MEMES[:3]) is None and llm.style_rules(None) == []
    prompt = llm.build_generation_prompt(llm.TweetRequest(**SCHOOLS_REQ), llm.VoiceProfile(**{
        k: v for k, v in analysis_payload().items() if k in llm.VoiceProfile.model_fields}), {}, MEMES[:3], None)
    assert "<voice_examples>" in prompt and "<observed_style>" not in prompt


# ---------- 2. quality and variety (separate from factual grounding) ----------

@pytest.mark.parametrize("text, issue", [
    ("Tip: practice daily to keep your streak", "Generic template opener"),
    ("teachers, let's dive into the free dashboard", "Generic marketing cliche"),
    ("picture this: you, back in the classroom with your students", "Generic template opener"),
    ("Ever wondered why teachers love dashboards?", "Generic template opener"),
    ("hot take: teachers deserve free dashboards", "Generic template opener"),  # the approach label leaked
    ("lol same", "Too vague"),
    ("Duolingo for Schools is here. Duolingo makes teachers happy", "Brand name repeated"),
    ("assign coursework, track progress, free dashboard, classrooms, teachers", "Reads like a feature list"),
    ("teachers, your students are about to love coursework (1/", "Incomplete thread marker"),
    ("teachers: the free dashboard nobody told you about", "Thread opener without a thread cue"),
    ("Teachers can assign coursework from one free dashboard", "Starts with a capital"),
    ("teachers can assign coursework from one free dashboard #edtech", "Uses hashtags"),
    ("teachers can assign coursework!! and track student progress!!", "Several exclamation marks"),
])
def test_quality_issues_are_flagged(text, issue):
    style = "thread opener" if "Thread opener" in issue else "hook"
    assert any(i.startswith(issue) for i in llm.tweet_quality(text, style, CTX)), llm.tweet_quality(text, style, CTX)


@pytest.mark.parametrize("text, style", [
    ("teachers, your coursework era starts now. the owl is watching your classroom", "hook"),
    ("one free dashboard, thirty students, zero excuses. the owl assigns homework now 🧵", "thread opener"),
    ("POV: you assigned coursework and every student actually did it", "relatable moment"),
    ("I made the free dashboard my whole personality and my students are thriving", "hot take"),
])
def test_on_voice_tweets_have_no_quality_issues(text, style):
    assert llm.tweet_quality(text, style, CTX) == []


def test_batch_flags_repeated_openings_ideas_and_brand_overuse():
    tweets = [tweet("duolingo knows teachers need a free dashboard to track progress"),
              tweet("duolingo knows students skip coursework when nobody checks"),
              tweet("track progress with a free dashboard teachers need"),
              tweet("duolingo for schools: assign coursework in one click"),
              tweet("ask any teacher about Duolingo and watch them assign coursework"),
              tweet("classrooms run on Duolingo and quiet student panic")]
    out = llm._batch_quality(tweets, CTX)
    assert out[0].quality_issues == []
    assert "Same opening as another tweet: “tweet 1”" in out[1].quality_issues
    assert "Repeats the idea of another tweet: “tweet 1”" in out[2].quality_issues
    assert not any(i.startswith("Brand name used in too many") for t in out[:4] for i in t.quality_issues)
    assert any(i.startswith("Brand name used in too many tweets") for i in out[5].quality_issues)  # 5th of 10 > a third


def test_selection_prefers_clean_candidates_over_templates():
    styles = llm.TWEET_STYLES
    good = [tweet(f"clean idea {s} {i} for teachers", style=s) for i, s in enumerate(styles)]
    template = tweet("Tip: templates for teachers", style="hook", quality_issues=["Generic template opener: “Tip:”"])
    chosen = llm._select_tweets([template] + good + [tweet("spare teacher idea one"), tweet("spare teacher idea two")])
    assert template not in chosen and len(chosen) == 10


def test_clean_alternates_beat_a_flagged_tweet_kept_only_to_cover_an_approach():
    # Seen live: the only "question" candidate opened with "ever wonder…" while clean alternates were unused.
    good = [tweet(f"clean idea {s} {i} for teachers", style=s) for i, s in enumerate(llm.TWEET_STYLES) if s != "question"]
    flagged = tweet("ever wonder why teachers skip dashboards?", style="question", quality_issues=["Generic template opener"])
    spares = [tweet("spare teacher idea one", style="alternate"), tweet("spare teacher idea two", style="alternate")]
    chosen = llm._select_tweets([flagged] + good + spares)
    assert flagged not in chosen and spares[0] in chosen and len(chosen) == 10


def test_vague_or_templated_tweets_need_review_end_to_end(monkeypatch):
    gen = tweets_payload(n=10)  # no spares, so the weak tweets must be returned (and marked) rather than replaced
    gen["tweets"][0]["text"] = "lol same"
    gen["tweets"][1]["text"] = "Tip: Teachers love dashboards! #EdTech #Teachers"
    r, _ = run(monkeypatch, gen)
    tweets = {t["text"]: t for t in r.json()["tweets"]}
    assert r.status_code == 200 and len(r.json()["tweets"]) == 10
    assert tweets["lol same"]["unverified_details"] == []  # no claims at all...
    assert tweets["lol same"]["quality_issues"]  # ...but still not ready to post
    issues = tweets["Tip: Teachers love dashboards! #EdTech #Teachers"]["quality_issues"]
    assert any(i.startswith("Generic template opener") for i in issues)
    assert any(i.startswith("Uses hashtags") for i in issues)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_frontend_treats_quality_issues_as_needing_review():
    vague = {"style": "hook", "text": "lol same", "source_ids": [], "basis": "inferred", "unverified_details": [],
             "quality_issues": ["Too vague: says almost nothing specific: “lol same”"]}
    clean = dict(vague, text="fine", quality_issues=[])
    assert run_review_js(f"R.needsReview({vague})".replace("False", "false")) is True
    assert run_review_js(f"R.summary([{vague}, {clean}])") == {"ready": 1, "review": 1}
    assert "Too vague" in run_review_js(f"R.copyOne({vague})")
    assert run_review_js("R.needsReview({unverified_details: []})") is False  # older responses still work


# ---------- 3. weak evidence stays tentative ----------

def test_single_post_doesnt_make_an_evidence_backed_trait(monkeypatch):
    posts = MEMES + [OUTAGE]
    analysis = analysis_payload(relevance=[])
    analysis["traits"][1] = {"name": "Supportive", "description": "Reassures users during problems.",
                             "evidence": [{"source_id": "S1", "quote": OUTAGE}]}
    analysis["traits"][0]["evidence"] = [{"source_id": "S1", "quote": MEMES[0]}, {"source_id": "S1", "quote": MEMES[4]}]
    use_router(monkeypatch, routes(posts={"data": [{"id": str(i), "text": t, "created_at": "x"}
                                                   for i, t in enumerate(posts)]}))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis
    traits = post_generate(make_settings(), SCHOOLS_REQ).json()["voice_profile"]["traits"]
    # An outage notice is support boilerplate: filtered out of the evidence entirely, so it can't back a trait at all.
    assert traits[1]["basis"] == "inferred" and traits[1]["evidence"] == []
    assert traits[0]["basis"] == "evidence"  # two different posts


def test_pattern_quotes_must_show_what_they_are_cited_for():
    segments = {"S1": ["love this #Duo", "no tags here at all", "another one without tags"]}
    item = {"evidence": [{"source_id": "S1", "quote": "love this #Duo"}, {"source_id": "S1", "quote": "no tags here"}]}
    out = llm._with_basis(item, segments, "Rarely uses hashtags")
    assert out["evidence"] == [{"source_id": "S1", "quote": "no tags here"}] and out["basis"] == "inferred"


def test_one_off_themes_are_dropped_recurring_and_briefed_ones_kept():
    posts = MEMES + [OUTAGE]
    themes = llm._supported_themes(["Technical support", "the owl", "classroom adoption"], posts,
                                   SCHOOLS_REQ["campaign_objective"] + " " + SCHOOLS_REQ["description"])
    assert themes == ["the owl", "classroom adoption"]  # 2+ posts mention the owl; classrooms come from the brief
    assert llm._supported_themes(["Technical support"], [], "brief") == ["Technical support"]  # nothing to check


# ---------- 4. claim checking: brief and official product evidence ----------

BRIEF = "Duolingo\nEducation\nGet lapsed learners back\nDuolingo is a free app for learning languages with daily streaks."


@pytest.mark.parametrize("claim, quote", [
    ("Duolingo is free", "a free app for learning languages"),
    ("You can learn a language in the app", "free app for learning languages"),  # paraphrase, stem-matched
    ("Duolingo offers daily streaks", "daily streaks"),  # short quote that covers the whole claim (seen live)
])
def test_brief_supported_claims_are_not_flagged(claim, quote):
    assert llm._check_claims([{"claim": claim, "support": "brief", "quote": quote}], BRIEF, brand="Duolingo") == []


def test_official_website_evidence_supports_claims_but_social_posts_dont():
    product = {"S2": "Duolingo Max adds Video Call practice with an AI character."}
    ok = {"claim": "Max has video call practice", "support": "S2", "quote": "Duolingo Max adds Video Call practice"}
    assert llm._check_claims([ok], BRIEF, product) == []
    assert llm._check_claims([dict(ok, support="S1")], BRIEF, product)  # an X source never proves a capability
    assert llm._check_claims([dict(ok, quote="Video Call practice with a human tutor")], BRIEF, product)  # not verbatim


@pytest.mark.parametrize("claim, quote", [
    ("Daily streaks come with a streak repair", "daily streaks"),  # short quote that covers only part of the claim
    ("Duolingo offers daily streaks", "streaks"),  # one word is never enough
])
def test_short_quotes_must_cover_the_whole_claim(claim, quote):
    assert llm._check_claims([{"claim": claim, "support": "brief", "quote": quote}], BRIEF, brand="Duolingo")
    # A short quote can't support a claim whose only content is the brand plus a verb.
    assert llm._check_claims([{"claim": "Duolingo offers", "support": "brief", "quote": "Duolingo is"}], BRIEF, brand="Duolingo")


@pytest.mark.parametrize("text, flag", [
    ("90% of learners who come back keep their streak", "Number not in your brief"),
    ("lessons take just 5 minutes", "Number not in your brief"),
    ("behind the scenes: the owl has been in meetings all week planning your comeback", "Behind-the-scenes story"),
    ("the team stayed up all night rebuilding your streak", "Behind-the-scenes story"),
])
def test_invented_numbers_and_behind_the_scenes_stories_are_flagged(text, flag):
    assert any(d.startswith(flag) for d in llm.unverified_details(text, BRIEF, brand="Duolingo"))


def test_jokes_and_brief_facts_are_not_flagged():
    for text in ["your streak misses you. one lesson. that's it.", "free app. daily streaks. no excuses left"]:
        assert llm.unverified_details(text, BRIEF, brand="Duolingo") == []


def test_website_product_facts_and_numbers_are_accepted_end_to_end(monkeypatch):
    site = {"url": "https://www.duolingo.com/", "title": "Duolingo", "metadata": {"url": "https://www.duolingo.com/"},
            "markdown": "Duolingo\nTeachers can set up a class in 2 minutes.\nThe free, fun and effective way to learn."}
    gen = tweets_payload(n=12)
    gen["tweets"][0].update(text="teachers, your class takes 2 minutes to set up. the owl timed it", claims=[
        {"claim": "Class setup takes 2 minutes", "support": "S2", "quote": "Teachers can set up a class in 2 minutes."}])
    gen["tweets"][1].update(text="teachers, your class takes 3 minutes to set up", claims=[
        {"claim": "Class setup takes 3 minutes", "support": "S1", "quote": "the owl does not forget."}])
    r, fake = run(monkeypatch, gen, web=(site,))
    out = r.json()["tweets"]
    assert "<product_evidence>\n[S2]" in fake.prompt("generation")
    assert out[0]["unverified_details"] == []
    assert any("3 minutes" in d for d in out[1]["unverified_details"])
    assert any(d.startswith("Claim not supported") for d in out[1]["unverified_details"])


# ---------- 5. contract: exactly 10, one top-up at most, same key routing ----------

def test_one_top_up_call_fills_a_short_first_batch(monkeypatch):
    first = tweets_payload(n=6)
    second = {"tweets": [dict(t, text=f"top-up idea {w} for teachers")
                         for t, w in zip(tweets_payload(n=6)["tweets"], ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])]}
    r, fake = run(monkeypatch, replies=[first, second])
    assert r.status_code == 200 and len(r.json()["tweets"]) == 10
    assert fake.stages == ["analysis", "generation", "generation"]
    topup = fake.calls[-1]["messages"][1]["content"]
    assert "<already_written>" in topup and "Tweet about dawn for night owls." in topup
    keys = {c.get("_key") for c in fake.calls}  # key routing is covered in test_config; nothing extra is sent here
    assert keys == {None}


def test_no_brand_name_means_no_examples_and_no_style(monkeypatch):
    req = with_prefs({k: SCHOOLS_REQ[k] for k in ("campaign_objective", "description")})
    r, fake = run(monkeypatch, tweets_payload(n=12), req=req)
    prompt = fake.prompt("generation")
    assert r.status_code == 200 and r.json()["research"]["status"] == "skipped"
    assert "<voice_examples>" not in prompt and "<observed_style>" not in prompt and "<product_evidence>" not in prompt


def test_repeated_openings_are_flagged_end_to_end(monkeypatch):
    gen = tweets_payload(n=10)
    gen["tweets"][0]["text"] = "teachers know the free dashboard tracks student progress"
    gen["tweets"][1]["text"] = "teachers know coursework disappears without a deadline"
    r, _ = run(monkeypatch, gen)
    out = r.json()["tweets"]
    assert r.status_code == 200 and "Same opening as another tweet: “tweet 1”" in out[1]["quality_issues"]
