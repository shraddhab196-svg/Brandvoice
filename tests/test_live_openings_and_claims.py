"""Regression tests from a live Duolingo run: (1) "Same opening as tweet 1" was reported for tweets that only shared
"streaks are"; (2) a supported product detail ("bite-size lessons") was flagged together with the unsupported
specifics ("5-minute", "3 words"). Offline only."""

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import Tweet
from conftest import analysis_payload, install_fake_groq, tweets_payload
from test_research import make_settings, post_generate, use_router
from test_research_pipeline import MEMES, feed, routes

CTX = llm.QualityContext(brand="Duolingo", style=None, campaign_terms=llm._terms("streak lesson learners come back"))
LIVE_SAME_WORD = ["streaks are like ice cream—just one more lesson and you’re back for seconds 😊",
                  "streaks are actually the real reason you keep learning, not the lessons.",
                  "streaks are the silent cheerleaders of your language journey."]
BRIEF = {"brand_name": "Duolingo", "industry": "Education technology",
         "campaign_objective": "Get lapsed learners to come back and restart their daily streak",
         "description": "Duolingo is a free app for learning languages, math and music through short, game-like lessons, "
                        "with daily streaks and reminders to keep learners on track."}
SITE = {"url": "https://www.duolingo.com/", "title": "Duolingo", "metadata": {"url": "https://www.duolingo.com/"},
        "markdown": "Learn a language with Duolingo\nBite-sized lessons that feel like a game.\n"
                    "Earn points for correct answers and unlock new levels."}
GROUNDING = "\n".join(BRIEF.values()) + "\n" + SITE["markdown"]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def openings_flagged(texts):
    out = llm._batch_quality([Tweet(style="hook", content_style="engaging", text=t, rationale="r") for t in texts], CTX)
    return [[q for q in t.quality_issues if q.startswith("Same opening")] for t in out]


# ---------- 1. duplicate openings ----------

def test_shared_first_word_is_not_a_repeated_opening():
    assert openings_flagged(LIVE_SAME_WORD) == [[], [], []]  # the live false positive


@pytest.mark.parametrize("texts", [
    ["your streak misses you", "your streak is waiting"],  # shared subject + filler
    ["missed a lesson? no stress", "missed a streak? it happens"],
    ["@duolingo streaks are fun", "#streaks streaks are back"],  # mentions/hashtags aren't the opening
])
def test_other_shared_subjects_are_not_flagged(texts):
    assert openings_flagged(texts) == [[], []]


@pytest.mark.parametrize("texts", [
    ["ever wonder why streaks work?", "Ever wonder how the owl knows?"],  # same meaningful two words, any case
    ["remember that time you skipped?", "remember that time the owl cried?"],  # same first three words
    ["streaks are the best", "streaks are the worst"],
    ["duolingo knows you skipped", "duolingo knows you're back"],
])
def test_genuinely_repeated_openings_are_still_flagged(texts):
    assert openings_flagged(texts) == [[], ["Same opening as another tweet: “tweet 1”"]]


def test_live_batch_end_to_end_has_no_false_opening_flags(monkeypatch):
    use_router(monkeypatch, routes(web=(SITE,)))
    fake = install_fake_groq(monkeypatch)
    gen = tweets_payload(n=10)
    for i, text in zip((0, 3, 9), LIVE_SAME_WORD):
        gen["tweets"][i]["text"] = text
    fake.generation = gen
    tweets = {t["text"]: t for t in post_generate(make_settings(), BRIEF).json()["tweets"]}
    # (The filler fixture tweets all start "Tweet about…", a genuine repeat, so only the live tweets are checked.)
    assert not any(q.startswith("Same opening") for text in LIVE_SAME_WORD for q in tweets[text]["quality_issues"])


# ---------- 2. claims: supported product facts vs unsupported specifics ----------

@pytest.mark.parametrize("text, number", [
    ("forgot to practice, but that 5-minute lesson still counts toward your streak.", "5-minute lesson"),
    ("learn 3 words in a bite-size lesson before your coffee is finished.", "3 words"),
])
def test_live_specifics_get_one_precise_flag_and_supported_parts_none(monkeypatch, text, number):
    use_router(monkeypatch, routes(web=(SITE,)))
    fake = install_fake_groq(monkeypatch)
    gen = tweets_payload(n=10)
    # What a model typically declares: the supported product fact, quoted from the official site.
    gen["tweets"][0].update(text=text, claims=[
        {"claim": "Duolingo has bite-size lessons", "support": "S2", "quote": "Bite-sized lessons"},
        {"claim": "Lessons take 5 minutes", "support": "S2", "quote": "Bite-sized lessons that feel like a game."}])
    fake.generation = gen
    tweet = post_generate(make_settings(), BRIEF).json()["tweets"][0]
    assert len(tweet["unverified_details"]) == 1, tweet["unverified_details"]
    assert tweet["unverified_details"][0].startswith(f"Number not in your brief or product evidence: “{number}")
    assert tweet["basis"] == "inferred"  # not ready to post
    assert not any("Number" in q for q in tweet["quality_issues"])  # fact checking stays out of style checks


@pytest.mark.parametrize("claim, quote", [
    ("Duolingo has bite-size lessons", "Bite-sized lessons"),  # "size" vs "sized"
    ("Lessons are bite-size", "Bite-sized lessons that feel like a game."),
    ("You earn points and unlock levels", "Earn points for correct answers and unlock new levels."),
    ("Class setup takes 2 minutes", "Teachers can set up a class in 2 minutes."),  # number in the quote supports it
])
def test_supported_claims_and_paraphrases_pass(claim, quote):
    product = {"S2": SITE["markdown"] + "\nTeachers can set up a class in 2 minutes."}
    assert llm._check_claims([{"claim": claim, "support": "S2", "quote": quote}], BRIEF["description"], product,
                             "Duolingo") == []


@pytest.mark.parametrize("claim, quote", [
    ("Lessons include live tutors", "Bite-sized lessons that feel like a game."),  # unrelated feature, real quote
    ("Duolingo has bite-size lessons", "streak freeze"),  # not in any product source
])
def test_unsupported_feature_claims_still_fail(claim, quote):
    assert llm._check_claims([{"claim": claim, "support": "S2", "quote": quote}], BRIEF["description"],
                             {"S2": SITE["markdown"]}, "Duolingo")


@pytest.mark.parametrize("text", [
    "earn XP and level up tonight",  # paraphrase of points/levels; "tonight" is not a feature
    "one bite-size lesson and your streak is back",
    "earn points, unlock new levels, repeat",
    "get today done, one lesson at a time",  # a time word is not a product feature
])
def test_paraphrased_supported_features_in_text_pass(text):
    assert llm.unverified_details(text, GROUNDING, brand="Duolingo") == []


@pytest.mark.parametrize("text, flag", [
    ("lessons take 5 minutes", "Number not in your brief or product evidence"),
    ("finish 2 lessons in 90 seconds", "Number not in your brief or product evidence"),
    ("level up tonight", "Product feature not in your brief or product evidence"),  # no levels in this grounding
    ("unlock live tutors with every lesson", "Product feature not in your brief or product evidence"),
])
def test_unsupported_specifics_and_features_are_flagged(text, flag):
    grounding = "\n".join(BRIEF.values())  # brief only: no website evidence of levels or durations
    assert any(d.startswith(flag) for d in llm.unverified_details(text, grounding, brand="Duolingo"))


@pytest.mark.parametrize("text", LIVE_SAME_WORD + ["your streak misses you. one lesson. that's it."])
def test_jokes_opinions_and_figurative_lines_are_not_fact_flagged(text):
    assert llm.unverified_details(text, GROUNDING, brand="Duolingo") == []


def test_social_posts_never_support_a_duration(monkeypatch):
    # The brand's own post says lessons take 5 minutes; posts shape voice, never product facts.
    use_router(monkeypatch, routes(posts=feed(MEMES + ["our lessons take 5 minutes, no excuses"]), web=(SITE,)))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(relevance=[{"source_id": "S1", "relevant": True, "reason": "official"},
                                                {"source_id": "S2", "relevant": True, "reason": "official"}])
    gen = tweets_payload(n=10)
    gen["tweets"][0].update(text="lessons take 5 minutes, so come back tonight", claims=[
        {"claim": "Lessons take 5 minutes", "support": "S1", "quote": "our lessons take 5 minutes"}])
    fake.generation = gen
    details = post_generate(make_settings(), BRIEF).json()["tweets"][0]["unverified_details"]
    assert any(d.startswith("Number not in your brief or product evidence: “5 minutes") for d in details)
    assert any(d.startswith("Claim not supported by your brief or product evidence") for d in details)
