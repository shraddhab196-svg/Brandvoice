"""Regression tests for two live-test reliability issues: (1) the analysis model returned valid JSON without the required
`themes` field (HTTP 502) and (2) an undeclared, unsupported product feature ("earn points") passed factual review.
Offline only: Groq is faked and research keys are absent unless a test routes research explicitly."""

from types import SimpleNamespace

import groq
import pytest
from fastapi.testclient import TestClient

from app import llm
from app.config import Settings, get_settings
from app.main import app
from conftest import analysis_payload, install_fake_groq, tweets_payload
from test_generate import VALID_INPUT, setup, status_error
from test_research import make_settings, post_generate, use_router
from test_research_pipeline import MEMES, routes

MISSING_THEMES = {k: v for k, v in analysis_payload().items() if k != "themes"}
DUO_BRIEF = ("Duolingo\nEducation technology\nGet lapsed learners to come back and restart their daily streak\n"
             "Duolingo is a free app for learning languages, math and music through short, game-like lessons, "
             "with daily streaks and reminders to keep learners on track.")


@pytest.fixture(autouse=True)
def clean():
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


# ---------- 1. missing / invalid required analysis fields ----------

def test_missing_themes_triggers_one_repair_and_succeeds(monkeypatch):
    client, fake = setup(monkeypatch, analysis=MISSING_THEMES)
    fake.field_repair = {"themes": ["late nights", "small-batch craft"]}
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 200, r.text
    assert fake.stages == ["analysis", "field_repair", "generation"]
    assert r.json()["voice_profile"]["themes"] == ["late nights", "small-batch craft"]
    prompt = fake.prompt("field_repair")
    assert "<fields_to_fix>\n- themes: 2-5 content themes (strings)\n</fields_to_fix>" in prompt
    assert '"tone": "Warm, wry and unhurried."' in prompt.split("<accepted_profile>")[1]  # accepted fields kept
    assert VALID_INPUT["description"] in prompt  # same brief (and research, when present) as the first call
    # The rest of the profile is the first answer's, unchanged (only the broken field was replaced).
    assert r.json()["voice_profile"]["tone"] == "Warm, wry and unhurried."


@pytest.mark.parametrize("field, broken, repair", [
    ("tone", "", {"tone": "Warm and wry."}),
    ("audience", None, {"audience": "Night-shift workers."}),
    ("traits", [{"name": "Only one", "description": "x"}],
     {"traits": [{"name": n, "description": "How it shows.", "evidence": []} for n in ("Dry wit", "Warmth", "Calm")]}),
    ("writing_patterns", [], {"writing_patterns": [{"pattern": "Short lines."}, {"pattern": "Few emoji."}]}),
    ("themes", "not a list", {"themes": ["late nights"]}),
])
def test_every_required_field_is_validated_and_repairable(monkeypatch, field, broken, repair):
    analysis = analysis_payload()
    analysis[field] = broken
    client, fake = setup(monkeypatch, analysis=analysis)
    fake.field_repair = repair
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 200, r.text
    assert fake.stages == ["analysis", "field_repair", "generation"]
    assert f"- {field}:" in fake.prompt("field_repair")


@pytest.mark.parametrize("repair", [
    {"themes": []},  # still invalid
    {"tone": "Wrong field only."},  # doesn't provide the missing field
    {"themes": "late nights"},  # wrong type
    "not json",  # malformed
    "",  # empty
])
def test_invalid_repair_fails_safely_without_generation(monkeypatch, repair):
    client, fake = setup(monkeypatch, analysis=MISSING_THEMES)
    fake.field_repair = repair
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 502 and "try again" in r.json()["detail"]
    assert fake.stages == ["analysis", "field_repair"]  # exactly one repair, never a loop, no generation


@pytest.mark.parametrize("error, code", [(status_error(groq.RateLimitError, 429), 429),
                                         (groq.APITimeoutError(request=None), 504)])
def test_repair_provider_errors_are_mapped(monkeypatch, error, code):
    client, fake = setup(monkeypatch, analysis=MISSING_THEMES)
    fake.field_repair = error
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == code and fake.stages == ["analysis", "field_repair"]


def test_repair_cannot_smuggle_in_unverified_evidence(monkeypatch):
    # Repaired traits go through the same exact-quote verification: an invented quote stays out, basis stays inferred.
    analysis = analysis_payload(traits=[])
    client, fake = setup(monkeypatch, analysis=analysis)
    fake.field_repair = {"traits": [{"name": n, "description": "d", "evidence": [{"source_id": "S1", "quote": "invented quote here ok"}]}
                                    for n in ("Dry wit", "Warmth", "Calm")]}
    traits = client.post("/generate", json=VALID_INPUT).json()["voice_profile"]["traits"]
    assert all(t["basis"] == "inferred" and t["evidence"] == [] for t in traits)


def test_summary_only_problem_still_uses_the_summary_repair(monkeypatch):
    client, fake = setup(monkeypatch, analysis=analysis_payload(summary=["only one"]))
    assert client.post("/generate", json=VALID_INPUT).status_code == 200
    assert fake.stages == ["analysis", "summary_repair", "generation"]  # one repair call in total


def test_unparseable_analysis_is_not_repaired(monkeypatch):
    client, fake = setup(monkeypatch, analysis="{not json")
    assert client.post("/generate", json=VALID_INPUT).status_code == 502 and fake.stages == ["analysis"]


def test_repair_uses_the_analysis_key_never_the_generation_key(monkeypatch):
    fake = install_fake_groq(monkeypatch)
    fake.analysis = MISSING_THEMES
    keys = []
    monkeypatch.setattr(llm, "get_client", lambda key, timeout: keys.append(key) or SimpleNamespace(
        chat=SimpleNamespace(completions=fake)))
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, groq_api_key="analysis-key-test", groq_api_key_brand_voice="generation-key-test")
    assert TestClient(app).post("/generate", json=VALID_INPUT).status_code == 200
    assert list(zip(fake.stages, keys)) == [("analysis", "analysis-key-test"), ("field_repair", "analysis-key-test"),
                                            ("generation", "generation-key-test")]


def test_repaired_themes_are_still_checked_against_the_posts(monkeypatch):
    # With real posts, a repaired one-off theme is filtered exactly like a first-answer theme.
    use_router(monkeypatch, routes())
    fake = install_fake_groq(monkeypatch)
    fake.analysis = {k: v for k, v in analysis_payload(relevance=[]).items() if k != "themes"}
    fake.field_repair = {"themes": ["the owl", "quantum physics"]}
    body = post_generate(make_settings(), {"brand_name": "Duolingo", "campaign_objective": "Bring learners back",
                                           "description": "Duolingo is a free language learning app."}).json()
    assert body["voice_profile"]["themes"] == ["the owl"]
    assert "<brand_research>" in fake.prompt("field_repair") and MEMES[0] in fake.prompt("field_repair")


# ---------- 2. unsupported product features in the tweet text ----------

def details(text, extra=""):
    return llm.unverified_details(text, DUO_BRIEF + ("\n" + extra if extra else ""), brand="Duolingo")


def test_earn_points_is_flagged_without_support():
    # The live case: no declared claim, and nothing in the brief mentions points.
    assert details("grab short lessons, earn points, and keep your streak alive.") == [
        "Product feature not in your brief or product evidence: “earn points”"]


@pytest.mark.parametrize("evidence", [
    "Earn points for every lesson you finish.",  # same words
    "Earn XP with every lesson and climb the leaderboard.",  # paraphrase: XP ~ points
])
def test_earn_points_is_accepted_with_product_evidence(evidence):
    assert details("grab short lessons, earn points, and keep your streak alive.", evidence) == []


@pytest.mark.parametrize("text", [
    "get daily reminders so you never forget your streak",
    "short, game-like lessons in languages, math and music",
    "keep learning on track with reminders",
])
def test_brief_supported_features_are_not_flagged(text):
    assert details(text) == []


@pytest.mark.parametrize("text", [
    "unlock badges by finishing lessons", "the app comes with offline mode", "compete in weekly leagues",
])
def test_other_unsupported_features_are_flagged(text):
    assert any(d.startswith("Product feature not in your brief") for d in details(text))


@pytest.mark.parametrize("text", [
    "earn the owl's respect. one lesson.",  # figurative
    "get back to it, your streak misses you",
    "get a life? no. get a streak.",
    "streaks are the best motivation ever.",  # opinion
    "missing that daily streak glow? jump back in, the free app's waiting!",
])
def test_jokes_and_opinions_are_not_product_claims(text):
    assert details(text) == []


def test_unsupported_numbers_still_flagged_even_with_product_evidence():
    assert any(d.startswith("Number not in your brief") for d in details("earn 50 points a day", "Earn XP with every lesson."))


def test_social_posts_never_support_a_feature(monkeypatch):
    # The brand's own X post says "earn points", but posts are voice evidence, not product proof.
    use_router(monkeypatch, routes(posts={"data": [{"id": str(i), "text": t, "created_at": "x"}
                                                   for i, t in enumerate(MEMES + ["earn points every day, bestie"])]}))
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(relevance=[{"source_id": "S1", "relevant": True, "reason": "official"}])
    gen = tweets_payload(n=10)
    gen["tweets"][0]["text"] = "grab short lessons, earn points, and keep your streak alive."
    req = {"brand_name": "Duolingo", "campaign_objective": "Get lapsed learners to restart their daily streak",
           "description": "Duolingo is a free app with short, game-like lessons and daily streaks."}
    fake.generation = gen
    tweet = post_generate(make_settings(), req).json()["tweets"][0]
    assert any("earn points" in d for d in tweet["unverified_details"])


def test_tweet_cannot_pass_because_the_model_declared_no_claims(monkeypatch):
    gen = tweets_payload(n=10)
    gen["tweets"][0].update(text="Grab a cold brew and earn points with every bottle.", claims=[])
    client, _ = setup(monkeypatch, generation=gen)
    tweet = client.post("/generate", json=VALID_INPUT).json()["tweets"][0]
    assert any(d.startswith("Product feature not in your brief") for d in tweet["unverified_details"])
    assert tweet["basis"] == "inferred"
