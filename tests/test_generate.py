import json

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app import llm
from app.config import Settings, get_settings
from app.main import app
from app.schemas import TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs

FAKE_KEY = "test-key-not-real"
VALID_INPUT = {
    "brand_name": "Moonbrew",
    "industry": "Coffee",
    "campaign_objective": "Launch our new decaf cold brew for late-night workers",
    "description": "Small-batch cold brew coffee for night owls, brewed in Pune.",
}


@pytest.fixture(autouse=True)
def block_real_groq(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("Real Groq client must not be created in tests")

    monkeypatch.setattr(groq, "Groq", _forbidden)
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def setup(monkeypatch, analysis=None, generation=None, api_key=FAKE_KEY):
    """Research keys are absent here, so research is skipped and no HTTP calls can happen."""
    fake = install_fake_groq(monkeypatch)
    if analysis is not None:
        fake.analysis = analysis
    if generation is not None:
        fake.generation = generation
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, groq_api_key=api_key, groq_api_key_brand_voice=api_key)
    return TestClient(app), fake


def status_error(cls, code):
    resp = httpx.Response(code, request=httpx.Request("POST", "https://api.groq.com/test"))
    return cls("err", response=resp, body=None)


def test_health(monkeypatch):
    client, _ = setup(monkeypatch)
    assert client.get("/health").json() == {"status": "ok", "app": "AI Tweet Generator", "groq_configured": True}


def test_analysis_runs_before_generation_and_feeds_it(monkeypatch):
    client, fake = setup(monkeypatch)
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 200
    body = r.json()

    assert set(body) == {"brand_voice_summary", "voice_source", "voice_profile", "research", "tweets"}
    assert fake.stages == ["analysis", "generation"]  # two separate, ordered steps
    assert all(c["response_format"] == {"type": "json_object"} for c in fake.calls)

    profile = body["voice_profile"]
    assert body["brand_voice_summary"] == profile["summary"] and 3 <= len(profile["summary"]) <= 4
    assert [t["name"] for t in profile["traits"]] == ["Dry wit", "Warmth", "Craft pride"]
    assert profile["tone"] and profile["audience"] and profile["writing_patterns"] and profile["themes"]

    gen_prompt = fake.prompt("generation")
    assert "<voice_profile>" in gen_prompt and "Dry wit" in gen_prompt
    for value in VALID_INPUT.values():
        assert value in fake.prompt("analysis") and value in gen_prompt

    assert len(body["tweets"]) == 10
    for t in body["tweets"]:
        assert len(t["text"]) <= 280 and t["style"] and t["rationale"]
        assert t["traits"] == ["Dry wit"]
    assert FAKE_KEY not in r.text


def test_without_research_everything_is_labelled_inferred(monkeypatch):
    # The model tries to cite evidence and sources that were never retrieved.
    client, fake = setup(
        monkeypatch,
        analysis=analysis_payload(evidence=[{"source_id": "S1", "quote": "We brewed this for night owls"}],
                                  insights=[{"source_id": "S1", "insights": ["Made up"]}]),
        generation=tweets_payload(source_ids=["S1"], insight="Posts use late-night humour."),
    )
    body = client.post("/generate", json=VALID_INPUT).json()

    assert body["research"]["status"] == "skipped" and body["research"]["sources"] == []
    assert all(t["basis"] == "inferred" and t["evidence"] == [] for t in body["voice_profile"]["traits"])
    assert all(t["basis"] == "inferred" and t["source_ids"] == [] and t["insight"] is None for t in body["tweets"])
    assert "<research_insights>" not in fake.prompt("generation")


def test_optional_fields_missing(monkeypatch):
    client, fake = setup(monkeypatch)
    payload = {k: VALID_INPUT[k] for k in ("campaign_objective", "description")}
    r = client.post("/generate", json=payload)
    assert r.status_code == 409 and fake.stages == []  # no brand: ask the voice questions first
    r = client.post("/generate", json=with_prefs(payload))
    assert r.status_code == 200
    assert "Brand name: Not provided" in fake.prompt("analysis")
    assert r.json()["research"]["status"] == "skipped"


def test_normalizes_harmless_deviations(monkeypatch):
    analysis = analysis_payload()
    analysis["summary"] = ["- Warm, witty tone", "b", "c", "d", "e"]
    generation = tweets_payload(n=12, traits=("dry WIT", "Not a trait", "Warmth", "Dry wit"))
    generation["tweets"][0]["style"] = "  Hot Take "
    client, _ = setup(monkeypatch, analysis=analysis, generation=generation)
    body = client.post("/generate", json=VALID_INPUT).json()

    assert body["brand_voice_summary"][0] == "Warm, witty tone" and len(body["brand_voice_summary"]) == 4
    assert len(body["tweets"]) == 10
    assert body["tweets"][0]["style"] == "hot take"
    assert body["tweets"][0]["traits"] == ["Dry wit", "Warmth"]  # canonical names, unknown dropped, deduped


@pytest.mark.parametrize("analysis", [
    "not json at all",
    "",
    json.dumps(["a", "list"]),
    analysis_payload(traits=[{"name": "One", "description": "Only one trait"}]),
    analysis_payload(writing_patterns=[]),
    analysis_payload(tone=""),
    # A short summary alongside another invalid field: one field repair (not the summary-only repair).
    analysis_payload(summary=["only one"], traits=[{"name": "One", "description": "Only one trait"}]),
])
def test_malformed_analysis_returns_502_and_skips_generation(monkeypatch, analysis):
    client, fake = setup(monkeypatch, analysis=analysis)  # the default repair reply doesn't fix these fields
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 502 and "detail" in r.json()
    # Unparseable output fails at once; parseable-but-invalid output gets exactly one field repair, never more.
    assert fake.stages == ["analysis"] + (["field_repair"] if isinstance(analysis, dict) else [])


# ---------- analysis summary: fewer than 3 items (live failure: "summary ... at least 3 items, not 1") ----------

def test_one_item_summary_gets_one_targeted_repair_on_the_analysis_key(monkeypatch):
    client, fake = setup(monkeypatch, analysis=analysis_payload(summary=["Warm, wry and calm."]))
    keys = []
    fake_client = llm.get_client("unused", 1.0)
    monkeypatch.setattr(llm, "get_client", lambda api_key, timeout: keys.append(api_key) or fake_client)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, groq_api_key="analysis-key", groq_api_key_brand_voice="generation-key")
    r = TestClient(app).post("/generate", json=VALID_INPUT)

    assert r.status_code == 200
    assert fake.stages == ["analysis", "summary_repair", "generation"]
    assert keys == ["analysis-key", "analysis-key", "generation-key"]  # repair stays on the analysis account
    assert r.json()["brand_voice_summary"] == fake.summary_repair["summary"]
    repair_prompt = fake.prompt("summary_repair")
    assert VALID_INPUT["campaign_objective"] in repair_prompt and '"draft_summary": ["Warm, wry and calm."]' in repair_prompt


@pytest.mark.parametrize("summary", [
    ["- Warm, wry and calm.\n- Built for night owls.\n- Themes: craft and late nights."],  # one string, newline bullets
    ["• Warm, wry and calm. • Built for night owls. • Themes: craft and late nights."],
    ["1. Warm, wry and calm. 2. Built for night owls. 3) Themes: craft and late nights."],
    "- Warm, wry and calm.\n- Built for night owls.\n- Themes: craft and late nights.",  # a string, not a list
])
def test_bullets_packed_into_one_string_are_split_without_an_extra_call(monkeypatch, summary):
    client, fake = setup(monkeypatch, analysis=analysis_payload(summary=summary))
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 200 and fake.stages == ["analysis", "generation"]
    assert r.json()["brand_voice_summary"] == ["Warm, wry and calm.", "Built for night owls.", "Themes: craft and late nights."]


def test_prose_is_never_split_into_invented_bullets(monkeypatch):
    # One paragraph with no bullet markers stays one item (sentences aren't chopped up); the repair is used instead.
    client, fake = setup(monkeypatch, analysis=analysis_payload(summary=["Warm and wry. Built for night owls. Craft themes."]))
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 200 and fake.stages == ["analysis", "summary_repair", "generation"]


@pytest.mark.parametrize("repair", [{"summary": ["still only one"]}, {"summary": []}, "not json"])
def test_failed_summary_repair_returns_502_without_generation(monkeypatch, repair):
    client, fake = setup(monkeypatch, analysis=analysis_payload(summary=["only", "two"]))
    fake.summary_repair = repair
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 502 and "detail" in r.json()
    assert fake.stages == ["analysis", "summary_repair"]


def test_summary_repair_provider_errors_are_mapped(monkeypatch):
    client, fake = setup(monkeypatch, analysis=analysis_payload(summary=["only one"]))
    fake.summary_repair = status_error(groq.RateLimitError, 429)
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 429 and fake.stages == ["analysis", "summary_repair"]


def test_analysis_prompt_example_shows_several_summary_items():
    example = llm.ANALYSIS_PROMPT.split('{"summary": [', 1)[1].split("]", 1)[0]
    assert example.count('", "') >= 2  # at least 3 separate example bullets, not one
    assert "one bullet per array element" in llm.ANALYSIS_PROMPT


@pytest.mark.parametrize("generation", [
    "not json at all",
    tweets_payload(n=8),
    tweets_payload(text_len=281),
    tweets_payload(rationale=""),
    tweets_payload(style=""),
    {"tweets": [{"text": "no style"}] * 10},
])
def test_malformed_generation_returns_502(monkeypatch, generation):
    client, fake = setup(monkeypatch, generation=generation)
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 502 and "detail" in r.json()
    # Unparseable output fails at once; too few usable tweets get exactly one top-up call, never more.
    assert fake.stages == ["analysis", "generation"] + (["generation"] if isinstance(generation, dict) else [])


@pytest.mark.parametrize(
    "error, expected",
    [
        (lambda: status_error(groq.RateLimitError, 429), 429),
        (lambda: status_error(groq.AuthenticationError, 401), 502),
        (lambda: status_error(groq.BadRequestError, 400), 502),
        (lambda: status_error(groq.InternalServerError, 500), 502),
        (lambda: groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com/test")), 504),
        (lambda: groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/test")), 502),
    ],
)
@pytest.mark.parametrize("stage", ["analysis", "generation"])
def test_groq_errors_mapped(monkeypatch, error, expected, stage):
    client, _ = setup(monkeypatch, **{stage: error()})
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == expected
    assert FAKE_KEY not in r.text


def test_missing_api_key_returns_503(monkeypatch):
    client, fake = setup(monkeypatch, api_key=None)
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 503
    assert fake.calls == []


def test_invalid_input_returns_422(monkeypatch):
    client, fake = setup(monkeypatch)
    r = client.post("/generate", json={"description": "short"})
    assert r.status_code == 422
    assert fake.calls == []


# ---------- one Groq key per stage ----------

ANALYSIS_KEY, GENERATION_KEY = "analysis-account-key", "generation-account-key"


def two_key_settings(analysis=ANALYSIS_KEY, generation=GENERATION_KEY) -> Settings:
    return Settings(_env_file=None, groq_api_key=analysis, groq_api_key_brand_voice=generation,
                    groq_model="openai/gpt-oss-20b")


def record_keys(monkeypatch, fake) -> list[tuple[str, str]]:
    """Patch get_client to record (stage, api_key) for every Groq client request."""
    fake_client = llm.get_client("unused", 1.0)  # the installed fake returns the fake client for any args
    used = []

    def recording_get_client(api_key, timeout):
        used.append(api_key)
        return fake_client

    monkeypatch.setattr(llm, "get_client", recording_get_client)
    return used


def test_each_stage_uses_only_its_own_key(monkeypatch):
    _, fake = setup(monkeypatch)
    keys = record_keys(monkeypatch, fake)
    app.dependency_overrides[get_settings] = lambda: two_key_settings()
    r = TestClient(app).post("/generate", json=VALID_INPUT)

    assert r.status_code == 200
    assert list(zip(fake.stages, keys)) == [("analysis", ANALYSIS_KEY), ("generation", GENERATION_KEY)]
    assert all(c["model"] == "openai/gpt-oss-20b" for c in fake.calls)  # model unchanged for both
    assert ANALYSIS_KEY not in r.text and GENERATION_KEY not in r.text


def test_analysis_stage_receives_groq_api_key(monkeypatch):
    _, fake = setup(monkeypatch)
    keys = record_keys(monkeypatch, fake)
    llm.analyze_voice(TweetRequest(**VALID_INPUT), two_key_settings(), None)
    assert keys == [ANALYSIS_KEY] and fake.stages == ["analysis"]


def test_generation_stage_receives_groq_api_key_brand_voice(monkeypatch):
    _, fake = setup(monkeypatch)
    keys = record_keys(monkeypatch, fake)
    profile, insights = llm.analyze_voice(TweetRequest(**VALID_INPUT), two_key_settings(), None)
    keys.clear()
    llm.write_tweets(TweetRequest(**VALID_INPUT), two_key_settings(), profile, None, insights)
    assert keys == [GENERATION_KEY] and fake.stages[-1] == "generation"


@pytest.mark.parametrize("missing, message", [
    ({"analysis": None}, "GROQ_API_KEY (used for brand voice analysis) is not configured on the server."),
    ({"analysis": "   "}, "GROQ_API_KEY (used for brand voice analysis) is not configured on the server."),
    ({"generation": None}, "GROQ_API_KEY_BRAND_VOICE (used for tweet generation) is not configured on the server."),
    ({"analysis": None, "generation": None},
     "GROQ_API_KEY (used for brand voice analysis) and GROQ_API_KEY_BRAND_VOICE (used for tweet generation) are not configured on the server."),
])
def test_missing_key_is_stage_specific_and_never_falls_back(monkeypatch, missing, message):
    _, fake = setup(monkeypatch)
    app.dependency_overrides[get_settings] = lambda: two_key_settings(**missing)
    client = TestClient(app)

    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 503 and r.json()["detail"] == message
    assert fake.calls == []  # no Groq call with the other stage's key, and no wasted analysis call
    assert ANALYSIS_KEY not in r.text and GENERATION_KEY not in r.text
    assert client.get("/health").json()["groq_configured"] is False


@pytest.mark.parametrize("stage, message", [
    ("analysis", "GROQ_API_KEY (used for brand voice analysis) is not configured on the server."),
    ("generation", "GROQ_API_KEY_BRAND_VOICE (used for tweet generation) is not configured on the server."),
])
def test_stage_call_refuses_to_run_without_its_own_key(monkeypatch, stage, message):
    _, fake = setup(monkeypatch)
    settings = two_key_settings(**{stage: None})  # the other stage's key is present but must not be used
    profile, insights = llm.analyze_voice(TweetRequest(**VALID_INPUT), two_key_settings(), None)
    fake.calls.clear()
    with pytest.raises(llm.LLMError) as exc:
        if stage == "analysis":
            llm.analyze_voice(TweetRequest(**VALID_INPUT), settings, None)
        else:
            llm.write_tweets(TweetRequest(**VALID_INPUT), settings, profile, None, insights)
    assert exc.value.status_code == 503 and exc.value.message == message
    assert fake.calls == []


@pytest.mark.parametrize("stage, env_name", [("analysis", "GROQ_API_KEY"), ("generation", "GROQ_API_KEY_BRAND_VOICE")])
def test_rejected_key_names_the_stage(monkeypatch, stage, env_name):
    client, _ = setup(monkeypatch, **{stage: status_error(groq.AuthenticationError, 401)})
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 502 and env_name in r.json()["detail"]
    other = "GROQ_API_KEY_BRAND_VOICE" if stage == "analysis" else "GROQ_API_KEY ("
    assert other not in r.json()["detail"]
    assert FAKE_KEY not in r.text
