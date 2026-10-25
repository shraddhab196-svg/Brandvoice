"""Brand voice analysis visibility: evidence verification, source attribution and research status."""

import groq
import pytest

from app import llm, research
from app.main import app
from app.research import ResearchResult
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs
from test_research import FC_KEY, GROQ_KEY, REQ, SB_KEY, make_settings, ok_routes, post_generate, use_router

POST = "Night shift? We brewed this for you."  # a real post in the mocked ScrapeBadger response


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def run(monkeypatch, analysis=None, generation=None, routes=None, settings=None, payload=REQ):
    use_router(monkeypatch, routes if routes is not None else ok_routes())
    fake = install_fake_groq(monkeypatch)
    if analysis is not None:
        fake.analysis = analysis
    if generation is not None:
        fake.generation = generation
    r = post_generate(settings or make_settings(), payload)
    assert r.status_code == 200, r.text
    return r.json(), fake


# ---------- research visibility ----------

def test_report_lists_only_retrieved_sources_with_excerpts_and_insights(monkeypatch):
    body, _ = run(monkeypatch, analysis=analysis_payload(insights=[
        {"source_id": "S1", "insights": [
            {"text": "Posts speak directly to night-shift workers.", "quote": "Night shift? We brewed this for you."},
            {"text": "Posts poke at a rival.", "quote": None},  # no quote -> an inference, not a finding
        ]},
        {"source_id": "S2", "insights": [{"text": "Site leans on night owls imagery.", "quote": "Cold brew for night owls."}]},
        {"source_id": "S9", "insights": [{"text": "A source that was never retrieved.", "quote": None}]},
    ]))
    report = body["research"]
    assert report["status"] == "completed"
    assert [(s["id"], s["kind"], s["url"]) for s in report["sources"]] == [
        ("S1", "x_posts", "https://x.com/moonbrew"),
        ("S2", "website", "https://moonbrew.com/"),
    ]
    assert POST in report["sources"][0]["excerpts"]
    assert report["sources"][0]["insights"] == [
        {"id": "S1.1", "text": "Posts speak directly to night-shift workers.", "quote": "Night shift? We brewed this for you.",
         "basis": "evidence"},
        {"id": "S1.2", "text": "Posts poke at a rival.", "quote": None, "basis": "inferred"},
    ]
    assert report["sources"][1]["insights"][0]["basis"] == "evidence"
    assert "S9" not in str(report)  # never claim a source that wasn't retrieved
    for secret in (SB_KEY, FC_KEY, GROQ_KEY):
        assert secret not in str(body)


def test_analysis_prompt_labels_sources_and_keeps_them_contained(monkeypatch):
    _, fake = run(monkeypatch)
    prompt = fake.prompt("analysis")
    assert "[S1] X posts from @moonbrew" in prompt and "[S2] Website: moonbrew.com" in prompt
    assert prompt.count("</brand_research>") == 1  # scraped text can't close the research block


def test_sources_are_numbered_from_what_was_retrieved():
    only_site = ResearchResult(website_url="https://moonbrew.com/", website_text="Cold brew for night owls, brewed slowly in small batches.")
    assert [(s.id, s.kind) for s in only_site.sources()] == [("S1", "website")]
    assert ResearchResult(x_handle="moonbrew", posts=[]).sources() == []  # handle alone is not a source


@pytest.mark.parametrize("result, status", [
    (ResearchResult(skipped_reason="not_configured"), "skipped"),
    (ResearchResult(skipped_reason="no_brand_name"), "skipped"),
    (ResearchResult(providers=["scrapebadger"], had_error=True), "failed"),
    (ResearchResult(providers=["scrapebadger"], limit_reached=True), "failed"),
    (ResearchResult(providers=["scrapebadger", "firecrawl"]), "completed"),  # ran cleanly, nothing matched
    (ResearchResult(providers=["scrapebadger", "firecrawl"], x_handle="m", posts=["A real post here."]), "partial"),
    (ResearchResult(providers=["scrapebadger"], x_handle="m", posts=["A real post here."], limit_reached=True), "partial"),
    (ResearchResult(providers=["scrapebadger"], x_handle="m", posts=["A real post here."]), "completed"),
])
def test_research_status(result, status):
    report = result.report()
    assert report.status == status and report.message
    if result.limit_reached:
        assert research.LIMIT_MESSAGE in report.message


# ---------- evidence-backed vs inferred voice traits ----------

def test_only_verbatim_quotes_count_as_evidence(monkeypatch):
    analysis = analysis_payload()
    analysis["traits"][0]["evidence"] = [
        {"source_id": "S1", "quote": "We brewed this for you"},  # real, from S1
        {"source_id": "S1", "quote": "We never sleep and neither should you"},  # fabricated
    ]
    analysis["traits"][1]["evidence"] = [{"source_id": "S2", "quote": "We brewed this for you"}]  # real text, wrong source
    analysis["traits"][2]["evidence"] = [{"source_id": "S1", "quote": "night SHIFT? we brewed"}]  # different capitalisation
    body, _ = run(monkeypatch, analysis=analysis)
    traits = body["voice_profile"]["traits"]

    assert traits[0]["evidence"] == [{"source_id": "S1", "quote": "We brewed this for you"}]
    assert traits[0]["basis"] == "inferred"  # verified, but a single post is not enough for an evidence-backed trait
    assert traits[1]["basis"] == "inferred" and traits[1]["evidence"] == []
    assert traits[2]["basis"] == "inferred"  # quotes must match exactly, including case


def test_whitespace_and_typographic_quotes_are_the_only_tolerated_differences(monkeypatch):
    analysis = analysis_payload()
    analysis["traits"][0]["evidence"] = [{"source_id": "S1", "quote": "  Night   shift? We brewed  this  "}]
    body, _ = run(monkeypatch, analysis=analysis)
    assert body["voice_profile"]["traits"][0]["evidence"] == [{"source_id": "S1", "quote": "Night shift? We brewed this"}]


def test_writing_patterns_are_verified_too(monkeypatch):
    analysis = analysis_payload()
    analysis["writing_patterns"][0]["evidence"] = [{"source_id": "S1", "quote": "Night shift? We brewed this for you."}]
    analysis["writing_patterns"][1]["evidence"] = [{"source_id": "S1", "quote": "Award-winning since 1999"}]
    body, _ = run(monkeypatch, analysis=analysis)
    patterns = body["voice_profile"]["writing_patterns"]
    assert patterns[0]["evidence"] and patterns[1]["evidence"] == []
    assert patterns[0]["basis"] == patterns[1]["basis"] == "inferred"  # one post can't make a pattern evidence-backed


def test_website_copy_is_never_voice_evidence(monkeypatch):
    # "Cold brew for night owls." is verbatim website copy (S2): real text, but product context, not social voice.
    analysis = analysis_payload()
    analysis["traits"][0]["evidence"] = [{"source_id": "S2", "quote": "Cold brew for night owls."}]
    analysis["writing_patterns"][0]["evidence"] = [{"source_id": "S2", "quote": "Cold brew for night owls."}]
    body, _ = run(monkeypatch, analysis=analysis)
    assert body["voice_profile"]["traits"][0]["basis"] == "inferred"
    assert body["voice_profile"]["writing_patterns"][0]["basis"] == "inferred"


# ---------- tweet reasoning and attribution ----------

def test_tweet_citations_must_point_at_verified_findings_the_tweet_uses(monkeypatch):
    analysis = analysis_payload(insights=[{"source_id": "S1", "insights": [
        {"text": "Posts speak to night-shift workers.", "quote": "Night shift? We brewed this for you."},  # S1.1 finding
        {"text": "Posts poke at a rival.", "quote": None},  # S1.2 inference: never citable
    ]}])
    generation = tweets_payload()
    tweets = generation["tweets"]
    tweets[0].update(insight_ids=["S1.1"], text="Night shift? This decaf was brewed for you.", insight="MODEL PARAPHRASE")
    tweets[1].update(insight_ids=["S9.1"])  # never retrieved
    tweets[2].update(insight_ids=["S1.2"])  # an inference, not a verified finding
    tweets[3].update(insight_ids=["S1.1"], text="Fresh beans, slow roast.", rationale="Highlights roasting craft.")  # unrelated
    tweets[4].update(insight_ids=["S1.1", "S1.1", "S9.1"])  # deduped, bogus id dropped
    body, fake = run(monkeypatch, analysis=analysis, generation=generation)
    out = body["tweets"]

    assert out[0]["basis"] == "evidence" and out[0]["source_ids"] == ["S1"]
    assert out[0]["insight"] == "Posts speak to night-shift workers."  # the verified finding, not the model's words
    assert out[0]["insight_quote"] == "Night shift? We brewed this for you."
    for t in out[1:4]:
        assert t["basis"] == "inferred" and t["source_ids"] == [] and t["insight"] is None
    assert out[4]["source_ids"] == ["S1"] and out[4]["basis"] == "evidence"
    assert all(t["rationale"] for t in out)

    gen_prompt = fake.prompt("generation")
    insights = gen_prompt.split("<research_insights>", 1)[1].split("</research_insights>", 1)[0]
    assert "[S1.1]" in insights and "S1.2" not in insights and "[S2" not in insights  # only verified findings

def test_failed_research_still_explains_tweets_honestly(monkeypatch):
    routes = {k: (500, {"error": "down"}) for k in ok_routes()}
    body, fake = run(monkeypatch, routes=routes, generation=tweets_payload(insight_ids=["S1.1"]), payload=with_prefs(REQ))
    assert body["research"]["status"] == "failed" and body["research"]["sources"] == []
    assert all(t["basis"] == "inferred" and not t["source_ids"] for t in body["tweets"])
    assert all(t["basis"] == "inferred" for t in body["voice_profile"]["traits"])
    assert "<brand_research>" not in fake.prompt("analysis")


def test_missing_brand_name_skips_research_but_still_analyses(monkeypatch):
    payload = with_prefs({k: REQ[k] for k in ("campaign_objective", "description")})
    use_router_requests = use_router(monkeypatch, ok_routes())
    fake = install_fake_groq(monkeypatch)
    r = post_generate(make_settings(), payload)
    assert r.status_code == 200 and use_router_requests.requests == []
    assert r.json()["research"]["status"] == "skipped"
    assert fake.stages == ["analysis", "generation"] and len(r.json()["tweets"]) == 10
