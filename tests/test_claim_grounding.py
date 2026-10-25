"""Regression tests: research shapes voice, never product claims; claim validation; honest review status.

Uses a neutral brief (class planning) that says nothing about meetings, transcripts or trials, while the brand's
own (accepted) research promotes AI meeting notes: the exact situation that previously leaked into tweets."""

import json

import groq
import pytest

from app import llm, research
from app.main import app
from app.schemas import Tweet, TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload, with_prefs
from test_research import make_settings, post_generate, use_router

STUDENT_REQ = {
    "brand_name": "Notion",
    "industry": "Productivity software",
    "campaign_objective": "Get more students to use Notion for class planning",
    "description": "Notion is a connected workspace for notes, docs and projects.",
}
HQ = {"id": "1", "username": "NotionHQ", "name": "Notion", "followers_count": 900000, "verified": True,
      "url": None, "description": "", "created_at": "x"}
POSTS = {"data": [
    {"id": "1", "text": "AI Meeting Notes now summarize every call and keep the transcript updated for your team.", "created_at": "x"},
    {"id": "2", "text": "Plan your semester with templates for classes, notes and projects in one workspace.", "created_at": "x"},
    {"id": "3", "text": "Your docs and notes, connected in one calm workspace.", "created_at": "x"},
]}
ROUTES = {("GET", "/v1/twitter/users/notion/by_username"): (200, HQ),
          ("GET", "/v1/twitter/users/NotionHQ/latest_tweets"): (200, POSTS)}
FEATURE_FINDING = {"text": "Posts promote AI meeting notes that summarize calls.",
                   "quote": "AI Meeting Notes now summarize every call"}
VOICE_FINDING = {"text": "Posts use a calm, connected-workspace framing.", "quote": "Your docs and notes, connected in one calm workspace."}
BRIEF = "\n".join(STUDENT_REQ.values())
C = "Claim not supported by your brief or product evidence: “{}”".format  # review items explain the risk, quoting the claim


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def run(monkeypatch, tweets: list[dict], req=STUDENT_REQ, insights=(FEATURE_FINDING, VOICE_FINDING), **analysis_kw):
    router = use_router(monkeypatch, ROUTES)
    fake = install_fake_groq(monkeypatch)
    fake.analysis = analysis_payload(insights=[{"source_id": "S1", "insights": list(insights)}], **analysis_kw)
    gen = tweets_payload(text="Plan your semester in one workspace.", rationale="Class planning for students.")
    for target, override in zip(gen["tweets"], tweets):
        target.update(override)
    fake.generation = gen
    r = post_generate(make_settings(fc=None), req)
    assert r.status_code == 200, r.text
    return r.json(), fake, router


# ---------- 1 + 2. declared-claim validation ----------

def test_finding_supported_claim_with_empty_quote_is_flagged(monkeypatch):
    body, _, _ = run(monkeypatch, [{"claims": [{"claim": "Summarizes every meeting", "support": "S1.1", "quote": ""}],
                                    "insight_ids": ["S1.1"]}])
    t = body["tweets"][0]
    assert C("Summarizes every meeting") in t["unverified_details"] and t["basis"] == "inferred"


@pytest.mark.parametrize("quote", ["for notes", "a connected workspace for", "", "Notion is a connected workspace"])
def test_unrelated_or_empty_brief_quote_cannot_support_a_claim(quote):
    claims = [{"claim": "Free trial available", "support": "brief", "quote": quote}]
    assert llm._check_claims(claims, BRIEF) == [C("Free trial available")]


def test_too_short_brief_quote_cannot_support_a_claim():
    # "for notes" is verbatim and covers half of "Notes app", but two words are too thin to count as support.
    assert llm._check_claims([{"claim": "Notes app", "support": "brief", "quote": "for notes"}], BRIEF) == [C("Notes app")]


def test_finding_id_alone_never_verifies_a_product_claim():
    claims = [{"claim": "Summarizes every call", "support": "S1.1", "quote": "AI Meeting Notes now summarize every call"}]
    assert llm._check_claims(claims, BRIEF) == [C("Summarizes every call")]


def test_properly_quoted_brief_claim_is_not_flagged(monkeypatch):
    body, _, _ = run(monkeypatch, [{
        "text": "Notes, docs and projects in one connected workspace.",
        "claims": [{"claim": "Connected workspace for notes, docs and projects", "support": "brief",
                    "quote": "a connected workspace for notes, docs and projects"}]}])
    assert body["tweets"][0]["unverified_details"] == []


# ---------- 3. research-only product features (declared or not) ----------

@pytest.mark.parametrize("text, claims", [
    ("Notion AI summarizes every meeting and keeps the transcript updated.",  # declared, backed only by a finding
     [{"claim": "AI summarizes meetings", "support": "S1.1", "quote": "AI Meeting Notes now summarize every call"}]),
    ("Every call, summarized. Every transcript, updated.", []),  # not declared, no trigger words
    ("Notion AI automatically summarizes every meeting.", []),  # not declared, capability wording
])
def test_feature_found_only_in_research_is_flagged(monkeypatch, text, claims):
    body, _, _ = run(monkeypatch, [{"text": text, "claims": claims, "insight_ids": ["S1.1"],
                                    "rationale": "Shows the AI meeting notes that summarize calls."}])
    t = body["tweets"][0]
    assert t["unverified_details"], "research-only feature must be flagged"
    assert t["basis"] == "inferred"
    assert t["source_ids"] == ["S1"]  # citation kept as voice attribution, not as proof


@pytest.mark.parametrize("text, flag", [
    ("Works with Slack, so your class notes stay in sync.", "Slack"),  # "Slack" only in research chatter
    ("Join 100 students planning smarter.", "100 students planning"),  # "100" only in research
])
def test_backstops_ground_against_the_brief_not_research(monkeypatch, text, flag):
    # These research lines don't read like product statements, so only brief-only grounding catches the reuse.
    chatter = {"data": POSTS["data"] + [
        {"id": "4", "text": "Thanks to everyone who joined our Slack community for class notes and docs today.", "created_at": "x"},
        {"id": "5", "text": "Over 100 classes planned with notes and docs this week.", "created_at": "x"},
    ]}
    monkeypatch.setitem(ROUTES, ("GET", "/v1/twitter/users/NotionHQ/latest_tweets"), (200, chatter))
    body, _, _ = run(monkeypatch, [{"text": text}])
    assert any(flag in d for d in body["tweets"][0]["unverified_details"]), body["tweets"][0]["unverified_details"]


@pytest.mark.parametrize("text", [
    "Plan your semester like a pro.",  # one research word ("semester") is not a product claim
    "Your docs and notes, all in one calm place.",  # voice echo of a non-product research line
    "Class planning that finally feels calm.",
    "Picture this: every note for every class, right where you left it.",
])
def test_ordinary_voice_language_is_not_flagged(monkeypatch, text):
    body, _, _ = run(monkeypatch, [{"text": text}])
    assert body["tweets"][0]["unverified_details"] == []


def test_research_only_check_is_conservative_by_design():
    # Documents a known limit: a single research-only word can slip through (needs >= 2 in one sentence).
    sources = research.ResearchResult(x_handle="NotionHQ", posts=[p["text"] for p in POSTS["data"]]).sources()
    assert llm.research_only_details("Transcripts for every class.", BRIEF, sources) == []
    assert llm.research_only_details("Transcripts, summarized for every class.", BRIEF, sources) != []


# ---------- 5 + 6. research still shapes voice; the brief owns the campaign ----------

def test_research_still_informs_voice_profile_and_prompt(monkeypatch):
    analysis_kw = {"evidence": [{"source_id": "S1", "quote": "Your docs and notes, connected in one calm workspace."},
                                {"source_id": "S1", "quote": "templates for classes, notes and projects in one workspace."}],
                   "themes": ["AI meeting notes", "calm organisation"]}
    body, fake, _ = run(monkeypatch, [{"text": "Your docs and notes, all in one calm place.", "insight_ids": ["S1.2"],
                                       "rationale": "Echoes the calm connected-workspace voice."}], **analysis_kw)
    assert body["voice_profile"]["traits"][0]["basis"] == "evidence"
    prompt = fake.prompt("generation")
    assert "[S1.2]" in prompt and "Voice and style findings only" in prompt
    t = body["tweets"][0]
    assert t["basis"] == "evidence" and t["source_ids"] == ["S1"] and t["unverified_details"] == []


def test_campaign_objective_is_intact_and_themes_are_voice_context(monkeypatch):
    _, fake, _ = run(monkeypatch, [], themes=["AI meeting notes", "calm organisation"])
    for stage in ("analysis", "generation"):
        prompt = fake.prompt(stage)
        assert STUDENT_REQ["campaign_objective"] in prompt and STUDENT_REQ["description"] in prompt
    gen = fake.prompt("generation")
    assert '"voice_themes (style context only; not campaign topics or product facts)": ["AI meeting notes"' in gen
    assert '"themes"' not in gen
    assert "may come only from <brand_input> or from" in llm.GENERATION_PROMPT and "<product_evidence> (the brand\'s official website)" in llm.GENERATION_PROMPT
    assert "Social posts and research findings cannot support a product claim" in llm.GENERATION_PROMPT


# ---------- 9 + 11. honest status, exactly 10, unchanged shape and calls ----------

def test_ten_tweets_with_some_needing_review_and_shape_unchanged(monkeypatch):
    body, fake, _ = run(monkeypatch, [{"text": "Every call, summarized. Every transcript, updated."},
                                      {"text": "Start your free trial today."}])
    assert len(body["tweets"]) == 10
    assert set(body) == {"brand_voice_summary", "voice_source", "voice_profile", "research", "tweets"}
    assert all(set(t) == set(Tweet.model_fields) for t in body["tweets"])
    flagged = [t for t in body["tweets"] if t["unverified_details"]]
    assert len(flagged) == 2 and all(t["basis"] == "inferred" for t in flagged)
    # Claim-flagged tweets get ONE bounded regeneration call (the fake returns the same candidates, so they stay flagged).
    assert fake.stages == ["analysis", "generation", "generation"]


def test_missing_brand_name_still_skips_scraping(monkeypatch):
    req = with_prefs({k: v for k, v in STUDENT_REQ.items() if k != "brand_name"})  # answers to the voice questions
    body, _, router = run(monkeypatch, [], req=req)
    assert router.requests == [] and body["research"]["status"] == "skipped"


# ---------- 8. Swagger no longer prefills a sample campaign ----------

def test_openapi_request_has_no_prefilled_campaign():
    props = app.openapi()["components"]["schemas"]["TweetRequest"]["properties"]
    assert all("examples" not in p and "example" not in p for p in props.values())
    blob = json.dumps(props).lower()
    assert "notion" not in blob and "meeting notes" not in blob
    schema = app.openapi()["components"]["schemas"]["TweetRequest"]
    assert sorted(schema["required"]) == ["campaign_objective", "description"]  # input rules unchanged


def test_tweet_request_rules_unchanged():
    assert TweetRequest(campaign_objective="Grow sign-ups", description="A notes app for students.").brand_name is None
    with pytest.raises(Exception):
        TweetRequest(description="A notes app for students.")
