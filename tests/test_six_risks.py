"""Focused tests for the six risk areas: unsupported claims, generic tweets, uncertain brand voice, example tweets,
exact count/length with selective regeneration, and distinct voices across brands. Offline only (fake Groq, mocked
HTTP). They check rules and data flow; they can't prove the model's writing quality."""

import json

import groq
import pytest

from app import llm, research
from app.main import FRONTEND_DIR, app
from app.schemas import Tweet, TweetRequest
from conftest import analysis_payload, install_fake_groq, tweets_payload
from test_frontend import NODE, run_review_js
from test_generate import VALID_INPUT, setup, status_error
from test_research import make_settings, post_generate, use_router
from test_research_pipeline import MEMES, SCHOOLS_REQ, routes

BRIEF = "Acme\nCoffee\nLaunch our decaf\nAcme roasts small-batch decaf cold brew for night owls."
EXAMPLES = ["Night shift? We brewed this for you.", "Slow-steeped for 18 hours. Worth the wait.",
            "Decaf that still tastes like a decision."]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real Groq client")))
    research._cache.clear()
    llm.get_client.cache_clear()
    yield
    app.dependency_overrides.clear()


def details(text, grounding=BRIEF):
    return llm.unverified_details(text, grounding, brand="Acme")


FLAG_SUFFIXES = [" Rated 4.9 by 900 baristas.", " Loved by 70% of students.", " Brewed 3x faster than rivals.",
                 " Ships to 12 cities next week."]


def flag_first_four(gen):
    for t, suffix in zip(gen["tweets"][:4], FLAG_SUFFIXES):
        t["text"] += suffix  # invented specifics -> flagged (each different, so they aren't merged as duplicates)
    return gen


def clean_payload(n=12):
    gen = tweets_payload(n=n)
    for t in gen["tweets"]:
        if t["style"] == "thread opener":
            t["text"] += " 1/"
    return gen


# ---------- 1. hallucinations and unsupported claims ----------

@pytest.mark.parametrize("text, label", [
    ("our decaf is free this week", "Free offer not in your brief"),
    ("get 20% off your first bottle", "Discount or sale not in your brief"),
    ("half off every decaf tonight", "Discount or sale not in your brief"),
    ("freshness guaranteed or your money back", "Guarantee not in your brief"),
    ("the new decaf launches tomorrow", "Launch or availability date not in your brief"),
    ("the only app that brews for you", "Superlative claim not in your brief"),
])
def test_offer_discount_guarantee_and_launch_wording_are_review_triggers(text, label):
    assert any(d.startswith(label) for d in details(text)), details(text)


@pytest.mark.parametrize("text, grounding", [
    ("our decaf is free this week", BRIEF + "\nThe first bottle is free for new customers."),
    ("get 20% off your first bottle", BRIEF + "\n20% off your first order."),
    ("freshness guaranteed", BRIEF + "\nFreshness guarantee on every bottle."),
    ("the new decaf launches tomorrow", BRIEF + "\nThe decaf launches tomorrow."),
])
def test_the_same_wording_is_fine_when_the_evidence_supports_it(text, grounding):
    assert not any("not in your brief" in d for d in details(text, grounding) if not d.startswith("Number")), details(text, grounding)


def test_a_different_date_than_the_evidence_is_flagged():
    # "launch" is in the evidence, but its date is Friday, not tomorrow: the date itself must be supported.
    flags = details("the new decaf launches tomorrow", BRIEF + " The decaf launch is this Friday.")
    assert any(d.startswith("Launch or availability date not in your brief") for d in flags)


@pytest.mark.parametrize("text", [
    "you're the only one who gets the 3 a.m. struggle",  # figurative "the only"
    "feel free to sip slower tonight",  # "free" not an offer
    "decaf, but make it a decision",
])
def test_jokes_and_figurative_lines_are_not_claims(text):
    assert details(text) == []


def test_typographic_hyphens_dont_break_verbatim_quotes():
    brief = "Acme is a free app for game-like lessons."
    quote = "Acme is a free app for game‑like lessons."  # U+2011 non-breaking hyphen, as models emit
    assert llm._check_claims([{"claim": "free app", "support": "brief", "quote": quote}], brief, {}, "Acme") == []


def test_unsupported_claims_are_regenerated_once_and_valid_tweets_kept(monkeypatch):
    first = flag_first_four(clean_payload())
    second = {"tweets": [dict(first["tweets"][4], text=f"Fresh decaf idea {w} for night owls, slow brewed.",
                              style="hook") for w in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")]}
    client, fake = setup(monkeypatch, generation=first)
    fake.generation_replies = [first, second]
    r = client.post("/generate", json=VALID_INPUT)
    tweets = r.json()["tweets"]
    assert r.status_code == 200 and len(tweets) == 10
    assert fake.stages == ["analysis", "generation", "generation"]  # exactly one bounded regeneration
    assert not any(t["unverified_details"] for t in tweets)  # flagged ones were replaced, not passed off as valid
    kept = {t["text"] for t in first["tweets"][4:]}
    assert len(kept & {t["text"] for t in tweets}) >= 6  # valid originals preserved
    topup = fake.calls[-1]["messages"][1]["content"]
    assert "<rejected_candidates>" in topup and "Number not in your brief" in topup and FLAG_SUFFIXES[0].strip() in topup  # told what was unsupported


def test_regeneration_is_bounded_and_failures_stay_flagged(monkeypatch):
    first = flag_first_four(clean_payload())
    client, fake = setup(monkeypatch, generation=first)  # the "retry" returns the same flagged tweets
    r = client.post("/generate", json=VALID_INPUT)
    tweets = r.json()["tweets"]
    assert r.status_code == 200 and fake.stages == ["analysis", "generation", "generation"]  # never a loop
    flagged = [t for t in tweets if t["unverified_details"]]
    assert len(tweets) == 10 and len(flagged) == 2 and all(t["basis"] == "inferred" for t in flagged)


def test_a_rate_limited_regeneration_keeps_ten_flagged_tweets(monkeypatch):
    first = flag_first_four(clean_payload())
    client, fake = setup(monkeypatch)
    fake.generation_replies = [first, status_error(groq.RateLimitError, 429)]
    r = client.post("/generate", json=VALID_INPUT)
    assert r.status_code == 200 and len(r.json()["tweets"]) == 10  # the repair is optional; flags stay visible
    assert sum(bool(t["unverified_details"]) for t in r.json()["tweets"]) == 2


def test_prompt_names_every_restricted_claim_type():
    p = llm.GENERATION_PROMPT
    assert 'Discounts, percentages, "free", "% off", sales, guarantees and launch or availability dates' in p
    assert "never swap in a different made-up one" in p


# ---------- 2. generic tweets ----------

CTX = llm.QualityContext(brand="Acme", style=None, campaign_terms=llm._terms("launch our decaf small-batch cold brew night owls"))


@pytest.mark.parametrize("text", [
    "discover the magic of slow decaf for night owls",
    "say goodbye to bitter decaf, night owls",
    "whether you're a night owl or an early bird, decaf awaits",
])
def test_generic_filler_is_flagged(text):
    assert any(q.startswith("Generic marketing cliche") for q in llm.tweet_quality(text, "hook", CTX))


def test_a_brand_mention_is_not_a_specific_detail():
    assert any(q.startswith("No clear link") for q in llm.tweet_quality("acme is simply the whole mood every single evening", "hook", CTX))
    assert not any(q.startswith("No clear link") for q in llm.tweet_quality("acme decaf for night owls tonight", "hook", CTX))


def test_details_from_official_product_evidence_count_as_specific():
    ctx = llm.QualityContext(brand="Acme", style=None, campaign_terms=llm._terms("decaf night owls 18-hour steep"))
    assert not any(q.startswith("No clear link") for q in llm.tweet_quality("that 18-hour steep hits different", "hook", ctx))


def test_repeated_hooks_and_ctas_are_flagged_but_common_filler_is_not():
    t = lambda text: Tweet(style="hook", content_style="engaging", text=text, rationale="r")
    out = llm._batch_quality([t("restart your trial now, night owls"), t("cold brew season: restart your trial now"),
                              t("your order is brewing, night owls"), t("your order is waiting by the door")], CTX)
    assert any(q == "Repeats a phrase from tweet 1: “restart your trial”" for q in out[1].quality_issues)
    assert not any(q.startswith("Repeats a phrase") for q in out[3].quality_issues)  # "your order is": filler


def test_reusing_the_briefs_product_details_is_not_a_repeated_hook():
    # Seen live: "A5 dot-grid" and "160 numbered pages" were flagged as repeats, but tweets are meant to use them.
    desc = "Fieldnote makes A5 dot-grid notebooks with 160 numbered pages and lay-flat binding."
    ctx = llm.QualityContext(brand="Fieldnote", style=None, campaign_terms=set(), detail_phrases=llm._content_runs(desc))
    t = lambda text: Tweet(style="hook", content_style="engaging", text=text, rationale="r")
    out = llm._batch_quality([t("Your week, laid out on an A5 dot-grid page. 160 numbered pages wait."),
                              t("Open the A5 dot-grid notebook tonight. All 160 numbered pages are yours."),
                              t("Open the notebook tonight, start your weekly review now"),
                              t("Coffee first, then start your weekly review now")], ctx)
    assert not any(q.startswith("Repeats a phrase") for q in out[1].quality_issues)  # product details: fine
    assert any(q.startswith("Repeats a phrase from tweet 3") for q in out[3].quality_issues)  # a repeated CTA: flagged


# ---------- 3. uncertainty and confidence ----------

def test_no_brand_writing_means_no_confidence_and_a_suggested_voice(monkeypatch):
    client, fake = setup(monkeypatch)  # research not configured: no posts, no examples
    body = client.post("/generate", json=VALID_INPUT).json()
    assert body["voice_profile"]["voice_confidence"] == "none" and body["voice_source"] == "brief"
    assert '"voice_confidence": "none"' in fake.prompt("generation")
    assert "describe a\n  suggested voice that fits the brief" in llm.ANALYSIS_PROMPT


def test_confidence_levels_follow_the_evidence():
    traits = lambda basis: analysis_payload()["traits"] and [
        {"name": n, "description": "d", "basis": basis} for n in ("a", "b", "c")]
    profile = lambda basis: llm.VoiceProfile(summary=["a", "b", "c"], tone="t", audience="a", themes=["x"],
                                             traits=traits(basis), writing_patterns=[{"pattern": "p"}, {"pattern": "q"}])
    assert llm._voice_confidence([], profile("evidence")) == "none"
    assert llm._voice_confidence(["a", "b"], profile("evidence")) == "low"  # too few posts
    assert llm._voice_confidence(["a", "b", "c", "d"], profile("inferred")) == "low"  # no evidence-backed traits
    assert llm._voice_confidence(["a", "b", "c", "d"], profile("evidence")) == "high"


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("data, brand, expected", [
    ({"voice_source": "brief", "voice_profile": {"voice_confidence": "none"}}, "Acme", "Suggested voice (from your brief)"),
    ({"voice_source": "research", "voice_profile": {"voice_confidence": "low"}}, "Acme", "How Acme sounds (limited evidence)"),
    ({"voice_source": "research", "voice_profile": {"voice_confidence": "high"}}, "Acme", "How Acme sounds"),
    ({"voice_source": "user_examples", "voice_profile": {"voice_confidence": "low"}}, "", "Voice from your example tweets (limited evidence)"),
    ({"voice_source": "user_preferences", "voice_profile": {"voice_confidence": "none"}}, "", "Your chosen voice"),
])
def test_ui_never_presents_low_confidence_as_fact(data, brand, expected):
    assert run_review_js(f"R.voiceTitle({json.dumps(data)}, {json.dumps(brand)})") == expected


def test_no_brand_name_still_skips_scraping(monkeypatch):
    payload = {k: SCHOOLS_REQ[k] for k in ("campaign_objective", "description")} | {"example_tweets": EXAMPLES}
    router = use_router(monkeypatch, routes())
    install_fake_groq(monkeypatch)
    r = post_generate(make_settings(), payload)
    assert r.status_code == 200 and router.requests == []


# ---------- 4. example tweets ----------

@pytest.mark.parametrize("examples, ok", [
    (EXAMPLES, True), (EXAMPLES + ["Fourth one.", "Fifth one."], True), ([], True), (["  ", ""], True),
    (EXAMPLES[:2], False), (EXAMPLES + ["4.", "5.", "6."], False), (EXAMPLES[:2] + ["x" * 281], False),
    (EXAMPLES[:2] + [EXAMPLES[0].upper()], False),
])
def test_example_tweet_limits(examples, ok):
    if ok:
        req = TweetRequest(campaign_objective="Launch decaf", description="Acme decaf cold brew.", example_tweets=examples)
        assert req.example_tweets in (None, examples) and (req.example_tweets is None) == (not any(x.strip() for x in examples))
    else:
        with pytest.raises(ValueError):
            TweetRequest(campaign_objective="Launch decaf", description="Acme decaf cold brew.", example_tweets=examples)


def test_examples_reach_analysis_and_generation_and_count_as_evidence(monkeypatch):
    analysis = analysis_payload(evidence=[{"source_id": "EX", "quote": EXAMPLES[0]}, {"source_id": "EX", "quote": EXAMPLES[2]}])
    client, fake = setup(monkeypatch, analysis=analysis, generation=clean_payload())
    body = client.post("/generate", json=VALID_INPUT | {"example_tweets": EXAMPLES}).json()
    a, g = fake.prompt("analysis"), fake.prompt("generation")
    assert "<example_tweets>\n[EX]" in a and all(x in a for x in EXAMPLES)
    assert all(x in g.split("<voice_examples>")[1].split("</voice_examples>")[0] for x in EXAMPLES)
    trait = body["voice_profile"]["traits"][0]
    assert trait["basis"] == "evidence" and {e["source_id"] for e in trait["evidence"]} == {"EX"}
    assert body["voice_source"] == "user_examples"


def test_examples_replace_the_voice_questions(monkeypatch):
    use_router(monkeypatch, routes(posts={"data": []}))  # brand named, no usable posts
    install_fake_groq(monkeypatch)
    assert post_generate(make_settings(), SCHOOLS_REQ).status_code == 409  # without examples: ask
    research._cache.clear()
    r = post_generate(make_settings(), SCHOOLS_REQ | {"example_tweets": EXAMPLES})
    assert r.status_code == 200 and r.json()["voice_source"] == "user_examples"


def test_copying_an_example_is_flagged():
    ctx = llm.QualityContext(brand="Acme", style=None, campaign_terms=llm._terms("decaf night"), examples=EXAMPLES)
    assert any(q.startswith("Copies an example post") for q in llm.tweet_quality(EXAMPLES[0], "hook", ctx))
    assert not any(q.startswith("Copies") for q in llm.tweet_quality("night owls deserve decaf with a backbone", "hook", ctx))


def test_without_examples_the_analysis_is_unchanged(monkeypatch):
    client, fake = setup(monkeypatch)
    client.post("/generate", json=VALID_INPUT)
    assert "<example_tweets>" not in fake.prompt("analysis")


def test_frontend_has_an_optional_examples_input():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="example_tweets"' in html and "Optional" in html.split('id="example_tweets"')[0][-300:]
    assert "example_tweets: exampleLines(" in js and "LIMITS.example_tweets" in js


# ---------- 5. exact count, character limit ----------

def test_count_and_length_use_the_final_text(monkeypatch):
    gen = clean_payload()
    long_text = ("night owls get decaf cold brew steeped slow for the long shift ahead and the quiet hours after "
                 * 3)[:280].strip()
    gen["tweets"][0]["text"] = "   " + long_text + "   "  # padded beyond 280; the final (stripped) text is what counts
    gen["tweets"][1]["text"] = "z " * 141  # 282 characters even after stripping: invalid, dropped
    client, _ = setup(monkeypatch, generation=gen)
    tweets = client.post("/generate", json=VALID_INPUT).json()["tweets"]
    texts = {t["text"] for t in tweets}
    assert len(tweets) == 10 and all(len(t["text"]) <= 280 for t in tweets)
    assert long_text in texts and len(long_text) <= 280 and gen["tweets"][1]["text"].strip() not in texts


# ---------- 6. distinct voices across brands ----------

def test_different_brands_get_different_voice_instructions(monkeypatch):
    formal = ["Introducing our Spring Collection. #Style #New", "Shop the Summer Edit today. #Style",
              "Our Autumn Lookbook is here. #Fashion", "Discover Winter Essentials now. #Style #Gifts"]
    prompts = {}
    for name, posts in (("memes", MEMES), ("formal", formal)):
        req = TweetRequest(campaign_objective="Launch the new season", description="A brand with a new season.",
                           example_tweets=None)
        profile = llm.VoiceProfile(summary=["a", "b", "c"], tone=f"{name} tone", audience="a", themes=["x"],
                                   traits=[{"name": f"{name} {i}", "description": "d"} for i in range(3)],
                                   writing_patterns=[{"pattern": "p"}, {"pattern": "q"}])
        prompts[name] = llm.build_generation_prompt(req, profile, {}, posts, research.observed_style(posts))
    m, f = prompts["memes"], prompts["formal"]
    assert "Start tweets in lowercase" in m and "Start tweets in lowercase" not in f
    assert "Hashtags are part of the style" in f and "Don't use hashtags" in m
    assert MEMES[0] in m and MEMES[0] not in f and formal[0] in f
    assert '"memes tone"' in m and '"formal tone"' in f
    for word in ("duolingo", "owl", "streak"):  # no Duolingo default leaks into another brand's instructions
        assert word not in f.lower()
