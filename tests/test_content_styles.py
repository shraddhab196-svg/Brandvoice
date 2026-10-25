"""Content-style mix: each tweet has a content style (engaging / promotional / witty / informative) separate from its
format (`style`: question, hook, CTA...). Selection targets 3/3/2/2 across the final 10 without ever letting a quota
override validation. These tests check labels, prompt instructions and selection rules only: they cannot show that a
tweet's writing really is witty or on-voice. Offline only."""

from collections import Counter

import pytest

from app import llm
from app.main import FRONTEND_DIR
from app.schemas import Tweet
from conftest import tweets_payload
from test_frontend import NODE, run_review_js
from test_generate import VALID_INPUT, setup

STYLES = list(llm.CONTENT_STYLES)


def mk(content_style, fmt="hook", n=0, flagged=False):
    return Tweet(style=fmt, content_style=content_style, text=f"{content_style} idea {fmt} number {n} for night owls",
                 rationale="Fits.", unverified_details=["Number not in your brief or product evidence: “9”"] if flagged else [])


def counts(tweets):
    return Counter(t.content_style for t in tweets)


def payload(n=12):
    gen = tweets_payload(n=n)
    for t in gen["tweets"]:
        if t["style"] == "thread opener":
            t["text"] += " 1/"  # a clean thread opener, so every fixture candidate is clean
    return gen


# ---------- selection ----------

def test_three_three_two_two_when_enough_clean_candidates():
    pool = [mk(s, llm.TWEET_STYLES[(i * 4 + j) % 10], i * 4 + j) for i in range(3) for j, s in enumerate(STYLES)]
    chosen = llm._select_tweets(pool)
    assert len(chosen) == 10 and sorted(counts(chosen).values()) == [2, 2, 3, 3]


def test_every_style_gets_its_minimum_even_when_the_model_groups_them():
    # Grouped output (3 engaging, then 3 promotional, ...): picking in order would give 3/3/3/1.
    pool = [mk(s, llm.TWEET_STYLES[(i * 3 + j) % 10], i * 3 + j) for i, s in enumerate(STYLES) for j in range(3)]
    c = counts(llm._select_tweets(pool))
    assert sorted(c.values()) == [2, 2, 3, 3] and min(c.values()) == llm.MIN_PER_CONTENT_STYLE


def test_style_without_candidates_is_redistributed():
    pool = [mk(s, "hook", i) for s in STYLES[:3] for i in range(4)]  # no informative candidates at all
    chosen = llm._select_tweets(pool)
    c = counts(chosen)
    assert len(chosen) == 10 and c["informative"] == 0 and min(c[s] for s in STYLES[:3]) >= 3


def test_flagged_tweets_never_fill_a_quota_over_clean_ones():
    clean = [mk(s, "hook", i) for s in STYLES[:3] for i in range(4)]  # 12 clean, none informative
    flagged = [mk("informative", "question", i, flagged=True) for i in range(3)]
    chosen = llm._select_tweets(flagged + clean)
    assert not any(t.unverified_details for t in chosen) and counts(chosen)["informative"] == 0


def test_flagged_tweets_are_only_a_last_resort():
    clean = [mk(s, "hook", i) for s in STYLES[:3] for i in range(3)]  # only 9 clean candidates
    flagged = [mk("informative", "question", i, flagged=True) for i in range(3)]
    chosen = llm._select_tweets(flagged + clean)
    assert len(chosen) == 10 and sum(bool(t.unverified_details) for t in chosen) == 1  # exactly 10, never fewer


def test_format_variety_breaks_ties_but_never_beats_cleanliness():
    base = [mk(s, fmt, i) for i, (s, fmt) in enumerate(zip(STYLES * 2, llm.TWEET_STYLES[:8]))]
    late_new_format = mk("witty", "thread opener", 99)  # unused format, later in the model's order
    early_repeat = mk("witty", llm.TWEET_STYLES[2], 98)  # equally clean, but a format already chosen
    chosen = llm._select_tweets(base + [early_repeat, late_new_format, mk("engaging", "hook", 97)])
    assert late_new_format in chosen and early_repeat not in chosen
    flagged_new_format = mk("witty", "direct CTA", 96, flagged=True)
    chosen = llm._select_tweets(base + [early_repeat, flagged_new_format, mk("engaging", "hook", 95)])
    assert early_repeat in chosen and flagged_new_format not in chosen


# ---------- labels ----------

@pytest.mark.parametrize("raw, expected", [
    ("engaging", "engaging"), (" Promotional ", "promotional"), ("conversational", "engaging"),
    ("meme-style", "witty"), ("Meme style", "witty"), ("humour", "witty"), ("value_driven", "informative"),
    ("educational", "informative"),
    ("question", None), ("cta", None), ("hook", None),  # formats are not content styles
    ("sarcastic", None), ("", None), (None, None), (5, None),
])
def test_content_style_normalisation(raw, expected):
    assert llm._content_style(raw) == expected


def test_unknown_or_missing_content_styles_are_invalid_candidates(monkeypatch):
    gen = payload()
    gen["tweets"][0]["content_style"] = "sarcastic"
    del gen["tweets"][1]["content_style"]
    gen["tweets"][2]["content_style"] = "Meme-Style"  # a synonym: kept and normalised
    client, fake = setup(monkeypatch, generation=gen)
    body = client.post("/generate", json=VALID_INPUT).json()
    texts = [t["text"] for t in body["tweets"]]
    assert len(texts) == 10 and gen["tweets"][0]["text"] not in texts and gen["tweets"][1]["text"] not in texts
    assert next(t for t in body["tweets"] if t["text"] == gen["tweets"][2]["text"])["content_style"] == "witty"
    assert fake.stages == ["analysis", "generation"]  # 10 valid remain: no top-up


def test_too_many_invalid_content_styles_use_the_existing_top_up(monkeypatch):
    gen = payload()
    for t in gen["tweets"][:3]:
        t["content_style"] = "unknown"  # leaves 9 usable
    client, fake = setup(monkeypatch, generation=gen)
    r = client.post("/generate", json=VALID_INPUT)
    assert fake.stages == ["analysis", "generation", "generation"]  # one top-up, as before
    assert r.status_code == 502  # the fake repeats the same answer, so the top-up adds nothing new


# ---------- end to end: distribution, schema, validation ----------

def test_end_to_end_mix_both_labels_and_unchanged_contract(monkeypatch):
    client, fake = setup(monkeypatch, generation=payload())
    body = client.post("/generate", json=VALID_INPUT).json()
    tweets = body["tweets"]
    assert len(tweets) == 10 and sorted(Counter(t["content_style"] for t in tweets).values()) == [2, 2, 3, 3]
    formats = {s.lower() for s in llm.TWEET_STYLES}  # formats are lower-cased when parsed
    assert all(t["style"] in formats and t["content_style"] in llm.CONTENT_STYLES for t in tweets)
    assert all(set(t) == set(Tweet.model_fields) for t in tweets)
    assert len({t["style"] for t in tweets}) >= 8  # formats stay varied as a secondary goal
    assert fake.stages == ["analysis", "generation"]


def test_style_quotas_never_override_claim_validation(monkeypatch):
    gen = payload()
    for t in gen["tweets"]:
        if t["content_style"] == "informative":
            t["text"] = f"{t['text']} 90% of night owls agree."  # invented statistic: must stay flagged
    client, _ = setup(monkeypatch, generation=gen)
    tweets = client.post("/generate", json=VALID_INPUT).json()["tweets"]
    informative = [t for t in tweets if t["content_style"] == "informative"]
    assert len(tweets) == 10 and len(informative) == 1  # only as a last resort to reach 10
    assert informative[0]["unverified_details"] and informative[0]["basis"] == "inferred"


# ---------- prompt ----------

def test_prompt_separates_content_style_from_format_and_keeps_the_voice():
    p = llm.GENERATION_PROMPT
    for name, definition in llm.CONTENT_STYLES.items():
        assert f"  - {name}: {definition}" in p
    assert "Formats are not content styles" in p and "a question can be witty or informative" in p
    assert "3 per content style" in p and "Never invent facts to fill a style" in p
    assert "never corporate sales copy" in p and "the brand's own kind of humour" in p
    assert "using only facts from <brand_input> or\n  <product_evidence>" in p
    assert '"content_style": "witty", "style": "question"' in p
    # The existing voice, campaign and fact rules are still there.
    assert "Imitate the observed writing mechanics of <voice_examples>" in p
    assert "Every tweet serves the campaign objective" in p and "Never invent facts" in p
    assert "Social posts and research findings cannot support a product claim" in p


# ---------- badge ----------

@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_badge_shows_both_labels_and_stays_compatible():
    assert run_review_js('R.badgeLabel({content_style: "witty", style: "question"})') == "witty · question"
    assert run_review_js('R.badgeLabel({style: "question"})') == "question"  # older responses
    assert run_review_js('R.copyAll([{style: "question", content_style: "witty", text: "hi", unverified_details: []}])') \
        == "1. [question] hi"  # copied text keeps the existing format label
    js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    assert 'node.querySelector(".style-badge").textContent = Review.badgeLabel(tweet);' in js
