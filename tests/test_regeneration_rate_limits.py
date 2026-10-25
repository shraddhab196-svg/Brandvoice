"""Regeneration (repair) reliability under Groq's tokens-per-minute limits: a compact repair request, valid tweets
kept, one Retry-After wait at most, bounded attempts, and the safe flagged result when repair can't complete.
Offline only (fake Groq; waiting is recorded, never real)."""

import groq
import httpx
import pytest

from app import llm
from conftest import tweets_payload
from test_generate import VALID_INPUT, setup

SUFFIXES = [" Rated 4.9 by 900 baristas.", " Loved by 70% of students.", " Brewed 3x faster than rivals.",
            " Ships to 12 cities next week."]


def first_batch():
    gen = tweets_payload(n=12)
    for t in gen["tweets"]:
        if t["style"] == "thread opener":
            t["text"] += " 1/"
    for t, s in zip(gen["tweets"][:4], SUFFIXES):
        t["text"] += s  # four distinct unsupported claims -> 8 clean of 12 -> repair needed
    return gen


def replacements(n=3):
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    return {"tweets": [{"content_style": cs, "style": "hook", "text": f"Fresh decaf idea {w} for night owls, slow brewed.",
                        "traits": ["Dry wit"], "insight_ids": [], "claims": [], "rationale": "Fits."}
                       for w, cs in zip(words[:n], ["informative", "witty", "engaging", "promotional", "witty"])]}


def rate_limit(retry_after=None, message="Rate limit reached for model on tokens per minute (TPM)."):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.groq.com/test"))
    return groq.RateLimitError(message, response=resp, body={"error": {"message": message}})


def server_error():
    resp = httpx.Response(500, request=httpx.Request("POST", "https://api.groq.com/test"))
    return groq.InternalServerError("boom", response=resp, body=None)


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(llm, "_sleep", waits.append)
    return waits


def run(monkeypatch, replies):
    client, fake = setup(monkeypatch)
    fake.generation_replies = list(replies)
    r = client.post("/generate", json=VALID_INPUT)
    return r, fake


def assert_ten_valid(r):
    tweets = r.json()["tweets"]
    assert r.status_code == 200 and len(tweets) == 10 and all(len(t["text"]) <= 280 for t in tweets)
    return tweets


# ---------- compact repair, valid tweets kept ----------

def test_repair_is_compact_and_keeps_valid_tweets(monkeypatch, sleeps):
    first = first_batch()
    r, fake = run(monkeypatch, [first, replacements()])
    tweets = assert_ten_valid(r)
    assert fake.stages == ["analysis", "generation", "generation"] and sleeps == []
    gen_call, repair_call = fake.calls[1], fake.calls[2]
    assert repair_call["messages"][0]["content"] == llm.REPAIR_PROMPT  # not the full generation system prompt
    assert repair_call["max_tokens"] == llm.REPAIR_MAX_TOKENS < gen_call["max_tokens"]
    size = lambda c: len(c["messages"][0]["content"]) + len(c["messages"][1]["content"])
    assert size(repair_call) < 0.6 * size(gen_call)  # far smaller request
    user = repair_call["messages"][1]["content"]
    assert "<research_insights>" not in user and "<voice_profile>" not in user  # no unnecessary context
    assert "<rejected_candidates>" in user and SUFFIXES[0].strip() in user and "Number not in your brief" in user
    assert "<already_written>" in user and "Write exactly 3 new tweets." in user  # 10 - 8 clean + 1 spare
    kept = {t["text"] for t in first["tweets"][4:]}
    assert len(kept & {t["text"] for t in tweets}) >= 7  # valid originals preserved, not regenerated
    assert not any(t["unverified_details"] for t in tweets)  # flagged ones replaced


def test_repair_prompt_carries_voice_brief_and_trimmed_evidence():
    req = llm.TweetRequest(brand_name="Acme", campaign_objective="Launch decaf", description="Acme decaf cold brew.")
    profile = llm.VoiceProfile(summary=["a", "b", "c"], tone="dry and calm", audience="a", themes=["x"],
                               traits=[{"name": n, "description": "d"} for n in ("Dry wit", "Calm", "Warm")],
                               writing_patterns=[{"pattern": "Short lines."}, {"pattern": "No hashtags."}])
    src = llm.Source(id="S2", kind="website", name="Website: acme.com", url="https://acme.com/", text="x" * 5000,
                     excerpts=[], segments=[])
    posts = ["night shift? we brewed this.", "slow-steeped. never rushed.", "decaf with a backbone.", "3 a.m. called.",
             "sip, don't gulp.", "one more for the road."]
    prompt = llm.build_repair_prompt(req, profile, posts, llm.observed_style(posts), [src], [], [], 5)
    assert "<brand_input>" in prompt and "Acme decaf cold brew." in prompt  # the facts
    assert '"tone": "dry and calm"' in prompt and "Dry wit" in prompt  # the voice
    assert "<style_rules>" in prompt and "Start tweets in lowercase" in prompt
    assert prompt.count("- ") >= llm.REPAIR_MAX_EXAMPLES and posts[5] not in prompt  # at most 5 examples
    assert "x" * llm.REPAIR_MAX_EVIDENCE_CHARS in prompt and "x" * (llm.REPAIR_MAX_EVIDENCE_CHARS + 1) not in prompt


# ---------- HTTP 429 ----------

@pytest.mark.parametrize("error, expected_wait", [
    (rate_limit(retry_after=6), 6.0),  # Retry-After header
    (rate_limit(message="Rate limit reached ... Please try again in 6.3075s. Need more tokens?"), 6.3075),  # body hint
])
def test_short_rate_limit_waits_once_then_repairs(monkeypatch, sleeps, error, expected_wait):
    r, fake = run(monkeypatch, [first_batch(), error, replacements()])
    tweets = assert_ten_valid(r)
    assert sleeps == [expected_wait + 0.5]  # exactly one wait, as long as Groq asked
    assert fake.stages == ["analysis", "generation", "generation", "generation"]  # one retry, no more
    assert not any(t["unverified_details"] for t in tweets)


@pytest.mark.parametrize("error", [rate_limit(retry_after=30), rate_limit()])  # too long to wait / no hint at all
def test_long_or_unknown_rate_limit_is_not_waited_out(monkeypatch, sleeps, error):
    r, fake = run(monkeypatch, [first_batch(), error])
    tweets = assert_ten_valid(r)
    assert sleeps == [] and fake.stages == ["analysis", "generation", "generation"]  # no blind retry
    assert sum(bool(t["unverified_details"]) for t in tweets) == 2  # safe result: warnings stay visible


def test_repeated_rate_limit_stops_after_one_retry(monkeypatch, sleeps):
    r, fake = run(monkeypatch, [first_batch(), rate_limit(retry_after=5), rate_limit(retry_after=5)])
    tweets = assert_ten_valid(r)
    assert len(sleeps) == 1 and fake.stages == ["analysis", "generation", "generation", "generation"]
    assert sum(bool(t["unverified_details"]) for t in tweets) == 2


# ---------- other API failures ----------

@pytest.mark.parametrize("error", [server_error(), groq.APITimeoutError(request=httpx.Request("POST", "https://x"))])
def test_other_repair_failures_keep_the_safe_result_without_retrying(monkeypatch, sleeps, error):
    r, fake = run(monkeypatch, [first_batch(), error])
    tweets = assert_ten_valid(r)
    assert sleeps == [] and fake.stages == ["analysis", "generation", "generation"]
    assert sum(bool(t["unverified_details"]) for t in tweets) == 2


def test_malformed_repair_output_keeps_the_safe_result(monkeypatch, sleeps):
    r, fake = run(monkeypatch, [first_batch(), "not json"])
    assert sum(bool(t["unverified_details"]) for t in assert_ten_valid(r)) == 2


def test_without_ten_usable_tweets_a_failed_repair_is_still_an_error(monkeypatch, sleeps):
    short = tweets_payload(n=6)
    r, fake = run(monkeypatch, [short, rate_limit(retry_after=30)])
    assert r.status_code == 429  # can't return 10: the existing contract (clear error, never a short list)
    assert fake.stages == ["analysis", "generation", "generation"] and sleeps == []
