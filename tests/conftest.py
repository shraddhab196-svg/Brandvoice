"""Shared test helpers: a fake Groq client that answers the analysis and generation steps separately."""

import json
from types import SimpleNamespace

import pytest

from app import llm


def analysis_payload(evidence=None, insights=None, relevance=None, **overrides) -> dict:
    data = {
        "summary": ["Warm, wry and calm.", "Built for night owls.", "Themes: craft and late-night rituals."],
        "tone": "Warm, wry and unhurried.",
        "audience": "Night-shift workers, creatives and students.",
        "traits": [
            {"name": "Dry wit", "description": "Understated, knowing jokes.", "evidence": evidence or []},
            {"name": "Warmth", "description": "Talks like a friend who is also awake.", "evidence": []},
            {"name": "Craft pride", "description": "Quietly proud of the brewing process.", "evidence": []},
        ],
        "writing_patterns": [
            {"pattern": "Short sentences with a twist at the end.", "evidence": []},
            {"pattern": "Rarely uses emojis.", "evidence": []},
        ],
        "themes": ["late nights", "small-batch craft"],
        "avoid": ["hype", "hustle-culture cliches"],
        "source_relevance": relevance if relevance is not None else [
            {"source_id": "S1", "relevant": True, "reason": "The brand's own account."},
            {"source_id": "S2", "relevant": True, "reason": "The brand's own site."},
        ],
        "source_insights": insights or [],
    }
    data.update(overrides)
    return data


VARIANTS = ["dawn", "dusk", "midnight", "moonrise", "sunrise", "twilight", "starlight", "daybreak", "nightfall",
            "evening", "noon", "morning", "lamplight", "afterglow"]


def tweets_payload(n=10, text_len=None, traits=("Dry wit",), insight_ids=None, claims=None, text=None, **overrides) -> dict:
    """n candidate tweets. Texts are always distinct (the app rejects duplicates and near-duplicates): when `text` is
    given, tweet 1 uses it exactly and the others append hashtags unique to them (ignored by claim checks)."""
    tweets = []
    for i in range(n):
        if text_len:
            body = "x" * text_len
        elif text is not None:
            # Four tags unique to each variant keep word overlap well below the near-duplicate threshold.
            body = text if i == 0 else f"{text} #v{i}a #v{i}b #v{i}c #v{i}d"
        else:
            body = f"Tweet about {VARIANTS[i % len(VARIANTS)]} for night owls."
        tweet = {
            "style": llm.TWEET_STYLES[i % len(llm.TWEET_STYLES)],
            "content_style": list(llm.CONTENT_STYLES)[i % len(llm.CONTENT_STYLES)],  # 12 candidates: 3 per style
            "text": body,
            "traits": list(traits),
            "insight_ids": list(insight_ids or []),
            "claims": list(claims or []),
            "rationale": "Uses the brand's dry wit to sell the launch to night-shift workers.",
        }
        tweet.update(overrides)
        tweets.append(tweet)
    return {"tweets": tweets}


class FakeGroq:
    """Stands in for client.chat.completions; picks the reply by which step's system prompt was sent."""

    def __init__(self):
        self.analysis: object = analysis_payload()
        self.summary_repair: object = {"summary": ["Repaired tone bullet.", "Repaired audience bullet.",
                                                   "Repaired themes bullet."]}
        self.field_repair: object = {"themes": ["late nights", "small-batch craft"]}
        self.generation: object = tweets_payload()
        self.generation_replies: list[object] = []  # optional per-call generation replies (used first, in order)
        self.calls: list[dict] = []

    @staticmethod
    def _stage(system: str) -> str:
        return {llm.ANALYSIS_PROMPT: "analysis", llm.SUMMARY_REPAIR_PROMPT: "summary_repair",
                llm.FIELD_REPAIR_PROMPT: "field_repair"}.get(system, "generation")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        stage = self._stage(kwargs["messages"][0]["content"])
        reply = self.generation_replies.pop(0) if stage == "generation" and self.generation_replies else getattr(self, stage)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def prompt(self, stage: str) -> str:
        return next(c["messages"][1]["content"] for c in self.calls if self._stage(c["messages"][0]["content"]) == stage)

    @property
    def stages(self) -> list[str]:
        return [self._stage(c["messages"][0]["content"]) for c in self.calls]


def install_fake_groq(monkeypatch) -> FakeGroq:
    fake = FakeGroq()
    monkeypatch.setattr(llm, "get_client", lambda *a: SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    return fake


@pytest.fixture
def groq_fake(monkeypatch) -> FakeGroq:
    return install_fake_groq(monkeypatch)


# Voice answers a user gives after a 409 "voice_input_needed" (brand named, but no usable X posts).
VOICE_PREFS = {"tone": "playful", "formality": "casual", "length": "short", "emoji_hashtags": "none"}


def with_prefs(payload: dict, **prefs) -> dict:
    return {**payload, "voice_preferences": {**VOICE_PREFS, **prefs}}
