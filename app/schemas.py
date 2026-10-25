from typing import Literal

from pydantic import BaseModel, Field, field_validator

TWEET_COUNT = 10
MAX_TWEET_CHARS = 280

# "evidence": backed by a quote verified against retrieved research text.
# "inferred": derived from the user's brief only (no verified research evidence).
Basis = Literal["evidence", "inferred"]


class VoicePreferences(BaseModel):
    """The user's own answers about how to sound. Asked for only when a brand name was given but none of the brand's
    own X posts could be used; never presented as researched voice."""
    tone: Literal["playful", "friendly", "professional", "bold", "calm"]
    formality: Literal["casual", "polished"]
    length: Literal["short", "medium"]
    emoji_hashtags: Literal["none", "emoji", "hashtags", "both"]
    notes: str | None = Field(default=None, max_length=200, description="Optional: anything to always do or avoid.")

    @field_validator("notes", mode="before")
    @classmethod
    def blank_notes(cls, v: str | None) -> str | None:
        return (v.strip() or None) if isinstance(v, str) else v


# The questions behind VoicePreferences, returned with the 409 "voice_input_needed" response so clients can ask them.
VOICE_QUESTIONS = [
    {"field": "tone", "question": "How should your tweets sound?",
     "options": ["playful", "friendly", "professional", "bold", "calm"]},
    {"field": "formality", "question": "Casual or polished?", "options": ["casual", "polished"]},
    {"field": "length", "question": "Short and punchy, or a bit longer?", "options": ["short", "medium"]},
    {"field": "emoji_hashtags", "question": "Use emoji or hashtags?", "options": ["none", "emoji", "hashtags", "both"]},
    {"field": "notes", "question": "Anything to always do or avoid? (optional)", "options": []},
]


class TweetRequest(BaseModel):
    # No request examples on purpose: Swagger's "Try it out" prefills examples, and a sample campaign is easily
    # mistaken for the user's own input (and then drives what the tweets promote).
    brand_name: str | None = Field(default=None, max_length=100,
                                   description="Optional. Your brand's name. Without it, brand research is skipped.")
    industry: str | None = Field(default=None, max_length=100, description="Optional. Your industry or category.")
    campaign_objective: str = Field(min_length=3, max_length=500,
                                    description="Required. What this campaign should achieve, in your own words.")
    description: str = Field(min_length=10, max_length=3000,
                             description="Required. Your product or brand. Tweets may only state product facts written here.")
    voice_preferences: VoicePreferences | None = Field(
        default=None, description="Optional. Your answers to the voice questions, sent after a 409 voice_input_needed "
                                  "response (no usable posts were found for the brand).")

    @field_validator("brand_name", "industry", mode="before")
    @classmethod
    def blank_to_none(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("campaign_objective", "description", mode="before")
    @classmethod
    def strip_required(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


# ---------- Research ----------

class SourceInsight(BaseModel):
    """An interpretation of a source. It's a verified finding only if `quote` (verbatim from the source) backs it."""
    id: str = Field(examples=["S1.1"])
    text: str = Field(description="The interpretation, in the analyst's words.")
    quote: str | None = Field(default=None, description="Verbatim words from one post/line of the source that show it.")
    basis: Basis = "inferred"


class ResearchSource(BaseModel):
    id: str = Field(examples=["S1"])
    kind: Literal["x_posts", "website"]
    name: str
    url: str | None = None
    user_supplied: bool = Field(default=False, description="The user linked this source in their brief (still verified).")
    excerpts: list[str] = Field(default_factory=list, description="Verbatim snippets of the retrieved content.")
    insights: list[SourceInsight] = Field(default_factory=list, description="Findings (quoted) and inferences drawn from it.")


class ResearchReport(BaseModel):
    status: Literal["completed", "partial", "skipped", "failed"]
    message: str
    sources: list[ResearchSource] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list, description="Retrieved content rejected because it didn't match the brand.")


# ---------- Voice profile ----------

class EvidenceQuote(BaseModel):
    source_id: str
    quote: str


class VoiceTrait(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=300)
    basis: Basis = "inferred"
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class WritingPattern(BaseModel):
    pattern: str = Field(min_length=1, max_length=300)
    basis: Basis = "inferred"
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class VoiceProfile(BaseModel):
    summary: list[str] = Field(min_length=3, max_length=4)
    tone: str = Field(min_length=1, max_length=300)
    audience: str = Field(min_length=1, max_length=300)
    traits: list[VoiceTrait] = Field(min_length=3, max_length=5)
    writing_patterns: list[WritingPattern] = Field(min_length=2, max_length=4)
    themes: list[str] = Field(min_length=1, max_length=5)
    avoid: list[str] = Field(default_factory=list, max_length=5)


# ---------- Tweets ----------

class Tweet(BaseModel):
    style: str = Field(min_length=1, max_length=50, examples=["question"], description="The tweet's format (structure).")
    content_style: Literal["engaging", "promotional", "witty", "informative"] = Field(
        description="What the tweet does for the reader; independent of its format (style).")
    text: str = Field(min_length=1, max_length=MAX_TWEET_CHARS)
    traits: list[str] = Field(default_factory=list, description="Voice-profile trait names this tweet applies.")
    source_ids: list[str] = Field(default_factory=list, description="Research sources that influenced it; empty if inferred.")
    insight: str | None = Field(default=None, description="The verified research finding that shaped the voice (not proof of product claims).")
    insight_quote: str | None = Field(default=None, description="The verbatim source quote behind that finding.")
    rationale: str = Field(min_length=1, max_length=400, description="Why it fits the brand and campaign objective.")
    basis: Basis = Field(
        default="inferred",
        description='"evidence" = a verified research finding shaped the voice AND no unsupported claim was found. It does '
                    "not mean every factual statement was verified; checks are heuristic.")
    unverified_details: list[str] = Field(
        default_factory=list,
        description="Claims or specifics not supported by the user's brief. Non-empty = needs review before posting.")
    quality_issues: list[str] = Field(
        default_factory=list,
        description="Style/quality problems, separate from factual grounding (generic template, off-voice mechanics, "
                    "vague, repeated idea). Non-empty = needs review before posting.")


class TweetResponse(BaseModel):
    brand_voice_summary: list[str] = Field(min_length=3, max_length=4)
    voice_source: Literal["research", "user_preferences", "brief"] = Field(
        default="brief", description='Where the voice came from: "research" = the brand\'s own X posts; '
                                     '"user_preferences" = your answers (not research); "brief" = inferred from your brief.')
    voice_profile: VoiceProfile
    research: ResearchReport
    tweets: list[Tweet] = Field(min_length=TWEET_COUNT, max_length=TWEET_COUNT)


class HealthResponse(BaseModel):
    status: str
    app: str
    groq_configured: bool
