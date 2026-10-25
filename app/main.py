from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings, missing_groq_key_message
from app.llm import LLMError, analyze_voice, write_tweets
from app.research import ResearchResult, gather_research, remember_research
from app.schemas import VOICE_QUESTIONS, HealthResponse, TweetRequest, TweetResponse

app = FastAPI(title=get_settings().app_name, version="0.4.0")


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name, groq_configured=settings.groq_configured)


def _needs_voice_input(req: TweetRequest, settings: Settings, research: ResearchResult | None) -> HTTPException:
    """Fallback: there is no brand to research, or research ran but none of the brand's own X posts could be used.
    Ask the user how to sound instead of guessing; research (if any) is kept so the follow-up doesn't spend credits."""
    if research is not None:
        remember_research(req, settings, research)
    message = (f"We couldn't find usable X posts from {req.brand_name}, so we can't learn its voice from research."
               if req.brand_name else "Without a brand name there's nothing to research.")
    return HTTPException(status_code=409, detail={
        "code": "voice_input_needed",
        "message": f"{message} Answer a few quick questions and we'll write in the style you choose.",
        "questions": VOICE_QUESTIONS,
    })


def _from_answers(message: str) -> str:
    """Research messages say the voice was inferred from the brief; in fallback mode it came from the user's answers."""
    for old, new in (("the voice profile is inferred from your brief", "the voice comes from your answers, not research"),
                     ("The voice profile is inferred from your brief", "The voice comes from your answers, not research")):
        if old in message:
            return message.replace(old, new)
    return f"{message} The voice comes from your answers, not research."


@app.post("/generate", response_model=TweetResponse)
def generate(req: TweetRequest, settings: Settings = Depends(get_settings)) -> TweetResponse:
    # Skip research (and its credits) when generation can't run anyway.
    if settings.missing_groq_stages:  # name exactly which stage's key is missing
        raise HTTPException(status_code=503, detail=missing_groq_key_message(settings.missing_groq_stages))

    # The user's own example tweets are voice evidence: with them, there's no need to ask how to sound.
    has_examples = bool(req.example_tweets)
    if not req.brand_name and req.voice_preferences is None and not has_examples:
        raise _needs_voice_input(req, settings, None)  # no brand: no research at all, so ask how to sound

    research = gather_research(req, settings)  # 1. research (best-effort, credit-guarded; skipped without a brand)
    # Fallback only applies when research actually looked for the brand (not when it's unconfigured or skipped).
    looked = bool(req.brand_name) and not research.skipped_reason
    if looked and not research.posts and req.voice_preferences is None and not has_examples:
        raise _needs_voice_input(req, settings, research)  # no posts at all: ask before spending a Groq call
    try:
        profile, review = analyze_voice(req, settings, research)  # 2. voice analysis + source review
        has_voice = any(s.kind == "x_posts" and s.id in review.accepted for s in research.sources())
        if looked and not has_voice and req.voice_preferences is None and not has_examples:
            raise _needs_voice_input(req, settings, research)  # posts existed but weren't the brand's own
        tweets = write_tweets(req, settings, profile, research, review)  # 3. content that applies it
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    report = research.report(review)
    if has_voice:
        voice_source = "research"
    elif has_examples:
        voice_source = "user_examples"
    elif req.voice_preferences is not None:
        voice_source = "user_preferences"
        report.message = _from_answers(report.message)
    else:
        voice_source = "brief"
    return TweetResponse(
        brand_voice_summary=profile.summary,
        voice_source=voice_source,
        voice_profile=profile,
        research=report,
        tweets=tweets,
    )


# Serve the static frontend from the same origin (registered last so API routes take precedence).
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
