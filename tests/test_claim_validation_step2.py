"""Step 2 claim validation: false positives seen live (reflective "We…" lines, figurative "the only way", product
terminology flagged as a repeated hook) and false negatives (outcome, performance and evidence claims). Texts come
from live runs or mirror them; the rules are brand-independent (two unrelated briefs below). Offline only."""

import pytest

from app import llm
from app.schemas import Tweet

NOTEBOOK = ("Fieldnote\nStationery\nGet remote workers to try paper planning for their weekly reviews\n"
            "Fieldnote makes A5 dot-grid notebooks with 160 numbered pages and lay-flat binding, designed for weekly "
            "planning and reviews.")
LANGUAGE = ("Duolingo\nEducation technology\nGet lapsed learners to come back and restart their daily streak\n"
            "Duolingo is a free app for learning languages, math and music through short, game-like lessons, with "
            "daily streaks and reminders to keep learners on track.")


def details(text, grounding=NOTEBOOK, brand="Fieldnote"):
    return llm.unverified_details(text, grounding, brand=brand)


# ---------- false positives: reflective and figurative language ----------

@pytest.mark.parametrize("text", [
    "We often ask how a simple notebook can shape remote work habits.",  # seen live (flagged before)
    "We notice how quiet Sunday evenings feel once the laptop is closed.",
    "We wonder where last week's ideas went.",
    "We hear remote workers whisper that paper is too slow.",  # seen live
    "It's a myth that digital calendars are the only way to track a remote week.",  # seen live (flagged before)
    "Let ideas settle faster than a notification ping.",  # figurative comparison: no measurable outcome
])
def test_reflective_and_figurative_lines_are_not_claims(text):
    assert details(text) == [], details(text)


# ---------- unsupported claims stay flagged (and new ones are caught) ----------

@pytest.mark.parametrize("text, grounding, brand", [
    # seen live: an invented outcome hidden in a reflective opening
    ("We notice remote teams that set a weekly paper review report higher focus scores.", NOTEBOOK, "Fieldnote"),
    ("Paper reviews boost productivity for remote teams.", NOTEBOOK, "Fieldnote"),
    ("Writing by hand improves memory, so your weekly review sticks.", NOTEBOOK, "Fieldnote"),
    ("Studies show handwritten plans beat digital ones.", NOTEBOOK, "Fieldnote"),
    ("Remote teams using paper report better results every week.", NOTEBOOK, "Fieldnote"),
    ("Planning on paper is twice as fast as typing.", NOTEBOOK, "Fieldnote"),
    ("daily streaks boost retention by keeping practice consistent.", LANGUAGE, "Duolingo"),  # seen live earlier
    ("learners who keep streaks report higher scores.", LANGUAGE, "Duolingo"),
    ("Fieldnote is the only way to plan a week that sticks.", NOTEBOOK, "Fieldnote"),  # about the brand
    ("the only app that makes math feel like a game", LANGUAGE, "Duolingo"),
    ("We added offline sync last week.", NOTEBOOK, "Fieldnote"),  # plain company claim, not reflective
])
def test_unsupported_outcome_performance_evidence_and_company_claims_are_flagged(text, grounding, brand):
    assert details(text, grounding, brand), text


@pytest.mark.parametrize("text, grounding", [
    ("Lay-flat binding improves focus during your weekly review.",
     NOTEBOOK + "\nLay-flat binding improves focus during reviews."),
    ("Short lessons boost retention.", LANGUAGE + "\nShort lessons boost retention."),
])
def test_outcome_claims_stated_in_the_evidence_are_supported(text, grounding):
    assert details(text, grounding) == [], details(text, grounding)


# ---------- false positives: unavoidable product terminology in repeated phrases ----------

def batch(texts, brief_parts):
    ctx = llm.QualityContext(brand="Fieldnote", style=None, campaign_terms=set(),
                             detail_phrases=llm._content_runs("\n".join(brief_parts)))
    return llm._batch_quality([Tweet(style="hook", content_style="engaging", text=t, rationale="r") for t in texts], ctx)


def test_brief_terminology_is_not_a_repeated_hook_even_when_inflected():
    objective = "Get remote workers to try paper planning for their weekly reviews"
    desc = ("Fieldnote makes A5 dot-grid notebooks with 160 numbered pages and lay-flat binding, designed for weekly "
            "planning and reviews.")
    out = batch([  # seen live: "dot grid notebook", "lay flat design", "a weekly review" were flagged
        "Start your weekly review with a fresh dot-grid notebook and a quiet desk.",
        "A dot-grid notebook keeps a weekly review honest, page by page.",
        "Its lay-flat design stays open on the desk.",
        "The lay-flat design means no wrestling the spine.",
    ], [objective, desc])
    assert not any(q.startswith("Repeats a phrase") for t in out for q in t.quality_issues), [t.quality_issues for t in out]


def test_campaign_objective_wording_is_treated_as_terminology_end_to_end(monkeypatch):
    # The objective's own wording ("weekly reviews") reused across tweets is not a repeated hook (whole pipeline).
    from conftest import tweets_payload
    from test_generate import setup
    gen = tweets_payload(n=12)
    for t in gen["tweets"]:
        if t["style"] == "thread opener":
            t["text"] += " 1/"
    gen["tweets"][0]["text"] = "Start your weekly review on paper tonight, remote workers."
    gen["tweets"][1]["text"] = "A paper planner makes your weekly review feel calm for remote workers."
    client, _ = setup(monkeypatch, generation=gen)
    body = client.post("/generate", json={
        "brand_name": "Fieldnote", "campaign_objective": "Get remote workers to try paper planning for their weekly reviews",
        "description": "Fieldnote makes A5 dot-grid notebooks with lay-flat binding."}).json()
    second = next(t for t in body["tweets"] if t["text"] == gen["tweets"][1]["text"])
    assert not any(q.startswith("Repeats a phrase") for q in second["quality_issues"]), second["quality_issues"]


def test_genuinely_repeated_hooks_are_still_flagged():
    out = batch(["Do you feel the week slipping through notifications?",
                 "Do we not feel the week stretch thin across screens?"],  # seen live: a real repeat
                ["Get remote workers to try paper planning", "Fieldnote makes A5 dot-grid notebooks."])
    assert any(q.startswith("Repeats a phrase from tweet 1") for q in out[1].quality_issues)
