"""Stabilization: issues reported from a live beauty-brand run (tweets all opening with the same word, "lasts all
day"-style performance claims marked ready, "Tag us with your lash selfie" flagged as an invented person). The briefs
below are approximations written for these tests; the rules are brand-independent (two unrelated products). Offline."""

import pytest

from app import llm
from app.schemas import Tweet

MASCARA = ("Lumen\nBeauty\nGet Gen Z beauty fans to try Lumen Lift mascara\n"
           "Lumen Lift Washable Mascara gives limitless length and volume. Its flex brush bends to reach every lash "
           "from root to tip. Formula infused with bamboo extract and fibers. Buildable.")
SNEAKER = ("Stride\nFootwear\nLaunch the Stride Air runner\n"
           "Stride Air is a lightweight running shoe with a breathable knit upper and a recycled foam midsole.")


def details(text, grounding=MASCARA, brand="Lumen"):
    return llm.unverified_details(text, grounding, brand=brand)


# ---------- CTAs and questions are not invented people or stories ----------

@pytest.mark.parametrize("text", [
    "Tag us with your lash selfie.",  # reported live
    "Show us your lifted lashes and tag a friend.",
    "Send us your before-and-after.",
    "Tell us which brush you'd pick.",
    "Who's ready for sky-high lashes?",
])
def test_ctas_and_questions_are_not_invented_people(text):
    assert not any(d.startswith("Possible invented person") for d in details(text)), details(text)


@pytest.mark.parametrize("text", ["Priya switched to Lumen last week.", "Marco runs in Stride Air every morning."])
def test_genuine_invented_people_are_still_flagged(text):
    assert any(d.startswith("Possible invented person") for d in details(text, MASCARA + SNEAKER)), details(text)


# ---------- performance and wear claims need support ----------

@pytest.mark.parametrize("text, grounding", [
    ("Lashes that last all day.", MASCARA),  # reported live
    ("Lasts all day, no flakes.", MASCARA),
    ("Smudge-proof volume from morning to midnight.", MASCARA),
    ("Just one coat and you're done.", MASCARA),
    ("Limitless length, zero clumps.", MASCARA),
    ("Waterproof enough for a rainy day.", MASCARA),
    ("Two coats for full drama.", MASCARA),  # invented number of coats
    ("Long-lasting comfort for every run.", SNEAKER),
    ("Stride Air is waterproof for wet trails.", SNEAKER),
])
def test_unsupported_performance_claims_are_flagged(text, grounding):
    assert details(text, grounding, brand=grounding.split()[0]), text


@pytest.mark.parametrize("text, grounding", [
    ("Lashes that last all day.", MASCARA + "\nLasts all day without flaking."),
    ("Smudge-proof volume.", MASCARA + "\nSmudge-proof, washable formula."),
    ("Just one coat for lift.", MASCARA + "\nOne coat gives instant lift."),
])
def test_performance_claims_stated_in_the_brief_pass(text, grounding):
    assert details(text, grounding) == [], details(text, grounding)


@pytest.mark.parametrize("text", [
    "Get limitless length with the flex brush.",
    "Bamboo extract and fibers for buildable length.",
    "The flex brush bends to reach every lash from root to tip.",
    "Washable formula, buildable volume.",
])
def test_details_from_the_brief_are_not_flagged(text):
    assert details(text) == [], details(text)


# ---------- variety: repeated first words ----------

CTX = llm.QualityContext(brand="Lumen", style=None, campaign_terms=llm._terms(MASCARA))


def tweets(texts, **kw):
    return [Tweet(style=kw.get("style", "hook"), content_style="engaging", text=t, rationale="r") for t in texts]


def test_the_same_first_word_in_many_tweets_is_flagged_beyond_two():
    out = llm._batch_quality(tweets([
        "Maybe it's the flex brush.", "Maybe your lashes needed a lift.", "Maybe limitless length is the answer.",
        "Bamboo extract, buildable length.", "Maybe one more swipe of volume?"]), CTX)
    flags = [[q for q in t.quality_issues if q.startswith("Starts with the same word")] for t in out]
    assert flags[0] == flags[1] == flags[3] == []  # the first two uses are fine
    assert flags[2] and flags[4]  # the third and later uses are flagged
    assert "maybe" in flags[2][0].lower()


def test_selection_prefers_varied_openings_when_equally_clean():
    styles = llm.TWEET_STYLES
    same = [Tweet(style=s, content_style=list(llm.CONTENT_STYLES)[i % 4], text=f"Maybe idea {i} for root to tip length",
                  rationale="r") for i, s in enumerate(styles)]
    varied = [Tweet(style="hook", content_style=cs, text=f"{w} idea for root to tip length", rationale="r")
              for w, cs in zip(["Bamboo", "Flex", "Washable", "Buildable"], llm.CONTENT_STYLES)]
    chosen = llm._select_tweets(same + varied)
    firsts = [t.text.split()[0] for t in chosen]
    assert len(chosen) == 10 and firsts.count("Maybe") <= 6 and all(v in chosen for v in varied)


def test_analysis_prompt_qualifies_campaigns_platform_and_brief_only_traits():
    p = llm.ANALYSIS_PROMPT
    assert "most recent X posts (at most 15)" in p and "not as the brand's\n  permanent voice" in p
    assert "X-specific habits" in p and "platform style" in p
    assert '("suggested by the\n  brief"); never present it as observed' in p
    assert "suggested voice" in p  # no evidence at all: an honest suggestion (unchanged rule)


def test_prompts_ask_for_varied_openings_and_supported_performance():
    for prompt in (llm.GENERATION_PROMPT, llm.REPAIR_PROMPT):
        assert "no more than two tweets may start with the same word" in prompt
        assert "how long it lasts" in prompt
