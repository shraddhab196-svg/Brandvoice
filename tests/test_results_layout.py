"""Results view: tweets first, a concise voice summary, then traits/evidence and sources as collapsible details; labels
that keep evidence, limited evidence, inference and the brief apart. Offline (static markup + Node for review.js)."""

import pytest

from app.main import FRONTEND_DIR
from test_frontend import NODE, run_review_js

HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
SUCCESS = HTML[HTML.index('data-state="success"'):HTML.index("</main>")]


def test_tweets_come_first_then_the_voice_summary_then_collapsible_details():
    order = [SUCCESS.index(m) for m in ('id="results-title"', 'id="tweet-grid"', 'class="voice-card"',
                                        'id="profile-title"', 'id="research-title"')]
    assert order == sorted(order)


@pytest.mark.parametrize("panel", ["profile-panel", "research-panel"])
def test_traits_and_sources_are_collapsible_and_closed_by_default(panel):
    tag = SUCCESS[SUCCESS.index(f'class="panel-block {panel}"') - 20:][:120]
    assert "<details" in tag and " open" not in tag
    head = SUCCESS[SUCCESS.index(f'class="panel-block {panel}"'):]
    assert head.index('<summary class="block-head">') < head.index("</summary>") < head.index("</details>")


def test_distinct_information_is_preserved():
    for marker in ('id="voice-list"', 'id="voice-tone"', 'id="voice-audience"', 'id="trait-grid"', 'id="pattern-list"',
                   'id="theme-chips"', 'id="avoid-chips"', 'id="source-list"', 'id="excluded-list"', 'id="research-status"'):
        assert marker in SUCCESS


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("basis, quotes, source, expected", [
    ("evidence", 2, "research", "Evidence-backed"),
    ("inferred", 1, "research", "Limited evidence"),  # quoted, but a single post: not "from brief"
    ("inferred", 0, "research", "Inferred"),
    ("inferred", 0, "brief", "Inferred from brief"),
    ("inferred", 0, "user_preferences", "From your answers"),
    ("inferred", 0, "user_examples", "Inferred"),
])
def test_labels_separate_evidence_inference_and_the_brief(basis, quotes, source, expected):
    assert run_review_js(f'R.basisTagLabel("{basis}", {quotes}, "{source}")') == expected
