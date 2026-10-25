"""UI/UX fixes: accurate review copy, one list item per review reason (safely rendered), results brought into view
with focus, visible example-tweet instructions, a rate-limit retry countdown (429 only), and tag contrast.
Pure logic runs in Node against frontend/review.js; app.js wiring is checked statically. Offline only."""

import re

import pytest

from app.main import FRONTEND_DIR
from test_frontend import NODE, run_review_js

CLAIM = {"text": "Rated 4.9 by 900 users.", "style": "hook",
         "unverified_details": ["Number not in your brief or product evidence: “4.9 by”"], "quality_issues": []}
QUALITY = {"text": "your weekly review, again.", "style": "hook", "unverified_details": [],
           "quality_issues": ["Repeats a phrase from tweet 2: “weekly review”", "Same opening as another tweet: “tweet 1”."]}
CLEAN = {"text": "Night shift, sorted.", "style": "hook", "unverified_details": [], "quality_issues": []}
HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
APP = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
CSS = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
node = pytest.mark.skipif(NODE is None, reason="node not installed")


# ---------- Fix 1: accurate review warnings ----------

@node
def test_copy_reminder_depends_on_the_kind_of_flag():
    claim = run_review_js(f"R.copyOne({CLAIM})")
    quality = run_review_js(f"R.copyOne({QUALITY})")
    assert claim.startswith("[NEEDS REVIEW] ") and "Check against your product brief before publishing." in claim
    assert quality.startswith("[NEEDS REVIEW] ") and "Review before publishing." in quality
    assert "product brief" not in quality  # repetition isn't a fact-check
    assert run_review_js(f"R.copyOne({CLEAN})") == CLEAN["text"]  # clean tweets copy as-is


def test_review_note_covers_claims_and_quality_issues():
    note = re.search(r'id="review-note"[^>]*>(.*?)</p>', HTML, re.S).group(1)
    assert "unsupported claim" in note and "quality issue" in note and "repeated idea" in note


# ---------- Fix 3: one list item per reason, safely rendered ----------

@node
def test_review_reasons_are_separate_items_without_trailing_punctuation():
    assert run_review_js(f"R.reviewReasons({QUALITY})") == [
        "Repeats a phrase from tweet 2: “weekly review”", "Same opening as another tweet: “tweet 1”"]
    assert run_review_js(f"R.reviewReasons({CLEAN})") == []


def test_reasons_render_as_list_items_via_text_only():
    assert '<ul class="needs-review-text"></ul>' in HTML and '<div class="needs-review" role="note" hidden>' in HTML
    render = APP[APP.index('querySelector(".needs-review-text")'):][:200]
    assert "Review.reviewReasons(tweet).map((reason) => el(\"li\"" in render  # el() uses textContent
    assert "innerHTML" not in APP and "insertAdjacentHTML" not in APP  # no HTML parsing of any text


# ---------- Fix 2: results brought into view ----------

def test_results_heading_is_scrolled_to_and_focused_after_success():
    assert '<h3 id="results-title" tabindex="-1">' in HTML  # focusable by script, not in the tab order
    success = APP[APP.index("function renderSuccess"):APP.index("const ERROR_COPY")]
    assert success.index('showState("success")') < success.index("resultsTitle.scrollIntoView")
    assert "resultsTitle.focus({ preventScroll: true })" in success
    assert "prefers-reduced-motion: reduce" in success
    # The heading clears the success view's 14px fade-up offset (seen in a browser: it ended 14px above the fold).
    assert "#results-title { scroll-margin-top: 32px; }" in CSS and "translateY(14px)" in CSS
    # A later loading/error state must not appear off-screen after the page was scrolled to the results.
    assert 'matches || output.getBoundingClientRect().top < 0)' in APP


# ---------- Fix 4: example instructions stay visible ----------

def test_example_instructions_are_permanent_helper_text():
    field = HTML[HTML.index('id="example_tweets"'):HTML.index('id="generate-btn"')]
    assert 'placeholder="Paste 3–5 tweets…"' in field and 'aria-describedby="example_tweets-hint"' in field
    assert '<p class="hint" id="example_tweets-hint">' in field
    assert "One tweet per line. Put multi-line tweets on a single line." in field


# ---------- Fix 5: rate-limit recovery ----------

@node
@pytest.mark.parametrize("status, retry_after, expected", [
    (429, "7", 7), (429, "6.2", 7), (429, None, 20), (429, "abc", 20), (429, "0", 20), (429, "9999", 120),
    (502, "30", 0), (500, None, 0), (0, None, 0), (504, None, 0),  # only rate limits delay the retry
])
def test_retry_delay(status, retry_after, expected):
    arg = "null" if retry_after is None else f'"{retry_after}"'
    assert run_review_js(f"R.retryDelaySeconds({status}, {arg})") == expected


def test_rate_limit_uses_friendly_copy_and_cleans_up_its_countdown():
    error = APP[APP.index("function renderError"):APP.index("function applyServerFieldErrors")]
    assert "status !== 429 && typeof detail" in error  # 429 never shows the provider's raw message
    assert "startRetryCountdown(Review.retryDelaySeconds(status, retryAfter))" in error
    assert 'res.headers.get("retry-after")' in APP  # server-supplied delay preferred
    state = APP[APP.index("function showState"):][:200]
    assert 'if (name !== "error") clearRetryCountdown();' in state  # leaving the error view stops the timer
    start = APP[APP.index("function startRetryCountdown"):][:400]
    assert start.index("clearRetryCountdown()") < start.index("setInterval")  # never two timers at once
    assert "clearInterval(retryTimer)" in APP[APP.index("function clearRetryCountdown"):][:200]


# ---------- accessibility: optional-tag contrast ----------

def _lum(h):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _mix(a, b, p):  # CSS color-mix(in srgb, a p, b)
    return "#" + "".join(f"{round(int(a[i:i + 2], 16) * p + int(b[i:i + 2], 16) * (1 - p)):02x}" for i in (1, 3, 5))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_optional_tag_contrast_meets_wcag_aa(theme):
    assert ".tag-optional { background: var(--bg-2); color: color-mix(in srgb, var(--muted) 85%, var(--ink)); }" in CSS
    block = CSS if theme == "light" else CSS[CSS.index("prefers-color-scheme: dark"):]
    token = lambda name: re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", block).group(1)  # noqa: E731
    fg, bg = _mix(token("muted"), token("ink"), 0.85), token("bg-2")
    hi, lo = sorted([_lum(fg), _lum(bg)], reverse=True)
    assert (hi + 0.05) / (lo + 0.05) >= 4.5
