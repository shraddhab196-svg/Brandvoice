import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.main import FRONTEND_DIR, app

client = TestClient(app)
NODE = shutil.which("node")
OFFER = "Offer or sign-up step not in your brief: “free trial”"  # backend items explain the risk
FLAGGED = {"style": "humor", "text": "Start your free trial today.", "source_ids": ["S1"], "basis": "inferred",
           "unverified_details": [OFFER]}
CLEAN = {"style": "question", "text": "Plan your semester in one workspace.", "source_ids": [], "basis": "inferred",
         "unverified_details": []}


def run_review_js(expr: str):
    """Evaluate an expression against frontend/review.js in Node (the same file the browser loads)."""
    script = f"const R = require({json.dumps(str(FRONTEND_DIR / 'review.js'))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout)


def test_frontend_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert 'id="brief-form"' in r.text


def test_frontend_assets_served():
    assert client.get("/app.js").status_code == 200
    assert "text/css" in client.get("/styles.css").headers["content-type"]


def test_api_routes_not_shadowed_by_frontend():
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/generate", json={}).status_code == 422  # validation, no external calls
    assert client.get("/docs").status_code == 200


def test_review_assets_and_markup_present():
    assert client.get("/review.js").status_code == 200
    html = client.get("/").text
    assert html.index('src="review.js"') < html.index('src="app.js"')  # helpers load before the app
    for marker in ('class="needs-review"', 'id="review-note"', 'id="results-status"', "Needs review"):
        assert marker in html
    assert "against your brief before publishing" in html


def test_research_backed_wording_is_gone():
    for name in ("index.html", "app.js", "review.js"):
        assert "Research-backed" not in (FRONTEND_DIR / name).read_text(encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_copy_text_keeps_review_warnings():
    tweets = json.dumps([CLEAN, FLAGGED])
    copy_all = run_review_js(f"R.copyAll({tweets})")
    lines = copy_all.split("\n\n")
    assert lines[0] == "1. [question] Plan your semester in one workspace."
    assert lines[1].startswith("2. [NEEDS REVIEW] [humor] Start your free trial today.")
    assert f"({OFFER})" in lines[1]

    one = run_review_js(f"R.copyOne({json.dumps(FLAGGED)})")
    assert one.startswith(f"[NEEDS REVIEW] {OFFER}. Check against your product brief") and one.endswith("Start your free trial today.")
    assert run_review_js(f"R.copyOne({json.dumps(CLEAN)})") == CLEAN["text"]


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_review_status_and_labels():
    assert run_review_js(f"R.summary({json.dumps([CLEAN, FLAGGED, CLEAN])})") == {"ready": 2, "review": 1}
    assert run_review_js(f"[R.needsReview({json.dumps(FLAGGED)}), R.needsReview({json.dumps(CLEAN)})]") == [True, False]
    assert run_review_js(f"R.basisLabel({json.dumps(FLAGGED)})") == "Voice from research · S1"
    assert run_review_js(f"R.basisLabel({json.dumps(CLEAN)})") == "Based on voice profile"


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_review_reason_has_no_double_full_stop():
    tweet = {**FLAGGED, "unverified_details": ["Every call, summarized."]}
    assert run_review_js(f"R.copyOne({json.dumps(tweet)})").startswith("[NEEDS REVIEW] Every call, summarized. Check")
