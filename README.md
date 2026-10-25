# AI Tweet Generator — Backend

FastAPI backend. Endpoints: `GET /health` and `POST /generate` (research → voice analysis → 10 tweets with reasoning).

## Run locally (Windows PowerShell)

Run these from the `backend` folder (quote the path because it contains a space):

```powershell
cd "D:\100x Engineers\Brand _Voice\backend"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then paste GROQ_API_KEY (analysis) and GROQ_API_KEY_BRAND_VOICE (generation)
uvicorn app.main:app --reload
```

If PowerShell blocks `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` from cmd.

The key is optional for now; the server starts without it.

## Open the app

With the server running, open **http://127.0.0.1:8000/** — the backend serves the frontend in `../frontend` (plain HTML/CSS/JS, no build step) on the same origin, so no CORS setup is needed.

## Test

- Browser: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs
- PowerShell: `Invoke-RestMethod http://127.0.0.1:8000/health`

Expected response:

```json
{"status": "ok", "app": "AI Tweet Generator", "groq_configured": false}
```

`groq_configured` becomes `true` once both `GROQ_API_KEY` and `GROQ_API_KEY_BRAND_VOICE` are set in `.env` (restart the server after editing `.env`). The key itself is never returned or logged.

### POST /generate

Use Swagger UI (`/docs` → `POST /generate` → Try it out), or PowerShell:

```powershell
$body = @{
  brand_name = "Moonbrew"
  industry = "Coffee"
  campaign_objective = "Launch our new decaf cold brew for late-night workers"
  description = "Small-batch cold brew coffee for night owls."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/generate -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

Each request runs three steps: optional brand research, a voice-analysis Groq call, then a tweet-writing Groq call (two Groq calls total).
The response contains `brand_voice_summary`, `voice_profile` (traits/patterns marked `evidence` or `inferred`), `research` (status, retrieved sources, insights) and `tweets` (each with traits, rationale and any source it drew on).
A trait counts as `evidence` only if its quote appears verbatim in retrieved research; a tweet cites a source only if that source was retrieved and analysed. Without a brand name, research is skipped.

Errors: `503` key not set, `429` Groq rate limit, `504` timeout, `502` Groq error or malformed model output, `422` invalid input.

### Optional brand research

Set `SCRAPEBADGER_API_KEY` and/or `FIRECRAWL_API_KEY` in `.env` to let `/generate` research the brand first. With neither key it behaves exactly as before. Research is best-effort: any failure, timeout or uncertain match is skipped and tweets are generated from your inputs.

- **X posts (ScrapeBadger) = voice evidence:** uses an `x.com/<handle>` link in the description if present; otherwise tries the brand name as a handle, then a user search. Posts are the only evidence for how the brand writes, whatever they're about (memes and off-campaign posts still show the voice). Individual posts are filtered for authenticity/quality (retweets, other authors, older than 3 years, too short, duplicates); up to 8 are kept, originals first, with the brand's own replies filling in when it has fewer than 3 originals. Counts per reason are logged.
- **Website (Firecrawl) = product and company context only:** scrapes one page — a URL/domain in the description, else the profile link of a strongly identified X account, else a search result on the brand's domain (main site preferred over subdomains). Customer testimonials are removed. Website copy is never used as evidence of the social voice.
- **Identity, not campaign topic (no extra API calls):** an X account is trusted outright only if it's strongly identified (exact handle/name with organisation verification, or a profile link to the brand's own domain). Anything else — including links you put in the description — is "uncertain" and is used only if the voice-analysis call confirms it's the brand's official account. Lookalike websites (brand name inside another domain) are rejected if they describe a different business. Rejected/unreachable sources are listed under "Not used". Research results are cached per full brief (brand, industry, objective, description).
- **Evidence:** insights are findings only with a verbatim quote from one post/line; otherwise they're shown as inferences. Tweets cite findings by ID, declare factual claims with an exact supporting quote from the brief or a finding, and are additionally scanned for unsupported numbers, names, customer stories, integrations, platforms, signup/offer mechanics, superlatives and capability claims. Unsupported items are flagged "check before posting".
- **Credits:** each `/generate` run makes at most 2 ScrapeBadger and 1 Firecrawl request (failed attempts count). ScrapeBadger bills 1 credit + 1 per item returned. Successful results are cached in memory for 6 hours per brand. Research is skipped entirely if either Groq key or the brand name is missing.

**Groq keys (one per stage, separate accounts):** `GROQ_API_KEY` is used only for the brand voice analysis call; `GROQ_API_KEY_BRAND_VOICE` is used only for the tweet generation call. Neither is a fallback for the other: if one is missing, `/generate` returns `503` naming the missing key and its stage, before any research or Groq call.

## Automated tests (mocked, no API credits used)

```powershell
pip install -r requirements-dev.txt
python -m pytest
```
