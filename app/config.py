from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, resolved relative to this file so it works from any cwd
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Stage -> (env var holding its Groq key, human-readable stage name)
GROQ_STAGES = {
    "analysis": ("GROQ_API_KEY", "brand voice analysis"),
    "generation": ("GROQ_API_KEY_BRAND_VOICE", "tweet generation"),
}


def missing_groq_key_message(stages: list[str]) -> str:
    parts = [f"{GROQ_STAGES[s][0]} (used for {GROQ_STAGES[s][1]})" for s in stages]
    verb = "is" if len(parts) == 1 else "are"
    return f"{' and '.join(parts)} {verb} not configured on the server."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Tweet Generator"
    # One Groq key per stage (separate accounts), never used as a fallback for each other.
    # Optional so the app boots without them; SecretStr keeps them out of logs/reprs.
    groq_api_key: SecretStr | None = None  # env GROQ_API_KEY: brand voice analysis only
    groq_api_key_brand_voice: SecretStr | None = None  # env GROQ_API_KEY_BRAND_VOICE: tweet generation only
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_seconds: float = 45.0

    # Optional research providers; generation works without them.
    scrapebadger_api_key: SecretStr | None = None
    firecrawl_api_key: SecretStr | None = None
    research_timeout_seconds: float = 12.0  # per HTTP call
    research_budget_seconds: float = 30.0  # skip remaining research steps after this
    research_cache_ttl_seconds: int = 6 * 60 * 60  # avoid re-spending credits on the same brand
    # Credit guard: max API requests (attempted, incl. failures) per provider per /generate run.
    research_max_scrapebadger_calls: int = 2
    research_max_firecrawl_calls: int = 1

    def groq_key_for(self, stage: str) -> SecretStr | None:
        """The key for one stage only: "analysis" -> GROQ_API_KEY, "generation" -> GROQ_API_KEY_BRAND_VOICE."""
        return self.groq_api_key if stage == "analysis" else self.groq_api_key_brand_voice

    @property
    def missing_groq_stages(self) -> list[str]:
        return [stage for stage in GROQ_STAGES if not _has_value(self.groq_key_for(stage))]

    @property
    def groq_configured(self) -> bool:
        return not self.missing_groq_stages

    @property
    def scrapebadger_configured(self) -> bool:
        return _has_value(self.scrapebadger_api_key)

    @property
    def firecrawl_configured(self) -> bool:
        return _has_value(self.firecrawl_api_key)


def _has_value(secret: SecretStr | None) -> bool:
    return bool(secret and secret.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
