import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("GROQ_MODEL", "GROQ_API_KEY", "GROQ_API_KEY_BRAND_VOICE"):
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path, text: str):
    env = tmp_path / ".env"
    env.write_text(text, encoding="utf-8")
    return env


def test_groq_model_loaded_from_env_file(tmp_path):
    env = write_env(tmp_path, "GROQ_API_KEY=a-key\nGROQ_API_KEY_BRAND_VOICE=g-key\nGROQ_MODEL=openai/gpt-oss-20b\n")
    s = Settings(_env_file=env)
    assert s.groq_model == "openai/gpt-oss-20b"
    assert s.groq_configured
    assert "a-key" not in repr(s) and "g-key" not in repr(s)


def test_groq_model_default_when_unset(tmp_path):
    assert Settings(_env_file=write_env(tmp_path, "")).groq_model == "llama-3.3-70b-versatile"


def test_each_stage_reads_its_own_variable(tmp_path):
    s = Settings(_env_file=write_env(tmp_path, "GROQ_API_KEY=analysis-key\nGROQ_API_KEY_BRAND_VOICE=generation-key\n"))
    assert s.groq_key_for("analysis").get_secret_value() == "analysis-key"
    assert s.groq_key_for("generation").get_secret_value() == "generation-key"
    assert s.missing_groq_stages == []


@pytest.mark.parametrize("contents, missing", [
    ("GROQ_API_KEY=analysis-key\n", ["generation"]),
    ("GROQ_API_KEY_BRAND_VOICE=generation-key\n", ["analysis"]),
    ("GROQ_API_KEY=analysis-key\nGROQ_API_KEY_BRAND_VOICE=   \n", ["generation"]),
    ("", ["analysis", "generation"]),
])
def test_no_cross_fallback_between_keys(tmp_path, contents, missing):
    s = Settings(_env_file=write_env(tmp_path, contents))
    assert s.missing_groq_stages == missing and not s.groq_configured
    present = {"analysis", "generation"} - set(missing)
    for stage in missing:  # a missing stage never borrows the other stage's key
        key = s.groq_key_for(stage)
        assert key is None or not key.get_secret_value().strip()
    for stage in present:
        assert s.groq_key_for(stage).get_secret_value().endswith("-key")


def test_keys_from_os_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-analysis-key")
    monkeypatch.setenv("GROQ_API_KEY_BRAND_VOICE", "env-generation-key")
    s = Settings(_env_file=tmp_path / "missing.env")
    assert s.groq_key_for("analysis").get_secret_value() == "env-analysis-key"
    assert s.groq_key_for("generation").get_secret_value() == "env-generation-key"
