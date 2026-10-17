import pytest

from app.config import get_settings

REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_MODEL": "gpt-test",
    "NYLAS_API_KEY": "nylas-test",
    "NYLAS_CLIENT_ID": "client-test",
    "NYLAS_GRANT_ID": "grant-test",
    "WEBHOOK_SECRET": "webhook-test",
    "SERVER_URL": "http://localhost:8000",
    "DATABASE_URL": "postgresql://user:pass@localhost/db",
    "APP_API_KEY": "app-test",
    "PROCESSING_MODE": "manual",
}


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Run every test with a clean cwd (no repo .env) and clear cache.

    Changing cwd to an empty tmp_path means `Settings`' relative
    `env_file=".env"` points at a directory with no .env file, so only
    variables set via monkeypatch in each test are visible. This keeps
    tests independent of the developer's local .env contents.
    """
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_required_env(monkeypatch, **overrides):
    values = {**REQUIRED_ENV, **overrides}
    for key, value in values.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_loads_all_variables_from_environment(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("TEST_EMAIL", "candidate@example.com")

    settings = get_settings()

    assert settings.openai_api_key == "sk-test"
    assert settings.openai_model == "gpt-test"
    assert settings.nylas_api_key == "nylas-test"
    assert settings.nylas_client_id == "client-test"
    assert settings.nylas_grant_id == "grant-test"
    assert settings.webhook_secret == "webhook-test"
    assert settings.server_url == "http://localhost:8000"
    assert settings.database_url == "postgresql://user:pass@localhost/db"
    assert settings.app_api_key == "app-test"
    assert settings.processing_mode == "manual"
    assert settings.test_email == "candidate@example.com"


def test_test_email_is_optional(monkeypatch):
    _set_required_env(monkeypatch, TEST_EMAIL=None)

    settings = get_settings()

    assert settings.test_email is None


def test_confidence_threshold_defaults_to_point_seven(monkeypatch):
    _set_required_env(monkeypatch)

    settings = get_settings()

    assert settings.confidence_threshold == 0.7


def test_confidence_threshold_is_overridable(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.85")

    settings = get_settings()

    assert settings.confidence_threshold == 0.85


def test_missing_required_variable_raises_clear_error(monkeypatch):
    _set_required_env(monkeypatch, OPENAI_API_KEY=None)

    with pytest.raises(RuntimeError) as exc_info:
        get_settings()

    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_missing_multiple_required_variables_names_all_of_them(monkeypatch):
    _set_required_env(monkeypatch, OPENAI_API_KEY=None, DATABASE_URL=None)

    with pytest.raises(RuntimeError) as exc_info:
        get_settings()

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "DATABASE_URL" in message


def test_get_settings_is_cached(monkeypatch):
    _set_required_env(monkeypatch)

    first = get_settings()
    second = get_settings()

    assert first is second
