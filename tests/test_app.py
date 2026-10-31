"""App factory: docs exposure per ENVIRONMENT and the startup safety log."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings

DOCS_PATHS = ["/docs", "/redoc", "/openapi.json"]

# `app.main` builds its module-level app (reading settings) on import.
REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_MODEL": "gpt-test",
    "NYLAS_API_KEY": "nylas-test",
    "NYLAS_CLIENT_ID": "client-test",
    "NYLAS_GRANT_ID": "grant-test",
    "WEBHOOK_SECRET": "super-secret",
    "SERVER_URL": "http://localhost:8000",
    "DATABASE_URL": "postgresql://user:pass@localhost/db",
    "APP_API_KEY": "app-test",
    "PROCESSING_MODE": "test",
}

SECRETS = {
    "openai_api_key": "sk-secret-openai",
    "nylas_api_key": "nylas-secret-key",
    "webhook_secret": "webhook-secret-value",
    "app_api_key": "app-secret-key",
    "database_url": "postgresql://user:db-secret-pw@localhost/db",
}


def _settings(**overrides) -> Settings:
    values = dict(
        openai_model="gpt-test",
        nylas_client_id="client-test",
        nylas_grant_id="grant-test",
        server_url="http://localhost:8000",
        processing_mode="test",
        **SECRETS,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize("path", DOCS_PATHS)
def test_docs_are_served_locally(path):
    from app.main import create_app

    client = TestClient(create_app(_settings(environment="local")))

    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", DOCS_PATHS)
def test_docs_are_disabled_in_production(path):
    from app.main import create_app

    client = TestClient(create_app(_settings(environment="production")))

    assert client.get(path).status_code == 404


def test_production_app_still_serves_health():
    from app.main import create_app

    client = TestClient(create_app(_settings(environment="production")))

    assert client.get("/health").json() == {"status": "ok"}


def test_startup_logs_safety_flags_without_secrets(monkeypatch, caplog):
    from app.main import create_app

    settings = _settings(environment="production", dry_run=True)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr(logging.getLogger("app.main"), "disabled", False)

    with caplog.at_level(logging.INFO, logger="app.main"), TestClient(
        create_app(settings)
    ):
        pass

    assert "ENVIRONMENT=production" in caplog.text
    assert "PROCESSING_MODE=test" in caplog.text
    assert "DRY_RUN=True" in caplog.text
    assert "PROCESS_EVENTS_INLINE=False" in caplog.text
    for secret in SECRETS.values():
        assert secret not in caplog.text
    assert "db-secret-pw" not in caplog.text
