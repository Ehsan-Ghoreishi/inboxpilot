"""`/health` stays DB-free; `/ready` checks the database with SELECT 1."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings

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


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.main import create_app

    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test",
        openai_model="gpt-test",
        nylas_api_key="nylas-test",
        nylas_client_id="client-test",
        nylas_grant_id="grant-test",
        webhook_secret="super-secret",
        server_url="http://localhost:8000",
        database_url="postgresql://user:pass@localhost/db",
        app_api_key="app-test",
        processing_mode="test",
    )
    yield TestClient(create_app(settings))
    get_settings.cache_clear()


def _engine(execute: MagicMock) -> MagicMock:
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute = execute
    return engine


def test_health_never_touches_the_database(client, monkeypatch):
    get_engine = MagicMock(side_effect=AssertionError("DB used by /health"))
    monkeypatch.setattr("app.api.health.get_engine", get_engine)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    get_engine.assert_not_called()


def test_ready_runs_select_1(client, monkeypatch):
    execute = MagicMock()
    monkeypatch.setattr("app.api.health.get_engine", lambda: _engine(execute))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    (statement,), _ = execute.call_args
    assert str(statement) == "SELECT 1"


def test_ready_returns_503_without_leaking_the_error(client, monkeypatch, caplog):
    error = RuntimeError("connection to postgresql://user:db-secret-pw@host failed")
    monkeypatch.setattr(
        "app.api.health.get_engine", lambda: _engine(MagicMock(side_effect=error))
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
    assert "db-secret-pw" not in response.text
    assert "db-secret-pw" not in caplog.text
