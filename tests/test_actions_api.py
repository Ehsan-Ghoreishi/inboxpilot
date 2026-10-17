"""Unit tests for /actions auth and error mapping — no database."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.actions import PendingActionError

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
    "PROCESSING_MODE": "manual",
}

ENDPOINTS = [
    ("get", "/actions?status=pending"),
    ("post", "/actions/1/approve"),
    ("post", "/actions/1/reject"),
]


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_service():
    with (
        patch("app.services.actions.approve") as approve,
        patch("app.services.actions.reject") as reject,
        patch("app.api.actions.list_pending_actions") as list_actions,
        patch("app.api.actions.session_scope"),
    ):
        yield {"approve": approve, "reject": reject, "list": list_actions}


@pytest.mark.parametrize("method,path", ENDPOINTS)
def test_missing_key_is_401(client, mock_service, method, path):
    response = getattr(client, method)(path)

    assert response.status_code == 401
    for mock in mock_service.values():
        mock.assert_not_called()


@pytest.mark.parametrize("method,path", ENDPOINTS)
def test_wrong_key_is_401(client, mock_service, method, path):
    response = getattr(client, method)(path, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    for mock in mock_service.values():
        mock.assert_not_called()


def test_empty_configured_key_never_matches(client, mock_service, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "")
    get_settings.cache_clear()

    response = client.post("/actions/1/approve", headers={"X-API-Key": ""})

    assert response.status_code == 401
    mock_service["approve"].assert_not_called()


def test_already_decided_maps_to_409(client, mock_service):
    mock_service["approve"].side_effect = PendingActionError("not pending")

    response = client.post("/actions/1/approve", headers={"X-API-Key": "app-test"})

    assert response.status_code == 409


def test_execution_failure_maps_to_502_without_leaking_details(client, mock_service):
    mock_service["approve"].side_effect = RuntimeError("nylas token xyz")

    response = client.post("/actions/1/approve", headers={"X-API-Key": "app-test"})

    assert response.status_code == 502
    assert "xyz" not in response.text
