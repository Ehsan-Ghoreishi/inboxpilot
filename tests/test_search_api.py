"""Unit tests for POST /search auth, request validation, and snippets —
no database, no OpenAI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.search import SNIPPET_CHARS, build_snippet
from tests.test_actions_api import REQUIRED_ENV

API_KEY = {"X-API-Key": "app-test"}


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
def mock_search():
    with (
        patch("app.api.search.search_emails", return_value=[]) as search,
        patch("app.api.search.get_embedder") as get_embedder,
        patch("app.api.search.session_scope"),
    ):
        yield {"search": search, "get_embedder": get_embedder}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_missing_or_wrong_key_is_401(client, mock_search, headers):
    response = client.post("/search", json={"query": "interview"}, headers=headers)

    assert response.status_code == 401
    for mock in mock_search.values():
        mock.assert_not_called()


def test_valid_key_passes_defaults_and_min_score(client, mock_search, monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.25")
    get_settings.cache_clear()

    response = client.post("/search", json={"query": "  interview  "}, headers=API_KEY)

    assert response.status_code == 200
    assert response.json() == []
    _, request, _, min_score = mock_search["search"].call_args.args
    assert (request.query, request.top_k, min_score) == ("interview", 5, 0.25)


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 21},
        {"query": "x", "category": "not_a_category"},
        {
            "query": "x",
            "date_from": "2026-02-01T00:00:00Z",
            "date_to": "2026-01-01T00:00:00Z",
        },
        {"query": "x", "unknown": 1},
    ],
)
def test_invalid_body_is_422(client, mock_search, body):
    response = client.post("/search", json=body, headers=API_KEY)

    assert response.status_code == 422
    mock_search["search"].assert_not_called()


def test_snippet_collapses_whitespace():
    assert build_snippet("Hello\n\n  there\tfriend") == "Hello there friend"


def test_snippet_is_truncated():
    snippet = build_snippet("word " * 200)

    assert len(snippet) <= SNIPPET_CHARS
    assert snippet.endswith("…")
