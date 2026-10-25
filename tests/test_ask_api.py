"""Unit tests for POST /ask auth and request validation — no database,
no OpenAI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.schemas.ask import AskRequest, AskResponse
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
def mock_ask():
    with (
        patch(
            "app.api.ask.ask_question",
            return_value=AskResponse(
                answerable=True, answer="Friday.", citations=[], sources=[]
            ),
        ) as ask,
        patch("app.api.ask.get_embedder") as get_embedder,
    ):
        yield {"ask": ask, "get_embedder": get_embedder}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_missing_or_wrong_key_is_401(client, mock_ask, headers):
    response = client.post("/ask", json={"question": "When?"}, headers=headers)

    assert response.status_code == 401
    for mock in mock_ask.values():
        mock.assert_not_called()


def test_valid_key_passes_request_and_min_score(client, mock_ask, monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.3")
    get_settings.cache_clear()

    response = client.post(
        "/ask",
        json={"question": "  When is my interview?  ", "category": "interview_invite"},
        headers=API_KEY,
    )

    assert response.status_code == 200
    assert response.json() == {
        "answerable": True,
        "answer": "Friday.",
        "citations": [],
        "sources": [],
    }
    request, _, min_score = mock_ask["ask"].call_args.args
    assert request.question == "When is my interview?"
    assert request.category.value == "interview_invite"
    assert (request.top_k, min_score) == (5, 0.3)


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"query": "x"},
        {"question": "x", "top_k": 21},
        {"question": "x", "category": "not_a_category"},
        {
            "question": "x",
            "date_from": "2026-02-01T00:00:00Z",
            "date_to": "2026-01-01T00:00:00Z",
        },
    ],
)
def test_invalid_body_is_422(client, mock_ask, body):
    response = client.post("/ask", json=body, headers=API_KEY)

    assert response.status_code == 422
    mock_ask["ask"].assert_not_called()


def test_to_search_request_carries_question_and_filters():
    request = AskRequest(question="When?", top_k=3, sender="acme")

    search = request.to_search_request()

    assert (search.query, search.top_k, search.sender) == ("When?", 3, "acme")
