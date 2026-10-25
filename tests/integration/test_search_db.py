"""Integration tests: POST /search against a real test database, with
hand-built embeddings and a stub query embedder (no OpenAI, no Nylas)."""

import math
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import get_settings
from app.database.models import EMBEDDING_DIM, Classification, EmailRecord
from app.database.repository import upsert_embedding
from app.database.session import session_scope

API_KEY = {"X-API-Key": "app-test"}
TABLES = "raw_events, emails, classifications, pending_actions, audit_log"
MODEL = "stub-embedder"


def _vector(angle: float) -> list[float]:
    """Unit vector at `angle` from the query vector (`_vector(0)`), so its
    cosine similarity to the query is exactly cos(angle)."""
    return [math.cos(angle), math.sin(angle)] + [0.0] * (EMBEDDING_DIM - 2)


class StubEmbedder:
    model = MODEL

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_vector(0.0) for _ in texts]


def _store(
    session,
    *,
    key: str,
    angle: float,
    category: str = "other",
    sender_email: str = "sender@example.com",
    sender_name: str | None = "Test Sender",
    received_at: datetime = datetime(2026, 1, 15, tzinfo=timezone.utc),
    model: str = MODEL,
) -> int:
    email = EmailRecord(
        nylas_message_id=f"msg-{key}",
        thread_id=f"thread-{key}",
        grant_id="grant-test",
        sender_name=sender_name,
        sender_email=sender_email,
        recipients=[{"name": None, "email": "candidate@example.com"}],
        subject=f"Subject {key}",
        body=f"Synthetic body for {key}.",
        received_at=received_at,
    )
    session.add(email)
    session.flush()
    session.add(
        Classification(
            email_id=email.id,
            category=category,
            confidence=0.9,
            model="gpt-test",
            prompt_version="v1",
        )
    )
    upsert_embedding(
        session,
        email_id=email.id,
        model=model,
        content_hash=key,
        embedding=_vector(angle),
    )
    return email.id


def _truncate(test_engine) -> None:
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))


@pytest.fixture
def client(app_db_env, test_engine, monkeypatch) -> TestClient:
    # Stub vectors aren't on the real embedder's score scale, so don't
    # apply its calibrated threshold; tests opt into one explicitly.
    monkeypatch.setenv("RAG_MIN_SCORE", "0.0")
    monkeypatch.setattr("app.api.search.get_embedder", lambda: StubEmbedder())
    _truncate(test_engine)
    from app.main import app

    yield TestClient(app)
    _truncate(test_engine)


def _search(client, **body) -> list[dict]:
    response = client.post("/search", json={"query": "q", **body}, headers=API_KEY)
    assert response.status_code == 200
    return response.json()


def test_results_are_ranked_by_score(client):
    with session_scope() as session:
        far = _store(session, key="far", angle=1.2)
        near = _store(session, key="near", angle=0.1)
        mid = _store(session, key="mid", angle=0.6)

    results = _search(client)

    assert [r["email_id"] for r in results] == [near, mid, far]
    assert [r["score"] for r in results] == pytest.approx(
        [math.cos(0.1), math.cos(0.6), math.cos(1.2)], abs=1e-5
    )
    assert results[0] == {
        "email_id": near,
        "score": pytest.approx(math.cos(0.1), abs=1e-5),
        "subject": "Subject near",
        "sender": "Test Sender <sender@example.com>",
        "received_at": "2026-01-15T00:00:00Z",
        "snippet": "Synthetic body for near.",
    }


def test_top_k_limits_results(client):
    with session_scope() as session:
        ids = [_store(session, key=f"e{i}", angle=0.1 * i) for i in range(4)]

    assert [r["email_id"] for r in _search(client, top_k=2)] == ids[:2]


def test_other_models_embeddings_are_ignored(client):
    with session_scope() as session:
        _store(session, key="other-model", angle=0.0, model="another-model")
        kept = _store(session, key="kept", angle=0.5)

    assert [r["email_id"] for r in _search(client)] == [kept]


def test_category_filter_uses_latest_classification(client):
    with session_scope() as session:
        invite = _store(session, key="invite", angle=0.5, category="interview_invite")
        reclassified = _store(
            session, key="recls", angle=0.1, category="interview_invite"
        )
        session.add(
            Classification(
                email_id=reclassified,
                category="rejection",
                confidence=0.9,
                model="gpt-test",
                prompt_version="v1",
                created_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            )
        )
        _store(session, key="other", angle=0.0)

    results = _search(client, category="interview_invite")

    assert [r["email_id"] for r in results] == [invite]


def test_sender_filter_matches_address_or_name_case_insensitively(client):
    with session_scope() as session:
        by_address = _store(
            session, key="a", angle=0.1, sender_email="jane@acme.example"
        )
        by_name = _store(
            session,
            key="b",
            angle=0.2,
            sender_email="hr@x.example",
            sender_name="ACME Talent",
        )
        _store(session, key="c", angle=0.0, sender_email="bob@other.example")

    results = _search(client, sender="acme")

    assert [r["email_id"] for r in results] == [by_address, by_name]


def test_sender_filter_treats_like_wildcards_literally(client):
    with session_scope() as session:
        _store(session, key="a", angle=0.1, sender_email="jane@acme.example")

    assert _search(client, sender="%") == []


def test_date_filters_are_inclusive(client):
    def at(day: int) -> datetime:
        return datetime(2026, 1, day, tzinfo=timezone.utc)

    with session_scope() as session:
        _store(session, key="before", angle=0.0, received_at=at(1))
        start = _store(session, key="start", angle=0.1, received_at=at(10))
        end = _store(session, key="end", angle=0.2, received_at=at(20))
        _store(session, key="after", angle=0.0, received_at=at(30))

    results = _search(
        client, date_from=at(10).isoformat(), date_to=at(20).isoformat()
    )

    assert [r["email_id"] for r in results] == [start, end]


def test_filters_with_no_match_return_empty_list(client):
    with session_scope() as session:
        _store(session, key="a", angle=0.1, category="other")

    assert _search(client, category="rejection") == []


def test_no_embeddings_returns_empty_list(client):
    assert _search(client) == []


def test_min_score_drops_weak_matches(client, monkeypatch):
    with session_scope() as session:
        near = _store(session, key="near", angle=0.1)
        far = _store(session, key="far", angle=1.4)  # cos ~= 0.17
        _store(session, key="opposite", angle=math.pi)  # cos = -1

    assert [r["email_id"] for r in _search(client)] == [near, far]
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")
    get_settings.cache_clear()

    assert [r["email_id"] for r in _search(client)] == [near]
