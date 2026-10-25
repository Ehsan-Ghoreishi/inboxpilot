"""Integration tests: email embeddings against a real test database, with
`FakeEmbedder` (no OpenAI) and the triage agent mocked (no Nylas)."""

import importlib.util
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select, text

from app.database.models import EmailEmbedding, EmailRecord
from app.integrations.nylas.webhooks import compute_signature
from app.services.embedding import FakeEmbedder, embed_email
from tests.integration.conftest import REPO_ROOT
from tests.integration.test_triage_db import EVENT, _run

TABLES = "raw_events, emails, classifications, pending_actions, audit_log"


def _email(session, message_id: str = "msg-embed-1") -> EmailRecord:
    record = EmailRecord(
        nylas_message_id=message_id,
        thread_id="thread-embed-1",
        grant_id="grant-test",
        sender_name="Test Sender",
        sender_email="sender@example.com",
        recipients=[{"name": None, "email": "candidate@example.com"}],
        subject="Synthetic subject",
        body="Synthetic body.",
        received_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    session.add(record)
    session.flush()
    return record


def _embeddings(session) -> list[EmailEmbedding]:
    return list(session.scalars(select(EmailEmbedding)))


class FailingEmbedder(FakeEmbedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("synthetic embedding failure")


# --- idempotency -----------------------------------------------------------------


def test_unchanged_email_is_embedded_once(db_session):
    email = _email(db_session)
    embedder = FakeEmbedder()

    assert embed_email(db_session, email, embedder) is True
    assert embed_email(db_session, email, embedder) is False

    assert len(embedder.calls) == 1
    [row] = _embeddings(db_session)
    assert (row.email_id, row.model, row.dim) == (email.id, embedder.model, 1536)


def test_changed_content_replaces_the_embedding(db_session):
    email = _email(db_session)
    embedder = FakeEmbedder()
    embed_email(db_session, email, embedder)
    [before] = _embeddings(db_session)
    old_hash, old_vector = before.content_hash, list(before.embedding)

    email.body = "Edited synthetic body."
    assert embed_email(db_session, email, embedder) is True

    db_session.expire_all()
    [after] = _embeddings(db_session)
    assert after.id == before.id
    assert after.content_hash != old_hash
    assert list(after.embedding) != old_vector


def test_embeddings_are_kept_per_model(db_session):
    email = _email(db_session)
    other = FakeEmbedder()
    other.model = "fake-other-model"

    embed_email(db_session, email, FakeEmbedder())
    embed_email(db_session, email, other)

    assert {row.model for row in _embeddings(db_session)} == {
        "fake-hash-embedder",
        "fake-other-model",
    }


# --- failure isolation through POST /events ----------------------------------------


@pytest.fixture
def client(live_app_client, test_engine, monkeypatch):
    monkeypatch.setattr(
        "app.workflows.triage_workflow.analyze_with_metadata",
        MagicMock(return_value=_run()),
    )
    monkeypatch.setattr(
        "app.integrations.nylas.client.get_client",
        MagicMock(side_effect=AssertionError("Nylas client used")),
    )
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))
    yield live_app_client
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))


def _post(client) -> None:
    body = json.dumps(EVENT).encode("utf-8")
    response = client.post(
        "/events",
        content=body,
        headers={
            "x-nylas-signature": compute_signature(body, "super-secret"),
            "content-type": "application/json",
        },
    )
    assert response.status_code == 200


def _count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def test_event_email_is_embedded_after_triage(client, test_engine, fake_embedder):
    _post(client)

    assert _count(test_engine, "classifications") == 1
    assert _count(test_engine, "email_embeddings") == 1
    assert len(fake_embedder.calls) == 1


def test_embedding_failure_does_not_break_triage(client, test_engine, monkeypatch):
    monkeypatch.setattr(
        "app.workflows.background.get_embedder", lambda: FailingEmbedder()
    )

    _post(client)

    assert _count(test_engine, "emails") == 1
    assert _count(test_engine, "classifications") == 1
    assert _count(test_engine, "pending_actions") == 1
    assert _count(test_engine, "email_embeddings") == 0


# --- backfill script ------------------------------------------------------------


def _load_backfill_script():
    path = REPO_ROOT / "scripts" / "embed_emails.py"
    spec = importlib.util.spec_from_file_location("embed_emails", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stored_emails(app_db_env, test_engine):
    """Three committed emails; the first already has an embedding."""
    from app.database.session import session_scope

    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))
    with session_scope() as session:
        emails = [_email(session, f"msg-backfill-{i}") for i in range(3)]
        embed_email(session, emails[0], FakeEmbedder())
    yield
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))


def test_backfill_embeds_only_emails_lacking_one(stored_emails, test_engine, capsys):
    script = _load_backfill_script()
    embedder = FakeEmbedder()

    assert script.backfill(embedder) == {"embedded": 2, "skipped": 0, "failed": 0}
    assert len(embedder.calls) == 2
    assert _count(test_engine, "email_embeddings") == 3

    assert script.backfill(embedder) == {"embedded": 0, "skipped": 0, "failed": 0}


def test_backfill_counts_failures_and_retries_them_later(
    stored_emails, test_engine, monkeypatch, capsys
):
    script = _load_backfill_script()
    monkeypatch.setattr(script, "get_embedder", lambda: FailingEmbedder())

    script.main()

    assert capsys.readouterr().out == "embedded: 0\nskipped: 0\nfailed: 2\n"
    assert _count(test_engine, "email_embeddings") == 1
    assert script.backfill(FakeEmbedder())["embedded"] == 2
