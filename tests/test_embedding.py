"""Unit tests for embedding text, the fake embedder, and the background
hook's failure isolation. No database, no OpenAI, no Nylas."""

import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.database.models import EMBEDDING_DIM, EmailRecord
from app.services.embedding import (
    MAX_EMBEDDING_CHARS,
    FakeEmbedder,
    build_embedding_text,
)
from app.workflows.background import run_triage_safely
from app.workflows.triage_workflow import TriageOutcome

EVENT = json.loads(
    (Path(__file__).parent / "fixtures" / "event_normal.json").read_text()
)
SECRET = "synthetic-secret-content"


def _record(body: str = "Synthetic body.", sender_name: str | None = "Test Sender"):
    return EmailRecord(
        id=1,
        subject="Synthetic subject",
        sender_name=sender_name,
        sender_email="sender@example.com",
        received_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        body=body,
    )


def test_embedding_text_layout():
    assert build_embedding_text(_record()) == (
        "Subject: Synthetic subject\n"
        "From: Test Sender <sender@example.com>\n"
        "Date: 2026-01-02T03:04:05+00:00\n\n"
        "Synthetic body."
    )


def test_embedding_text_without_sender_name():
    assert "From: sender@example.com\n" in build_embedding_text(_record(sender_name=None))


def test_embedding_text_is_truncated():
    text = build_embedding_text(_record(body="x" * 20_000))

    assert len(text) == MAX_EMBEDDING_CHARS
    assert text.startswith("Subject: Synthetic subject\n")


def test_fake_embedder_is_deterministic_unit_vectors():
    embedder = FakeEmbedder()

    first, other = embedder.embed_texts(["a", "b"])
    [again] = embedder.embed_texts(["a"])

    assert len(first) == EMBEDDING_DIM
    assert first == again
    assert first != other
    assert math.isclose(sum(v * v for v in first), 1.0)
    assert embedder.calls == [["a", "b"], ["a"]]


@pytest.fixture
def background(monkeypatch, caplog) -> MagicMock:
    """Run `run_triage_safely` with triage stubbed to succeed; returns the
    `embed_email` mock."""
    monkeypatch.setattr(logging.getLogger("app.workflows.background"), "disabled", False)
    monkeypatch.setattr("app.workflows.background.session_scope", nullcontext)
    monkeypatch.setattr("app.workflows.background.get_settings", lambda: None)
    monkeypatch.setattr(
        "app.workflows.background.process_raw_event",
        lambda session, payload, settings: TriageOutcome.ACTION_PROPOSED,
    )
    monkeypatch.setattr(
        "app.workflows.background.get_email_by_message_id",
        lambda session, message_id: SimpleNamespace(id=1),
    )
    embed = MagicMock()
    monkeypatch.setattr("app.workflows.background.embed_email", embed)
    caplog.set_level(logging.INFO)
    return embed


def test_saved_email_is_embedded_after_triage(background, fake_embedder):
    assert run_triage_safely("evt-1", EVENT) is TriageOutcome.ACTION_PROPOSED
    background.assert_called_once()
    assert background.call_args.args[2] is fake_embedder


def test_embedding_failure_keeps_triage_outcome_and_logs_type_only(
    background, caplog
):
    background.side_effect = RuntimeError(SECRET)

    assert run_triage_safely("evt-1", EVENT) is TriageOutcome.ACTION_PROPOSED
    assert "Embedding failed for event_id=evt-1: RuntimeError" in caplog.text
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_skipped_event_type_is_not_embedded(background, monkeypatch):
    monkeypatch.setattr(
        "app.workflows.background.process_raw_event",
        lambda session, payload, settings: TriageOutcome.SKIPPED_TYPE,
    )

    run_triage_safely("evt-1", EVENT)

    background.assert_not_called()
