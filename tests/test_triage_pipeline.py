"""Unit tests for the raw-event triage pipeline (`process_raw_event`) and its
never-raising background wrapper. No database, no OpenAI, no Nylas:
persistence is recorded in memory and the agent is a fake."""

import copy
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.agents.email_analyzer import AnalysisRun
from app.config import Settings
from app.database.models import ActionType
from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailCategory,
    SuggestedAction,
    Urgency,
)
from app.workflows.background import run_triage_safely
from app.workflows.triage_workflow import TriageOutcome, process_raw_event

SECRET_SUBJECT = "Synthetic subject that must never be logged"
SECRET_BODY = "Synthetic body that must never be logged"

EVENT = {
    "specversion": "1.0",
    "type": "message.created",
    "source": "/google/emails/realtime",
    "id": "evt-pipeline-1",
    "time": 1706800000,
    "data": {
        "application_id": "app-synthetic",
        "object": {
            "id": "msg-pipeline-1",
            "grant_id": "grant-synthetic",
            "thread_id": "thread-pipeline-1",
            "subject": SECRET_SUBJECT,
            "from": [{"name": "Test Sender", "email": "sender@example.com"}],
            "to": [{"name": "Candidate", "email": "candidate@example.com"}],
            "date": 1706800000,
            "body": SECRET_BODY,
        },
    },
}


def _settings(**overrides) -> Settings:
    values = dict(
        openai_api_key="sk-test",
        openai_model="gpt-test",
        nylas_api_key="nylas-test",
        nylas_client_id="client-test",
        nylas_grant_id="grant-test",
        webhook_secret="super-secret",
        server_url="http://localhost:8000",
        database_url="postgresql://user:pass@localhost/db",
        app_api_key="app-test",
        processing_mode="live",
        confidence_threshold=0.7,
    )
    values.update(overrides)
    return Settings(**values)


def _run(
    category: EmailCategory = EmailCategory.RECRUITER_OUTREACH,
    confidence: float = 0.9,
    suggested_action: SuggestedAction = SuggestedAction.DRAFT_REPLY,
) -> AnalysisRun:
    return AnalysisRun(
        analysis=EmailAnalysis(
            category=category,
            company="Example Corp",
            role="Data Scientist",
            urgency=Urgency.MEDIUM,
            suggested_action=suggested_action,
            reasoning="Synthetic analysis for testing.",
            confidence=confidence,
        ),
        model="gpt-test",
        prompt_version="analyze_v1",
        input_tokens=120,
        output_tokens=40,
        latency_ms=850,
    )


@pytest.fixture(autouse=True)
def _enable_triage_logger(monkeypatch):
    """Alembic's `fileConfig` (run by the integration suite in the same
    process) disables already-created loggers; re-enable ours for caplog."""
    monkeypatch.setattr(logging.getLogger("app.workflows.background"), "disabled", False)


@pytest.fixture(autouse=True)
def no_nylas(monkeypatch):
    """Any Nylas client use in this flow is a bug."""
    forbidden = MagicMock(side_effect=AssertionError("Nylas client used"))
    for module in ("client", "actions", "messages"):
        monkeypatch.setattr(f"app.integrations.nylas.{module}.get_client", forbidden)
    yield
    assert forbidden.call_count == 0


@pytest.fixture
def store(monkeypatch) -> SimpleNamespace:
    """In-memory stand-ins for the pipeline's persistence calls."""
    recorded = SimpleNamespace(emails={}, classifications=[], actions=[])

    def insert_email_if_new(session, email: Email):
        if email.id in recorded.emails:
            return None
        recorded.emails[email.id] = email
        return SimpleNamespace(id=len(recorded.emails))

    def save_classification(session, email_record_id, analysis, **fields):
        recorded.classifications.append({"email_id": email_record_id, **fields})

    def propose_action(session, email_record_id, email, analysis, action_type):
        recorded.actions.append(action_type)
        return SimpleNamespace(id=len(recorded.actions))

    for name, fake in [
        ("insert_email_if_new", insert_email_if_new),
        ("save_classification", save_classification),
        ("propose_action", propose_action),
    ]:
        monkeypatch.setattr(f"app.workflows.triage_workflow.{name}", fake)
    return recorded


def test_high_confidence_actionable_email_records_classification_and_proposes_action(
    store,
):
    outcome = process_raw_event(None, EVENT, _settings(), analyzer=lambda e: _run())

    assert outcome is TriageOutcome.ACTION_PROPOSED
    assert list(store.emails) == ["msg-pipeline-1"]
    assert store.classifications == [
        {
            "email_id": 1,
            "model": "gpt-test",
            "prompt_version": "analyze_v1",
            "input_tokens": 120,
            "output_tokens": 40,
            "latency_ms": 850,
            "triage_status": "draft_reply",
        }
    ]
    assert store.actions == [ActionType.DRAFT_REPLY]


def test_confidence_exactly_at_threshold_proposes_action(store):
    outcome = process_raw_event(
        None, EVENT, _settings(), analyzer=lambda e: _run(confidence=0.7)
    )

    assert outcome is TriageOutcome.ACTION_PROPOSED


def test_low_confidence_needs_review_and_proposes_nothing(store):
    outcome = process_raw_event(
        None, EVENT, _settings(), analyzer=lambda e: _run(confidence=0.4)
    )

    assert outcome is TriageOutcome.NEEDS_REVIEW
    assert store.classifications[0]["triage_status"] == "needs_review"
    assert store.actions == []


def test_category_with_no_action_proposes_nothing_even_if_model_suggests_one(store):
    analyzer = lambda e: _run(  # noqa: E731
        category=EmailCategory.REJECTION,
        suggested_action=SuggestedAction.DRAFT_REPLY,
    )

    outcome = process_raw_event(None, EVENT, _settings(), analyzer=analyzer)

    assert outcome is TriageOutcome.NO_ACTION
    assert store.classifications[0]["triage_status"] == "none"
    assert store.actions == []


def test_duplicate_event_is_processed_once(store):
    analyzer = MagicMock(return_value=_run())

    first = process_raw_event(None, EVENT, _settings(), analyzer=analyzer)
    second = process_raw_event(None, EVENT, _settings(), analyzer=analyzer)

    assert first is TriageOutcome.ACTION_PROPOSED
    assert second is TriageOutcome.DUPLICATE
    analyzer.assert_called_once()
    assert len(store.classifications) == 1
    assert len(store.actions) == 1


def test_sender_filtered_out_stores_email_but_never_calls_agent(store):
    analyzer = MagicMock(return_value=_run())
    settings = _settings(processing_mode="test", test_email="someone-else@example.com")

    outcome = process_raw_event(None, EVENT, settings, analyzer=analyzer)

    assert outcome is TriageOutcome.FILTERED
    assert list(store.emails) == ["msg-pipeline-1"]
    analyzer.assert_not_called()
    assert store.classifications == []
    assert store.actions == []


def test_non_message_event_is_skipped(store):
    event = {**EVENT, "type": "grant.updated"}
    analyzer = MagicMock()

    outcome = process_raw_event(None, event, _settings(), analyzer=analyzer)

    assert outcome is TriageOutcome.SKIPPED_TYPE
    assert store.emails == {}
    analyzer.assert_not_called()


# --- run_triage_safely ---------------------------------------------------------


@pytest.fixture
def wrapper_env(monkeypatch, store):
    """Let `run_triage_safely` run without a real DB or environment."""
    monkeypatch.setattr("app.workflows.background.session_scope", nullcontext)
    monkeypatch.setattr("app.workflows.background.get_settings", _settings)


def test_wrapper_logs_only_event_id_and_exception_type_when_agent_fails(
    monkeypatch, wrapper_env, caplog
):
    monkeypatch.setattr(
        "app.workflows.triage_workflow.analyze_with_metadata",
        MagicMock(side_effect=RuntimeError(f"failed on {SECRET_SUBJECT}")),
    )

    with caplog.at_level(logging.INFO):
        outcome = run_triage_safely("evt-pipeline-1", EVENT)

    assert outcome is None
    assert "evt-pipeline-1" in caplog.text
    assert "RuntimeError" in caplog.text
    assert SECRET_SUBJECT not in caplog.text
    assert SECRET_BODY not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_wrapper_logs_malformed_payload_without_its_content(wrapper_env, caplog):
    event = copy.deepcopy(EVENT)
    del event["data"]["object"]["date"]

    with caplog.at_level(logging.INFO):
        outcome = run_triage_safely("evt-pipeline-1", event)

    assert outcome is None
    assert "ValidationError" in caplog.text
    assert SECRET_SUBJECT not in caplog.text


def test_wrapper_returns_outcome_on_success(monkeypatch, wrapper_env):
    monkeypatch.setattr(
        "app.workflows.triage_workflow.analyze_with_metadata", lambda e: _run()
    )

    assert run_triage_safely("evt-pipeline-1", EVENT) is TriageOutcome.ACTION_PROPOSED
