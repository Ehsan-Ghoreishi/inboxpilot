"""Integration tests: POST /events -> background triage against a real test
database, with the agent mocked (no OpenAI, no Nylas)."""

import importlib.util
import json
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from app.agents.email_analyzer import AnalysisRun
from app.config import get_settings
from app.integrations.nylas.webhooks import compute_signature
from app.schemas.email import EmailAnalysis, EmailCategory, SuggestedAction, Urgency
from tests.integration.conftest import REPO_ROOT

TABLES = "raw_events, emails, classifications, pending_actions, audit_log"

EVENT = {
    "specversion": "1.0",
    "type": "message.created",
    "source": "/google/emails/realtime",
    "id": "evt-triage-1",
    "time": 1706811644,
    "data": {
        "application_id": "app-1",
        "object": {
            "id": "msg-triage-1",
            "grant_id": "grant-test",
            "thread_id": "thread-triage-1",
            "subject": "Synthetic recruiter outreach",
            "from": [{"name": "Test Recruiter", "email": "recruiter@example.com"}],
            "to": [{"email": "candidate@example.com"}],
            "date": 1706811644,
            "body": "Synthetic body.",
        },
    },
}


def _run(confidence: float = 0.9) -> AnalysisRun:
    return AnalysisRun(
        analysis=EmailAnalysis(
            category=EmailCategory.RECRUITER_OUTREACH,
            company="Example Corp",
            role="Data Scientist",
            urgency=Urgency.MEDIUM,
            suggested_action=SuggestedAction.DRAFT_REPLY,
            reasoning="Synthetic analysis for testing.",
            confidence=confidence,
        ),
        model="gpt-test",
        prompt_version="analyze_v1",
        input_tokens=120,
        output_tokens=40,
        latency_ms=850,
    )


@pytest.fixture
def client(live_app_client, test_engine):
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))
    yield live_app_client
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))


@pytest.fixture
def analyzer(monkeypatch) -> MagicMock:
    mock = MagicMock(return_value=_run())
    monkeypatch.setattr("app.workflows.triage_workflow.analyze_with_metadata", mock)
    return mock


@pytest.fixture(autouse=True)
def no_nylas(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError("Nylas client used"))
    for module in ("client", "actions", "messages"):
        monkeypatch.setattr(f"app.integrations.nylas.{module}.get_client", forbidden)


def _post(client, event: dict = EVENT):
    body = json.dumps(event).encode("utf-8")
    return client.post(
        "/events",
        content=body,
        headers={
            "x-nylas-signature": compute_signature(body, "super-secret"),
            "content-type": "application/json",
        },
    )


def _count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def test_event_is_classified_and_action_proposed(client, analyzer, test_engine):
    response = _post(client)

    assert response.status_code == 200
    with test_engine.connect() as conn:
        classification = conn.execute(
            text(
                "SELECT model, prompt_version, input_tokens, output_tokens, "
                "latency_ms, triage_status FROM classifications"
            )
        ).one()
        action = conn.execute(
            text("SELECT action_type, status FROM pending_actions")
        ).one()
    assert tuple(classification) == (
        "gpt-test", "analyze_v1", 120, 40, 850, "draft_reply"
    )
    assert tuple(action) == ("draft_reply", "pending")
    assert _count(test_engine, "audit_log") == 1


def test_duplicate_event_is_processed_once(client, analyzer, test_engine):
    first = _post(client)
    second = _post(client)

    assert (first.status_code, second.status_code) == (200, 200)
    analyzer.assert_called_once()
    assert _count(test_engine, "emails") == 1
    assert _count(test_engine, "classifications") == 1
    assert _count(test_engine, "pending_actions") == 1


def test_filtered_sender_is_stored_but_not_classified(
    client, analyzer, test_engine, monkeypatch
):
    monkeypatch.setenv("PROCESSING_MODE", "test")
    monkeypatch.setenv("TEST_EMAIL", "someone-else@example.com")
    get_settings.cache_clear()

    response = _post(client)

    assert response.status_code == 200
    analyzer.assert_not_called()
    assert _count(test_engine, "emails") == 1
    assert _count(test_engine, "classifications") == 0
    assert _count(test_engine, "pending_actions") == 0


def test_low_confidence_is_marked_needs_review(client, analyzer, test_engine):
    analyzer.return_value = _run(confidence=0.4)

    response = _post(client)

    assert response.status_code == 200
    with test_engine.connect() as conn:
        status = conn.execute(
            text("SELECT triage_status FROM classifications")
        ).scalar_one()
    assert status == "needs_review"
    assert _count(test_engine, "pending_actions") == 0


def _load_reprocess_script():
    path = REPO_ROOT / "scripts" / "reprocess_raw_events.py"
    spec = importlib.util.spec_from_file_location("reprocess_raw_events", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_failure_rolls_back_then_reprocess_script_picks_it_up(
    client, analyzer, test_engine, monkeypatch, capsys
):
    analyzer.side_effect = RuntimeError("synthetic agent failure")

    response = _post(client)

    assert response.status_code == 200
    assert _count(test_engine, "raw_events") == 1
    assert _count(test_engine, "emails") == 0

    analyzer.side_effect = None
    script = _load_reprocess_script()
    monkeypatch.setattr("sys.argv", ["reprocess_raw_events.py"])
    script.main()

    assert "action_proposed: 1" in capsys.readouterr().out
    assert _count(test_engine, "emails") == 1
    assert _count(test_engine, "classifications") == 1
    assert _count(test_engine, "pending_actions") == 1

    script.main()
    assert "0 unprocessed raw event(s)" in capsys.readouterr().out
