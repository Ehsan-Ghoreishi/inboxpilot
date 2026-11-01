"""`scripts/send_fake_event.py` and `scripts/reprocess_raw_events.py`,
without a network or a database."""

import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.integrations.nylas.webhooks import compute_signature
from app.workflows.triage_workflow import TriageOutcome
from scripts import reprocess_raw_events, send_fake_event

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- send_fake_event -----------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://inboxpilot-abc.a.run.app", "https://inboxpilot-abc.a.run.app/events"),
        ("https://inboxpilot-abc.a.run.app/", "https://inboxpilot-abc.a.run.app/events"),
        ("https://example.test/events", "https://example.test/events"),
        ("http://localhost:8000/events", "http://localhost:8000/events"),
    ],
)
def test_resolve_url_appends_events_to_a_base_url(url, expected):
    assert send_fake_event.resolve_url(url) == expected


@pytest.mark.parametrize("url", ["example.test", "ftp://example.test/events", ""])
def test_resolve_url_rejects_non_http_urls(url):
    with pytest.raises(ValueError):
        send_fake_event.resolve_url(url)


def test_build_event_is_synthetic_and_fresh():
    first = send_fake_event.build_event("Subject", "Body", "sender@example.com")
    second = send_fake_event.build_event("Subject", "Body", "sender@example.com")

    assert first["id"] != second["id"]
    assert first["data"]["object"]["id"] != second["data"]["object"]["id"]
    assert first["data"]["object"]["subject"] == "Subject"
    assert first["data"]["object"]["from"] == [
        {"name": "Test Sender", "email": "sender@example.com"}
    ]


def test_signature_matches_the_apps_verification():
    raw = b'{"id":"evt-1"}'

    assert send_fake_event.sign(raw, "secret") == compute_signature(raw, "secret")


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr(send_fake_event, "load_dotenv", lambda *a, **k: None)


def test_posts_signed_event_to_the_given_url_with_env_secret(
    monkeypatch, no_dotenv, capsys
):
    monkeypatch.setenv("WEBHOOK_SECRET", "deployed-secret")
    sent = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        sent["request"] = request
        return Response()

    monkeypatch.setattr(send_fake_event.urllib.request, "urlopen", fake_urlopen)

    code = send_fake_event.main(["--preset", "recruiter", "--url", "https://svc.example.test"])

    assert code == 0
    request = sent["request"]
    assert request.full_url == "https://svc.example.test/events"
    assert request.get_method() == "POST"
    assert request.get_header("X-nylas-signature") == compute_signature(
        request.data, "deployed-secret"
    )
    out = capsys.readouterr().out
    assert json.loads(request.data)["id"] in out
    assert "deployed-secret" not in out


def test_missing_secret_exits_non_zero_without_sending(monkeypatch, no_dotenv):
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    urlopen = MagicMock()
    monkeypatch.setattr(send_fake_event.urllib.request, "urlopen", urlopen)

    assert send_fake_event.main([]) == 2
    urlopen.assert_not_called()


def test_http_error_exits_non_zero(monkeypatch, no_dotenv, capsys):
    monkeypatch.setenv("WEBHOOK_SECRET", "secret")

    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(send_fake_event.urllib.request, "urlopen", fail)

    assert send_fake_event.main([]) == 1


def test_unreachable_url_exits_non_zero(monkeypatch, no_dotenv):
    monkeypatch.setenv("WEBHOOK_SECRET", "secret")
    monkeypatch.setattr(
        send_fake_event.urllib.request,
        "urlopen",
        MagicMock(side_effect=urllib.error.URLError("refused")),
    )

    assert send_fake_event.main([]) == 1


# --- reprocess_raw_events ------------------------------------------------------


@pytest.fixture
def stored(monkeypatch):
    """Replace the DB read with an in-memory list; records the limit used."""
    state = {"events": [], "limits": []}

    def load(limit):
        state["limits"].append(limit)
        return state["events"][:limit]

    monkeypatch.setattr(reprocess_raw_events, "_load_unprocessed", load)
    return state


def test_reprocess_defaults_to_a_limit_of_50(stored, monkeypatch):
    monkeypatch.setattr(reprocess_raw_events, "run_triage_safely", MagicMock())

    assert reprocess_raw_events.main([]) == 0
    assert stored["limits"] == [50]


def test_reprocess_prints_counts_and_honours_limit(stored, monkeypatch, capsys):
    stored["events"] = [(f"evt-{i}", {"id": f"evt-{i}"}) for i in range(3)]
    triage = MagicMock(
        side_effect=[TriageOutcome.ACTION_PROPOSED, TriageOutcome.NO_ACTION]
    )
    monkeypatch.setattr(reprocess_raw_events, "run_triage_safely", triage)

    code = reprocess_raw_events.main(["--limit", "2"])

    out = capsys.readouterr().out
    assert code == 0
    assert stored["limits"] == [2]
    assert triage.call_count == 2
    assert "2 unprocessed raw event(s)" in out
    assert "action_proposed: 1" in out
    assert "no_action: 1" in out


def test_reprocess_exits_non_zero_when_an_event_fails(stored, monkeypatch, capsys):
    stored["events"] = [("evt-1", {}), ("evt-2", {})]
    monkeypatch.setattr(
        reprocess_raw_events,
        "run_triage_safely",
        MagicMock(side_effect=[None, TriageOutcome.NO_ACTION]),
    )

    assert reprocess_raw_events.main([]) == 1
    out = capsys.readouterr().out
    assert "error: 1" in out
    assert "no_action: 1" in out


def test_reprocess_exits_non_zero_when_the_database_is_unreachable(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        reprocess_raw_events,
        "_load_unprocessed",
        MagicMock(side_effect=RuntimeError("postgresql://user:db-secret-pw@host")),
    )

    assert reprocess_raw_events.main([]) == 1
    out = capsys.readouterr().out
    assert "RuntimeError" in out
    assert "db-secret-pw" not in out


def test_reprocess_dry_run_counts_without_processing(stored, monkeypatch, capsys):
    stored["events"] = [("evt-1", {})]
    triage = MagicMock()
    monkeypatch.setattr(reprocess_raw_events, "run_triage_safely", triage)

    assert reprocess_raw_events.main(["--dry-run"]) == 0
    triage.assert_not_called()
    assert "1 unprocessed raw event(s)" in capsys.readouterr().out


@pytest.mark.parametrize("limit", ["0", "-1", "abc"])
def test_reprocess_rejects_invalid_limits(limit):
    with pytest.raises(SystemExit) as exc_info:
        reprocess_raw_events.main(["--limit", limit])

    assert exc_info.value.code != 0


def test_reprocess_is_runnable_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reprocess_raw_events", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--limit" in result.stdout
