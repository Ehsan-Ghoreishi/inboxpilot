"""Parse the hand-written synthetic fixture events all the way to `Email`."""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.integrations.nylas.mapper import nylas_payload_to_email

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_normal_event_fixture_parses_into_email():
    payload = _load_fixture("event_normal.json")

    email = nylas_payload_to_email(payload)

    assert email.id == "msg-synthetic-normal-001"
    assert email.thread_id == "thread-synthetic-normal-001"
    assert email.grant_id == "grant-synthetic-demo"
    assert email.sender.email == "sam.rivera@example.com"
    assert email.subject == "Lunch next week?"
    assert [r.email for r in email.recipients] == ["candidate@example.com"]
    assert email.received_at == datetime.fromtimestamp(1706800000, tz=timezone.utc)


def test_recruiter_outreach_event_fixture_parses_into_email():
    payload = _load_fixture("event_recruiter_outreach.json")

    email = nylas_payload_to_email(payload)

    assert email.id == "msg-synthetic-recruiter-001"
    assert email.sender.name == "Priya Shah"
    assert email.sender.email == "priya.shah@brightforge.io"
    assert email.subject == "Quick question about your background"
    assert "Brightforge" in email.body
    assert email.received_at == datetime.fromtimestamp(1706811644, tz=timezone.utc)


def test_interview_invite_event_fixture_parses_into_email():
    payload = _load_fixture("event_interview_invite.json")

    email = nylas_payload_to_email(payload)

    assert email.id == "msg-synthetic-interview-001"
    assert email.sender.email == "recruiting@nimbuscloud.example"
    assert email.subject == "Interview confirmed: Thursday 2pm PT"
    # Only "to" maps to canonical recipients, not "cc".
    assert [r.email for r in email.recipients] == ["candidate@example.com"]
    assert email.received_at == datetime.fromtimestamp(1707120000, tz=timezone.utc)


def test_all_fixtures_produce_schema_valid_emails():
    for fixture_name in (
        "event_normal.json",
        "event_recruiter_outreach.json",
        "event_interview_invite.json",
    ):
        payload = _load_fixture(fixture_name)
        email = nylas_payload_to_email(payload)
        # Round-trips through the canonical schema's own validation again.
        assert email.model_dump() == email.model_copy().model_dump()
