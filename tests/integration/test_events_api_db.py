"""Integration test: POST /events against a real test database."""

import json

from sqlalchemy import text

from app.integrations.nylas.webhooks import compute_signature

SAMPLE_EVENT = {
    "specversion": "1.0",
    "type": "message.created",
    "source": "/google/emails/realtime",
    "id": "evt-db-1",
    "time": 1706811644,
    "data": {
        "application_id": "app-1",
        "object": {
            "id": "msg-1",
            "grant_id": "grant-test",
            "thread_id": "thread-1",
            "subject": "Interview invite",
        },
    },
}


def _signed_post(client, body: bytes, secret: str = "super-secret"):
    signature = compute_signature(body, secret)
    return client.post(
        "/events",
        content=body,
        headers={"x-nylas-signature": signature, "content-type": "application/json"},
    )


def test_post_events_persists_raw_event_row(live_app_client, test_engine):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = _signed_post(live_app_client, body)

    assert response.status_code == 200

    with test_engine.connect() as conn:
        row = conn.execute(
            text("SELECT event_id, payload FROM raw_events WHERE event_id = :id"),
            {"id": "evt-db-1"},
        ).one()

    assert row.event_id == "evt-db-1"
    assert row.payload == SAMPLE_EVENT


def test_post_events_duplicate_delivery_creates_only_one_row(
    live_app_client, test_engine
):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    first = _signed_post(live_app_client, body)
    second = _signed_post(live_app_client, body)

    assert first.status_code == 200
    assert second.status_code == 200

    with test_engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM raw_events WHERE event_id = :id"),
            {"id": "evt-db-1"},
        ).scalar()

    assert count == 1


def test_post_events_file_dump_is_optional(live_app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_RAW_EVENT_FILES", "false")
    from app.config import get_settings

    get_settings.cache_clear()

    body = json.dumps(SAMPLE_EVENT).encode("utf-8")
    response = _signed_post(live_app_client, body)

    assert response.status_code == 200
    assert not (tmp_path / "requests").exists()

    get_settings.cache_clear()
