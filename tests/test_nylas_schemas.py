import pytest
from pydantic import ValidationError

from app.schemas.nylas_email import EmailObject
from app.schemas.nylas_webhook import WebhookEvent

VALID_EMAIL_OBJECT = {
    "id": "msg-1",
    "object": "message",
    "grant_id": "grant-1",
    "thread_id": "thread-1",
    "subject": "Hello",
    "from": [{"name": "Jane", "email": "jane@example.com"}],
    "to": [{"email": "candidate@example.com"}],
    "date": 1706811644,
    "body": "Hi there",
}

VALID_EVENT = {
    "specversion": "1.0",
    "type": "message.created",
    "source": "/google/emails/realtime",
    "id": "evt-1",
    "time": 1706811644,
    "webhook_delivery_attempt": 1,
    "data": {"application_id": "app-1", "object": VALID_EMAIL_OBJECT},
}


# --- EmailObject ---------------------------------------------------------


def test_email_object_valid_payload_parses():
    obj = EmailObject.model_validate(VALID_EMAIL_OBJECT)

    assert obj.id == "msg-1"
    assert obj.from_[0].email == "jane@example.com"
    assert obj.to[0].email == "candidate@example.com"


def test_email_object_defaults_missing_optional_lists_and_fields():
    minimal = {
        "id": "msg-1",
        "grant_id": "grant-1",
        "thread_id": "thread-1",
        "date": 1706811644,
    }

    obj = EmailObject.model_validate(minimal)

    assert obj.from_ == []
    assert obj.to == []
    assert obj.subject is None
    assert obj.body is None


def test_email_object_ignores_unknown_fields():
    obj = EmailObject.model_validate({**VALID_EMAIL_OBJECT, "some_new_field": "x"})

    assert obj.id == "msg-1"


def test_email_object_missing_required_field_raises():
    broken = {k: v for k, v in VALID_EMAIL_OBJECT.items() if k != "grant_id"}

    with pytest.raises(ValidationError):
        EmailObject.model_validate(broken)


def test_email_object_malformed_address_raises():
    broken = {**VALID_EMAIL_OBJECT, "from": [{"email": "not-an-email"}]}

    with pytest.raises(ValidationError):
        EmailObject.model_validate(broken)


# --- WebhookEvent ----------------------------------------------------------


def test_webhook_event_valid_payload_parses():
    event = WebhookEvent.model_validate(VALID_EVENT)

    assert event.type == "message.created"
    assert event.data.application_id == "app-1"
    assert event.data.object.id == "msg-1"


def test_webhook_event_missing_data_object_raises():
    broken = {**VALID_EVENT, "data": {"application_id": "app-1"}}

    with pytest.raises(ValidationError):
        WebhookEvent.model_validate(broken)


def test_webhook_event_missing_top_level_id_raises():
    broken = {k: v for k, v in VALID_EVENT.items() if k != "id"}

    with pytest.raises(ValidationError):
        WebhookEvent.model_validate(broken)


def test_webhook_event_ignores_unknown_top_level_fields():
    event = WebhookEvent.model_validate({**VALID_EVENT, "extra_field": "x"})

    assert event.id == "evt-1"
