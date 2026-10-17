from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailCategory,
    EmailParticipant,
    SuggestedAction,
    Urgency,
)

VALID_EMAIL = {
    "id": "msg-1",
    "thread_id": "thread-1",
    "grant_id": "grant-1",
    "sender": {"name": "Jane Recruiter", "email": "jane@example.com"},
    "recipients": [{"name": "Candidate", "email": "candidate@example.com"}],
    "subject": "Interview invite",
    "body": "Are you free Tuesday?",
    "received_at": "2026-01-15T10:00:00Z",
}

VALID_ANALYSIS = {
    "category": "interview_invite",
    "company": "Acme Corp",
    "role": "Backend Engineer",
    "deadline": "2026-01-20",
    "urgency": "high",
    "suggested_action": "create_event",
    "reasoning": "Sender is proposing a specific interview time.",
    "confidence": 0.92,
}


# --- Email: valid ---------------------------------------------------------


def test_email_valid_payload_parses():
    email = Email.model_validate(VALID_EMAIL)

    assert email.id == "msg-1"
    assert email.sender == EmailParticipant(name="Jane Recruiter", email="jane@example.com")
    assert len(email.recipients) == 1
    assert isinstance(email.received_at, datetime)


def test_email_allows_multiple_recipients_and_blank_subject_and_body():
    payload = {
        **VALID_EMAIL,
        "recipients": [
            {"name": "A", "email": "a@example.com"},
            {"email": "b@example.com"},
        ],
        "subject": "",
        "body": "",
    }

    email = Email.model_validate(payload)

    assert len(email.recipients) == 2
    assert email.recipients[1].name is None
    assert email.subject == ""
    assert email.body == ""


# --- Email: invalid --------------------------------------------------------


def test_email_missing_required_field_raises():
    payload = {k: v for k, v in VALID_EMAIL.items() if k != "received_at"}

    with pytest.raises(ValidationError):
        Email.model_validate(payload)


def test_email_empty_id_raises():
    with pytest.raises(ValidationError):
        Email.model_validate({**VALID_EMAIL, "id": ""})


def test_email_malformed_sender_address_raises():
    with pytest.raises(ValidationError):
        Email.model_validate({**VALID_EMAIL, "sender": {"name": "Jane", "email": "not-an-email"}})


def test_email_empty_recipients_raises():
    with pytest.raises(ValidationError):
        Email.model_validate({**VALID_EMAIL, "recipients": []})


def test_email_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        Email.model_validate({**VALID_EMAIL, "unexpected": "surprise"})


# --- EmailAnalysis: valid ---------------------------------------------------


def test_email_analysis_valid_payload_parses():
    analysis = EmailAnalysis.model_validate(VALID_ANALYSIS)

    assert analysis.category == EmailCategory.INTERVIEW_INVITE
    assert analysis.urgency == Urgency.HIGH
    assert analysis.suggested_action == SuggestedAction.CREATE_EVENT
    assert analysis.deadline == date(2026, 1, 20)


def test_email_analysis_allows_null_company_role_and_deadline():
    payload = {
        **VALID_ANALYSIS,
        "category": "other",
        "company": None,
        "role": None,
        "deadline": None,
        "urgency": "low",
        "suggested_action": "none",
    }

    analysis = EmailAnalysis.model_validate(payload)

    assert analysis.company is None
    assert analysis.role is None
    assert analysis.deadline is None


def test_email_analysis_serializes_enums_as_plain_strings():
    analysis = EmailAnalysis.model_validate(VALID_ANALYSIS)

    dumped = analysis.model_dump(mode="json")

    assert dumped["category"] == "interview_invite"
    assert dumped["urgency"] == "high"
    assert dumped["suggested_action"] == "create_event"


# --- EmailAnalysis: invalid --------------------------------------------------


def test_email_analysis_invalid_category_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "category": "not_a_category"})


def test_email_analysis_invalid_urgency_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "urgency": "extreme"})


def test_email_analysis_invalid_suggested_action_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "suggested_action": "call_them"})


def test_email_analysis_empty_reasoning_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "reasoning": ""})


def test_email_analysis_reasoning_too_long_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "reasoning": "x" * 2001})


def test_email_analysis_empty_company_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "company": ""})


def test_email_analysis_malformed_deadline_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "deadline": "not-a-date"})


def test_email_analysis_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "unexpected": "surprise"})


def test_email_analysis_confidence_boundary_values_are_valid():
    low = EmailAnalysis.model_validate({**VALID_ANALYSIS, "confidence": 0.0})
    high = EmailAnalysis.model_validate({**VALID_ANALYSIS, "confidence": 1.0})

    assert low.confidence == 0.0
    assert high.confidence == 1.0


def test_email_analysis_confidence_below_zero_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "confidence": -0.01})


def test_email_analysis_confidence_above_one_raises():
    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate({**VALID_ANALYSIS, "confidence": 1.01})


def test_email_analysis_missing_confidence_raises():
    payload = {k: v for k, v in VALID_ANALYSIS.items() if k != "confidence"}

    with pytest.raises(ValidationError):
        EmailAnalysis.model_validate(payload)
