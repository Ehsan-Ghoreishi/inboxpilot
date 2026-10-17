from datetime import datetime, timezone
from unittest.mock import patch

from app.config import Settings
from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailCategory,
    EmailParticipant,
    SuggestedAction,
    Urgency,
)
from app.workflows.triage_workflow import TriageStatus, classify, route

SAMPLE_EMAIL = Email(
    id="msg-1",
    thread_id="thread-1",
    grant_id="grant-1",
    sender=EmailParticipant(name="Jane Recruiter", email="jane@example.com"),
    recipients=[EmailParticipant(email="candidate@example.com")],
    subject="Interview invite",
    body="Are you free Tuesday at 2pm?",
    received_at=datetime(2026, 2, 5, 9, 0, tzinfo=timezone.utc),
)


def _analysis(category: EmailCategory, confidence: float) -> EmailAnalysis:
    return EmailAnalysis(
        category=category,
        company="Acme Corp",
        role="Backend Engineer",
        deadline=None,
        urgency=Urgency.MEDIUM,
        suggested_action=SuggestedAction.NONE,
        reasoning="Synthetic analysis for testing.",
        confidence=confidence,
    )


def _settings(confidence_threshold: float = 0.7) -> Settings:
    return Settings(
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
        confidence_threshold=confidence_threshold,
    )


# --- classify() ------------------------------------------------------------


def test_classify_delegates_to_the_agent():
    expected = _analysis(EmailCategory.INTERVIEW_INVITE, confidence=0.95)

    with patch(
        "app.workflows.triage_workflow.analyze", return_value=expected
    ) as mock_analyze:
        result = classify(SAMPLE_EMAIL)

    mock_analyze.assert_called_once_with(SAMPLE_EMAIL)
    assert result is expected


# --- route(): low confidence -> needs_review, regardless of category --------


def test_route_below_threshold_needs_review():
    analysis = _analysis(EmailCategory.INTERVIEW_INVITE, confidence=0.5)
    settings = _settings(confidence_threshold=0.7)

    assert route(analysis, settings) == TriageStatus.NEEDS_REVIEW


def test_route_below_threshold_needs_review_even_for_rejection():
    analysis = _analysis(EmailCategory.REJECTION, confidence=0.1)
    settings = _settings(confidence_threshold=0.7)

    assert route(analysis, settings) == TriageStatus.NEEDS_REVIEW


def test_route_exactly_at_threshold_is_not_needs_review():
    # confidence < threshold triggers review; confidence == threshold does not.
    analysis = _analysis(EmailCategory.OTHER, confidence=0.7)
    settings = _settings(confidence_threshold=0.7)

    assert route(analysis, settings) == TriageStatus.NONE


# --- route(): confident enough -> category decides the status --------------


def test_route_recruiter_outreach_maps_to_draft_reply():
    analysis = _analysis(EmailCategory.RECRUITER_OUTREACH, confidence=0.9)
    settings = _settings()

    assert route(analysis, settings) == TriageStatus.DRAFT_REPLY


def test_route_interview_invite_maps_to_create_event():
    analysis = _analysis(EmailCategory.INTERVIEW_INVITE, confidence=0.9)
    settings = _settings()

    assert route(analysis, settings) == TriageStatus.CREATE_EVENT


def test_route_assessment_deadline_maps_to_create_event():
    analysis = _analysis(EmailCategory.ASSESSMENT_DEADLINE, confidence=0.9)
    settings = _settings()

    assert route(analysis, settings) == TriageStatus.CREATE_EVENT


def test_route_rejection_maps_to_none():
    analysis = _analysis(EmailCategory.REJECTION, confidence=0.9)
    settings = _settings()

    assert route(analysis, settings) == TriageStatus.NONE


def test_route_other_maps_to_none():
    analysis = _analysis(EmailCategory.OTHER, confidence=0.9)
    settings = _settings()

    assert route(analysis, settings) == TriageStatus.NONE


def test_route_respects_custom_threshold_setting():
    analysis = _analysis(EmailCategory.INTERVIEW_INVITE, confidence=0.8)

    assert route(analysis, _settings(confidence_threshold=0.9)) == (
        TriageStatus.NEEDS_REVIEW
    )
    assert route(analysis, _settings(confidence_threshold=0.5)) == (
        TriageStatus.CREATE_EVENT
    )
