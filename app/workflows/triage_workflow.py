"""Triage workflow: classify an email, then route it to a next step.

`classify()` delegates to the LLM agent (`app.agents.email_analyzer`).
`route()` is the deterministic decision on top of that output: below the
confidence threshold, nothing is trusted enough to act on automatically
and the item goes to a human; otherwise, the email's `category` (not
the model's own `suggested_action`) decides the `TriageStatus` via a
fixed lookup table here, so routing stays a single, auditable place
independent of whatever the model happened to suggest.
"""

from enum import Enum

from app.agents.email_analyzer import analyze
from app.config import Settings
from app.schemas.email import Email, EmailAnalysis, EmailCategory


class TriageStatus(str, Enum):
    """Where an analyzed email ends up after routing."""

    NEEDS_REVIEW = "needs_review"
    NONE = "none"
    DRAFT_REPLY = "draft_reply"
    CREATE_EVENT = "create_event"


_CATEGORY_TO_STATUS: dict[EmailCategory, TriageStatus] = {
    EmailCategory.RECRUITER_OUTREACH: TriageStatus.DRAFT_REPLY,
    EmailCategory.INTERVIEW_INVITE: TriageStatus.CREATE_EVENT,
    EmailCategory.ASSESSMENT_DEADLINE: TriageStatus.CREATE_EVENT,
    EmailCategory.REJECTION: TriageStatus.NONE,
    EmailCategory.OTHER: TriageStatus.NONE,
}


def classify(email: Email) -> EmailAnalysis:
    """Classify and extract structured information from `email` via the agent."""
    return analyze(email)


def route(analysis: EmailAnalysis, settings: Settings) -> TriageStatus:
    """Decide what should happen next with an already-classified email.

    `analysis.confidence` is model-reported, not a calibrated
    probability (see the field's docstring in `app/schemas/email.py`) —
    it's only used here as a relative signal for whether to trust the
    categorization at all, not treated as a precise likelihood.
    """
    if analysis.confidence < settings.confidence_threshold:
        return TriageStatus.NEEDS_REVIEW
    return _CATEGORY_TO_STATUS[analysis.category]

