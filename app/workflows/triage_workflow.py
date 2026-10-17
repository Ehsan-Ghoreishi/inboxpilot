"""Triage workflow: classify an email, then route it to a next step.

`classify()` delegates to the LLM agent (`app.agents.email_analyzer`).
`route()` is the deterministic decision on top of that output: below the
confidence threshold, nothing is trusted enough to act on automatically
and the item goes to a human; otherwise, the email's `category` (not
the model's own `suggested_action`) decides the `TriageStatus` via a
fixed lookup table here, so routing stays a single, auditable place
independent of whatever the model happened to suggest.
"""

from dataclasses import dataclass
from enum import Enum

from app.agents.email_analyzer import PROMPT_VERSION, analyze
from app.config import Settings
from app.database.models import ActionType
from app.schemas.email import Email, EmailAnalysis, EmailCategory
from app.services.actions import create_pending_action
from app.services.classifications import record_classification
from app.workflows.filters import should_process


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

# The only action each routed status may propose. NEEDS_REVIEW proposes
# nothing: the email is just shown, classified, for a human to look at.
_STATUS_TO_ACTION_TYPE: dict[TriageStatus, ActionType] = {
    TriageStatus.NONE: ActionType.ARCHIVE,
    TriageStatus.DRAFT_REPLY: ActionType.DRAFT_REPLY,
    TriageStatus.CREATE_EVENT: ActionType.CREATE_EVENT,
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


@dataclass(frozen=True)
class TriageResult:
    analysis: EmailAnalysis
    status: TriageStatus
    pending_action_id: int | None


def process_email(email: Email, settings: Settings) -> TriageResult | None:
    """Classify `email`, record the classification, and propose at most one
    pending action. Returns `None` if the processing-mode filter skips it.

    Never executes anything: a proposed action only runs when a human
    approves it via `app.services.actions.approve()`.
    """
    if not should_process(email, settings):
        return None

    analysis = classify(email)
    record_classification(
        email, analysis, model=settings.openai_model, prompt_version=PROMPT_VERSION
    )
    status = route(analysis, settings)
    action_type = _STATUS_TO_ACTION_TYPE.get(status)
    if action_type is None:
        return TriageResult(analysis, status, pending_action_id=None)

    pending_action = create_pending_action(email, analysis, action_type)
    return TriageResult(analysis, status, pending_action_id=pending_action.id)
