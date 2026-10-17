"""Triage workflow: classify an email, then route it to a next step.

`classify()` delegates to the LLM agent (`app.agents.email_analyzer`).
`route()` is the deterministic decision on top of that output: below the
confidence threshold, nothing is trusted enough to act on automatically
and the item goes to a human; otherwise, the email's `category` (not
the model's own `suggested_action`) decides the `TriageStatus` via a
fixed lookup table here, so routing stays a single, auditable place
independent of whatever the model happened to suggest.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from app.agents.email_analyzer import AnalysisRun, analyze, analyze_with_metadata
from app.config import Settings
from app.database.models import ActionType
from app.database.repository import (
    get_or_create_email,
    insert_email_if_new,
    save_classification,
)
from app.database.session import session_scope
from app.integrations.nylas.mapper import nylas_payload_to_email
from app.schemas.email import Email, EmailAnalysis, EmailCategory
from app.services.actions import propose_action
from app.workflows.filters import should_process

Analyzer = Callable[[Email], AnalysisRun]


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

# The only action each routed status may propose. NEEDS_REVIEW and NONE
# propose nothing: the email is just stored, classified, for a human to
# look at (NEEDS_REVIEW) or ignore (NONE).
_STATUS_TO_ACTION_TYPE: dict[TriageStatus, ActionType] = {
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


class TriageOutcome(str, Enum):
    """How processing one raw event ended (for logs and the reprocess script)."""

    SKIPPED_TYPE = "skipped_type"
    DUPLICATE = "duplicate"
    FILTERED = "filtered"
    NEEDS_REVIEW = "needs_review"
    NO_ACTION = "no_action"
    ACTION_PROPOSED = "action_proposed"


def _triage(
    session: Session,
    email_record_id: int,
    email: Email,
    settings: Settings,
    analyzer: Analyzer | None,
) -> TriageResult | None:
    """Filter, classify, record, and propose at most one pending action,
    all in `session`. Returns `None` if the processing-mode filter skips it."""
    if not should_process(email, settings):
        return None

    # Resolved at call time (not as a default argument) so tests can
    # patch the module-level `analyze_with_metadata`.
    run = (analyzer or analyze_with_metadata)(email)
    status = route(run.analysis, settings)
    save_classification(
        session,
        email_record_id,
        run.analysis,
        model=run.model,
        prompt_version=run.prompt_version,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        latency_ms=run.latency_ms,
        triage_status=status.value,
    )
    action_type = _STATUS_TO_ACTION_TYPE.get(status)
    if action_type is None:
        return TriageResult(run.analysis, status, pending_action_id=None)

    pending_action = propose_action(
        session, email_record_id, email, run.analysis, action_type
    )
    return TriageResult(run.analysis, status, pending_action_id=pending_action.id)


def process_email(
    email: Email, settings: Settings, analyzer: Analyzer | None = None
) -> TriageResult | None:
    """Classify `email`, record the classification, and propose at most one
    pending action, in one transaction. Returns `None` if the
    processing-mode filter skips it.

    Never executes anything: a proposed action only runs when a human
    approves it via `app.services.actions.approve()`.
    """
    with session_scope() as session:
        email_record = get_or_create_email(session, email)
        return _triage(session, email_record.id, email, settings, analyzer)


def process_raw_event(
    session: Session,
    payload: dict,
    settings: Settings,
    analyzer: Analyzer | None = None,
) -> TriageOutcome:
    """Run a stored raw webhook event through the full triage flow in `session`.

    Idempotent by message id: if an `emails` row for the message already
    exists, nothing else happens. Filtered-out emails are still stored
    (so they aren't re-examined), but never classified. If any step
    raises, the caller's transaction rolls back — including the email
    row — so the event stays eligible for reprocessing.

    Never calls Nylas: the email comes from the stored payload only.
    """
    if not str(payload.get("type", "")).startswith("message."):
        return TriageOutcome.SKIPPED_TYPE

    email = nylas_payload_to_email(payload)
    email_record = insert_email_if_new(session, email)
    if email_record is None:
        return TriageOutcome.DUPLICATE

    result = _triage(session, email_record.id, email, settings, analyzer)
    if result is None:
        return TriageOutcome.FILTERED
    if result.status is TriageStatus.NEEDS_REVIEW:
        return TriageOutcome.NEEDS_REVIEW
    if result.pending_action_id is None:
        return TriageOutcome.NO_ACTION
    return TriageOutcome.ACTION_PROPOSED
