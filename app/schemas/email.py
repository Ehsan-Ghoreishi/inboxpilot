"""Canonical email schemas.

These are the internal, provider-independent shapes everything past the
Nylas boundary works against. Code outside `app/integrations/` should
never see a raw Nylas payload — only `Email` and `EmailAnalysis`.
"""

from datetime import date, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class EmailParticipant(BaseModel):
    """A single sender/recipient address with an optional display name."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    email: EmailStr


class Email(BaseModel):
    """Canonical internal representation of an email."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    thread_id: NonEmptyStr
    grant_id: NonEmptyStr
    sender: EmailParticipant
    recipients: list[EmailParticipant] = Field(min_length=1)
    subject: str = ""
    body: str = ""
    received_at: datetime


class EmailCategory(str, Enum):
    RECRUITER_OUTREACH = "recruiter_outreach"
    INTERVIEW_INVITE = "interview_invite"
    ASSESSMENT_DEADLINE = "assessment_deadline"
    REJECTION = "rejection"
    OTHER = "other"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SuggestedAction(str, Enum):
    NONE = "none"
    DRAFT_REPLY = "draft_reply"
    CREATE_EVENT = "create_event"


class EmailAnalysis(BaseModel):
    """Structured output of LLM classification/extraction over an `Email`."""

    model_config = ConfigDict(extra="forbid")

    category: EmailCategory
    company: NonEmptyStr | None = None
    role: NonEmptyStr | None = None
    deadline: date | None = None
    urgency: Urgency
    suggested_action: SuggestedAction
    reasoning: Annotated[
        str, StringConstraints(min_length=1, max_length=2000, strip_whitespace=True)
    ]
    # Model-reported self-assessment of its own confidence, not a
    # calibrated probability (e.g. 0.9 does not mean "correct 90% of the
    # time" in any statistical sense) — treat it only as a relative
    # signal for routing, not as ground truth.
    confidence: float = Field(ge=0.0, le=1.0)
