"""Raw Nylas message shape ("EmailObject").

This models Nylas's own schema for a message — the thing found at
`data.object` inside a `message.*` webhook notification (see
`nylas_webhook.py`), not our canonical `Email` (see `email.py`). It
exists to validate untrusted external data (a saved webhook event) at
the boundary, before anything is mapped to the canonical schema.

Fields are a subset of Nylas's documented Message object
(developer.nylas.com/docs/v3/email/messages/): id, grant_id, thread_id,
subject, from/to/cc/bcc (arrays of {name, email}), date (Unix seconds),
snippet, body, folders, unread, starred. `extra="ignore"` throughout,
since this is Nylas's schema, not ours — new fields Nylas adds later
shouldn't break validation of the fields we do care about.
"""

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class NylasEmailAddress(BaseModel):
    """A `{name, email}` participant, as Nylas represents one."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    email: EmailStr


class EmailObject(BaseModel):
    """A Nylas Message object, as delivered in a webhook's `data.object`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    grant_id: str
    thread_id: str
    subject: str | None = None
    from_: list[NylasEmailAddress] = Field(default_factory=list, alias="from")
    to: list[NylasEmailAddress] = Field(default_factory=list)
    cc: list[NylasEmailAddress] = Field(default_factory=list)
    bcc: list[NylasEmailAddress] = Field(default_factory=list)
    date: int
    snippet: str | None = None
    body: str | None = None
    folders: list[str] = Field(default_factory=list)
    unread: bool | None = None
    starred: bool | None = None
