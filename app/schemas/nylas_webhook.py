"""Nylas webhook notification envelope ("WebhookEvent").

Validates the JSON that `app.integrations.nylas.webhooks.store_raw_event`
writes to `requests/events/<id>.json` — i.e. a raw, untrusted webhook
delivery — before anything downstream trusts its shape.

Confirmed envelope shape (CloudEvents-style; see the research notes in
`app/integrations/nylas/webhooks.py` and the plan for that feature):
top-level `specversion`, `type` (the trigger, e.g. "message.created"),
`source`, `id` (the notification/event id), `time`,
`webhook_delivery_attempt`, and `data: {application_id, object}`. This
module models the `object` as a message (`EmailObject`), since it's only
used for `message.*` triggers so far — other trigger types (calendar,
grant, ...) aren't modeled yet.
"""

from pydantic import BaseModel, ConfigDict

from app.schemas.nylas_email import EmailObject


class WebhookEventData(BaseModel):
    """The `data` wrapper: which application, and the changed object."""

    model_config = ConfigDict(extra="ignore")

    application_id: str
    object: EmailObject


class WebhookEvent(BaseModel):
    """A Nylas webhook notification for a message trigger."""

    model_config = ConfigDict(extra="ignore")

    specversion: str
    type: str
    source: str
    id: str
    time: int
    webhook_delivery_attempt: int | None = None
    data: WebhookEventData
