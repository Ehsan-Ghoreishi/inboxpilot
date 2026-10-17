"""Reading messages through the Nylas SDK."""

from nylas.models.messages import Message

from app.config import get_settings
from app.integrations.nylas.client import get_client


def list_recent_messages(limit: int = 10) -> list[Message]:
    """List the most recent messages for the configured grant, via the SDK."""
    client = get_client()
    response = client.messages.list(
        get_settings().nylas_grant_id,
        query_params={"limit": limit},
    )
    return response.data
