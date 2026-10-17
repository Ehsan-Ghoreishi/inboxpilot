"""Shared Nylas SDK client construction.

This is one of only two modules in the codebase allowed to read Nylas
secrets (the other is `webhooks.py`, for the webhook secret) — everything
else takes a client or plain data as a parameter.
"""

from functools import lru_cache

from nylas import Client

from app.config import get_settings


@lru_cache
def get_client() -> Client:
    """Return a cached Nylas SDK client built from app settings."""
    return Client(api_key=get_settings().nylas_api_key)
