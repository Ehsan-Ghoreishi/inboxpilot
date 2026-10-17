"""Shared-secret API key check for human-facing endpoints."""

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    """Reject the request with 401 unless `X-API-Key` matches
    `settings.app_api_key`. Compared in constant time; an empty configured
    key never matches (fail closed)."""
    expected = get_settings().app_api_key
    if (
        not provided
        or not expected
        or not hmac.compare_digest(provided.encode(), expected.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
