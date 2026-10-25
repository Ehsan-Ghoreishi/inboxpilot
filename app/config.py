"""Typed application configuration.

This is the only place in the codebase that should read environment
variables / secrets. Everything else should depend on `get_settings()`.
"""

from functools import lru_cache

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration the app needs, loaded from the environment/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    openai_model: str
    nylas_api_key: str
    nylas_client_id: str
    nylas_grant_id: str
    webhook_secret: str
    server_url: str
    database_url: str
    app_api_key: str
    processing_mode: str
    # Test-only convenience value (e.g. a mailbox to run manual checks
    # against); not required for the app to run.
    test_email: str | None = None
    # Below this, a classification's confidence is treated as too low to
    # act on automatically (see app/workflows/triage_workflow.py).
    confidence_threshold: float = 0.7
    # Whether the webhook endpoint also dumps each raw event to
    # requests/events/<id>.json, in addition to the database (which is
    # the required, authoritative store). Default on; set to false to
    # turn off the optional file copy.
    store_raw_event_files: bool = True
    # When true, approving a pending action never calls Nylas: the action
    # is logged (type + id only), gets a fake "dry-run-<uuid>" id, and is
    # still marked executed, with its audit rows flagged dry_run. For
    # offline development without a live grant.
    dry_run: bool = False
    # OpenAI embedding model for email embeddings. Must produce vectors
    # of `app.database.models.EMBEDDING_DIM` (1536) dimensions.
    embedding_model: str = "text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated `Settings` instance.

    Raises a `RuntimeError` naming the missing/invalid environment
    variable(s) if required configuration is absent, instead of letting
    the app start in a half-configured state.
    """
    try:
        return Settings()
    except ValidationError as exc:
        missing = sorted(
            str(error["loc"][0]).upper() for error in exc.errors() if error["loc"]
        )
        names = ", ".join(missing) if missing else str(exc)
        raise RuntimeError(
            f"Missing or invalid required environment variable(s): {names}. "
            "Check your .env against .env.example."
        ) from exc
