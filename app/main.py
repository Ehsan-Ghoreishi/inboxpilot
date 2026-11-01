import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.actions import router as actions_router
from app.api.ask import router as ask_router
from app.api.dashboard import router as dashboard_router
from app.api.events import router as events_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.config import Settings, get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def log_safety_flags(settings: Settings) -> None:
    """Log the flags that decide what the app may do. Never secret values."""
    logger.info(
        "Starting with ENVIRONMENT=%s PROCESSING_MODE=%s DRY_RUN=%s "
        "PROCESS_EVENTS_INLINE=%s",
        settings.environment,
        settings.processing_mode,
        settings.dry_run,
        settings.process_events_inline,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log_safety_flags(get_settings())
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    docs_enabled = settings.environment != "production"
    app = FastAPI(
        title="InboxPilot",
        lifespan=_lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.include_router(health_router)
    app.include_router(events_router)
    app.include_router(actions_router)
    app.include_router(dashboard_router)
    app.include_router(search_router)
    app.include_router(ask_router)
    return app


app = create_app()
