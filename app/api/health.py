"""Liveness (`/health`) and readiness (`/ready`) checks.

`/health` never touches the database, so it stays cheap and doesn't flap
when Postgres is briefly unavailable. `/ready` runs `SELECT 1` and
reports 503 if the database can't be reached.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.database.session import get_engine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> JSONResponse:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        # Type only: driver messages can include the connection string.
        logger.warning("Readiness check failed: %s", type(exc).__name__)
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return JSONResponse(status_code=200, content={"status": "ready"})
