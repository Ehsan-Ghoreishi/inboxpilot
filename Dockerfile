# Python minor version must match pyproject's requires-python (>=3.11,<3.12).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

# The venv lives outside /app so docker-compose's `.:/app` bind mount
# doesn't hide it. Raw-event files are off: the container's filesystem
# is ephemeral and not writable by the app user, and the database is the
# authoritative raw-event store.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    STORE_RAW_EVENT_FILES=false \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

RUN useradd --system --uid 10001 --no-create-home inboxpilot
USER inboxpilot

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
