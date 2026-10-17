"""Fixtures for integration tests that need a real Postgres database.

These tests are deliberately separate from the fast, fully-mocked tests
in `tests/` — they need Docker (to bring up the `db` Compose service)
and talk to real Postgres. If Docker isn't available or Postgres never
becomes reachable, the whole integration suite is skipped rather than
erroring the run.

A dedicated `<db>_test` database is created on the same Postgres
instance used for dev (never the real dev database) and migrated with
Alembic before any test runs.
"""

import subprocess
import time
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database.session import get_engine, get_sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_MODEL": "gpt-test",
    "NYLAS_API_KEY": "nylas-test",
    "NYLAS_CLIENT_ID": "client-test",
    "NYLAS_GRANT_ID": "grant-test",
    "WEBHOOK_SECRET": "super-secret",
    "SERVER_URL": "http://localhost:8000",
    "APP_API_KEY": "app-test",
    "PROCESSING_MODE": "manual",
}


def _docker_compose_up_db() -> None:
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "db"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        pytest.skip("docker is not available for integration tests")
    if result.returncode != 0:
        pytest.skip(
            "Could not start the 'db' Postgres service via docker compose: "
            f"{result.stderr.strip()}"
        )


def _wait_for_postgres(url, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            engine = create_engine(url)
            with engine.connect():
                pass
            engine.dispose()
            return
        except Exception as exc:  # noqa: BLE001 - report whatever broke
            last_error = exc
            time.sleep(0.5)
    pytest.skip(f"Postgres never became reachable at {url}: {last_error}")


def _ensure_database_exists(base_url, db_name: str) -> None:
    admin_engine = create_engine(
        base_url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        admin_engine.dispose()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """URL of a dedicated `<db>_test` database, migrated to head."""
    _docker_compose_up_db()

    base_url = make_url(get_settings().database_url)
    _wait_for_postgres(base_url.set(database="postgres"))

    test_db_name = f"{base_url.database}_test"
    _ensure_database_exists(base_url, test_db_name)
    test_url = base_url.set(database=test_db_name)

    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    alembic_cfg.set_main_option(
        "sqlalchemy.url", test_url.render_as_string(hide_password=False)
    )
    command.upgrade(alembic_cfg, "head")

    # render_as_string(hide_password=False) — plain str(url) masks the
    # password as "***", which would silently break the real connection.
    return test_url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def test_engine(test_database_url):
    engine = create_engine(test_database_url, future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """A `Session` bound to a transaction rolled back after the test —
    isolates tests without needing to truncate tables manually."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def app_db_env(test_database_url, tmp_path, monkeypatch):
    """Point the *app's own* cached settings/engine (the ones
    `session_scope()` and `get_settings()` actually use) at the test
    database, in an isolated cwd. For code that manages its own
    sessions internally (e.g. `app.services.actions`, the FastAPI app)
    rather than taking a `Session` parameter."""
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATABASE_URL", test_database_url)

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()

    try:
        yield
    finally:
        get_settings.cache_clear()
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()


@pytest.fixture
def live_app_client(app_db_env, test_engine):
    """A `TestClient` for the real FastAPI app, pointed at the test database."""
    with test_engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE raw_events RESTART IDENTITY CASCADE"))

    from app.main import app

    yield TestClient(app)
