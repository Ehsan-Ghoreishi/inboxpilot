"""Engine pool configuration (no database connection is made)."""

import pytest

from app.config import get_settings
from app.database.session import get_engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-test",
        "NYLAS_API_KEY": "nylas-test",
        "NYLAS_CLIENT_ID": "client-test",
        "NYLAS_GRANT_ID": "grant-test",
        "WEBHOOK_SECRET": "super-secret",
        "SERVER_URL": "http://localhost:8000",
        "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/db",
        "APP_API_KEY": "app-test",
        "PROCESSING_MODE": "test",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("DB_POOL_SIZE", raising=False)
    monkeypatch.delenv("DB_MAX_OVERFLOW", raising=False)
    get_settings.cache_clear()
    get_engine.cache_clear()
    yield monkeypatch
    get_engine.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def engine(env):
    return get_engine()


def test_engine_pool_is_bounded_and_resilient(engine):
    pool = engine.pool

    assert pool.size() == 5
    assert pool._max_overflow == 0
    assert pool._pre_ping is True
    assert pool._recycle == 300


def test_engine_pool_uses_db_pool_settings(env):
    env.setenv("DB_POOL_SIZE", "2")
    env.setenv("DB_MAX_OVERFLOW", "1")

    pool = get_engine().pool

    assert pool.size() == 2
    assert pool._max_overflow == 1
