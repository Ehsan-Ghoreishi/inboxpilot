"""Engine pool configuration (no database connection is made)."""

import pytest

from app.config import get_settings
from app.database.session import get_engine


@pytest.fixture
def engine(tmp_path, monkeypatch):
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
    get_settings.cache_clear()
    get_engine.cache_clear()
    yield get_engine()
    get_engine.cache_clear()
    get_settings.cache_clear()


def test_engine_pool_is_bounded_and_resilient(engine):
    pool = engine.pool

    assert pool.size() == 5
    assert pool._max_overflow == 0
    assert pool._pre_ping is True
    assert pool._recycle == 300
