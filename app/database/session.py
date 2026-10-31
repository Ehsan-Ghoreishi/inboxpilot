"""Database engine/session, following the same cached-getter pattern used
by `app.config.get_settings()` and `app.integrations.nylas.client.get_client()`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    # Sized for a small serverless instance: a fixed pool with no
    # overflow caps connections per instance (instances x 5 must fit the
    # database's limit). pre_ping and a short recycle drop connections
    # that the database or a proxy closed while the instance was idle.
    return create_engine(
        get_settings().database_url,
        future=True,
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=300,
    )


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """A `Session` that commits on success, rolls back and re-raises on
    error, and always closes."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
