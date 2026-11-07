"""Suite-wide guards."""

import functools
import json
from typing import Any

import openai
import pytest
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.config import Settings, get_settings
from app.integrations.langfuse import (
    flush_tracing,
    init_tracing,
    shutdown_tracing,
    tracing,
)
from app.services.embedding import FakeEmbedder


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch) -> FakeEmbedder:
    """The background task embeds with a `FakeEmbedder`, never OpenAI."""
    embedder = FakeEmbedder()
    monkeypatch.setattr("app.workflows.background.get_embedder", lambda: embedder)
    return embedder


@pytest.fixture(autouse=True)
def no_openai_network(monkeypatch) -> None:
    """Any real OpenAI request fails the test: tests must mock the client
    or use `FakeEmbedder`, never need a real key or spend money."""

    def forbidden(*args, **kwargs):
        raise RuntimeError("Tests must not call the OpenAI API; mock the client.")

    monkeypatch.setattr(openai.OpenAI, "request", forbidden)


_LANGFUSE_ENV = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
)


def _reset_tracing() -> None:
    # Tracing reads the cached settings lazily: don't let one test's (which
    # may have had tracing on) decide for the next.
    get_settings.cache_clear()
    shutdown_tracing()
    LangfuseResourceManager.reset()


@pytest.fixture(autouse=True)
def tracing_disabled(monkeypatch):
    """Tracing is off in every test, whatever the developer's shell or .env
    holds, and no test can build a Langfuse client that would export over
    the network: a client needs `tracing_enabled=False` or an in-memory
    `span_exporter` (see `tests/test_tracing.py`)."""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    for name in _LANGFUSE_ENV:
        monkeypatch.delenv(name, raising=False)

    real_client = tracing.Langfuse

    def offline_only(**kwargs):
        if kwargs.get("tracing_enabled") is False or "span_exporter" in kwargs:
            return real_client(**kwargs)
        # Keeps the keys out of pytest's failure output.
        del kwargs
        raise RuntimeError("Tests must not create a Langfuse client that exports.")

    monkeypatch.setattr(tracing, "Langfuse", offline_only)
    _reset_tracing()
    yield
    _reset_tracing()


class Traces:
    """Tracing switched on for one test, exported to memory only."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter
        # Scores are sent by a separate network consumer, so they are
        # recorded here instead of being queued.
        self.scores: list[dict[str, Any]] = []

    @staticmethod
    def settings(**overrides: Any) -> Settings:
        values: dict[str, Any] = dict(
            openai_api_key="sk-test",
            openai_model="gpt-test",
            nylas_api_key="nylas-test",
            nylas_client_id="client-test",
            nylas_grant_id="grant-test",
            webhook_secret="super-secret",
            server_url="http://localhost:8000",
            database_url="postgresql://user:pass@localhost/db",
            app_api_key="app-test",
            processing_mode="live",
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            langfuse_tracing_enabled=True,
        )
        values.update(overrides)
        return Settings(_env_file=None, **values)

    def start(self, **overrides: Any) -> Settings:
        """Create the client from `settings(**overrides)` and return them."""
        settings = self.settings(**overrides)
        assert init_tracing(settings)
        tracing._client.score_current_trace = lambda **kwargs: self.scores.append(kwargs)
        return settings

    def spans(self) -> list[ReadableSpan]:
        flush_tracing(timeout_seconds=10)
        return list(self._exporter.get_finished_spans())

    def named(self, name: str) -> ReadableSpan:
        [span] = [span for span in self.spans() if span.name == name]
        return span


def attribute(span: ReadableSpan, key: str) -> Any:
    """A span attribute with JSON-encoded values (input, output, metadata)
    decoded."""
    value = span.attributes[key]
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


@pytest.fixture
def traces(monkeypatch) -> Traces:
    """Tracing on, with a real Langfuse client whose exporter is in memory:
    the real mask hook runs, nothing leaves the process."""
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setattr(
        tracing, "Langfuse", functools.partial(tracing.Langfuse, span_exporter=exporter)
    )
    return Traces(exporter)
