"""Suite-wide guards."""

import openai
import pytest

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
