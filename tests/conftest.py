"""Suite-wide guards."""

import pytest

from app.services.embedding import FakeEmbedder


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch) -> FakeEmbedder:
    """The background task embeds with a `FakeEmbedder`, never OpenAI."""
    embedder = FakeEmbedder()
    monkeypatch.setattr("app.workflows.background.get_embedder", lambda: embedder)
    return embedder
