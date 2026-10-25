"""Service layer: compute and store vector embeddings of emails.

An `Embedder` turns texts into vectors. `OpenAIEmbedder` is the real
one; `FakeEmbedder` is deterministic and offline, so tests never call
OpenAI. `embed_email()` is idempotent: it hashes the text it would
embed and skips the embedder entirely when the stored embedding for
that (email, model) already has the same hash.
"""

import hashlib
import math
import random
from functools import lru_cache
from typing import Protocol

from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import EMBEDDING_DIM, EmailRecord
from app.database.repository import get_embedding_content_hash, upsert_embedding

MAX_EMBEDDING_CHARS = 8000


class Embedder(Protocol):
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


class FakeEmbedder:
    """Deterministic unit vectors seeded by each text's SHA-256: the same
    text always gets the same vector, different texts (almost surely)
    different ones. Records every call in `calls`."""

    model = "fake-hash-embedder"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        values = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return OpenAIEmbedder(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )


def build_embedding_text(email: EmailRecord) -> str:
    sender = (
        f"{email.sender_name} <{email.sender_email}>"
        if email.sender_name
        else email.sender_email
    )
    text = (
        f"Subject: {email.subject}\n"
        f"From: {sender}\n"
        f"Date: {email.received_at.isoformat()}\n\n"
        f"{email.body}"
    )
    return text[:MAX_EMBEDDING_CHARS]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_email(session: Session, email: EmailRecord, embedder: Embedder) -> bool:
    """Embed `email` with `embedder` and store it, in `session`.

    Returns `False` (without calling the embedder) when the stored
    embedding for this email and model is already up to date.
    """
    text = build_embedding_text(email)
    digest = content_hash(text)
    if get_embedding_content_hash(session, email.id, embedder.model) == digest:
        return False

    [vector] = embedder.embed_texts([text])
    upsert_embedding(
        session,
        email_id=email.id,
        model=embedder.model,
        content_hash=digest,
        embedding=vector,
    )
    return True
