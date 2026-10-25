"""Service layer: semantic search over stored email embeddings."""

from sqlalchemy.orm import Session

from app.database.models import EmailRecord
from app.database.repository import search_email_embeddings
from app.schemas.search import SearchRequest, SearchResult
from app.services.embedding import Embedder

SNIPPET_CHARS = 200


def build_snippet(body: str) -> str:
    collapsed = " ".join(body.split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[: SNIPPET_CHARS - 1].rstrip() + "…"


def format_sender(email: EmailRecord) -> str:
    if email.sender_name:
        return f"{email.sender_name} <{email.sender_email}>"
    return email.sender_email


def retrieve_emails(
    session: Session, request: SearchRequest, embedder: Embedder, min_score: float
) -> list[tuple[EmailRecord, float]]:
    """Embed `request.query` with `embedder` and return `(email, score)`
    pairs for the closest emails embedded with that same model, best first."""
    [query_embedding] = embedder.embed_texts([request.query])
    return search_email_embeddings(
        session,
        query_embedding=query_embedding,
        model=embedder.model,
        top_k=request.top_k,
        min_score=min_score,
        category=request.category.value if request.category else None,
        sender=request.sender,
        date_from=request.date_from,
        date_to=request.date_to,
    )


def to_search_result(email: EmailRecord, score: float) -> SearchResult:
    return SearchResult(
        email_id=email.id,
        score=score,
        subject=email.subject,
        sender=format_sender(email),
        received_at=email.received_at,
        snippet=build_snippet(email.body),
    )


def search_emails(
    session: Session, request: SearchRequest, embedder: Embedder, min_score: float
) -> list[SearchResult]:
    return [
        to_search_result(email, score)
        for email, score in retrieve_emails(session, request, embedder, min_score)
    ]
