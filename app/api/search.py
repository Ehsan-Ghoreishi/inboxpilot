"""Semantic search over stored emails. Requires `X-API-Key`."""

from fastapi import APIRouter, Depends

from app.api.auth import require_api_key
from app.config import get_settings
from app.database.session import session_scope
from app.schemas.search import SearchRequest, SearchResult
from app.services.embedding import get_embedder
from app.services.search import search_emails

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/search")
def search(request: SearchRequest) -> list[SearchResult]:
    with session_scope() as session:
        return search_emails(
            session, request, get_embedder(), get_settings().rag_min_score
        )
