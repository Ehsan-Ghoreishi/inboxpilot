"""Question answering over stored emails. Requires `X-API-Key`."""

from fastapi import APIRouter, Depends

from app.api.auth import require_api_key
from app.config import get_settings
from app.integrations.langfuse import flush_tracing
from app.schemas.ask import AskRequest, AskResponse
from app.services.ask import ask_question
from app.services.embedding import get_embedder

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    try:
        return ask_question(request, get_embedder(), get_settings().rag_min_score)
    finally:
        # Cloud Run throttles CPU after the response; send the trace now.
        flush_tracing()
