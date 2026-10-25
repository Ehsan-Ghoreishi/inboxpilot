"""Service layer: answer a question from the emails most similar to it.

Retrieval is the same as /search, including `RAG_MIN_SCORE`. When no
email passes it, the answer is a fixed "insufficient evidence" response
and the model is never called, so it has nothing to invent an answer
from. The model's citations are checked against the emails it was given;
an answer that fails the check is retried once, then refused.
Read-only: nothing here proposes or takes an action.
"""

import logging

from app.agents.email_qa import answer_question
from app.database.session import session_scope
from app.schemas.ask import AskRequest, AskResponse, AskResult, ContextEmail
from app.schemas.search import SearchResult
from app.services.embedding import Embedder
from app.services.search import format_sender, retrieve_emails, to_search_result

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = "Insufficient evidence in stored emails."
INVALID_ANSWER = "Could not produce a verifiable answer from stored emails."
# Total model calls per question: the first attempt plus one retry.
MAX_ATTEMPTS = 2


def _retrieve_context(
    request: AskRequest, embedder: Embedder, min_score: float
) -> list[tuple[ContextEmail, SearchResult]]:
    with session_scope() as session:
        matches = retrieve_emails(
            session, request.to_search_request(), embedder, min_score
        )
        return [
            (
                ContextEmail(
                    email_id=email.id,
                    sender=format_sender(email),
                    subject=email.subject,
                    received_at=email.received_at,
                    body=email.body,
                ),
                to_search_result(email, score),
            )
            for email, score in matches
        ]


def _refusal(answer: str) -> AskResponse:
    return AskResponse(answerable=False, answer=answer, citations=[], sources=[])


def citations_are_valid(result: AskResult, context_size: int) -> bool:
    """Every citation is one of the numbers 1..`context_size` the model was
    given, and an answerable result cites at least one email."""
    if result.answerable and not result.citations:
        return False
    return all(1 <= number <= context_size for number in result.citations)


def _attempt_answer(question: str, emails: list[ContextEmail]) -> AskResult | None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = answer_question(question, emails)
        except ValueError:
            logger.warning("Unparsed /ask answer (attempt %d)", attempt)
            continue
        if citations_are_valid(result, len(emails)):
            return result
        logger.warning(
            "Invalid /ask citations %s for %d emails (attempt %d)",
            result.citations,
            len(emails),
            attempt,
        )
    return None


def ask_question(
    request: AskRequest,
    embedder: Embedder,
    min_score: float,
) -> AskResponse:
    # The model call happens after the session closes, so a slow answer
    # never holds a database connection.
    context = _retrieve_context(request, embedder, min_score)
    if not context:
        return _refusal(INSUFFICIENT_EVIDENCE_ANSWER)

    result = _attempt_answer(request.question, [email for email, _ in context])
    if result is None:
        return _refusal(INVALID_ANSWER)

    # Citation numbers are 1-based positions in `context`; dedupe, keep order.
    cited = [context[number - 1] for number in dict.fromkeys(result.citations)]
    return AskResponse(
        answerable=result.answerable,
        answer=result.answer,
        citations=[email.email_id for email, _ in cited],
        sources=[hit for _, hit in cited],
    )
