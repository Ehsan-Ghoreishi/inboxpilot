"""API and agent schemas for answering questions over stored emails."""

from datetime import datetime

from pydantic import BaseModel

from app.schemas.search import QueryText, SearchFilters, SearchRequest, SearchResult


class AskRequest(SearchFilters):
    question: QueryText

    def to_search_request(self) -> SearchRequest:
        return SearchRequest(
            query=self.question, **self.model_dump(exclude={"question"})
        )


class ContextEmail(BaseModel):
    """A retrieved email as the answering agent sees it."""

    email_id: int
    sender: str
    subject: str
    received_at: datetime
    body: str


class AskResult(BaseModel):
    """The answering agent's structured output."""

    answerable: bool
    answer: str
    # 1-based positions of the context emails the answer relies on, as
    # numbered in the prompt ([1]..[k]) — not database ids.
    citations: list[int]


class AskResponse(BaseModel):
    answerable: bool
    answer: str
    # Database ids of the cited emails, in citation order.
    citations: list[int]
    # The cited emails, in the same order as `citations`.
    sources: list[SearchResult]
