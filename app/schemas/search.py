"""API request/response schemas for semantic email search."""

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.email import EmailCategory

MAX_TOP_K = 20


# Non-blank text of at most 2000 characters, surrounding whitespace removed.
QueryText = Annotated[
    str, StringConstraints(min_length=1, max_length=2000, strip_whitespace=True)
]


class SearchFilters(BaseModel):
    """Retrieval options shared by /search and /ask."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    # Matches the email's latest classification.
    category: EmailCategory | None = None
    # Case-insensitive substring of the sender's address or display name.
    sender: Annotated[
        str, StringConstraints(min_length=1, max_length=320, strip_whitespace=True)
    ] | None = None
    # Inclusive bounds on `received_at`.
    date_from: datetime | None = None
    date_to: datetime | None = None

    @model_validator(mode="after")
    def _date_range_is_ordered(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class SearchRequest(SearchFilters):
    query: QueryText


class SearchResult(BaseModel):
    email_id: int
    # Cosine similarity: 1 - cosine distance.
    score: float
    subject: str
    sender: str
    received_at: datetime
    snippet: str
