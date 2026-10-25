"""Integration tests: POST /ask against a real test database, with
hand-built embeddings, a stub query embedder, and a stub answering agent
(no OpenAI, no Nylas)."""

import math

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import get_settings
from app.database.models import AuditLog, EmailRecord, PendingAction
from app.database.session import session_scope
from app.schemas.ask import AskResult, ContextEmail
from app.services.ask import INSUFFICIENT_EVIDENCE_ANSWER, INVALID_ANSWER
from tests.integration.test_search_db import StubEmbedder, _store, _truncate
from tests.test_ask_service import INJECTION_BODY

API_KEY = {"X-API-Key": "app-test"}


class StubAnswerer:
    """Records what the agent was given; answers with fixed citation numbers
    (all of them, 1..k, by default)."""

    def __init__(self, citations: list[int] | None = None) -> None:
        self.citations = citations
        self.calls: list[tuple[str, list[ContextEmail]]] = []

    def __call__(self, question: str, emails: list[ContextEmail]) -> AskResult:
        self.calls.append((question, emails))
        numbers = self.citations
        return AskResult(
            answerable=True,
            answer="Stub answer.",
            citations=list(range(1, len(emails) + 1)) if numbers is None else numbers,
        )

    def context_ids(self) -> list[int]:
        [(_, emails), *_] = self.calls
        return [email.email_id for email in emails]


@pytest.fixture
def answerer(monkeypatch) -> StubAnswerer:
    stub = StubAnswerer()
    monkeypatch.setattr("app.services.ask.answer_question", stub)
    return stub


@pytest.fixture
def client(app_db_env, test_engine, monkeypatch, answerer) -> TestClient:
    monkeypatch.setattr("app.api.ask.get_embedder", lambda: StubEmbedder())
    _truncate(test_engine)
    from app.main import app

    yield TestClient(app)
    _truncate(test_engine)


def _ask(client, **body) -> dict:
    response = client.post("/ask", json={"question": "q", **body}, headers=API_KEY)
    assert response.status_code == 200
    return response.json()


def test_no_matching_emails_answers_without_calling_the_model(client, answerer):
    result = _ask(client)

    assert result == {
        "answerable": False,
        "answer": INSUFFICIENT_EVIDENCE_ANSWER,
        "citations": [],
        "sources": [],
    }
    assert answerer.calls == []


def test_context_is_retrieved_emails_best_first_with_full_body(client, answerer):
    with session_scope() as session:
        far = _store(session, key="far", angle=0.9)
        near = _store(session, key="near", angle=0.1)

    result = _ask(client, question="When is the interview?")

    [(question, emails)] = answerer.calls
    assert question == "When is the interview?"
    assert [e.email_id for e in emails] == [near, far]
    assert emails[0].body == "Synthetic body for near."
    assert emails[0].sender == "Test Sender <sender@example.com>"
    assert result["answer"] == "Stub answer."
    assert result["citations"] == [near, far]
    assert [s["email_id"] for s in result["sources"]] == [near, far]
    assert result["sources"][0]["score"] == pytest.approx(math.cos(0.1), abs=1e-5)


def test_duplicate_content_reaches_the_model_once(client, answerer):
    with session_scope() as session:
        near = _store(session, key="near", angle=0.1, content_hash="same")
        _store(session, key="copy", angle=0.2, content_hash="same")
        other = _store(session, key="other", angle=0.5)

    result = _ask(client, top_k=2)

    assert answerer.context_ids() == [near, other]
    assert result["citations"] == [near, other]


def test_min_score_gates_which_emails_reach_the_model(client, answerer, monkeypatch):
    with session_scope() as session:
        near = _store(session, key="near", angle=0.1)
        _store(session, key="weak", angle=1.4)  # cos ~= 0.17
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")
    get_settings.cache_clear()

    _ask(client)

    assert answerer.context_ids() == [near]


def test_min_score_excluding_everything_skips_the_model(client, answerer, monkeypatch):
    with session_scope() as session:
        _store(session, key="weak", angle=1.4)
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")
    get_settings.cache_clear()

    assert _ask(client)["sources"] == []
    assert answerer.calls == []


def test_filters_restrict_the_context(client, answerer):
    with session_scope() as session:
        invite = _store(session, key="invite", angle=0.5, category="interview_invite")
        _store(session, key="other", angle=0.0)

    _ask(client, category="interview_invite")

    assert answerer.context_ids() == [invite]


def test_only_cited_emails_are_sources(client, answerer):
    with session_scope() as session:
        near = _store(session, key="near", angle=0.1)
        cited = _store(session, key="cited", angle=0.5)
    answerer.citations = [2]

    result = _ask(client)

    assert answerer.context_ids() == [near, cited]
    assert result["citations"] == [cited]
    assert [s["email_id"] for s in result["sources"]] == [cited]


def test_out_of_range_citation_is_retried_then_refused(client, answerer):
    with session_scope() as session:
        given = _store(session, key="given", angle=0.1)
        _store(session, key="not-given", angle=0.2, category="rejection")
    # Only one email passes the filter, so 2 is out of range.
    answerer.citations = [1, 2]

    result = _ask(client, category="other")

    assert len(answerer.calls) == 2
    assert answerer.context_ids() == [given]
    assert result == {
        "answerable": False,
        "answer": INVALID_ANSWER,
        "citations": [],
        "sources": [],
    }


def test_answerable_answer_citing_nothing_is_refused(client, answerer):
    with session_scope() as session:
        _store(session, key="a", angle=0.1)
    answerer.citations = []

    result = _ask(client)

    assert len(answerer.calls) == 2
    assert (result["answerable"], result["sources"]) == (False, [])


def _row_counts() -> tuple[int, int, int]:
    with session_scope() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (EmailRecord, PendingAction, AuditLog)
        )


def test_injection_email_in_corpus_creates_no_pending_actions(client, answerer):
    with session_scope() as session:
        honest = _store(session, key="honest", angle=0.1)
        attack = _store(session, key="attack", angle=0.05)
        session.get(EmailRecord, attack).body = INJECTION_BODY
    before = _row_counts()

    result = _ask(client, question="Did anyone reject me?")

    assert _row_counts() == before
    assert before[1] == 0
    [(_, emails)] = answerer.calls
    assert [e.email_id for e in emails] == [attack, honest]
    assert emails[0].body == INJECTION_BODY
    assert result["answerable"] is True
