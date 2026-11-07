"""Langfuse tracing: configuration, startup without keys, the traces the
email and ask flows produce, and what they leave out.

Nothing here reaches the network: with tracing off no client is built, and
with it on the client exports to an in-memory exporter (the `traces`
fixture in conftest.py)."""

import logging
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langfuse.openai import OpenAI

from app.agents import email_analyzer
from app.agents.email_analyzer import AnalysisRun, analyze_with_metadata
from app.integrations.langfuse import (
    annotate,
    content_tracing_allowed,
    flush_tracing,
    init_tracing,
    shutdown_tracing,
    traced,
    tracing,
    tracing_requested,
)
from app.integrations.langfuse.masking import mask_otel_spans
from app.schemas.ask import AskRequest, AskResult
from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailCategory,
    EmailParticipant,
    SuggestedAction,
    Urgency,
)
from app.services import ask as ask_service
from app.services.embedding import FakeEmbedder
from app.workflows.background import run_triage_safely
from tests.conftest import attribute

# The suite blocks `OpenAI.request`; the wrapper test below needs the real
# method, with its transport replaced by an in-process mock.
_REAL_OPENAI_REQUEST = openai.OpenAI.request

INPUT = "langfuse.observation.input"
OUTPUT = "langfuse.observation.output"
METADATA = "langfuse.observation.metadata."

SENDER = "sender@example.com"
SECRET_SUBJECT = "Synthetic subject that must not reach Langfuse"
SECRET_BODY = "Synthetic body that must not reach Langfuse"


# --- configuration ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "mode", "allowed"),
    [
        (True, "test", True),
        (True, "live", False),
        (True, "manual", False),
        (False, "test", False),
        (False, "live", False),
    ],
)
def test_email_content_is_traced_only_with_the_flag_in_test_mode(
    traces, flag, mode, allowed
):
    settings = traces.settings(trace_email_content=flag, processing_mode=mode)

    assert content_tracing_allowed(settings) is allowed


def test_content_flag_defaults_to_off(traces):
    assert traces.settings().trace_email_content is False


@pytest.mark.parametrize(
    ("overrides", "requested"),
    [
        ({}, True),
        ({"langfuse_tracing_enabled": False}, False),
        ({"langfuse_public_key": None}, False),
        ({"langfuse_secret_key": None}, False),
        ({"langfuse_public_key": "  ", "langfuse_secret_key": ""}, False),
    ],
)
def test_tracing_needs_the_flag_and_both_keys(traces, overrides, requested):
    assert tracing_requested(traces.settings(**overrides)) is requested


@pytest.fixture
def created(monkeypatch) -> list[dict]:
    """Record the arguments of every Langfuse client the app builds."""
    kwargs_seen: list[dict] = []

    def record(**kwargs):
        kwargs_seen.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tracing, "Langfuse", record)
    return kwargs_seen


def test_mask_hook_is_installed_when_content_is_not_allowed(traces, created):
    settings = traces.settings(trace_email_content=True, processing_mode="live")

    assert init_tracing(settings)

    [kwargs] = created
    assert kwargs["mask_otel_spans"] is mask_otel_spans
    assert kwargs["public_key"] == "pk-test"
    assert kwargs["secret_key"] == "sk-test"


def test_mask_hook_is_installed_by_default(traces, created):
    assert init_tracing(traces.settings(processing_mode="test"))

    assert created[0]["mask_otel_spans"] is mask_otel_spans


def test_no_mask_hook_only_with_the_flag_in_test_mode(traces, created):
    assert init_tracing(
        traces.settings(trace_email_content=True, processing_mode="test")
    )

    assert created[0]["mask_otel_spans"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"langfuse_public_key": None, "langfuse_secret_key": None},
        {"langfuse_tracing_enabled": False},
        {"langfuse_secret_key": None},
    ],
)
def test_unconfigured_tracing_builds_only_a_disabled_client(
    traces, created, overrides
):
    assert not init_tracing(traces.settings(**overrides))

    [kwargs] = created
    assert kwargs["tracing_enabled"] is False
    assert "mask_otel_spans" not in kwargs


def test_unloadable_settings_leave_tracing_off(created, monkeypatch):
    def broken():
        raise RuntimeError("Missing or invalid required environment variable(s)")

    monkeypatch.setattr(tracing, "get_settings", broken)

    assert not init_tracing()
    assert created[0]["tracing_enabled"] is False


def test_the_client_is_built_once(traces, created):
    settings = traces.settings()

    init_tracing(settings)
    init_tracing(settings)

    assert len(created) == 1


def test_blank_langfuse_values_mean_unset(traces):
    settings = traces.settings(
        langfuse_public_key="", langfuse_secret_key="  ", langfuse_base_url=" "
    )

    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_base_url is None


# --- starting without keys ---------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"langfuse_public_key": None, "langfuse_secret_key": None},
        {"langfuse_tracing_enabled": False},
    ],
    ids=["no-keys", "tracing-disabled"],
)
def test_app_starts_and_serves_without_langfuse(traces, monkeypatch, caplog, overrides):
    from app.main import create_app

    settings = traces.settings(**overrides)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr(logging.getLogger("app.main"), "disabled", False)

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").json() == {"status": "ok"}

    assert "Tracing: langfuse=off email_content=redacted" in caplog.text


def test_startup_logs_tracing_state_without_the_keys(traces, monkeypatch, caplog):
    from app.main import create_app

    settings = traces.settings(langfuse_secret_key="sk-secret-value")
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr(logging.getLogger("app.main"), "disabled", False)

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(create_app(settings)):
            pass

    assert "Tracing: langfuse=on email_content=redacted" in caplog.text
    assert "sk-secret-value" not in caplog.text
    assert "pk-test" not in caplog.text


def test_startup_warns_when_the_content_flag_cannot_take_effect(
    traces, monkeypatch, caplog
):
    from app.main import log_safety_flags

    monkeypatch.setattr(logging.getLogger("app.main"), "disabled", False)

    with caplog.at_level(logging.INFO, logger="app.main"):
        log_safety_flags(traces.settings(trace_email_content=True, processing_mode="live"))

    assert "email_content=redacted" in caplog.text
    assert "TRACE_EMAIL_CONTENT is ignored" in caplog.text


def test_startup_does_not_warn_when_the_content_flag_is_in_effect(
    traces, monkeypatch, caplog
):
    from app.main import log_safety_flags

    monkeypatch.setattr(logging.getLogger("app.main"), "disabled", False)

    with caplog.at_level(logging.INFO, logger="app.main"):
        log_safety_flags(traces.settings(trace_email_content=True, processing_mode="test"))

    assert "email_content=on" in caplog.text
    assert "ignored" not in caplog.text


# --- tracing off: behaviour is unchanged -------------------------------------------


@traced("step")
def _add(a: int, b: int) -> int:
    annotate(total=a + b)
    return a + b


@traced("step")
def _boom() -> None:
    raise ValueError("kept as is")


def test_traced_functions_run_normally_with_tracing_off():
    assert _add(1, 2) == 3
    assert _add.__name__ == "_add"


def test_traced_functions_still_raise_with_tracing_off():
    with pytest.raises(ValueError, match="kept as is"):
        _boom()


def test_flush_and_shutdown_are_harmless_with_tracing_off():
    flush_tracing()
    shutdown_tracing()
    shutdown_tracing()
    assert _add(2, 2) == 4


def test_a_slow_flush_does_not_hold_the_caller(traces, monkeypatch, caplog):
    traces.start()
    monkeypatch.setattr(tracing, "_flush", lambda client: time.sleep(0.5))
    monkeypatch.setattr(logging.getLogger(tracing.__name__), "disabled", False)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=tracing.__name__):
        flush_tracing(timeout_seconds=0.05)

    assert time.monotonic() - started < 0.4
    assert "not finished" in caplog.text


def test_a_failing_flush_is_logged_not_raised(traces, monkeypatch, caplog):
    traces.start()
    monkeypatch.setattr(
        tracing._client, "flush", MagicMock(side_effect=OSError("down"))
    )
    monkeypatch.setattr(logging.getLogger(tracing.__name__), "disabled", False)

    with caplog.at_level(logging.WARNING, logger=tracing.__name__):
        flush_tracing()

    assert "flush failed: OSError" in caplog.text


# --- generic steps ------------------------------------------------------------------


def test_steps_nest_under_a_named_root_and_capture_nothing_by_themselves(traces):
    traces.start()

    @traced("email", root=True)
    def root() -> str:
        child("alice@example.com")
        return "result for alice@example.com"

    @traced("triage")
    def child(address: str) -> None:
        annotate(decided_by="sender_filter", dropped=None)

    root()

    parent, step = traces.named("email"), traces.named("triage")
    assert step.parent.span_id == parent.context.span_id
    assert step.context.trace_id == parent.context.trace_id
    assert attribute(parent, "langfuse.trace.name") == "email"
    assert attribute(step, METADATA + "decided_by") == "sender_filter"
    assert METADATA + "dropped" not in step.attributes
    for span in (parent, step):
        assert INPUT not in span.attributes
        assert OUTPUT not in span.attributes


def test_a_failing_step_is_recorded_as_an_error_and_still_raises(traces):
    traces.start()

    with pytest.raises(ValueError):
        _boom()

    assert attribute(traces.named("step"), "langfuse.observation.level") == "ERROR"


# --- the email trace -----------------------------------------------------------------


def _event(sender: str = SENDER, event_type: str = "message.created") -> dict:
    return {
        "specversion": "1.0",
        "type": event_type,
        "source": "/google/emails/realtime",
        "id": "evt-trace-1",
        "time": 1706800000,
        "data": {
            "application_id": "app-synthetic",
            "object": {
                "id": "msg-trace-1",
                "grant_id": "grant-synthetic",
                "thread_id": "thread-trace-1",
                "subject": SECRET_SUBJECT,
                "from": [{"name": "Test Sender", "email": sender}],
                "to": [{"email": "candidate@example.com"}],
                "date": 1706800000,
                "body": SECRET_BODY,
            }
        },
    }


def _run(confidence: float = 0.9) -> AnalysisRun:
    return AnalysisRun(
        analysis=EmailAnalysis(
            category=EmailCategory.RECRUITER_OUTREACH,
            urgency=Urgency.MEDIUM,
            suggested_action=SuggestedAction.DRAFT_REPLY,
            reasoning="Synthetic analysis for testing.",
            confidence=confidence,
        ),
        model="gpt-test",
        prompt_version="analyze_v2",
        input_tokens=120,
        output_tokens=40,
        latency_ms=850,
    )


@pytest.fixture
def pipeline(monkeypatch):
    """The triage flow with persistence and the model stubbed out."""

    def wire(settings, run: AnalysisRun | None = None) -> None:
        run = run or _run()
        monkeypatch.setattr("app.workflows.background.get_settings", lambda: settings)
        # Step metadata (`run_flags`) reads the process settings itself.
        monkeypatch.setattr(tracing, "get_settings", lambda: settings)
        monkeypatch.setattr("app.workflows.background.session_scope", nullcontext)
        monkeypatch.setattr(
            "app.workflows.triage_workflow.insert_email_if_new",
            lambda session, email: SimpleNamespace(id=1),
        )
        monkeypatch.setattr(
            "app.workflows.triage_workflow.save_classification", MagicMock()
        )
        monkeypatch.setattr(
            "app.workflows.triage_workflow.propose_action",
            lambda *args, **kwargs: SimpleNamespace(id=42),
        )
        monkeypatch.setattr(
            "app.workflows.triage_workflow.analyze_with_metadata",
            lambda email: run,
        )
        monkeypatch.setattr(
            "app.workflows.background.get_email_by_message_id",
            lambda session, message_id: SimpleNamespace(id=1),
        )
        monkeypatch.setattr(
            "app.workflows.background.embed_email", lambda *args, **kwargs: True
        )

    return wire


def test_an_email_produces_one_trace_with_the_four_steps(traces, pipeline):
    settings = traces.start(processing_mode="test", test_email=SENDER, dry_run=True)
    pipeline(settings)

    run_triage_safely("evt-trace-1", _event())

    spans = traces.spans()
    assert {span.name for span in spans} == {
        "email",
        "triage",
        "classify",
        "propose_action",
        "embed",
    }
    assert len({span.context.trace_id for span in spans}) == 1
    root = traces.named("email")
    assert root.parent is None
    assert attribute(root, "langfuse.trace.name") == "email"
    assert attribute(root, METADATA + "outcome") == "action_proposed"
    for name in ("triage", "classify", "propose_action", "embed"):
        step = traces.named(name)
        assert step.parent.span_id == root.context.span_id
        assert attribute(step, METADATA + "processing_mode") == "test"
        assert attribute(step, METADATA + "dry_run") is True


def test_step_metadata_records_who_decided_and_what(traces, pipeline):
    settings = traces.start(processing_mode="test", test_email=SENDER)
    pipeline(settings)

    run_triage_safely("evt-trace-1", _event())

    triage = traces.named("triage")
    assert attribute(triage, METADATA + "decided_by") == "sender_filter"
    assert attribute(triage, METADATA + "decision") == "process"
    classify = traces.named("classify")
    assert attribute(classify, METADATA + "decided_by") == "llm"
    assert attribute(classify, METADATA + "category") == "recruiter_outreach"
    assert attribute(classify, METADATA + "confidence") == 0.9
    propose = traces.named("propose_action")
    assert attribute(propose, METADATA + "decided_by") == "category_table"
    assert attribute(propose, METADATA + "status") == "draft_reply"
    assert attribute(propose, METADATA + "proposed") is True
    assert attribute(propose, METADATA + "pending_action_id") == 42
    assert attribute(traces.named("embed"), METADATA + "embedded") is True


def test_low_confidence_is_attributed_to_the_threshold(traces, pipeline):
    settings = traces.start(processing_mode="live")
    pipeline(settings, _run(confidence=0.2))

    run_triage_safely("evt-trace-1", _event())

    propose = traces.named("propose_action")
    assert attribute(propose, METADATA + "decided_by") == "confidence_threshold"
    assert attribute(propose, METADATA + "status") == "needs_review"
    assert attribute(propose, METADATA + "proposed") is False
    assert attribute(traces.named("email"), METADATA + "outcome") == "needs_review"


def test_a_filtered_email_stops_after_the_triage_step(traces, pipeline):
    settings = traces.start(processing_mode="test", test_email=SENDER)
    pipeline(settings)

    run_triage_safely("evt-trace-1", _event(sender="stranger@example.net"))

    names = {span.name for span in traces.spans()}
    assert names == {"email", "triage", "embed"}
    assert attribute(traces.named("triage"), METADATA + "decision") == "skip"
    assert attribute(traces.named("email"), METADATA + "outcome") == "filtered"


def test_events_that_are_not_messages_produce_no_trace(traces, pipeline):
    settings = traces.start()
    pipeline(settings)

    run_triage_safely("evt-1", _event(event_type="grant.expired"))

    assert traces.spans() == []


def test_a_failure_in_the_flow_ends_the_trace_as_an_error_outcome(
    traces, pipeline, monkeypatch
):
    settings = traces.start(processing_mode="live")
    pipeline(settings)

    def broken(email):
        raise RuntimeError(f"model said no about {SECRET_BODY}")

    monkeypatch.setattr("app.workflows.triage_workflow.analyze_with_metadata", broken)

    assert run_triage_safely("evt-trace-1", _event()) is None

    root = traces.named("email")
    assert attribute(root, METADATA + "outcome") == "error"
    assert attribute(traces.named("classify"), "langfuse.observation.level") == "ERROR"


def test_a_trace_in_redacted_mode_carries_no_email_text(traces, pipeline, monkeypatch):
    settings = traces.start(processing_mode="live", trace_email_content=True)
    pipeline(settings)

    def broken(email):
        raise RuntimeError(f"model said no about {SECRET_BODY} to {SENDER}")

    monkeypatch.setattr("app.workflows.triage_workflow.analyze_with_metadata", broken)
    run_triage_safely("evt-trace-1", _event())

    for span in traces.spans():
        text = " ".join(str(value) for value in span.attributes.values())
        for secret in (SECRET_SUBJECT, SECRET_BODY, SENDER, "candidate@example.com"):
            assert secret not in text
    assert attribute(traces.named("classify"), "langfuse.observation.status_message") == (
        "[redacted]"
    )


# --- the model call, through the real Langfuse OpenAI wrapper -------------------------

_PARSED_ANSWER = (
    '{"category":"interview_invite","company":null,"role":null,"deadline":null,'
    '"start_time":null,"timezone":null,"duration_minutes":null,'
    '"needs_clarification":false,"urgency":"low","suggested_action":"none",'
    '"reasoning":"Synthetic reasoning about jane@example.com.","confidence":0.8}'
)


def _responses_payload() -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": _PARSED_ANSWER, "annotations": []}
                ],
            }
        ],
        "usage": {
            "input_tokens": 123,
            "output_tokens": 45,
            "total_tokens": 168,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


@pytest.fixture
def wrapped_client(monkeypatch):
    """The agent's client: Langfuse's OpenAI class, answering from memory."""
    monkeypatch.setattr(openai.OpenAI, "request", _REAL_OPENAI_REQUEST)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_responses_payload()))
    client = OpenAI(api_key="sk-test", http_client=httpx.Client(transport=transport))
    monkeypatch.setattr(email_analyzer, "_get_client", lambda: client)
    return client


ANALYZED_EMAIL = Email(
    id="m1",
    thread_id="t1",
    grant_id="g1",
    sender=EmailParticipant(name="Jane Recruiter", email="jane@example.com"),
    recipients=[EmailParticipant(email="candidate@example.org")],
    subject=SECRET_SUBJECT,
    body=SECRET_BODY,
    received_at=datetime(2026, 2, 5, 9, 0, tzinfo=timezone.utc),
)


def test_classification_generation_reports_model_and_tokens(
    traces, wrapped_client, monkeypatch
):
    settings = traces.start()
    monkeypatch.setattr(email_analyzer, "get_settings", lambda: settings)

    run = analyze_with_metadata(ANALYZED_EMAIL)

    assert run.input_tokens == 123
    generation = traces.named("classify_llm")
    assert attribute(generation, "langfuse.observation.type") == "generation"
    assert attribute(generation, "langfuse.observation.model.name") == "gpt-test"
    usage = attribute(generation, "langfuse.observation.usage_details")
    assert usage["input_tokens"] == 123
    assert usage["output_tokens"] == 45


def test_classification_generation_is_redacted_by_default(
    traces, wrapped_client, monkeypatch
):
    settings = traces.start(processing_mode="live", trace_email_content=True)
    monkeypatch.setattr(email_analyzer, "get_settings", lambda: settings)

    analyze_with_metadata(ANALYZED_EMAIL)

    generation = traces.named("classify_llm")
    exported = generation.attributes[INPUT] + generation.attributes[OUTPUT]
    for secret in (
        SECRET_SUBJECT,
        SECRET_BODY,
        "jane@example.com",
        "candidate@example.org",
        "Jane Recruiter",
    ):
        assert secret not in exported
    messages = attribute(generation, INPUT)
    assert f"[redacted: body {len(SECRET_BODY)} chars]" in messages[-1]["content"]
    assert attribute(generation, OUTPUT)["redacted"] is True
    usage = attribute(generation, "langfuse.observation.usage_details")
    assert usage["input_tokens"] == 123


def test_classification_generation_has_content_with_flag_in_test_mode(
    traces, wrapped_client, monkeypatch
):
    settings = traces.start(processing_mode="test", trace_email_content=True)
    monkeypatch.setattr(email_analyzer, "get_settings", lambda: settings)

    analyze_with_metadata(ANALYZED_EMAIL)

    generation = traces.named("classify_llm")
    assert SECRET_BODY in generation.attributes[INPUT]
    assert "jane@example.com" in generation.attributes[INPUT]
    assert "interview_invite" in generation.attributes[OUTPUT]


def test_embedding_calls_never_export_the_text_by_default(
    traces, monkeypatch
):
    traces.start(processing_mode="live")
    monkeypatch.setattr(openai.OpenAI, "request", _REAL_OPENAI_REQUEST)
    payload = {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = OpenAI(api_key="sk-test", http_client=httpx.Client(transport=transport))

    client.embeddings.create(
        model="text-embedding-3-small", input=[f"Subject: x\n\n{SECRET_BODY} {SENDER}"]
    )

    span = traces.named("OpenAI-embedding")
    assert SECRET_BODY not in span.attributes[INPUT]
    assert SENDER not in span.attributes[INPUT]
    assert attribute(span, INPUT)["chars"] == [len(f"Subject: x\n\n{SECRET_BODY} {SENDER}")]
    assert attribute(span, "langfuse.observation.usage_details") == {"input": 4}


# --- the ask trace ---------------------------------------------------------------------


@pytest.fixture
def ask_flow(monkeypatch):
    """`ask_question` with retrieval and the model stubbed out."""

    def wire(matches: list[tuple[SimpleNamespace, float]], answers: list[AskResult]):
        monkeypatch.setattr(ask_service, "session_scope", nullcontext)
        monkeypatch.setattr(
            ask_service, "retrieve_emails", lambda *args, **kwargs: matches
        )
        replies = iter(answers)
        monkeypatch.setattr(
            ask_service, "answer_question", lambda question, emails: next(replies)
        )

    return wire


def _record(email_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=email_id,
        subject=SECRET_SUBJECT,
        sender_name="Jane Recruiter",
        sender_email="jane@example.com",
        received_at=datetime(2026, 2, 5, 9, 0, tzinfo=timezone.utc),
        body=SECRET_BODY,
    )


def _ask(min_score: float = 0.30):
    request = AskRequest(question="When is my interview?", top_k=3)
    return ask_service.ask_question(request, FakeEmbedder(), min_score)


def test_ask_traces_retrieval_and_the_answer(traces, ask_flow):
    traces.start()
    ask_flow(
        [(_record(7), 0.512345), (_record(8), 0.4)],
        [AskResult(answerable=True, answer="Friday.", citations=[1])],
    )

    response = _ask()

    assert response.citations == [7]
    root = traces.named("ask")
    assert root.parent is None
    assert attribute(root, "langfuse.trace.name") == "ask"
    assert attribute(root, METADATA + "answerable") is True
    assert attribute(root, METADATA + "citations_valid") is True
    assert traces.scores == [
        dict(name="answerable", value=1.0, data_type="BOOLEAN"),
        dict(name="citations_valid", value=1.0, data_type="BOOLEAN"),
    ]
    retrieve = traces.named("retrieve")
    assert retrieve.parent.span_id == root.context.span_id
    assert attribute(retrieve, METADATA + "top_k") == 3
    assert attribute(retrieve, METADATA + "scores") == [0.5123, 0.4]
    assert attribute(retrieve, METADATA + "threshold") == 0.30


def test_ask_without_evidence_is_traced_as_a_refusal(traces, ask_flow):
    traces.start()
    ask_flow([], [])

    assert not _ask().answerable

    assert attribute(traces.named("ask"), METADATA + "answerable") is False
    assert attribute(traces.named("ask"), METADATA + "refusal") == "insufficient_evidence"
    assert attribute(traces.named("retrieve"), METADATA + "scores") == []
    assert traces.scores == [dict(name="answerable", value=0.0, data_type="BOOLEAN")]


def test_ask_with_invalid_citations_is_traced_as_invalid(traces, ask_flow):
    traces.start()
    bad = AskResult(answerable=True, answer="Friday.", citations=[9])
    ask_flow([(_record(7), 0.5)], [bad, bad])

    assert not _ask().answerable

    root = traces.named("ask")
    assert attribute(root, METADATA + "citations_valid") is False
    assert attribute(root, METADATA + "refusal") == "invalid_answer"
    assert traces.scores == [
        dict(name="answerable", value=0.0, data_type="BOOLEAN"),
        dict(name="citations_valid", value=0.0, data_type="BOOLEAN"),
    ]


def test_ask_trace_carries_no_email_or_answer_text(traces, ask_flow):
    traces.start(processing_mode="live", trace_email_content=True)
    ask_flow(
        [(_record(7), 0.5)],
        [AskResult(answerable=True, answer="Friday, says jane@example.com", citations=[1])],
    )

    _ask()

    for span in traces.spans():
        text = " ".join(str(value) for value in span.attributes.values())
        for secret in (SECRET_SUBJECT, SECRET_BODY, "jane@example.com", "Friday"):
            assert secret not in text


def test_the_secret_key_reaches_the_client_as_plain_text_only_there(traces, created):
    settings = traces.settings(langfuse_secret_key="sk-secret-value")

    assert "sk-secret-value" not in repr(settings)
    assert init_tracing(settings)

    assert created[0]["secret_key"] == "sk-secret-value"


def test_scores_are_not_sent_with_tracing_off():
    from app.integrations.langfuse import score_trace

    score_trace("answerable", True)  # no client: must be a no-op


def test_a_failing_client_constructor_cannot_print_the_keys(tmp_path):
    """With the repo's pytest config, a failure shows no arguments or locals
    (the earlier leak was the failing frame's keyword arguments)."""
    (tmp_path / "test_leak.py").write_text(
        "KEY = 'sk-lf-' + 'leaked-value'\n"
        "def build(**kwargs):\n"
        "    raise RuntimeError('boom')\n"
        "def test_it():\n"
        "    build(secret_key=KEY)\n"
    )
    config = Path(__file__).resolve().parents[1] / "pyproject.toml"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(config), "--rootdir", str(tmp_path),
         "-p", "no:cacheprovider", str(tmp_path / "test_leak.py")],
        capture_output=True, text=True, cwd=tmp_path,
    )

    assert "boom" in result.stdout  # it did fail, and was reported
    assert "sk-lf-leaked-value" not in result.stdout + result.stderr
