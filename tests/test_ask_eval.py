"""Ask eval: metric math, refusal attribution, usage recording, and a full
run with a fake asker (no database, no OpenAI calls)."""

import json
from types import SimpleNamespace

import pytest

from app.services.ask import INSUFFICIENT_EVIDENCE_ANSWER, INVALID_ANSWER
from eval.run_ask import (
    INVALID,
    LLM,
    THRESHOLD,
    AskItemResult,
    AskRun,
    _parse_args,
    citation_precision,
    citation_recall,
    citations_valid,
    question_cost,
    recording_llm_usage,
    refused_by,
    run,
    summarize,
)
from eval.run_retrieval import DEFAULT_DATASET, load_questions


def _run(
    answerable: bool = True,
    cited: list[str] | None = None,
    calls: int = 1,
    answer: str = "An answer.",
    tokens: tuple[int | None, int | None] = (1000, 100),
) -> AskRun:
    return AskRun(
        answerable=answerable,
        answer=answer,
        cited_ids=cited or [],
        llm_calls=calls,
        input_tokens=tokens[0] if calls else 0,
        output_tokens=tokens[1] if calls else 0,
        latency_ms=100,
    )


def _item(id_: str, expected: list[str], run_: AskRun | None) -> AskItemResult:
    return AskItemResult(
        id=id_,
        question=f"question {id_}",
        expected_email_ids=expected,
        run=run_,
        error=None if run_ else "RuntimeError: boom",
    )


# --- metrics -------------------------------------------------------------------


def test_refusal_is_attributed_to_threshold_model_or_invalid_answer():
    assert refused_by(_run(answerable=True, cited=["a"])) is None
    assert refused_by(_run(False, calls=0, answer=INSUFFICIENT_EVIDENCE_ANSWER)) == THRESHOLD
    assert refused_by(_run(False, calls=1)) == LLM
    assert refused_by(_run(False, calls=2, answer=INVALID_ANSWER)) == INVALID


def test_citations_are_valid_only_when_all_are_expected():
    assert citations_valid(["a"], ["a", "b"])
    assert not citations_valid(["a", "x"], ["a", "b"])
    assert not citations_valid([], ["a"])


def test_citation_precision():
    assert citation_precision(["a", "x"], ["a"]) == 0.5
    assert citation_precision(["a", "a"], ["a"]) == 1.0
    assert citation_precision([], ["a"]) is None


def test_citation_recall():
    assert citation_recall(["a"], ["a", "b"]) == 0.5
    assert citation_recall(["a", "b", "x"], ["a", "b"]) == 1.0
    assert citation_recall([], ["a"]) == 0.0
    assert citation_recall(["a"], []) is None


def test_cost_needs_both_prices_and_is_zero_without_model_calls():
    assert question_cost(_run(tokens=(1_000_000, 1_000_000)), 1.0, 4.0) == 5.0
    assert question_cost(_run(), None, 4.0) is None
    assert question_cost(_run(False, calls=0), 1.0, 4.0) == 0.0
    assert question_cost(_run(tokens=(None, None)), 1.0, 4.0) is None


def test_summary_counts_answers_citations_and_refusal_sources():
    results = [
        _item("q-1", ["a"], _run(cited=["a"])),
        _item("q-2", ["b", "y"], _run(cited=["b", "x"])),
        _item("q-3", ["c"], _run(False, calls=0)),
        _item("q-4", ["d"], None),
        _item("u-1", [], _run(False, calls=0)),
        _item("u-2", [], _run(False, calls=1)),
        _item("u-3", [], _run(cited=["a"])),
    ]

    summary = summarize(results, 1.0, 10.0)

    a, u = summary["answerable"], summary["unanswerable"]
    assert summary["errors"] == 1
    assert (a["answered"], a["n"], a["answered_rate"]) == (2, 4, 0.5)
    assert a["false_refusals"] == {THRESHOLD: 1, LLM: 0, INVALID: 0}
    assert (a["citations_valid"], a["citation_validity_rate"]) == (1, 0.5)
    assert a["mean_citation_precision"] == 0.75
    assert a["mean_citation_recall"] == 0.75
    assert (u["refused"], u["refusal_rate"]) == (2, 2 / 3)
    assert (u["refused_by_threshold"], u["refused_by_llm"]) == (1, 1)
    # 4 model-calling questions at $0.002 each, 2 threshold refusals at $0.
    usage = summary["usage"]
    assert usage["estimated_cost_usd_per_question"] == pytest.approx(0.008 / 6)
    assert usage["mean_llm_calls"] == pytest.approx(4 / 6)


# --- usage recording -----------------------------------------------------------


def test_recording_llm_usage_records_each_call_and_passes_the_response_through(
    monkeypatch,
):
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=12, output_tokens=3))
    responses = SimpleNamespace(parse=lambda **_: response)
    monkeypatch.setattr(
        "eval.run_ask.email_qa._get_client", lambda: SimpleNamespace(responses=responses)
    )

    with recording_llm_usage() as calls:
        assert responses.parse(model="m") is response
        responses.parse(model="m")

    assert calls == [(12, 3), (12, 3)]
    assert responses.parse(model="m") is response  # unpatched afterwards


# --- full run with a fake asker ------------------------------------------------


def test_run_writes_results_and_reports_every_question(tmp_path):
    questions = load_questions(DEFAULT_DATASET)

    def asker(question: str) -> AskRun:
        match = next(q for q in questions if q.question == question)
        if match.answerable:
            return _run(cited=match.expected_email_ids[:1])
        return _run(False, calls=0, answer=INSUFFICIENT_EVIDENCE_ANSWER)

    args = _parse_args(
        [
            "--results-dir", str(tmp_path),
            "--min-score", "0.3",
            "--input-usd-per-mtok", "1",
            "--output-usd-per-mtok", "10",
        ]
    )

    out_path, report = run(
        args, asker, model="fake", embedding_model="fake", corpus_size=50
    )

    data = json.loads(out_path.read_text())
    assert out_path.name.startswith("ask-")
    assert len(data["items"]) == 25
    assert data["run"]["min_score"] == 0.3
    assert data["summary"]["answerable"]["answered"] == 20
    assert data["summary"]["answerable"]["citation_validity_rate"] == 1.0
    # The fake asker cites only the first expected email.
    items = {item["id"]: item for item in data["items"]}
    assert items["q-12"]["citation_recall"] == 0.5
    assert items["q-01"]["citation_recall"] == 1.0
    assert items["u-01"]["citation_recall"] is None
    assert data["summary"]["unanswerable"]["refused_by_threshold"] == 5
    for question in questions:
        assert question.id in report
    assert "refused ok (threshold)" in report
    assert "estimated cost/question: $" in report
