"""Evaluate /ask end to end over the retrieval eval questions.

Usage:
    uv run python -m eval.run_ask [--min-score X] \\
        [--input-usd-per-mtok X --output-usd-per-mtok Y]

Needs a configured `.env` and a reachable Postgres. The corpus is loaded
into the `<db>_eval` database exactly as `eval.run_retrieval` does it,
then every question in `eval/retrieval_dataset.jsonl` goes through
`ask_question()` with the real embedder and the real model, at
`settings.rag_min_score` unless overridden.

Answerable questions are scored on how often the model answers and
whether its citations are expected emails (validity: all of them;
precision: the fraction of citations that are expected; recall: the
fraction of expected emails that were cited). Unanswerable questions are scored on refusals,
split into those refused by the threshold (no model call) and those
refused by the model. Latency is the wall time of `ask_question()`;
cost counts model tokens only (not the question's embedding) and is only
estimated when both prices are passed. Writes
`eval/results/ask-<UTC timestamp>.json`.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents import email_qa
from app.config import get_settings
from app.schemas.ask import AskRequest
from app.services.ask import INVALID_ANSWER, ask_question
from app.services.embedding import Embedder, get_embedder
from eval.run import DEFAULT_RESULTS_DIR, load_dataset
from eval.run_retrieval import (
    DEFAULT_CORPUS,
    DEFAULT_DATASET,
    RetrievalItem,
    check_expected_ids,
    eval_database_url,
    ingest_corpus,
    load_questions,
    prepare_database,
)

THRESHOLD = "threshold"
LLM = "llm"
INVALID = "invalid_answer"


class AskRun(BaseModel):
    """One `ask_question()` call, with citations as corpus ids."""

    answerable: bool
    answer: str
    cited_ids: list[str]
    llm_calls: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


# A question in, its /ask outcome out.
Asker = Callable[[str], AskRun]


class AskItemResult(BaseModel):
    id: str
    question: str
    expected_email_ids: list[str]
    run: AskRun | None = None
    error: str | None = None

    @property
    def expects_answer(self) -> bool:
        return bool(self.expected_email_ids)


# --- metrics (pure) --------------------------------------------------------------


def refused_by(run: AskRun) -> str | None:
    """Why /ask refused: below the threshold (the model was never called),
    by the model's `answerable` flag, or because its citations failed
    validation on every attempt. `None` when it answered."""
    if run.answerable:
        return None
    if run.llm_calls == 0:
        return THRESHOLD
    return INVALID if run.answer == INVALID_ANSWER else LLM


def citations_valid(cited: Sequence[str], expected: Sequence[str]) -> bool:
    """At least one citation, and every one is an expected email."""
    return bool(cited) and set(cited) <= set(expected)


def citation_precision(cited: Sequence[str], expected: Sequence[str]) -> float | None:
    if not cited:
        return None
    return len(set(cited) & set(expected)) / len(set(cited))


def citation_recall(cited: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    return len(set(cited) & set(expected)) / len(set(expected))


def _answered_citation_scores(r: AskItemResult) -> tuple[float | None, float | None]:
    """`(precision, recall)` for an answered answerable question, else
    `(None, None)`: a refusal is already counted as a false refusal."""
    if not (r.expects_answer and r.run and r.run.answerable):
        return None, None
    return (
        citation_precision(r.run.cited_ids, r.expected_email_ids),
        citation_recall(r.run.cited_ids, r.expected_email_ids),
    )


def question_cost(
    run: AskRun, input_usd_per_mtok: float | None, output_usd_per_mtok: float | None
) -> float | None:
    if input_usd_per_mtok is None or output_usd_per_mtok is None:
        return None
    if run.llm_calls == 0:
        return 0.0
    if run.input_tokens is None or run.output_tokens is None:
        return None
    return (
        run.input_tokens * input_usd_per_mtok + run.output_tokens * output_usd_per_mtok
    ) / 1e6


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def summarize(
    results: Sequence[AskItemResult],
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> dict:
    """Errored questions stay in each group's denominator, counted as
    neither answered nor refused."""
    answerable = [r for r in results if r.expects_answer]
    unanswerable = [r for r in results if not r.expects_answer]
    answered = [r for r in answerable if r.run and r.run.answerable]
    valid = [
        r for r in answered if citations_valid(r.run.cited_ids, r.expected_email_ids)
    ]
    refusals = [refused_by(r.run) for r in unanswerable if r.run]
    runs = [r.run for r in results if r.run]
    return {
        "n": len(results),
        "errors": sum(r.error is not None for r in results),
        "answerable": {
            "n": len(answerable),
            "answered": len(answered),
            "answered_rate": _rate(len(answered), len(answerable)),
            "false_refusals": {
                reason: sum(r.run is not None and refused_by(r.run) == reason for r in answerable)
                for reason in (THRESHOLD, LLM, INVALID)
            },
            "citations_valid": len(valid),
            "citation_validity_rate": _rate(len(valid), len(answered)),
            "mean_citation_precision": _mean_or_none(
                [citation_precision(r.run.cited_ids, r.expected_email_ids) for r in answered]
            ),
            "mean_citation_recall": _mean_or_none(
                [citation_recall(r.run.cited_ids, r.expected_email_ids) for r in answered]
            ),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "refused": sum(reason is not None for reason in refusals),
            "refusal_rate": _rate(
                sum(reason is not None for reason in refusals), len(unanswerable)
            ),
            "refused_by_threshold": refusals.count(THRESHOLD),
            "refused_by_llm": refusals.count(LLM),
            "refused_invalid_answer": refusals.count(INVALID),
        },
        "usage": {
            "mean_latency_ms": _mean_or_none([run.latency_ms for run in runs]),
            "mean_llm_calls": _mean_or_none([run.llm_calls for run in runs]),
            "mean_input_tokens": _mean_or_none([run.input_tokens for run in runs]),
            "mean_output_tokens": _mean_or_none([run.output_tokens for run in runs]),
            "estimated_cost_usd_per_question": _mean_or_none(
                [question_cost(run, input_usd_per_mtok, output_usd_per_mtok) for run in runs]
            ),
        },
    }


# --- report ----------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "—" if value is None else format(value, spec)


def _outcome(r: AskItemResult) -> str:
    if r.run is None:
        return f"ERROR {r.error}"
    reason = refused_by(r.run)
    if r.expects_answer:
        if reason is not None:
            return f"FALSE REFUSAL ({reason})"
        if not citations_valid(r.run.cited_ids, r.expected_email_ids):
            return "answered, BAD CITATIONS"
        return "answered ok"
    return "NOT REFUSED" if reason is None else f"refused ok ({reason})"


def format_report(run_info: dict, results: Sequence[AskItemResult], summary: dict) -> str:
    lines = [
        f"model={run_info['model']}  prompt={run_info['prompt_version']}  "
        f"embedding_model={run_info['embedding_model']}  "
        f"questions={summary['n']}  min_score={run_info['min_score']}  "
        f"errors={summary['errors']}",
        "",
        f"{'question':<8}{'calls':>6}{'ms':>7}  {'cited':<24}{'expected':<24}"
        f"{'prec':>6}{'rec':>6}  outcome",
    ]
    for r in results:
        cited = ",".join(r.run.cited_ids) if r.run else ""
        calls = str(r.run.llm_calls) if r.run else "—"
        latency = str(r.run.latency_ms) if r.run else "—"
        precision, recall = _answered_citation_scores(r)
        lines.append(
            f"{r.id:<8}{calls:>6}{latency:>7}  {cited or '—':<24}"
            f"{','.join(r.expected_email_ids) or '—':<24}"
            f"{_fmt(precision, '.2f'):>6}{_fmt(recall, '.2f'):>6}  {_outcome(r)}"
        )

    a = summary["answerable"]
    u = summary["unanswerable"]
    usage = summary["usage"]
    cost = usage["estimated_cost_usd_per_question"]
    lines += [
        "",
        f"answerable (n={a['n']}): answered {a['answered']}/{a['n']} "
        f"(rate {_fmt(a['answered_rate'])}); false refusals "
        + "  ".join(f"{k}={v}" for k, v in a["false_refusals"].items()),
        f"  citations valid {a['citations_valid']}/{a['answered']} "
        f"(rate {_fmt(a['citation_validity_rate'])}); "
        f"mean citation precision {_fmt(a['mean_citation_precision'])}, "
        f"mean citation recall {_fmt(a['mean_citation_recall'])}",
        f"unanswerable (n={u['n']}): refused {u['refused']}/{u['n']} "
        f"(rate {_fmt(u['refusal_rate'])}); by threshold (0 model calls) "
        f"{u['refused_by_threshold']}, by model {u['refused_by_llm']}, "
        f"invalid answer {u['refused_invalid_answer']}",
        "",
        f"mean latency: {_fmt(usage['mean_latency_ms'], '.0f')} ms  "
        f"mean model calls: {_fmt(usage['mean_llm_calls'], '.2f')}",
        f"mean tokens/question: in={_fmt(usage['mean_input_tokens'], '.0f')} "
        f"out={_fmt(usage['mean_output_tokens'], '.0f')}",
        "estimated cost/question: "
        + (
            f"${cost:.6f}"
            if cost is not None
            else "n/a (pass --input-usd-per-mtok and --output-usd-per-mtok)"
        ),
    ]
    return "\n".join(lines)


# --- run -------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_ask", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--min-score",
        type=float,
        help="retrieval threshold (default: settings.rag_min_score)",
    )
    parser.add_argument("--input-usd-per-mtok", type=float, help="USD per 1M input tokens")
    parser.add_argument("--output-usd-per-mtok", type=float, help="USD per 1M output tokens")
    return parser.parse_args(argv)


def evaluate_item(item: RetrievalItem, asker: Asker) -> AskItemResult:
    """One question; a failed call is recorded as an error, not raised,
    so one bad call doesn't lose the rest of the run."""
    result = AskItemResult(
        id=item.id, question=item.question, expected_email_ids=item.expected_email_ids
    )
    try:
        result.run = asker(item.question)
    except Exception as exc:  # noqa: BLE001 - recorded per item
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _item_record(r: AskItemResult) -> dict:
    precision, recall = _answered_citation_scores(r)
    return {
        **r.model_dump(mode="json"),
        "refused_by": refused_by(r.run) if r.run else None,
        "citation_precision": precision,
        "citation_recall": recall,
        "outcome": _outcome(r),
    }


def run(
    args: argparse.Namespace,
    asker: Asker,
    *,
    model: str,
    embedding_model: str,
    corpus_size: int,
) -> tuple[Path, str]:
    """Ask every question, write the results file, and return its path
    and the printable report."""
    questions = load_questions(args.dataset)
    started = datetime.now(timezone.utc)
    results = []
    for i, item in enumerate(questions, 1):
        result = evaluate_item(item, asker)
        print(f"[{i:>2}/{len(questions)}] {item.id}  {_outcome(result)}", flush=True)
        results.append(result)

    run_info = {
        "timestamp": started.isoformat(),
        "model": model,
        "prompt_version": email_qa.PROMPT_VERSION,
        "embedding_model": embedding_model,
        "min_score": args.min_score,
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "corpus_size": corpus_size,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "prices_usd_per_mtok": {
            "input": args.input_usd_per_mtok,
            "output": args.output_usd_per_mtok,
        },
    }
    summary = summarize(results, args.input_usd_per_mtok, args.output_usd_per_mtok)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"ask-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps(
            {
                "run": run_info,
                "summary": summary,
                "items": [_item_record(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path, format_report(run_info, results, summary)


# --- real wiring -----------------------------------------------------------------


@contextmanager
def recording_llm_usage() -> Iterator[list[tuple[int | None, int | None]]]:
    """Record `(input_tokens, output_tokens)` for every model call made by
    the answering agent, without changing what it returns."""
    calls: list[tuple[int | None, int | None]] = []
    responses = email_qa._get_client().responses
    parse = responses.parse

    def recording_parse(*args: Any, **kwargs: Any) -> Any:
        response = parse(*args, **kwargs)
        usage = getattr(response, "usage", None)
        calls.append(
            (getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
        )
        return response

    with patch.object(responses, "parse", recording_parse):
        yield calls


def _sum_or_none(values: Sequence[int | None]) -> int | None:
    return None if any(v is None for v in values) else sum(values)  # type: ignore[arg-type]


def make_asker(
    embedder: Embedder,
    min_score: float,
    id_map: dict[int, str],
    calls: list[tuple[int | None, int | None]],
) -> Asker:
    """`ask_question()` exactly as /ask calls it, timed, with the model
    calls it made and its citations mapped to corpus ids."""

    def asker(question: str) -> AskRun:
        calls.clear()
        started = time.perf_counter()
        response = ask_question(AskRequest(question=question), embedder, min_score)
        latency_ms = round((time.perf_counter() - started) * 1000)
        return AskRun(
            answerable=response.answerable,
            answer=response.answer,
            cited_ids=[id_map[email_id] for email_id in response.citations],
            llm_calls=len(calls),
            input_tokens=_sum_or_none([c[0] for c in calls]),
            output_tokens=_sum_or_none([c[1] for c in calls]),
            latency_ms=latency_ms,
        )

    return asker


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = get_settings()
    if args.min_score is None:
        args.min_score = settings.rag_min_score

    corpus = load_dataset(args.corpus)
    check_expected_ids(load_questions(args.dataset), corpus)

    url = eval_database_url(settings.database_url)
    prepare_database(url)
    engine = create_engine(url)
    eval_sessions = sessionmaker(bind=engine, expire_on_commit=False)
    embedder = get_embedder()
    try:
        with eval_sessions() as session:
            print(f"embedding {len(corpus)} corpus emails into {url.database} ...", flush=True)
            id_map = ingest_corpus(session, corpus, embedder)
            session.commit()
        # `ask_question()` opens its own sessions; point them at the eval
        # database instead of the dev one.
        with (
            patch("app.database.session.get_sessionmaker", return_value=eval_sessions),
            recording_llm_usage() as calls,
        ):
            out_path, report = run(
                args,
                make_asker(embedder, args.min_score, id_map, calls),
                model=settings.openai_model,
                embedding_model=embedder.model,
                corpus_size=len(id_map),
            )
    finally:
        engine.dispose()

    print()
    print(report)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
