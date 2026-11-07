"""Run the email analyzer over the labeled eval dataset.

Usage:
    uv run python -m eval.run --prompt-version analyze_v1 \\
        [--split dev|test|all] [--model gpt-...] \\
        [--input-usd-per-mtok X --output-usd-per-mtok Y]

Makes one real OpenAI call per selected dataset item (needs a configured
`.env`). The dataset is split into dev and test with a fixed seed
(`assign_splits`); only dev may be used for prompt tuning, so the default
is `--split dev`. Run `--split test` to report a final number.
Writes `eval/results/<UTC timestamp>.json` and prints a summary report.
Cost is only estimated when both prices are passed: they change over
time and differ per model, so none are hardcoded here.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from app.agents.email_analyzer import (
    PROMPT_VERSION,
    AnalysisRun,
    analyze_with_metadata,
    available_prompt_versions,
)
from app.integrations.langfuse import (
    annotate,
    run_flags,
    score_trace,
    shutdown_tracing,
    traced,
)
from app.schemas.email import Email, EmailAnalysis, EmailCategory, EmailParticipant
from app.services.event_time import resolve_timing
from eval.metrics import (
    ERROR,
    ItemResult,
    accuracy,
    calibration_table,
    confusion_matrix,
    critical_recall,
    per_class_metrics,
    slice_accuracy,
    tag_accuracy,
    time_extraction_summary,
    usage_summary,
)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"
LABELS = [category.value for category in EmailCategory]

# Fixed "now" for every item, so relative dates in emails ("tomorrow",
# "within 5 days") resolve the same way on every run. A Monday.
RECEIVED_AT = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)

Analyzer = Callable[[Email], AnalysisRun]

# Seed and share for the dev/test split. Changing either (or the dataset's
# items) re-draws the split, so the test items would no longer be unseen.
SPLIT_SEED = 20260302
TEST_FRACTION = 0.5
SPLITS = ("dev", "test")
EVAL_TAGS = ("eval",)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    subject: str
    body: str
    sender: EmailParticipant
    label: EmailCategory
    # Slicing tags. `suite` separates the original corpus (which the
    # retrieval and ask evals also use) from the hard classification set.
    language: Literal["en", "de", "mixed"] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    suite: Literal["base", "hard"] = "base"
    # Why the label is what it is, for items where it is a judgment call.
    note: str | None = None
    # Time labels; `expected_needs_clarification` is `None` on items
    # that aren't scored for time extraction.
    expected_start_time: AwareDatetime | None = None
    expected_duration_minutes: int | None = None
    expected_needs_clarification: bool | None = None


def load_dataset(path: Path, suite: str | None = None) -> list[EvalItem]:
    """All items, or only those of one `suite`."""
    with path.open(encoding="utf-8") as f:
        items = [EvalItem.model_validate_json(line) for line in f if line.strip()]
    return [i for i in items if suite is None or i.suite == suite]


def assign_splits(items: Sequence[EvalItem]) -> dict[str, str]:
    """`{item id: "dev" | "test"}`, stratified by label and fixed by
    `SPLIT_SEED`, so every class appears on both sides and the split is
    the same on every run and machine."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for item in items:
        by_label[item.label.value].append(item.id)
    rng = random.Random(SPLIT_SEED)
    splits: dict[str, str] = {}
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        n_test = round(len(ids) * TEST_FRACTION)
        for i, item_id in enumerate(ids):
            splits[item_id] = "test" if i < n_test else "dev"
    return splits


def select_split(items: Sequence[EvalItem], split: str) -> list[EvalItem]:
    if split == "all":
        return list(items)
    splits = assign_splits(items)
    return [i for i in items if splits[i.id] == split]


def to_email(item: EvalItem) -> Email:
    return Email(
        id=item.id,
        thread_id=f"thread-{item.id}",
        grant_id="eval",
        sender=item.sender,
        recipients=[EmailParticipant(email="candidate@example.com")],
        subject=item.subject,
        body=item.body,
        received_at=RECEIVED_AT,
    )


def time_correct(item: EvalItem, analysis: EmailAnalysis | None) -> bool | None:
    """Scored on what would actually be scheduled (after the deterministic
    timezone check), not on the model's raw fields. The offset must match
    too, not just the instant, so a CET/CEST mix-up counts as wrong."""
    if item.expected_needs_clarification is None:
        return None
    if analysis is None:
        return False
    timing = resolve_timing(analysis)
    if item.expected_needs_clarification:
        return timing.needs_clarification and timing.start is None
    expected = item.expected_start_time
    # Instants compared in UTC: inside a DST fold, aware datetimes in
    # different zones never compare equal (PEP 495).
    return (
        timing.start is not None
        and expected is not None
        and timing.start.astimezone(timezone.utc) == expected.astimezone(timezone.utc)
        and timing.start.utcoffset() == expected.utcoffset()
    )


def duration_correct(item: EvalItem, analysis: EmailAnalysis | None) -> bool | None:
    if item.expected_needs_clarification is not False:
        return None
    if analysis is None:
        return False
    return resolve_timing(analysis).duration_minutes == item.expected_duration_minutes


def evaluate_item(item: EvalItem, analyzer: Analyzer, split: str | None = None) -> ItemResult:
    """One item; a failed analysis is recorded as an error, not raised,
    so one bad call doesn't lose the rest of the run."""
    result = _analyze_item(item, analyzer)
    result.language, result.difficulty, result.split = item.language, item.difficulty, split
    return result


@traced("eval_item", root=True, tags=EVAL_TAGS)
def evaluate_item_traced(item: EvalItem, analyzer: Analyzer, split: str | None) -> ItemResult:
    """`evaluate_item` as its own Langfuse trace, scored `correct`.

    Only labels and tags are attached, never the email's text; what the
    model call itself records is redacted by the tracing client unless
    TRACE_EMAIL_CONTENT is on in test mode (app/integrations/langfuse).
    """
    result = evaluate_item(item, analyzer, split)
    annotate(
        item_id=item.id,
        label=result.label,
        predicted=result.predicted,
        language=item.language,
        difficulty=item.difficulty,
        split=split,
        errored=result.error is not None,
        **run_flags(),
    )
    score_trace("correct", result.correct)
    return result


def _analyze_item(item: EvalItem, analyzer: Analyzer) -> ItemResult:
    try:
        run = analyzer(to_email(item))
    except Exception as exc:  # noqa: BLE001 - recorded per item
        return ItemResult(
            id=item.id,
            label=item.label.value,
            predicted=None,
            confidence=None,
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            error=f"{type(exc).__name__}: {exc}",
            time_correct=time_correct(item, None),
            duration_correct=duration_correct(item, None),
        )
    return ItemResult(
        id=item.id,
        label=item.label.value,
        predicted=run.analysis.category.value,
        confidence=run.analysis.confidence,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        latency_ms=run.latency_ms,
        model=run.model,
        analysis=run.analysis.model_dump(mode="json"),
        time_correct=time_correct(item, run.analysis),
        duration_correct=duration_correct(item, run.analysis),
    )


def summarize(
    results: Sequence[ItemResult],
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> dict:
    matrix = confusion_matrix(results, LABELS)
    return {
        "n": len(results),
        "errors": sum(r.error is not None for r in results),
        "accuracy": accuracy(results),
        "time_extraction": time_extraction_summary(results),
        "per_class": per_class_metrics(matrix, LABELS),
        "confusion_matrix": matrix,
        "calibration": calibration_table(results),
        "slices": slice_accuracy(results),
        "slices_language": tag_accuracy(results, "language"),
        "slices_difficulty": tag_accuracy(results, "difficulty"),
        "critical_recall": critical_recall(matrix),
        "usage": usage_summary(results, input_usd_per_mtok, output_usd_per_mtok),
    }


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def format_report(run_info: dict, summary: dict) -> str:
    time = summary["time_extraction"]
    lines = [
        f"prompt={run_info['prompt_version']}  model={run_info['model']}  "
        f"n={summary['n']}  errors={summary['errors']}",
        f"classification accuracy: {summary['accuracy']:.3f}",
        f"time extraction accuracy: {_fmt(time['accuracy'], '.3f')} "
        f"(n={time['n']}), duration accuracy: "
        f"{_fmt(time['duration_accuracy'], '.3f')} (n={time['duration_n']})",
        "",
        f"{'class':<22}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>5}",
    ]
    for label, m in summary["per_class"].items():
        lines.append(
            f"{label:<22}{_fmt(m['precision']):>7}{_fmt(m['recall']):>8}"
            f"{_fmt(m['f1']):>7}{m['support']:>5}"
        )

    matrix = summary["confusion_matrix"]
    columns = list(next(iter(matrix.values())).keys())
    short = {c: "error" if c == ERROR else c[:6] for c in columns}
    lines += ["", "confusion matrix (rows = actual, cols = predicted)"]
    lines.append(f"{'':<22}" + "".join(f"{short[c]:>8}" for c in columns))
    for actual, row in matrix.items():
        lines.append(f"{actual:<22}" + "".join(f"{row[c]:>8}" for c in columns))

    lines += ["", f"{'confidence':<12}{'n':>4}{'mean conf':>11}{'accuracy':>10}"]
    for b in summary["calibration"]:
        lines.append(
            f"{b['bin']:<12}{b['n']:>4}{_fmt(b['mean_confidence']):>11}"
            f"{_fmt(b['accuracy']):>10}"
        )

    for title, key in (
        ("slices", "slices"),
        ("by language", "slices_language"),
        ("by difficulty", "slices_difficulty"),
    ):
        if summary.get(key):
            lines += ["", f"{title}: " + "  ".join(
                f"{tag}={s['accuracy']:.2f} (n={s['n']})"
                for tag, s in summary[key].items()
            )]

    lines += ["", "recall of the costly classes (a miss means a missed interview or deadline)"]
    for label, m in summary["critical_recall"].items():
        missed_as = ", ".join(f"{k}={v}" for k, v in m["missed_as"].items()) or "-"
        lines.append(
            f"{label:<22}recall={_fmt(m['recall'])}  missed={m['missed']}/{m['support']}"
            f"  missed as: {missed_as}"
        )

    usage = summary["usage"]
    cost = usage["estimated_cost_usd_per_email"]
    lines += [
        "",
        f"mean tokens/email: in={_fmt(usage['mean_input_tokens'], '.0f')} "
        f"out={_fmt(usage['mean_output_tokens'], '.0f')}",
        "estimated cost/email: "
        + (f"${cost:.6f}" if cost is not None
           else "n/a (pass --input-usd-per-mtok and --output-usd-per-mtok)"),
        f"mean latency: {_fmt(usage['mean_latency_ms'], '.0f')} ms",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        choices=available_prompt_versions(),
        help=f"system prompt file under app/agents/prompts/ (default: {PROMPT_VERSION})",
    )
    parser.add_argument("--model", help="override settings.openai_model")
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="dev",
        help="which part of the dataset to run; tune prompts on dev only (default: dev)",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--input-usd-per-mtok", type=float, help="USD per 1M input tokens")
    parser.add_argument("--output-usd-per-mtok", type=float, help="USD per 1M output tokens")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, analyzer: Analyzer) -> tuple[Path, str]:
    """Evaluate every item, write the results file, and return its path
    and the printable report."""
    items = select_split(load_dataset(args.dataset), args.split)
    started = datetime.now(timezone.utc)
    results = []
    for i, item in enumerate(items, 1):
        result = evaluate_item_traced(item, analyzer, args.split)
        mark = "ok " if result.correct else ("ERR" if result.error else "x  ")
        print(f"[{i:>2}/{len(items)}] {mark} {item.id}", flush=True)
        results.append(result)

    models = sorted({r.model for r in results if r.model})
    run_info = {
        "timestamp": started.isoformat(),
        "prompt_version": args.prompt_version,
        "model": ", ".join(models) or args.model or "(no successful calls)",
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "split": args.split,
        "split_seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "received_at": RECEIVED_AT.isoformat(),
        "prices_usd_per_mtok": {
            "input": args.input_usd_per_mtok,
            "output": args.output_usd_per_mtok,
        },
    }
    summary = summarize(results, args.input_usd_per_mtok, args.output_usd_per_mtok)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps(
            {
                "run": run_info,
                "summary": summary,
                "items": [
                    {**r.model_dump(mode="json"), "correct": r.correct} for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path, format_report(run_info, summary)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    def analyzer(email: Email) -> AnalysisRun:
        return analyze_with_metadata(
            email, prompt_version=args.prompt_version, model=args.model
        )

    try:
        out_path, report = run(args, analyzer)
    finally:
        shutdown_tracing()
    print()
    print(report)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
