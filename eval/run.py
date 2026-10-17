"""Run the email analyzer over the labeled eval dataset.

Usage:
    uv run python -m eval.run --prompt-version analyze_v1 \\
        [--model gpt-...] \\
        [--input-usd-per-mtok X --output-usd-per-mtok Y]

Makes one real OpenAI call per dataset item (needs a configured `.env`).
Writes `eval/results/<UTC timestamp>.json` and prints a summary report.
Cost is only estimated when both prices are passed: they change over
time and differ per model, so none are hardcoded here.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.agents.email_analyzer import (
    PROMPT_VERSION,
    AnalysisRun,
    analyze_with_metadata,
    available_prompt_versions,
)
from app.schemas.email import Email, EmailCategory, EmailParticipant
from eval.metrics import (
    ERROR,
    ItemResult,
    accuracy,
    calibration_table,
    confusion_matrix,
    per_class_metrics,
    slice_accuracy,
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


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    subject: str
    body: str
    sender: EmailParticipant
    label: EmailCategory


def load_dataset(path: Path) -> list[EvalItem]:
    with path.open(encoding="utf-8") as f:
        return [EvalItem.model_validate_json(line) for line in f if line.strip()]


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


def evaluate_item(item: EvalItem, analyzer: Analyzer) -> ItemResult:
    """One item; a failed analysis is recorded as an error, not raised,
    so one bad call doesn't lose the rest of the run."""
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
        "per_class": per_class_metrics(matrix, LABELS),
        "confusion_matrix": matrix,
        "calibration": calibration_table(results),
        "slices": slice_accuracy(results),
        "usage": usage_summary(results, input_usd_per_mtok, output_usd_per_mtok),
    }


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def format_report(run_info: dict, summary: dict) -> str:
    lines = [
        f"prompt={run_info['prompt_version']}  model={run_info['model']}  "
        f"n={summary['n']}  errors={summary['errors']}",
        f"accuracy: {summary['accuracy']:.3f}",
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

    lines += ["", "slices: " + "  ".join(
        f"{tag}={s['accuracy']:.2f} (n={s['n']})" for tag, s in summary["slices"].items()
    )]

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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--input-usd-per-mtok", type=float, help="USD per 1M input tokens")
    parser.add_argument("--output-usd-per-mtok", type=float, help="USD per 1M output tokens")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, analyzer: Analyzer) -> tuple[Path, str]:
    """Evaluate every item, write the results file, and return its path
    and the printable report."""
    items = load_dataset(args.dataset)
    started = datetime.now(timezone.utc)
    results = []
    for i, item in enumerate(items, 1):
        result = evaluate_item(item, analyzer)
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

    out_path, report = run(args, analyzer)
    print()
    print(report)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
