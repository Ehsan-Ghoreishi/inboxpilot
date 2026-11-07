"""Write docs/eval-report.md from the latest classification results.

Usage:
    uv run python -m eval.report [--results-file PATH ...] [--output PATH]

Takes the newest `eval/results/<UTC timestamp>.json` of each split (dev and
test) and writes tables only, with the splits side by side and a combined
column over both. Without a dev and a test file it falls back to the newest
file alone (`ask-` and `retrieval-` files are other evals and are skipped).
Every number comes from the results files: metrics are recomputed from the
per-item results, so older files without summary sections still work.
Prices are read from the file's `run.prices_usd_per_mtok`; files without
them show "n/a" for cost. Error-analysis notes come from the dataset.
"""

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.metrics import (
    ERROR,
    CRITICAL_LABELS,
    ItemResult,
    accuracy,
    confusion_matrix,
    critical_recall,
    per_class_metrics,
    slice_accuracy,
    tag_accuracy,
    time_extraction_summary,
    usage_summary,
    wilson_interval,
)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
DEFAULT_OUTPUT = EVAL_DIR.parent / "docs" / "eval-report.md"
SPLIT_ORDER = ("dev", "test")
NOT_AVAILABLE = "n/a"
_RESULT_NAME = re.compile(r"^\d{8}T\d{6}Z\.json$")
_ITEM_FIELDS = set(ItemResult.model_fields)


def latest_results_file(results_dir: Path) -> Path:
    candidates = sorted(p for p in results_dir.glob("*.json") if _RESULT_NAME.match(p.name))
    if not candidates:
        raise FileNotFoundError(f"no classification results in {results_dir}")
    return candidates[-1]


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def numeric(i: int) -> bool:
        cells = [row[i] for row in rows if row[i] is not None]
        return i > 0 and all(isinstance(c, (int, float)) for c in cells)

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" if numeric(i) else "---" for i in range(len(header))) + "|",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _slice_rows(slices: dict[str, dict]) -> list[list[object]]:
    return [[tag, s["n"], s["accuracy"]] for tag, s in slices.items()]


@dataclass(frozen=True)
class Column:
    """One column of the side-by-side table: a split, or dev + test."""

    name: str
    runs: list[dict]
    results: list[ItemResult]


def _item_results(data: dict) -> list[ItemResult]:
    return [
        ItemResult(**{k: v for k, v in item.items() if k in _ITEM_FIELDS})
        for item in data["items"]
    ]


def _columns(datas: Sequence[dict]) -> list[Column]:
    columns = [
        Column(d["run"].get("split") or "all (not recorded)", [d["run"]], _item_results(d))
        for d in datas
    ]
    columns.sort(key=lambda c: SPLIT_ORDER.index(c.name) if c.name in SPLIT_ORDER else len(SPLIT_ORDER))
    names = {c.name for c in columns}
    if len(columns) > 1 and names.issuperset(SPLIT_ORDER):
        both = [c for c in columns if c.name in SPLIT_ORDER]
        columns.append(
            Column(
                " + ".join(SPLIT_ORDER),
                [r for c in both for r in c.runs],
                [r for c in both for r in c.results],
            )
        )
    return columns


def _distinct(runs: Sequence[dict], key: str) -> str:
    return ", ".join(sorted({str(r[key]) for r in runs if r.get(key) is not None})) or NOT_AVAILABLE


def _interval(successes: int, n: int) -> str:
    interval = wilson_interval(successes, n)
    return NOT_AVAILABLE if interval is None else f"{interval[0]:.3f} to {interval[1]:.3f}"


def _prices(runs: Sequence[dict]) -> tuple[float, float] | None:
    """The one price pair every run in the column was given, else `None`."""
    pairs = {
        (p.get("input"), p.get("output"))
        for r in runs
        for p in [r.get("prices_usd_per_mtok") or {}]
    }
    if len(pairs) != 1:
        return None
    price_in, price_out = pairs.pop()
    return None if price_in is None or price_out is None else (price_in, price_out)


def _usage_cells(column: Column) -> tuple[str, str, str]:
    prices = _prices(column.runs)
    usage = usage_summary(column.results, *(prices or (None, None)))
    mean_in, mean_out = usage["mean_input_tokens"], usage["mean_output_tokens"]
    cost = usage["estimated_cost_usd_per_email"]
    return (
        NOT_AVAILABLE if mean_in is None else f"{mean_in:.1f}",
        NOT_AVAILABLE if mean_out is None else f"{mean_out:.1f}",
        NOT_AVAILABLE if cost is None else f"${cost:.6f}",
    )


def _overview_table(columns: Sequence[Column], labels: Sequence[str]) -> str:
    cells: list[dict[str, object]] = []
    for c in columns:
        matrix = confusion_matrix(c.results, labels)
        correct = sum(r.correct for r in c.results)
        row: dict[str, object] = {
            "Results file timestamp": _distinct(c.runs, "timestamp"),
            "Prompt version": _distinct(c.runs, "prompt_version"),
            "Model": _distinct(c.runs, "model"),
            "Dataset SHA-256": _distinct(c.runs, "dataset_sha256"),
            "Items": len(c.results),
            "Errors": sum(r.error is not None for r in c.results),
            "Accuracy": accuracy(c.results),
            "Accuracy 95% CI (Wilson)": _interval(correct, len(c.results)),
        }
        for label, m in critical_recall(matrix, CRITICAL_LABELS).items():
            hits = matrix[label][label]
            row[f"{label} support"] = m["support"]
            row[f"{label} recall"] = m["recall"]
            row[f"{label} recall 95% CI (Wilson)"] = _interval(hits, m["support"])
        row["Mean input tokens per email"], row["Mean output tokens per email"], row[
            "Cost per email (USD)"
        ] = _usage_cells(c)
        cells.append(row)
    return table(
        ["Metric", *(c.name for c in columns)],
        [[metric, *(row[metric] for row in cells)] for metric in cells[0]],
    )


def load_notes(dataset: Path) -> dict[str, str]:
    """`{item id: note}` for the dataset items that have a note."""
    if not dataset.exists():
        return {}
    notes = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("note"):
                notes[item["id"]] = item["note"]
    return notes


def _error_analysis(columns: Sequence[Column], notes: Mapping[str, str]) -> str:
    # The combined column repeats its parts, so list each item once.
    parts = [c for c in columns if c.name in SPLIT_ORDER] or list(columns)
    rows = [
        [r.id, r.split or c.name, r.label, r.predicted or ERROR, r.confidence, notes.get(r.id)]
        for c in parts
        for r in c.results
        if not r.correct
    ]
    return table(
        ["Id", "Split", "Actual", "Predicted", "Confidence", "Dataset note"],
        rows or [[None] * 6],
    )


# Static wording only; the evidence column is computed from the results.
_LIMITATIONS = (
    (
        "Synthetic data",
        "Emails are written for the eval, not sampled from real inboxes (real email "
        "content is never committed), so real phrasing, noise and formatting are under-represented.",
    ),
    (
        "Small n",
        "Few items per class, so every rate has a wide interval; a difference smaller "
        "than the interval is not evidence of a difference.",
    ),
    (
        "Saturation",
        "Accuracy near the ceiling cannot rank prompt versions: remaining misses are "
        "few and may be label judgment calls, not model errors.",
    ),
    (
        "Test split used once",
        "The test split only stays an unbiased estimate while it is run once, after "
        "tuning; each extra run or follow-up change turns it into a second dev set.",
    ),
    (
        "Time and duration extraction n too small",
        "Only a handful of items carry time labels, so those accuracies are anecdotal "
        "and are not reported with an interval.",
    ),
    (
        "Same-model family only",
        "Only one model family has been evaluated, so results say nothing about other "
        "models, and a shared family may share blind spots with whoever wrote the labels.",
    ),
)


def _evidence(
    columns: Sequence[Column], runs_per_split: Mapping[str, int]
) -> dict[str, str]:
    overall = columns[-1]
    correct = sum(r.correct for r in overall.results)
    interval = wilson_interval(correct, len(overall.results))
    time = time_extraction_summary(overall.results)
    return {
        "Synthetic data": "By construction",
        "Small n": (
            "Items: " + ", ".join(f"{c.name} {len(c.results)}" for c in columns)
            + (f"; accuracy 95% CI {interval[0]:.3f} to {interval[1]:.3f}" if interval else "")
        ),
        "Saturation": "Accuracy: " + ", ".join(f"{c.name} {accuracy(c.results):.3f}" for c in columns),
        "Test split used once": (
            f"Test result files found: {runs_per_split['test']}"
            if "test" in runs_per_split
            else NOT_AVAILABLE
        ),
        "Time and duration extraction n too small": (
            f"Items with a time label: {time['n']}; with a duration label: {time['duration_n']}"
        ),
        "Same-model family only": f"Models: {_distinct(overall.runs, 'model')}",
    }


def _limitations(columns: Sequence[Column], runs_per_split: Mapping[str, int]) -> str:
    evidence = _evidence(columns, runs_per_split)
    return table(
        ["Limitation", "What it means", "Evidence from these results"],
        [[name, meaning, evidence[name]] for name, meaning in _LIMITATIONS],
    )


def render(
    datas: Sequence[dict],
    notes: Mapping[str, str] | None = None,
    runs_per_split: Mapping[str, int] | None = None,
) -> str:
    columns = _columns(datas)
    overall = columns[-1]
    labels = list(datas[0]["summary"]["per_class"])
    matrix = confusion_matrix(overall.results, labels)
    per_class = per_class_metrics(matrix, labels)
    results = overall.results
    cols = [*labels, ERROR]

    sections = [
        "# Classification eval report",
        _overview_table(columns, labels),
        "## Recall of the costly classes",
        table(
            ["Class", "Recall", "Missed", "Support", "Missed as"],
            [
                [
                    label,
                    m["recall"],
                    m["missed"],
                    m["support"],
                    ", ".join(f"{k}: {v}" for k, v in m["missed_as"].items()) or "—",
                ]
                for label, m in critical_recall(matrix, CRITICAL_LABELS).items()
            ],
        ),
        "## Per-class precision, recall and F1",
        table(
            ["Class", "Precision", "Recall", "F1", "Support"],
            [
                [label, m["precision"], m["recall"], m["f1"], m["support"]]
                for label, m in per_class.items()
            ],
        ),
        "## Confusion matrix (rows: actual, columns: predicted)",
        table(
            ["Actual", *cols],
            [[actual, *(row[c] for c in cols)] for actual, row in matrix.items()],
        ),
    ]
    for title, slices in (
        ("Accuracy by language", tag_accuracy(results, "language")),
        ("Accuracy by difficulty", tag_accuracy(results, "difficulty")),
        ("Accuracy by id slice", slice_accuracy(results)),
    ):
        if slices:
            sections += [f"## {title}", table(["Slice", "N", "Accuracy"], _slice_rows(slices))]

    sections += [
        "## Error analysis",
        _error_analysis(columns, notes or {}),
        "## Limitations",
        _limitations(columns, runs_per_split or {}),
    ]
    return "\n\n".join(sections) + "\n"


def _load_runs(results_dir: Path) -> list[tuple[Path, dict]]:
    """Every classification results file, newest first."""
    files = sorted(
        (p for p in results_dir.glob("*.json") if _RESULT_NAME.match(p.name)), reverse=True
    )
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in files]


def latest_by_split(runs: Sequence[tuple[Path, dict]]) -> dict[str, Path]:
    """Newest file for each of dev and test; files that don't record a
    split (older runs) or ran on all items are not eligible."""
    found: dict[str, Path] = {}
    for path, data in runs:
        split = data.get("run", {}).get("split")
        if split in SPLIT_ORDER:
            found.setdefault(split, path)
    return found


def count_by_split(runs: Sequence[tuple[Path, dict]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, data in runs:
        split = data.get("run", {}).get("split")
        if split in SPLIT_ORDER:
            counts[split] = counts.get(split, 0) + 1
    return counts


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results-file",
        type=Path,
        action="append",
        help="repeatable; default: the newest dev and test files in --results-dir",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="source of item notes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _sources(args: argparse.Namespace, runs: Sequence[tuple[Path, dict]]) -> list[Path]:
    if args.results_file:
        return args.results_file
    by_split = latest_by_split(runs)
    if set(by_split) == set(SPLIT_ORDER):
        return [by_split[s] for s in SPLIT_ORDER]
    return [latest_results_file(args.results_dir)]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    runs = _load_runs(args.results_dir) if args.results_dir.exists() else []
    sources = _sources(args, runs)
    datas = [json.loads(p.read_text(encoding="utf-8")) for p in sources]
    report = render(datas, load_notes(args.dataset), count_by_split(runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} from {', '.join(p.name for p in sources)}")


if __name__ == "__main__":
    main()
