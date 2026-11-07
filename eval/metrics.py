"""Pure metric functions over eval results (no I/O, no model calls)."""

from collections.abc import Sequence
from math import sqrt
from statistics import mean

from pydantic import BaseModel

ERROR = "(error)"

# A missed one of these is the costly error: the candidate never hears
# about an interview or a deadline.
CRITICAL_LABELS = ("interview_invite", "assessment_deadline")

# Half-open [lo, hi), except the last bin, which includes 1.0.
CALIBRATION_BINS: list[tuple[float, float]] = [
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.0),
]


class ItemResult(BaseModel):
    id: str
    label: str
    predicted: str | None
    confidence: float | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    model: str | None = None
    error: str | None = None
    analysis: dict | None = None
    # `None` for items without a time label (see `eval.run.EvalItem`).
    time_correct: bool | None = None
    duration_correct: bool | None = None
    # Dataset tags, for slicing (see `eval.run.EvalItem`).
    language: str | None = None
    difficulty: str | None = None
    split: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.label


def accuracy(results: Sequence[ItemResult]) -> float:
    """Over all items; an item whose analysis errored counts as wrong."""
    return sum(r.correct for r in results) / len(results) if results else 0.0


def confusion_matrix(
    results: Sequence[ItemResult], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """`matrix[actual][predicted]`. Errors get their own `(error)` column."""
    columns = [*labels, ERROR]
    matrix = {actual: dict.fromkeys(columns, 0) for actual in labels}
    for r in results:
        matrix[r.label][r.predicted or ERROR] += 1
    return matrix


def per_class_metrics(
    matrix: dict[str, dict[str, int]], labels: Sequence[str]
) -> dict[str, dict[str, float | int | None]]:
    """Precision/recall/F1 per label. `None` where the ratio is undefined
    (e.g. precision for a label that was never predicted)."""
    out: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        tp = matrix[label][label]
        predicted = sum(matrix[actual][label] for actual in labels)
        support = sum(matrix[label].values())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        elif precision == 0 or recall == 0:
            # A class that exists but was never caught (or only ever
            # predicted wrongly) has F1 = 0, not "undefined".
            f1 = 0.0
        else:
            f1 = None
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return out


def _bin_index(confidence: float) -> int:
    for i, (lo, hi) in enumerate(CALIBRATION_BINS):
        if lo <= confidence < hi:
            return i
    return len(CALIBRATION_BINS) - 1  # confidence == 1.0


def calibration_table(
    results: Sequence[ItemResult],
) -> list[dict[str, float | int | str | None]]:
    """Per confidence bin: count, mean stated confidence, actual accuracy.
    Errored items (no confidence) are excluded."""
    bins: list[list[ItemResult]] = [[] for _ in CALIBRATION_BINS]
    for r in results:
        if r.confidence is not None:
            bins[_bin_index(r.confidence)].append(r)
    return [
        {
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(members),
            "mean_confidence": mean(r.confidence for r in members) if members else None,
            "accuracy": accuracy(members) if members else None,
        }
        for (lo, hi), members in zip(CALIBRATION_BINS, bins)
    ]


def _mean_or_none(values: Sequence[int | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def usage_summary(
    results: Sequence[ItemResult],
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> dict[str, float | None]:
    """Mean tokens/latency per successfully analyzed email, and estimated
    cost per email — only when both prices are given."""
    mean_in = _mean_or_none([r.input_tokens for r in results])
    mean_out = _mean_or_none([r.output_tokens for r in results])
    cost = None
    if (
        input_usd_per_mtok is not None
        and output_usd_per_mtok is not None
        and mean_in is not None
        and mean_out is not None
    ):
        cost = (mean_in * input_usd_per_mtok + mean_out * output_usd_per_mtok) / 1e6
    return {
        "mean_input_tokens": mean_in,
        "mean_output_tokens": mean_out,
        "estimated_cost_usd_per_email": cost,
        "mean_latency_ms": _mean_or_none([r.latency_ms for r in results]),
    }


def slice_accuracy(results: Sequence[ItemResult]) -> dict[str, dict[str, float | int]]:
    """Accuracy per dataset slice, encoded as an id suffix (`-de`, `-amb`,
    `-adv`); items without a suffix are `plain`."""
    slices: dict[str, list[ItemResult]] = {}
    for r in results:
        tag = r.id.rsplit("-", 1)[-1]
        slices.setdefault(tag if tag in {"de", "amb", "adv"} else "plain", []).append(r)
    return {
        tag: {"n": len(members), "accuracy": accuracy(members)}
        for tag, members in sorted(slices.items())
    }


def tag_accuracy(
    results: Sequence[ItemResult], field: str
) -> dict[str, dict[str, float | int]]:
    """Accuracy per value of a tag field (`language`, `difficulty`).
    Items without the tag are left out."""
    groups: dict[str, list[ItemResult]] = {}
    for r in results:
        value = getattr(r, field)
        if value is not None:
            groups.setdefault(value, []).append(r)
    return {
        value: {"n": len(members), "accuracy": accuracy(members)}
        for value, members in sorted(groups.items())
    }


def critical_recall(
    matrix: dict[str, dict[str, int]], labels: Sequence[str] = CRITICAL_LABELS
) -> dict[str, dict[str, float | int | None | dict[str, int]]]:
    """Recall for the costly classes on their own, with the number missed
    and what the misses were taken for."""
    out: dict[str, dict[str, float | int | None | dict[str, int]]] = {}
    for label in labels:
        row = matrix[label]
        support = sum(row.values())
        missed_as = {col: n for col, n in row.items() if col != label and n}
        out[label] = {
            "recall": row[label] / support if support else None,
            "support": support,
            "missed": sum(missed_as.values()),
            "missed_as": missed_as,
        }
    return out


def wilson_interval(
    successes: int, n: int, z: float = 1.96
) -> tuple[float, float] | None:
    """Wilson score interval for a proportion (95% for the default `z`).
    `None` when there are no trials. Unlike the normal approximation it
    stays inside [0, 1] and is honest at small n and at 0% or 100%."""
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def time_extraction_summary(
    results: Sequence[ItemResult],
) -> dict[str, float | int | None]:
    """Over time-labeled items only, separate from category accuracy: is
    the scheduled start (or the call for clarification) right, and, where
    a start is expected, is the duration right."""
    times = [r.time_correct for r in results if r.time_correct is not None]
    durations = [r.duration_correct for r in results if r.duration_correct is not None]
    return {
        "n": len(times),
        "accuracy": _rate(times),
        "duration_n": len(durations),
        "duration_accuracy": _rate(durations),
    }
