"""Eval harness: dataset invariants, metric math, and a full run with a
fake analyzer (no OpenAI calls)."""

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agents.email_analyzer import (
    AnalysisRun,
    _load_system_prompt,
    analyze_with_metadata,
)
from app.config import get_settings
from app.schemas.email import Email, EmailAnalysis, EmailCategory, SuggestedAction, Urgency
from eval.metrics import (
    ItemResult,
    accuracy,
    calibration_table,
    confusion_matrix,
    per_class_metrics,
    slice_accuracy,
    usage_summary,
)
from eval.run import DEFAULT_DATASET, LABELS, _parse_args, load_dataset, run, to_email

# --- dataset -------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DEFAULT_DATASET)


def test_dataset_has_50_items_balanced_across_categories(dataset):
    assert len(dataset) == 50
    assert Counter(item.label for item in dataset) == {c: 10 for c in EmailCategory}


def test_dataset_ids_are_unique(dataset):
    assert len({item.id for item in dataset}) == len(dataset)


def test_dataset_slices(dataset):
    tags = Counter(item.id.rsplit("-", 1)[-1] for item in dataset)
    assert tags["adv"] == 5
    assert tags["de"] >= 5
    assert tags["amb"] >= 5


def test_dataset_is_synthetic_only(dataset):
    """Every address uses the reserved `.example` TLD (RFC 2606), so no
    item can point at a real mailbox."""
    for item in dataset:
        assert item.sender.email.endswith(".example"), item.id


def test_every_item_converts_to_a_canonical_email(dataset):
    for item in dataset:
        assert isinstance(to_email(item), Email)


# --- metrics -------------------------------------------------------------------


def _r(id_, label, predicted, confidence=0.9, **kw) -> ItemResult:
    return ItemResult(
        id=id_,
        label=label,
        predicted=predicted,
        confidence=confidence if predicted else None,
        input_tokens=kw.get("input_tokens"),
        output_tokens=kw.get("output_tokens"),
        latency_ms=kw.get("latency_ms"),
        error=None if predicted else "boom",
    )


RESULTS = [
    _r("a-01", "rejection", "rejection", 0.95),
    _r("a-02", "rejection", "other", 0.6),
    _r("a-03-de", "other", "other", 0.8),
    _r("a-04-adv", "other", None),
]


def test_accuracy_counts_errors_as_wrong():
    assert accuracy(RESULTS) == 0.5


def test_confusion_matrix_has_error_column():
    matrix = confusion_matrix(RESULTS, LABELS)

    assert matrix["rejection"]["rejection"] == 1
    assert matrix["rejection"]["other"] == 1
    assert matrix["other"]["other"] == 1
    assert matrix["other"]["(error)"] == 1


def test_per_class_precision_recall_f1():
    metrics = per_class_metrics(confusion_matrix(RESULTS, LABELS), LABELS)

    assert metrics["rejection"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert metrics["other"]["precision"] == 0.5
    assert metrics["other"]["recall"] == 0.5
    # Never predicted, no support: undefined rather than a misleading 0.
    assert metrics["interview_invite"]["precision"] is None
    assert metrics["interview_invite"]["recall"] is None
    assert metrics["interview_invite"]["f1"] is None


def test_f1_is_zero_for_a_class_that_is_never_caught():
    results = [_r("m-1", "rejection", "other"), _r("m-2", "other", "other")]
    metrics = per_class_metrics(confusion_matrix(results, LABELS), LABELS)

    assert metrics["rejection"]["precision"] is None
    assert metrics["rejection"]["recall"] == 0.0
    assert metrics["rejection"]["f1"] == 0.0


def test_calibration_bins_and_edges():
    results = [
        _r("x-1", "other", "other", 0.0),
        _r("x-2", "other", "rejection", 0.5),
        _r("x-3", "other", "other", 0.7),
        _r("x-4", "other", "other", 1.0),
        _r("x-5", "other", None),
    ]
    table = {row["bin"]: row for row in calibration_table(results)}

    assert [table[b]["n"] for b in ("0.0-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0")] == [
        1, 1, 1, 1,
    ]
    assert table["0.5-0.7"]["accuracy"] == 0.0
    assert table["0.9-1.0"]["accuracy"] == 1.0


def test_usage_summary_cost_only_with_both_prices():
    results = [
        _r("u-1", "other", "other", input_tokens=1000, output_tokens=100, latency_ms=200),
        _r("u-2", "other", "other", input_tokens=3000, output_tokens=300, latency_ms=400),
    ]

    priced = usage_summary(results, input_usd_per_mtok=2.0, output_usd_per_mtok=8.0)
    unpriced = usage_summary(results, input_usd_per_mtok=2.0, output_usd_per_mtok=None)

    assert priced["mean_input_tokens"] == 2000
    assert priced["mean_latency_ms"] == 300
    assert priced["estimated_cost_usd_per_email"] == pytest.approx(
        (2000 * 2.0 + 200 * 8.0) / 1e6
    )
    assert unpriced["estimated_cost_usd_per_email"] is None


def test_slice_accuracy():
    slices = slice_accuracy(RESULTS)

    assert slices["plain"] == {"n": 2, "accuracy": 0.5}
    assert slices["de"] == {"n": 1, "accuracy": 1.0}
    assert slices["adv"] == {"n": 1, "accuracy": 0.0}


# --- full run with a fake analyzer ---------------------------------------------


def _fake_analyzer(email: Email) -> AnalysisRun:
    """Predicts `other` for everything, and fails on one item."""
    if email.id == "rej-01":
        raise RuntimeError("simulated API error")
    return AnalysisRun(
        analysis=EmailAnalysis(
            category=EmailCategory.OTHER,
            urgency=Urgency.LOW,
            suggested_action=SuggestedAction.NONE,
            reasoning="fake",
            confidence=0.8,
        ),
        model="fake-model",
        prompt_version="analyze_v1",
        input_tokens=500,
        output_tokens=50,
        latency_ms=100,
    )


def test_run_writes_results_and_reports_all_sections(tmp_path, capsys):
    args = _parse_args(
        [
            "--results-dir", str(tmp_path),
            "--input-usd-per-mtok", "1",
            "--output-usd-per-mtok", "4",
        ]
    )

    out_path, report = run(args, _fake_analyzer)

    data = json.loads(out_path.read_text())
    assert out_path.parent == tmp_path
    assert data["run"]["prompt_version"] == "analyze_v1"
    assert data["run"]["model"] == "fake-model"
    assert len(data["items"]) == 50
    assert data["summary"]["errors"] == 1
    assert data["summary"]["accuracy"] == pytest.approx(10 / 50)
    for section in ("accuracy:", "prec", "confusion matrix", "0.7-0.9", "mean latency"):
        assert section in report
    assert "$0.000700" in report  # (500 * 1 + 50 * 4) / 1e6


def test_unknown_prompt_version_is_rejected_by_cli():
    with pytest.raises(SystemExit):
        _parse_args(["--prompt-version", "../../.env"])


# --- analyzer: prompt version + usage -------------------------------------------


@pytest.fixture
def _env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-test-model",
        "NYLAS_API_KEY": "nylas-test",
        "NYLAS_CLIENT_ID": "client-test",
        "NYLAS_GRANT_ID": "grant-test",
        "WEBHOOK_SECRET": "super-secret",
        "SERVER_URL": "http://localhost:8000",
        "DATABASE_URL": "postgresql://user:pass@localhost/db",
        "APP_API_KEY": "app-test",
        "PROCESSING_MODE": "manual",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_load_system_prompt_rejects_unknown_version():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        _load_system_prompt("../../.env")


def test_analyze_with_metadata_reports_usage_and_model(_env, dataset):
    parsed = _fake_analyzer(to_email(dataset[0])).analysis
    client = MagicMock()
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(input_tokens=812, output_tokens=95),
    )

    with patch("app.agents.email_analyzer._get_client", return_value=client):
        result = analyze_with_metadata(
            to_email(dataset[0]), prompt_version="analyze_v1", model="override-model"
        )

    assert result.analysis == parsed
    assert result.model == "override-model"
    assert (result.input_tokens, result.output_tokens) == (812, 95)
    assert result.latency_ms >= 0
    assert client.responses.parse.call_args.kwargs["model"] == "override-model"
