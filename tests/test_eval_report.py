"""eval/report.py: tables only, every number from the results file."""

import json

import pytest

from app.agents.email_analyzer import AnalysisRun
from app.schemas.email import Email, EmailAnalysis, EmailCategory, SuggestedAction, Urgency
from eval.metrics import wilson_interval
from eval.report import _LIMITATIONS, latest_results_file, load_notes, main, render
from eval.run import DEFAULT_DATASET, _parse_args, load_dataset, run

_LABEL_BY_ID = {i.id: i.label for i in load_dataset(DEFAULT_DATASET)}


def _analyzer(wrong: set[str]):
    """Right everywhere except on the ids in `wrong`, where it says `other`."""

    def analyze(email: Email) -> AnalysisRun:
        category = EmailCategory.OTHER if email.id in wrong else _LABEL_BY_ID[email.id]
        return AnalysisRun(
            analysis=EmailAnalysis(
                category=category,
                urgency=Urgency.LOW,
                suggested_action=SuggestedAction.NONE,
                reasoning="fake",
                confidence=0.9,
            ),
            model="fake-model",
            prompt_version="analyze_v2",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )

    return analyze


def _results(tmp_path, wrong=frozenset(), split="all", prices=()):
    args = _parse_args(["--split", split, "--results-dir", str(tmp_path), *prices])
    path, _ = run(args, _analyzer(set(wrong)))
    return path


def _load(path):
    return json.loads(path.read_text())


def _both_splits(tmp_path, wrong=frozenset(), prices=()):
    return [
        _load(_results(tmp_path / split, wrong, split, prices)) for split in ("dev", "test")
    ]


def test_report_is_tables_and_headings_only(tmp_path):
    out = tmp_path / "report.md"
    main(["--results-file", str(_results(tmp_path)), "--output", str(out)])

    for line in out.read_text().splitlines():
        assert not line.strip() or line.startswith(("#", "|")), line
    text = out.read_text()
    for heading in (
        "## Recall of the costly classes",
        "## Per-class precision, recall and F1",
        "## Confusion matrix",
        "## Accuracy by language",
        "## Accuracy by difficulty",
    ):
        assert heading in text


def test_numbers_come_from_the_results_file(tmp_path):
    ids = {"hard-01", "hard-02"}  # two interview invites
    perfect = render([_load(_results(tmp_path / "a"))])
    degraded = render([_load(_results(tmp_path / "b", wrong=ids))])

    assert "| interview_invite | 1.000 | 0 |" in perfect
    assert "| interview_invite | 0.9" in degraded
    assert "| interview_invite | 1.000 | 0 |" not in degraded
    assert perfect != degraded
    assert "| hard-01 | all | interview_invite | other | 0.900 |" in degraded
    assert "| hard-01 |" not in perfect


def test_report_recomputes_from_items_so_summary_sections_are_optional(tmp_path):
    data = _load(_results(tmp_path))
    for key in ("critical_recall", "slices_language", "slices_difficulty"):
        del data["summary"][key]
    for item in data["items"]:
        for key in ("language", "difficulty", "split"):
            del item[key]
    del data["run"]["split"]

    text = render([data])

    assert "## Recall of the costly classes" in text
    assert "## Accuracy by language" not in text  # untagged old results
    assert "all (not recorded)" in text


def test_latest_results_file_ignores_other_evals(tmp_path):
    for name in (
        "20261017T162140Z.json",
        "20261025T193701Z.json",
        "ask-20261025T194211Z.json",
        "retrieval-20261025T101312Z.json",
        "notes.json",
    ):
        (tmp_path / name).write_text("{}")

    assert latest_results_file(tmp_path).name == "20261025T193701Z.json"


def test_no_results_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no classification results"):
        latest_results_file(tmp_path)


def test_main_picks_the_newest_results_by_default(tmp_path, capsys):
    results_dir = tmp_path / "results"
    old = _results(results_dir / "x")
    newer = results_dir / "99991231T235959Z.json"
    newer.write_text(old.read_text())
    (results_dir / old.name).write_text(old.read_text())
    out = tmp_path / "docs" / "eval-report.md"

    main(["--results-dir", str(results_dir), "--output", str(out)])

    assert out.exists()
    assert newer.name in capsys.readouterr().out


def test_wilson_interval_matches_known_values():
    low, high = wilson_interval(8, 10)
    assert (round(low, 3), round(high, 3)) == (0.490, 0.943)
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(0, 0) is None


def _row(text: str, name: str) -> list[str]:
    line = next(l for l in text.splitlines() if l.startswith(f"| {name} |"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_splits_side_by_side_with_combined_and_intervals(tmp_path):
    dev, test = _both_splits(tmp_path)
    text = render([test, dev])  # input order does not matter

    header = next(l for l in text.splitlines() if l.startswith("| Metric |"))
    assert [c.strip() for c in header.strip("|").split("|")] == [
        "Metric", "dev", "test", "dev + test"
    ]
    items = _row(text, "Items")
    assert items[1:] == [str(len(dev["items"])), str(len(test["items"])), str(len(dev["items"]) + len(test["items"]))]
    for metric in (
        "Accuracy 95% CI (Wilson)",
        "interview_invite recall 95% CI (Wilson)",
        "assessment_deadline recall 95% CI (Wilson)",
    ):
        assert all(" to " in cell for cell in _row(text, metric)[1:]), metric


def test_combined_column_counts_misses_from_both_splits(tmp_path):
    dev, test = _both_splits(tmp_path, wrong={"hard-01", "hard-02"})

    text = render([dev, test])

    errors = _row(text, "Errors")
    accuracy = _row(text, "Accuracy")
    assert accuracy[3] != "1.000"
    assert float(accuracy[3]) == pytest.approx(
        1 - 2 / (len(dev["items"]) + len(test["items"])), abs=1e-3
    )
    assert errors[1:] == ["0", "0", "0"]  # wrong answers are not call errors


def test_cost_and_tokens_come_from_the_file(tmp_path):
    prices = ["--input-usd-per-mtok", "2", "--output-usd-per-mtok", "4"]
    dev, test = _both_splits(tmp_path, prices=prices)

    text = render([dev, test])

    # The fake analyzer reports 1 input and 1 output token per email.
    assert _row(text, "Mean input tokens per email")[1:] == ["1.0"] * 3
    assert _row(text, "Cost per email (USD)")[1:] == ["$0.000006"] * 3


def test_price_flags_are_persisted_and_old_files_show_na(tmp_path):
    prices = ["--input-usd-per-mtok", "2.5", "--output-usd-per-mtok", "10"]
    priced = _load(_results(tmp_path / "a", prices=prices))
    unpriced = _load(_results(tmp_path / "b"))
    assert priced["run"]["prices_usd_per_mtok"] == {"input": 2.5, "output": 10.0}
    assert unpriced["run"]["prices_usd_per_mtok"] == {"input": None, "output": None}

    legacy = _load(_results(tmp_path / "c"))
    del legacy["run"]["prices_usd_per_mtok"]
    for item in legacy["items"]:
        item["input_tokens"] = item["output_tokens"] = None

    for data in (unpriced, legacy):
        text = render([data])
        assert _row(text, "Cost per email (USD)")[1] == "n/a"
    assert _row(render([legacy]), "Mean input tokens per email")[1] == "n/a"


def test_mixed_prices_across_splits_do_not_give_a_combined_cost(tmp_path):
    prices = ["--input-usd-per-mtok", "2", "--output-usd-per-mtok", "4"]
    dev = _load(_results(tmp_path / "dev", split="dev", prices=prices))
    test = _load(_results(tmp_path / "test", split="test"))

    cost = _row(render([dev, test]), "Cost per email (USD)")

    assert cost[1:] == ["$0.000006", "n/a", "n/a"]


def test_error_analysis_lists_each_miss_once_with_split_and_note(tmp_path):
    dev, test = _both_splits(tmp_path, wrong={"hard-01"})
    notes = {"hard-01": "judgment | call"}

    text = render([dev, test], notes=notes)

    section = text.split("## Error analysis")[1].split("## Limitations")[0]
    assert "| Id | Split | Actual | Predicted | Confidence | Dataset note |" in section
    rows = [l for l in section.splitlines() if l.startswith("| hard-01 |")]
    assert len(rows) == 1
    split = "dev" if any(i["id"] == "hard-01" for i in dev["items"]) else "test"
    assert rows[0] == f"| hard-01 | {split} | interview_invite | other | 0.900 | judgment \\| call |"


def test_error_analysis_without_misses_is_an_empty_row(tmp_path):
    section = render([_load(_results(tmp_path))]).split("## Error analysis")[1].split("## Limitations")[0]
    assert "| — | — | — | — | — | — |" in section


def test_notes_are_read_from_the_dataset(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        json.dumps({"id": "a", "note": "why"}) + "\n" + json.dumps({"id": "b"}) + "\n"
    )
    assert load_notes(dataset) == {"a": "why"}
    assert load_notes(tmp_path / "missing.jsonl") == {}


def test_limitations_table_covers_every_caveat_without_hardcoded_numbers(tmp_path):
    text = render([_load(_results(tmp_path))], runs_per_split={"test": 3})

    section = text.split("## Limitations")[1]
    assert "| Limitation | What it means | Evidence from these results |" in section
    for name in (
        "Synthetic data",
        "Small n",
        "Saturation",
        "Test split used once",
        "Time and duration extraction n too small",
        "Same-model family only",
    ):
        assert f"| {name} |" in section
    assert "Test result files found: 3" in section
    for name, meaning in _LIMITATIONS:
        assert not any(ch.isdigit() for ch in meaning), name


def test_main_uses_newest_file_of_each_split(tmp_path, capsys):
    results = tmp_path / "results"
    old_dev = _results(results, split="dev")
    stamp = {"dev": "20300101T000000Z.json", "test": "20300102T000000Z.json"}
    for split, name in stamp.items():
        (results / name).write_text(_results(tmp_path / split, split=split).read_text())
    (results / old_dev.name).write_text(old_dev.read_text())
    out = tmp_path / "report.md"

    main(["--results-dir", str(results), "--output", str(out)])

    printed = capsys.readouterr().out
    assert stamp["dev"] in printed and stamp["test"] in printed
    assert old_dev.name not in printed
    assert "| Metric | dev | test | dev + test |" in out.read_text()
    assert "Test result files found: 1" in out.read_text()
