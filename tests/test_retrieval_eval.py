"""Retrieval eval: dataset invariants, metric math, and a full run with a
fake retriever (no database, no OpenAI calls)."""

import json

import pytest

from eval.run import DEFAULT_DATASET as CORPUS_PATH
from eval.run import load_dataset
from eval.run_retrieval import (
    DEFAULT_DATASET,
    QuestionResult,
    RetrievalItem,
    _parse_args,
    check_expected_ids,
    eval_database_url,
    first_relevant_rank,
    is_refused,
    load_questions,
    recall_at_k,
    reciprocal_rank,
    run,
    summarize,
)

# --- dataset -------------------------------------------------------------------


@pytest.fixture(scope="module")
def questions():
    return load_questions(DEFAULT_DATASET)


def test_dataset_has_20_answerable_and_5_unanswerable(questions):
    assert sum(q.answerable for q in questions) == 20
    assert sum(not q.answerable for q in questions) == 5


def test_dataset_ids_are_unique(questions):
    assert len({q.id for q in questions}) == len(questions)


def test_expected_ids_exist_in_corpus(questions):
    check_expected_ids(questions, load_dataset(CORPUS_PATH))


def test_unknown_expected_id_is_rejected():
    bad = [RetrievalItem(id="x", question="q", expected_email_ids=["nope-01"])]
    with pytest.raises(ValueError, match="nope-01"):
        check_expected_ids(bad, load_dataset(CORPUS_PATH))


# --- metrics -------------------------------------------------------------------


def test_recall_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, ["a"], 3) == 1.0
    assert recall_at_k(ranked, ["c", "e"], 3) == 0.5
    assert recall_at_k(ranked, ["c", "e"], 5) == 1.0
    assert recall_at_k(ranked, ["z"], 5) == 0.0


def test_reciprocal_rank_and_first_relevant_rank():
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, ["c", "b"]) == 0.5
    assert first_relevant_rank(ranked, ["c", "b"]) == 2
    assert reciprocal_rank(ranked, ["z"]) == 0.0
    assert first_relevant_rank(ranked, ["z"]) is None


def test_refusal_is_strictly_below_threshold():
    assert is_refused(0.29, 0.3)
    assert not is_refused(0.3, 0.3)
    assert is_refused(None, -1.0)


def _q(id_, expected, retrieved) -> QuestionResult:
    return QuestionResult(
        id=id_, question=f"question {id_}", expected_email_ids=expected, retrieved=retrieved
    )


RESULTS = [
    _q("q-1", ["a"], [("a", 0.6), ("b", 0.4)]),
    _q("q-2", ["b"], [("a", 0.5), ("b", 0.45)]),
    _q("u-1", [], [("a", 0.35)]),
    _q("u-2", [], [("b", 0.2)]),
]


def test_summary_metrics_cover_answerable_only():
    summary = summarize(RESULTS, min_score=0.3)

    assert summary["n_answerable"] == 2
    assert summary["recall@3"] == 1.0
    assert summary["mrr"] == pytest.approx(0.75)
    assert summary["top1_score_range"]["answerable"] == {"min": 0.5, "max": 0.6}
    assert summary["top1_score_range"]["unanswerable"] == {"min": 0.2, "max": 0.35}
    assert summary["at_min_score"] == {
        "min_score": 0.3,
        "answerable_kept": 2,
        "answerable_total": 2,
        "unanswerable_refused": 1,
        "unanswerable_total": 2,
    }


def test_sweep_trades_false_refusals_for_correct_refusals():
    sweep = {row["min_score"]: row for row in summarize(RESULTS, 0.0)["sweep"]}

    assert sweep[0.4]["unanswerable_refused"] == 2
    assert sweep[0.4]["answerable_kept"] == 2
    assert sweep[0.55]["answerable_kept"] == 1


def test_suggested_min_score_picks_widest_gap_among_best_cuts():
    # Only a cut between 0.35 and 0.5 handles all four correctly.
    assert summarize(RESULTS, 0.0)["suggested_min_score"] == pytest.approx(0.425)

    overlapping = [
        _q("q-1", ["a"], [("a", 0.6)]),
        _q("u-1", [], [("a", 0.5)]),
        _q("q-2", ["a"], [("a", 0.45)]),
        _q("u-2", [], [("a", 0.1)]),
    ]
    # 3/4 either way: cut in 0.1..0.45 (misses u-1) or 0.5..0.6 (loses
    # q-2); the wider gap wins.
    assert summarize(overlapping, 0.0)["suggested_min_score"] == pytest.approx(0.275)


def test_suggested_min_score_needs_two_distinct_scores():
    assert summarize([_q("q-1", ["a"], [("a", 0.5)])], 0.0)["suggested_min_score"] is None


# --- full run with a fake retriever --------------------------------------------


def test_run_writes_results_and_reports_every_question(tmp_path, questions):
    def retriever(question: str) -> list[tuple[str, float]]:
        # Rank every answerable question's first expected email top.
        match = next(q for q in questions if q.question == question)
        if match.answerable:
            return [(match.expected_email_ids[0], 0.6), ("oth-01", 0.3)]
        return [("oth-01", 0.25)]

    args = _parse_args(["--results-dir", str(tmp_path), "--min-score", "0.4"])

    out_path, report = run(args, retriever, embedding_model="fake", corpus_size=50)

    data = json.loads(out_path.read_text())
    assert out_path.name.startswith("retrieval-")
    assert len(data["items"]) == 25
    assert data["summary"]["mrr"] == 1.0
    assert data["summary"]["at_min_score"]["unanswerable_refused"] == 5
    assert all(item["top1_score"] is not None for item in data["items"])
    for question in questions:
        assert question.id in report
    assert "refused ok" in report
    assert "recall@5=" in report


def test_eval_database_is_never_the_dev_database():
    url = eval_database_url("postgresql+psycopg://u:p@localhost:5432/inboxpilot")
    assert url.database == "inboxpilot_eval"
