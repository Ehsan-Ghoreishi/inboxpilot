"""Evaluate semantic retrieval (the /search and /ask retrieval step) over
the classification eval emails.

Usage:
    uv run python -m eval.run_retrieval [--min-score X]

Needs a configured `.env` and a reachable Postgres. The corpus
(`eval/dataset.jsonl`) is loaded into a dedicated `<db>_eval` database
(created and migrated if missing, then truncated, never the dev
database) and embedded with the real configured embedder, one call per
email plus one per question.

Answerable questions are scored with recall@3, recall@5 and MRR; every
question's top-1 score is reported so `RAG_MIN_SCORE` can be calibrated,
and unanswerable questions count as correctly refused when their top-1
score is below the threshold (that is when /ask refuses without calling
the model). Writes `eval/results/retrieval-<UTC timestamp>.json`.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from alembic import command
from alembic.config import Config
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database.repository import get_or_create_email
from app.schemas.search import MAX_TOP_K, SearchRequest
from app.services.embedding import Embedder, embed_email, get_embedder
from app.services.search import retrieve_emails
from eval.run import DEFAULT_DATASET as DEFAULT_CORPUS
from eval.run import DEFAULT_RESULTS_DIR, EvalItem, load_dataset, to_email

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DEFAULT_DATASET = EVAL_DIR / "retrieval_dataset.jsonl"
TABLES = "raw_events, emails, classifications, pending_actions, audit_log"
RECALL_KS = (3, 5)
# Candidate RAG_MIN_SCORE values for the calibration sweep.
SWEEP = [round(0.05 * i, 2) for i in range(2, 15)]

# A question in, ranked `(corpus email id, score)` pairs out, best first.
Retriever = Callable[[str], list[tuple[str, float]]]


class RetrievalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    # Corpus ids (`eval/dataset.jsonl` ids); empty means unanswerable.
    expected_email_ids: list[str]

    @property
    def answerable(self) -> bool:
        return bool(self.expected_email_ids)


class QuestionResult(BaseModel):
    id: str
    question: str
    expected_email_ids: list[str]
    retrieved: list[tuple[str, float]]

    @property
    def answerable(self) -> bool:
        return bool(self.expected_email_ids)

    @property
    def ranked_ids(self) -> list[str]:
        return [email_id for email_id, _ in self.retrieved]

    @property
    def top1_score(self) -> float | None:
        return self.retrieved[0][1] if self.retrieved else None


def load_questions(path: Path) -> list[RetrievalItem]:
    with path.open(encoding="utf-8") as f:
        return [RetrievalItem.model_validate_json(line) for line in f if line.strip()]


# --- metrics (pure) --------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    return len(set(ranked[:k]) & set(expected)) / len(expected)


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    """1/rank of the first relevant email; 0 when none was retrieved."""
    for rank, email_id in enumerate(ranked, 1):
        if email_id in expected:
            return 1 / rank
    return 0.0


def first_relevant_rank(ranked: Sequence[str], expected: Sequence[str]) -> int | None:
    return next((i for i, e in enumerate(ranked, 1) if e in expected), None)


def is_refused(top1_score: float | None, min_score: float) -> bool:
    """Whether /ask would refuse: no retrieved email reaches `min_score`."""
    return top1_score is None or top1_score < min_score


def threshold_row(results: Sequence[QuestionResult], min_score: float) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    return {
        "min_score": min_score,
        "answerable_kept": sum(not is_refused(r.top1_score, min_score) for r in answerable),
        "answerable_total": len(answerable),
        "unanswerable_refused": sum(is_refused(r.top1_score, min_score) for r in unanswerable),
        "unanswerable_total": len(unanswerable),
    }


def suggest_min_score(results: Sequence[QuestionResult]) -> float | None:
    """The threshold that handles the most questions correctly (answerable
    kept + unanswerable refused), placed at the midpoint of the widest
    top-1 score gap among the equally good ones. Only cuts between two
    observed scores are considered; `None` with fewer than two."""
    scores = sorted({r.top1_score for r in results if r.top1_score is not None})

    def quality(lower: float, upper: float) -> tuple[int, float]:
        row = threshold_row(results, (lower + upper) / 2)
        return row["answerable_kept"] + row["unanswerable_refused"], upper - lower

    gaps = list(zip(scores, scores[1:]))
    if not gaps:
        return None
    lower, upper = max(gaps, key=lambda gap: quality(*gap))
    return round((lower + upper) / 2, 3)


def summarize(results: Sequence[QuestionResult], min_score: float) -> dict:
    answerable = [r for r in results if r.answerable]
    top1 = {
        kind: [r.top1_score for r in results if r.answerable == is_answerable and r.top1_score is not None]
        for kind, is_answerable in (("answerable", True), ("unanswerable", False))
    }
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(results) - len(answerable),
        **{
            f"recall@{k}": mean(recall_at_k(r.ranked_ids, r.expected_email_ids, k) for r in answerable)
            for k in RECALL_KS
        },
        "mrr": mean(reciprocal_rank(r.ranked_ids, r.expected_email_ids) for r in answerable),
        "top1_score_range": {
            kind: {"min": min(scores), "max": max(scores)} if scores else None
            for kind, scores in top1.items()
        },
        "at_min_score": threshold_row(results, min_score),
        "suggested_min_score": suggest_min_score(results),
        "sweep": [threshold_row(results, t) for t in SWEEP],
    }


# --- report ----------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "—" if value is None else format(value, spec)


def format_report(run_info: dict, results: Sequence[QuestionResult], summary: dict) -> str:
    min_score = run_info["min_score"]
    lines = [
        f"embedding_model={run_info['embedding_model']}  corpus={run_info['corpus_size']}  "
        f"questions={len(results)}  min_score={min_score}",
        "",
        f"{'question':<8}{'top1':>7}  {'top1 email':<12}{'rank':>5}{'R@3':>6}{'R@5':>6}  outcome",
    ]
    for r in results:
        top1_id = r.ranked_ids[0] if r.retrieved else "—"
        refused = is_refused(r.top1_score, min_score)
        if r.answerable:
            rank = first_relevant_rank(r.ranked_ids, r.expected_email_ids)
            metrics = (
                f"{rank or '—':>5}"
                f"{recall_at_k(r.ranked_ids, r.expected_email_ids, 3):>6.2f}"
                f"{recall_at_k(r.ranked_ids, r.expected_email_ids, 5):>6.2f}"
            )
            outcome = "FALSE REFUSAL" if refused else "answered"
        else:
            metrics = f"{'':>17}"
            outcome = "refused ok" if refused else "NOT REFUSED"
        lines.append(
            f"{r.id:<8}{_fmt(r.top1_score):>7}  {top1_id:<12}{metrics}  {outcome}"
        )

    at = summary["at_min_score"]
    ranges = summary["top1_score_range"]
    lines += [
        "",
        f"answerable (n={summary['n_answerable']}): "
        f"recall@3={summary['recall@3']:.3f}  recall@5={summary['recall@5']:.3f}  "
        f"MRR={summary['mrr']:.3f}",
        "top-1 score range: "
        + "  ".join(
            f"{kind}={_fmt(rng['min'])}..{_fmt(rng['max'])}" if rng else f"{kind}=—"
            for kind, rng in ranges.items()
        ),
        f"at min_score={min_score}: unanswerable refused "
        f"{at['unanswerable_refused']}/{at['unanswerable_total']}, answerable kept "
        f"{at['answerable_kept']}/{at['answerable_total']}",
        f"suggested min_score: {_fmt(summary['suggested_min_score'])} "
        "(most questions handled correctly, then widest score gap)",
        "",
        f"{'min_score':<11}{'answerable kept':>17}{'unanswerable refused':>22}",
    ]
    for row in summary["sweep"]:
        lines.append(
            f"{row['min_score']:<11.2f}"
            f"{row['answerable_kept']:>11}/{row['answerable_total']:<5}"
            f"{row['unanswerable_refused']:>16}/{row['unanswerable_total']:<5}"
        )
    return "\n".join(lines)


# --- run -------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_retrieval", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--min-score",
        type=float,
        help="threshold to report refusals at (default: settings.rag_min_score)",
    )
    return parser.parse_args(argv)


def check_expected_ids(questions: Sequence[RetrievalItem], corpus: Sequence[EvalItem]) -> None:
    corpus_ids = {item.id for item in corpus}
    unknown = sorted(
        {e for q in questions for e in q.expected_email_ids} - corpus_ids
    )
    if unknown:
        raise ValueError(f"expected_email_ids not in the corpus: {', '.join(unknown)}")


def run(
    args: argparse.Namespace,
    retriever: Retriever,
    *,
    embedding_model: str,
    corpus_size: int,
) -> tuple[Path, str]:
    """Retrieve for every question, write the results file, and return
    its path and the printable report."""
    questions = load_questions(args.dataset)
    started = datetime.now(timezone.utc)
    results = []
    for i, item in enumerate(questions, 1):
        results.append(
            QuestionResult(
                id=item.id,
                question=item.question,
                expected_email_ids=item.expected_email_ids,
                retrieved=retriever(item.question),
            )
        )
        print(f"[{i:>2}/{len(questions)}] {item.id}", flush=True)

    run_info = {
        "timestamp": started.isoformat(),
        "embedding_model": embedding_model,
        "min_score": args.min_score,
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "corpus_size": corpus_size,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    summary = summarize(results, args.min_score)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"retrieval-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps(
            {
                "run": run_info,
                "summary": summary,
                "items": [
                    {
                        **r.model_dump(mode="json"),
                        "top1_score": r.top1_score,
                        "refused": is_refused(r.top1_score, args.min_score),
                    }
                    for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path, format_report(run_info, results, summary)


# --- database wiring -------------------------------------------------------------


def eval_database_url(dev_url: str) -> URL:
    url = make_url(dev_url)
    return url.set(database=f"{url.database}_eval")


def prepare_database(url: URL) -> None:
    """Create the eval database if needed and migrate it to head."""
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        admin.dispose()

    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    alembic_cfg.set_main_option(
        "sqlalchemy.url", url.render_as_string(hide_password=False)
    )
    command.upgrade(alembic_cfg, "head")


def ingest_corpus(
    session: Session, corpus: Sequence[EvalItem], embedder: Embedder
) -> dict[int, str]:
    """Empty the database, store and embed every corpus email, and return
    a map from stored email id to corpus id."""
    session.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY CASCADE"))
    id_map = {}
    for item in corpus:
        record = get_or_create_email(session, to_email(item))
        embed_email(session, record, embedder)
        id_map[record.id] = item.id
    return id_map


def make_retriever(
    session: Session, embedder: Embedder, id_map: dict[int, str]
) -> Retriever:
    """Retrieval exactly as /search does it, but unthresholded and at the
    maximum depth, so every score is visible."""

    def retriever(question: str) -> list[tuple[str, float]]:
        matches = retrieve_emails(
            session,
            SearchRequest(query=question, top_k=MAX_TOP_K),
            embedder,
            min_score=-1.0,
        )
        return [(id_map[email.id], score) for email, score in matches]

    return retriever


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
    embedder = get_embedder()
    try:
        with sessionmaker(bind=engine)() as session:
            print(f"embedding {len(corpus)} corpus emails into {url.database} ...", flush=True)
            id_map = ingest_corpus(session, corpus, embedder)
            session.commit()
            out_path, report = run(
                args,
                make_retriever(session, embedder, id_map),
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
