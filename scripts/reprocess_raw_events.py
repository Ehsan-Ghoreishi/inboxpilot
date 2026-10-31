"""Run the triage flow for stored raw events that have no matching email.

Same flow as the `POST /events` background task (classification calls
OpenAI; Nylas is never called). Idempotent: already-processed messages
are skipped, so it is safe to re-run or schedule (e.g. as a Cloud Run
job). Prints event ids and outcomes only, never email content or
secrets. Exits non-zero if any event failed, so a scheduler can alert.

Usage:
    uv run python -m scripts.reprocess_raw_events [--limit N] [--dry-run]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.repository import list_unprocessed_raw_events  # noqa: E402
from app.database.session import session_scope  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.workflows.background import run_triage_safely  # noqa: E402

DEFAULT_LIMIT = 50


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _load_unprocessed(limit: int) -> list[tuple[str, dict]]:
    with session_scope() as session:
        return [
            (raw.event_id, raw.payload)
            for raw in list_unprocessed_raw_events(session, limit=limit)
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=DEFAULT_LIMIT,
        help=f"Process at most N events (default {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only count unprocessed events."
    )
    args = parser.parse_args(argv)
    configure_logging()

    try:
        events = _load_unprocessed(args.limit)
    except Exception as exc:  # noqa: BLE001 - report and fail the run
        print(f"Could not load unprocessed raw events: {type(exc).__name__}")
        return 1
    print(f"{len(events)} unprocessed raw event(s)")
    if args.dry_run:
        return 0

    counts: Counter[str] = Counter()
    for event_id, payload in events:
        outcome = run_triage_safely(event_id, payload)
        counts[outcome.value if outcome else "error"] += 1

    for outcome, count in sorted(counts.items()):
        print(f"{outcome}: {count}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
