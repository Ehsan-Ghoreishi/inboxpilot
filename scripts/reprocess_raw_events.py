"""Run the triage flow for every stored raw event that has no matching email.

Same flow as the `POST /events` background task (classification calls
OpenAI; Nylas is never called). Safe to re-run: already-processed
messages are skipped. Prints event ids and outcomes only, never email
content or secrets.

Usage:
    uv run python scripts/reprocess_raw_events.py [--dry-run]
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.repository import list_unprocessed_raw_events  # noqa: E402
from app.database.session import session_scope  # noqa: E402
from app.workflows.background import run_triage_safely  # noqa: E402


def _load_unprocessed() -> list[tuple[str, dict]]:
    with session_scope() as session:
        return [
            (raw.event_id, raw.payload)
            for raw in list_unprocessed_raw_events(session)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="Only count unprocessed events."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = _load_unprocessed()
    print(f"{len(events)} unprocessed raw event(s)")
    if args.dry_run:
        return

    counts: Counter[str] = Counter()
    for event_id, payload in events:
        outcome = run_triage_safely(event_id, payload)
        counts[outcome.value if outcome else "error"] += 1

    for outcome, count in sorted(counts.items()):
        print(f"{outcome}: {count}")


if __name__ == "__main__":
    main()
