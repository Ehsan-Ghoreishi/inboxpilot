"""Embed every stored email that has no embedding for the configured model.

Each email is embedded in its own transaction, so one failure doesn't
undo the others; failed emails stay eligible for the next run. Prints
counts and email ids only, never email content or secrets.

Usage:
    uv run python scripts/embed_emails.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.models import EmailRecord  # noqa: E402
from app.database.repository import list_email_ids_without_embedding  # noqa: E402
from app.database.session import session_scope  # noqa: E402
from app.services.embedding import Embedder, embed_email, get_embedder  # noqa: E402

logger = logging.getLogger(__name__)


def backfill(embedder: Embedder) -> dict[str, int]:
    counts = {"embedded": 0, "skipped": 0, "failed": 0}
    with session_scope() as session:
        email_ids = list_email_ids_without_embedding(session, embedder.model)

    for email_id in email_ids:
        try:
            with session_scope() as session:
                email = session.get_one(EmailRecord, email_id)
                embedded = embed_email(session, email, embedder)
        except Exception as exc:  # noqa: BLE001 - count it and keep going
            logger.error(
                "Embedding failed for email id=%s: %s", email_id, type(exc).__name__
            )
            counts["failed"] += 1
            continue
        counts["embedded" if embedded else "skipped"] += 1
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = backfill(get_embedder())
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
