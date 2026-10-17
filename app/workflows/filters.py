"""Processing-mode filter.

This is a **testing guard, not an authentication boundary**. It exists
so a developer can point a live Nylas grant at the app during
development/testing and have it only act on a single designated test
mailbox's emails, instead of accidentally processing real inbox
traffic. It does **not** verify anything cryptographically — `sender`
comes straight from the (already-canonical) `Email`, and the check is a
plain, case-insensitive string comparison against `TEST_EMAIL`. Anyone
who can get a message delivered with a matching `From` address
satisfies it. Never rely on this to restrict who can trigger
consequential actions in live mode — that's a separate concern (see
CLAUDE.md's human-approval rule).
"""

from app.config import Settings
from app.schemas.email import Email


def should_process(email: Email, settings: Settings) -> bool:
    """Decide whether `email` should be processed under the current mode.

    - `settings.processing_mode == "test"`: only process emails whose
      sender address case-insensitively matches `settings.test_email`.
      If `test_email` isn't set, nothing is processed (fail closed).
    - Any other mode (e.g. `"live"`): process everything.
    """
    if settings.processing_mode == "test":
        if not settings.test_email:
            return False
        return email.sender.email.lower() == settings.test_email.lower()
    return True
