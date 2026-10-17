"""Manual smoke test: list 5 recent Nylas messages, print sender + subject only.

Requires real NYLAS_API_KEY / NYLAS_GRANT_ID in .env. Run with:
    uv run python scripts/nylas_smoke_test.py
"""

from app.integrations.nylas.messages import list_recent_messages


def main() -> None:
    for message in list_recent_messages(limit=5):
        sender = (message.from_ or [{}])[0].get("email", "<unknown>")
        print(f"{sender} | {message.subject or '(no subject)'}")


if __name__ == "__main__":
    main()
