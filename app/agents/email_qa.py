"""LLM agent that answers a question from a set of retrieved emails.

Read-only: it returns text and the numbers of the emails it relied on,
and has no tools. Each email is rendered as a numbered, delimited,
quoted `<untrusted_email>` block (see `app.agents.untrusted`), and the
system prompt (`prompts/ask_v1.md`) tells the model never to follow
instructions found inside one.
"""

from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from app.agents.untrusted import quote, single_line
from app.config import get_settings
from app.schemas.ask import AskResult, ContextEmail

PROMPT_VERSION = "ask_v1"
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"
# Per email, so one long thread can't crowd the others out of the context.
MAX_BODY_CHARS = 3000


@lru_cache
def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache
def _get_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def _format_email(number: int, email: ContextEmail) -> str:
    return (
        f'<untrusted_email number="{number}">\n'
        f"From: {single_line(email.sender)}\n"
        f"Subject: {single_line(email.subject)}\n"
        f"Received: {email.received_at.isoformat()}\n"
        "Body (each line quoted with '> '):\n"
        f"{quote(email.body[:MAX_BODY_CHARS])}\n"
        "</untrusted_email>"
    )


def format_prompt_input(question: str, emails: list[ContextEmail]) -> str:
    """Number `emails` [1]..[k] in order, followed by the question."""
    blocks = "\n\n".join(
        _format_email(number, email) for number, email in enumerate(emails, start=1)
    )
    return f"{blocks}\n\n<question>\n{question}\n</question>"


def answer_question(question: str, emails: list[ContextEmail]) -> AskResult:
    """Answer `question` using only `emails`, with `settings.openai_model`.

    Citations are returned as the model gave them; the caller validates
    them against `len(emails)`.
    """
    response = _get_client().responses.parse(
        model=get_settings().openai_model,
        instructions=_load_system_prompt(),
        input=[{"role": "user", "content": format_prompt_input(question, emails)}],
        text_format=AskResult,
    )
    result = response.output_parsed
    if result is None:
        raise ValueError("Model did not return a parsed AskResult.")
    return result
