"""Rendering untrusted email text into prompts.

Email content is third-party data. These helpers keep it inside its
`<untrusted_email>` block: it can never contain the delimiter tag itself,
header fields stay on one line, and body lines are `> `-quoted.
"""

import re

# Any spelling of the delimiter tag (case, whitespace, attributes) that
# an email could use to close the untrusted block early, or open a fake one.
_DELIMITER_TAG = re.compile(r"<\s*/?\s*untrusted_email\b[^>]*>", re.IGNORECASE)


def neutralize_delimiters(text: str) -> str:
    return _DELIMITER_TAG.sub("[removed delimiter tag]", text)


def single_line(text: str) -> str:
    """Collapse all whitespace, so a header field can't start new
    prompt lines (e.g. a forged `Body:` or `SYSTEM:` line)."""
    return " ".join(neutralize_delimiters(text).split())


def quote(text: str) -> str:
    """Prefix every line with `> `, so no line of the body can pass for
    a line of the prompt's own structure (`Subject:`, `SYSTEM:`, ...)."""
    return "\n".join(f"> {line}" for line in neutralize_delimiters(text).splitlines())
