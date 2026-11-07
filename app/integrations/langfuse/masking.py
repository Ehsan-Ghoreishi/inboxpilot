"""Redaction of span attributes before they are exported to Langfuse.

Used as the client's `mask_otel_spans` hook whenever email content may
not be traced (see `content_tracing_allowed` in `tracing.py`). It runs
on the export thread, and Langfuse drops the whole batch if the hook
raises, so every span is handled in its own `try`: a span that can't be
redacted loses its content attributes instead of being exported as is.

What is redacted, in every string attribute: email addresses become
`[email]`. In addition:

- an `<untrusted_email>` block in a prompt keeps its opening tag and is
  reduced to the length of its body (sender, recipients, subject and
  body text are all dropped);
- the input of an embedding call becomes the length of each text;
- the output of a model call (a restatement of the email) becomes its
  length;
- status messages (which carry exception text) become `[redacted]`.

Token usage, model name, latency and step metadata are untouched.
"""

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

logger = logging.getLogger(__name__)

OBSERVATION_TYPE = "langfuse.observation.type"
INPUT_KEYS = ("langfuse.observation.input", "langfuse.trace.input")
OUTPUT_KEYS = ("langfuse.observation.output", "langfuse.trace.output")
STATUS_MESSAGE_KEY = "langfuse.observation.status_message"
METADATA_PREFIXES = ("langfuse.observation.metadata", "langfuse.trace.metadata")
# Attributes dropped when a span can't be redacted.
_CONTENT_PREFIXES = (*INPUT_KEYS, *OUTPUT_KEYS, STATUS_MESSAGE_KEY, *METADATA_PREFIXES)

REDACTED = "[redacted]"
_EMAIL_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# The agents' delimiter lines (`app/agents/untrusted.py` guarantees email
# text can't forge them), matched by line so quoted body text can't end
# the block early.
_EMAIL_BLOCK = re.compile(
    r"^(?P<open><untrusted_email\b[^>\n]*>)$.*?^</untrusted_email>$",
    re.MULTILINE | re.DOTALL,
)
_BODY_HEADER = re.compile(r"^Body\b[^\n]*:\n", re.MULTILINE)
# What `_replace_block` writes. Real body lines are always `> `-quoted, so
# an email can't produce this line itself.
_REDACTED_BLOCK = re.compile(
    r"<untrusted_email\b[^>\n]*>\n\[redacted: body \d+ chars\]\n</untrusted_email>"
)


def _body_chars(block: str) -> int:
    header = _BODY_HEADER.search(block)
    body = block[header.end() :] if header else block
    body = body.removesuffix("</untrusted_email>")
    lines = [line.removeprefix("> ") for line in body.splitlines()]
    return len("\n".join(lines))


def _replace_block(match: re.Match[str]) -> str:
    if _REDACTED_BLOCK.fullmatch(match[0]):
        return match[0]
    return (
        f"{match['open']}\n"
        f"[redacted: body {_body_chars(match[0])} chars]\n"
        "</untrusted_email>"
    )


def scrub_text(text: str) -> str:
    """Reduce email blocks to their body length, then mask addresses."""
    return _EMAIL_ADDRESS.sub("[email]", _EMAIL_BLOCK.sub(_replace_block, text))


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    if isinstance(value, dict):
        return {scrub_text(str(key)): _scrub_value(item) for key, item in value.items()}
    return value


def _scrub_json(raw: str) -> str:
    # SDK-written values are JSON strings; decode first so the block
    # regex sees real newlines. A raw (non-JSON) string is scrubbed as is.
    try:
        decoded = json.loads(raw)
    except ValueError:
        return scrub_text(raw)
    return json.dumps(_scrub_value(decoded))


def _embedding_input_summary(raw: str) -> str:
    try:
        decoded = json.loads(raw)
    except ValueError:
        decoded = raw
    texts = decoded if isinstance(decoded, list) else [decoded]
    return json.dumps({"chars": [len(t) if isinstance(t, str) else 0 for t in texts]})


def _redact_attribute(key: str, value: str, observation_type: object) -> str:
    if key == STATUS_MESSAGE_KEY:
        return REDACTED
    if key in OUTPUT_KEYS and observation_type == "generation":
        return json.dumps({"redacted": True, "chars": len(value)})
    if key in INPUT_KEYS and observation_type == "embedding":
        return _embedding_input_summary(value)
    if key in INPUT_KEYS or key in OUTPUT_KEYS or key.startswith(METADATA_PREFIXES):
        return _scrub_json(value)
    return scrub_text(value)


def redact_attributes(attributes: Mapping[str, Any]) -> dict[str, str]:
    """The attribute values that must change, with their redacted values."""
    observation_type = attributes.get(OBSERVATION_TYPE)
    updates: dict[str, str] = {}
    for key, value in attributes.items():
        if not isinstance(value, str):
            continue
        redacted = _redact_attribute(key, value, observation_type)
        if redacted != value:
            updates[key] = redacted
    return updates


def _patch_for(attributes: Mapping[str, Any]) -> OtelSpanPatch | None:
    try:
        updates = redact_attributes(attributes)
    except Exception as exc:  # noqa: BLE001 - must not raise; fail closed
        logger.warning(
            "Span redaction failed (%s); dropping the span's content attributes",
            type(exc).__name__,
        )
        return OtelSpanPatch(
            delete_attributes=[
                key for key in attributes if key.startswith(_CONTENT_PREFIXES)
            ]
        )
    return OtelSpanPatch(set_attributes=updates) if updates else None


def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    """Langfuse `mask_otel_spans` hook: redact every span in the batch."""
    patches = {
        identifier: patch
        for identifier, span in params.spans.items()
        if (patch := _patch_for(span.attributes)) is not None
    }
    return MaskOtelSpansResult(span_patches=patches)
