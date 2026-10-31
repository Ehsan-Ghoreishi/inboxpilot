"""Single-line JSON logs to stdout.

Each record becomes one JSON object per line, which log collectors such
as Cloud Logging parse into structured fields. Callers attach context
with `extra={"event_id": ..., "outcome": ...}`; only the keys in
`STRUCTURED_FIELDS` are emitted, so an arbitrary `extra` can't smuggle
other data into the logs.

Tracebacks are never written: exception messages and frame contents can
carry email content or provider responses. A record logged with
`exc_info` only gets the exception's type name.
"""

import json
import logging
import sys
from datetime import UTC, datetime

STRUCTURED_FIELDS = ("event_id", "outcome", "pending_action_id")

# uvicorn installs its own handlers on these; they're redirected to ours
# so its startup and access lines are JSON too.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            # Cloud Logging reads `severity` as the entry's log level.
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                entry[field] = value
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc_type"] = record.exc_info[0].__name__
        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Route the root logger (and uvicorn's) to one stdout JSON handler.

    Safe to call more than once. If the root logger already has handlers
    that aren't ours (e.g. a test runner capturing logs), it is left
    alone rather than overridden.
    """
    root = logging.getLogger()
    if any(not isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
