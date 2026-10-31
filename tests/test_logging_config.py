"""Single-line JSON logging, without tracebacks or arbitrary extras."""

import json
import logging

import pytest

from app.logging_config import JsonFormatter, configure_logging


def _record(msg: str = "hello", exc_info=None, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        "app.test", logging.INFO, __file__, 1, msg, None, exc_info
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formats_one_json_line_with_structured_fields():
    line = JsonFormatter().format(
        _record("Triage finished", event_id="evt-1", outcome="action_proposed")
    )

    assert "\n" not in line
    entry = json.loads(line)
    assert entry["severity"] == "INFO"
    assert entry["logger"] == "app.test"
    assert entry["message"] == "Triage finished"
    assert entry["event_id"] == "evt-1"
    assert entry["outcome"] == "action_proposed"
    assert "timestamp" in entry


def test_unlisted_extras_are_not_emitted():
    entry = json.loads(JsonFormatter().format(_record(body="Synthetic email body")))

    assert "body" not in entry
    assert "Synthetic email body" not in json.dumps(entry)


def test_exceptions_are_reduced_to_their_type():
    try:
        raise ValueError("Synthetic email body inside an exception")
    except ValueError:
        import sys

        line = JsonFormatter().format(_record("failed", exc_info=sys.exc_info()))

    entry = json.loads(line)
    assert entry["exc_type"] == "ValueError"
    assert "Synthetic email body" not in line
    assert "Traceback" not in line


def test_multiline_message_stays_on_one_line():
    line = JsonFormatter().format(_record("first\nsecond"))

    assert "\n" not in line
    assert json.loads(line)["message"] == "first\nsecond"


@pytest.fixture
def clean_root(monkeypatch):
    """Call inside the test body: pytest attaches its capture handlers to
    the root logger after fixtures run. Everything is restored afterwards."""

    def clear() -> logging.Logger:
        root = logging.getLogger()
        monkeypatch.setattr(root, "handlers", [])
        monkeypatch.setattr(root, "level", logging.WARNING)
        uvicorn = logging.getLogger("uvicorn.access")
        monkeypatch.setattr(uvicorn, "handlers", [logging.NullHandler()])
        monkeypatch.setattr(uvicorn, "propagate", False)
        return root

    return clear


def test_configure_logging_installs_one_json_stdout_handler(clean_root, capsys):
    root = clean_root()
    configure_logging()
    configure_logging()

    assert len(root.handlers) == 1
    logging.getLogger("app.test").info(
        "ready", extra={"event_id": "evt-1", "outcome": "ok"}
    )
    out = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line)["event_id"] for line in out] == ["evt-1"]


def test_configure_logging_routes_uvicorn_through_root(clean_root):
    clean_root()
    configure_logging()

    uvicorn = logging.getLogger("uvicorn.access")
    assert uvicorn.handlers == []
    assert uvicorn.propagate is True


def test_configure_logging_leaves_an_existing_setup_alone(clean_root):
    root = clean_root()
    existing = logging.StreamHandler()
    root.handlers.append(existing)

    configure_logging()

    assert root.handlers == [existing]
