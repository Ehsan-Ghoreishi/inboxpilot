"""Langfuse client lifecycle and the tracing helpers the app uses.

The rest of the app only sees `traced`, `annotate`, `flush_tracing` and
`shutdown_tracing`; nothing outside this package touches the Langfuse
SDK except the OpenAI client import (`langfuse.openai`).

The client is built once per process, lazily on the first traced call
(or explicitly by `init_tracing` at app startup), always from `Settings`
and never from whatever `LANGFUSE_*` variables happen to be exported:

- With tracing on and both keys set, a real client is created. Unless
  `content_tracing_allowed`, it carries the `mask_otel_spans` hook that
  redacts email content (see `masking.py`).
- Otherwise a placeholder client with `tracing_enabled=False` is
  registered. It exports nothing and makes no network calls, and it keeps
  the SDK's `get_client()` (used by the OpenAI wrapper) from creating a
  key-less client that logs a warning on every model call.
"""

import functools
import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from langfuse import Langfuse, observe, propagate_attributes

from app.config import Settings, get_settings
from app.integrations.langfuse.masking import mask_otel_spans

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Longest a request waits for pending spans to be sent.
FLUSH_TIMEOUT_SECONDS = 2.0
_PLACEHOLDER_KEY = "tracing-disabled"

_lock = threading.Lock()
_client: Langfuse | None = None
_active = False


def tracing_requested(settings: Settings) -> bool:
    """Tracing is on and fully configured (both keys present)."""
    return bool(
        settings.langfuse_tracing_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key is not None
    )


def content_tracing_allowed(settings: Settings) -> bool:
    """Email text and addresses may be traced only with TRACE_EMAIL_CONTENT
    on, and only in PROCESSING_MODE=test (where only the designated test
    mailbox's mail is processed)."""
    return settings.trace_email_content and settings.processing_mode == "test"


def _build_client(settings: Settings | None) -> tuple[Langfuse, bool]:
    if settings is None or not tracing_requested(settings):
        placeholder = Langfuse(
            public_key=_PLACEHOLDER_KEY,
            secret_key=_PLACEHOLDER_KEY,
            tracing_enabled=False,
        )
        return placeholder, False
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        mask_otel_spans=None if content_tracing_allowed(settings) else mask_otel_spans,
    )
    return client, True


def init_tracing(settings: Settings | None = None) -> bool:
    """Create the process-wide client if it doesn't exist yet.

    Returns whether tracing is active. Never raises for configuration
    reasons: if settings can't be loaded, tracing stays off.
    """
    global _client, _active
    with _lock:
        if _client is None:
            if settings is None:
                try:
                    settings = get_settings()
                except RuntimeError:
                    settings = None
            _client, _active = _build_client(settings)
        return _active


def _active_client() -> Langfuse | None:
    return _client if _active else None


def traced(
    name: str, *, root: bool = False, tags: Sequence[str] | None = None
) -> Callable[[F], F]:
    """Record each call of the decorated function as a Langfuse observation.

    Arguments and return value are never captured: what a step knows is
    attached explicitly with `annotate`. `root=True` also names the trace
    and applies `tags` to it.
    With tracing off the function is called directly.
    """

    def decorator(func: F) -> F:
        observed = observe(name=name, capture_input=False, capture_output=False)(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not init_tracing():
                return func(*args, **kwargs)
            if root:
                with propagate_attributes(trace_name=name, tags=list(tags) if tags else None):
                    return observed(*args, **kwargs)
            return observed(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def annotate(**metadata: Any) -> None:
    """Add metadata to the current step. `None` values are skipped."""
    client = _active_client()
    values = {key: value for key, value in metadata.items() if value is not None}
    if client is not None and values:
        client.update_current_span(metadata=values)


def score_trace(name: str, value: bool) -> None:
    """Record a BOOLEAN score on the current trace, so it can be filtered
    on in Langfuse."""
    client = _active_client()
    if client is not None:
        client.score_current_trace(name=name, value=float(value), data_type="BOOLEAN")


def run_flags(settings: Settings | None = None) -> dict[str, Any]:
    """The flags that decide what the app may do, for step metadata.

    Without `settings`, reads the process settings; empty if they can't
    be loaded (tracing never fails a step).
    """
    if settings is None:
        try:
            settings = get_settings()
        except RuntimeError:
            return {}
    return {"processing_mode": settings.processing_mode, "dry_run": settings.dry_run}


def _flush(client: Langfuse) -> None:
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001 - tracing must never break a request
        logger.warning("Langfuse flush failed: %s", type(exc).__name__)


def flush_tracing(timeout_seconds: float = FLUSH_TIMEOUT_SECONDS) -> None:
    """Send pending spans, waiting at most `timeout_seconds`.

    For the end of a request on hosts that throttle CPU once the response
    is sent (Cloud Run): spans still queued then would sit until the next
    request. A slow or unreachable Langfuse costs at most the timeout.
    """
    client = _active_client()
    if client is None:
        return
    worker = threading.Thread(
        target=_flush, args=(client,), name="langfuse-flush", daemon=True
    )
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        logger.warning("Langfuse flush not finished after %.1fs", timeout_seconds)


def shutdown_tracing() -> None:
    """Flush everything and stop the client's threads (end of a process)."""
    global _client, _active
    with _lock:
        client, _client, _active = _client, None, False
    if client is not None:
        client.shutdown()
