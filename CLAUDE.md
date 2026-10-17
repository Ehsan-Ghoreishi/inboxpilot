# InboxPilot

Event-driven AI email and calendar assistant for job-search email triage.

This file is background context for working in this repo. It does not
describe anything implemented yet — treat it as the intended shape of the
project, not a status report.

## Flow

Nylas webhook -> FastAPI -> canonical Pydantic `Email` -> LLM
classification/extraction -> pending action -> human approval -> Nylas
action -> Postgres audit trail.

## Stack

- Python 3.11, `uv` for dependency/env management
- FastAPI
- Pydantic v2
- OpenAI API (Responses API, structured output)
- SQLAlchemy
- PostgreSQL
- Docker Compose
- pytest

## Target structure

```
app/
  api/            # FastAPI routes, webhook entrypoints
  integrations/   # Nylas/OpenAI clients and other provider-specific code
  schemas/        # canonical Pydantic models
  services/       # business logic
  workflows/      # orchestration of multi-step flows
  agents/         # LLM classification/extraction logic
  database/       # SQLAlchemy models, session, migrations
tests/
eval/
docs/
```

## Rules

- Keep provider-specific (Nylas/OpenAI) logic behind integration
  boundaries; nothing outside `app/integrations/` should know their
  request/response shapes.
- Validate all external events at the boundary before they enter the
  system.
- Use canonical internal schemas (the Pydantic `Email` model, etc.)
  everywhere past the boundary — never pass raw provider payloads inward.
- Never commit secrets or real email content (including in tests,
  fixtures, or examples).
- Require explicit human confirmation before any consequential action
  (sending, deleting, calendar changes) — no automated action skips
  approval.
- Build one small, tested component at a time.
- Commit messages describe architecture/behavior changes, not course or
  week labels.

## Style

- Type hints everywhere.
- Small functions.
- Docstrings only where the behavior isn't obvious from the signature.
