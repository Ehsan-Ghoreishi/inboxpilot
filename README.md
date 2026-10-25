# InboxPilot

An event-driven assistant that triages job-search email: it classifies
each incoming message, extracts the facts that matter (company, role,
deadline, urgency), and proposes a next step — a draft reply, a calendar
event, or archiving — that only happens after a human approves it.
Stored emails are embedded with pgvector, so the mailbox can be searched
by meaning ("which companies asked for a take-home?"), not just keywords.

## Problem

During an active job search, the important emails (an interview slot to
confirm, a take-home due Friday) arrive mixed in with recruiter spam,
job alerts, and automated receipts. Missing one is costly; triaging all
of them by hand is tedious. An LLM can do the sorting, but an LLM that
can *act* on your mailbox is a liability: every email is
attacker-controlled text, and a single prompt injection could send,
delete, or forward mail. InboxPilot is built around that tension — the
model reads and suggests; it never acts.

## Architecture

```mermaid
flowchart LR
    N[Nylas webhook] -->|"HMAC-SHA256 signed"| W["POST /events<br/>verify signature"]
    W --> R[(raw_events)]
    R -->|"background task"| M["mapper → canonical Email"]
    M --> F{processing-mode<br/>filter}
    F --> A["analyzer (OpenAI)<br/>structured output, no tools"]
    A --> C[(classifications)]
    A --> RT["route()<br/>category + confidence"]
    RT -->|"at most one"| P[(pending_actions)]
    P --> UI["GET / dashboard<br/>/actions API (X-API-Key)"]
    UI -->|"human approves"| X["approve() → Nylas<br/>draft / event / archive"]
    X --> L[(audit_log)]
    A -->|"after triage commits"| E["embed (OpenAI)"]
    E --> V[(email_embeddings<br/>pgvector HNSW)]
    Q["POST /search (X-API-Key)"] --> V
```

| Layer | Path | Responsibility |
|---|---|---|
| API | `app/api/` | Webhook (`/events`), approval API (`/actions`), semantic search (`/search`), dashboard (`/`) |
| Integrations | `app/integrations/nylas/` | Signature check, SDK client, payload → `Email` mapper, actions |
| Schemas | `app/schemas/` | Canonical `Email` / `EmailAnalysis`, API responses |
| Agents | `app/agents/` | LLM classification/extraction; versioned prompts |
| Workflows | `app/workflows/` | `process_email()`: filter → classify → route → propose |
| Services | `app/services/` | Pending-action state machine (the only code that calls Nylas actions); embedding and search |
| Database | `app/database/`, `alembic/` | SQLAlchemy models, repository functions, migrations |

**Current state:** the webhook verifies and stores each raw event,
returns 200, and then runs triage (mapper → filter → classify → route →
propose) as a background task. Once triage commits, the email is
embedded in a separate transaction, so an embedding failure never
affects triage; `scripts/embed_emails.py` backfills anything missed.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker. First fill in
`.env` (`cp .env.example .env`); for the local database use
`DATABASE_URL=postgresql+psycopg://inboxpilot:inboxpilot@localhost:5432/inboxpilot`.

```sh
docker compose up -d db               # Postgres
uv run alembic upgrade head           # create tables
uv run uvicorn app.main:app --reload  # API on http://localhost:8000
```

Tests: `uv run pytest` (integration tests in `tests/integration/` use
the Compose database and are skipped if Docker isn't available).

## Offline development

Set `DRY_RUN=true` in `.env` to run the full approval flow without a
live Nylas grant:

- Approving a pending action (draft reply, archive, calendar event)
  never calls Nylas. The action is logged by type and pending-action id
  only — no email content — and given a fake id (`dry-run-<uuid>`).
- The action still moves `pending -> approved -> executed`, and its
  `audit_log` rows for those steps have `dry_run = true`, so dry-run
  activity is distinguishable from real actions.
- Repeat approvals still return 409, and every `/actions` route still
  requires `X-API-Key`.

The analysis pipeline (webhook -> classification -> pending action)
never calls Nylas regardless of this setting: it works from the stored
webhook payload. To feed it without a live grant, send signed synthetic
events with `scripts/send_fake_event.py`.

`DRY_RUN` defaults to `false`; apply migrations
(`uv run alembic upgrade head`) to add the `audit_log.dry_run` column.

## Design decisions

### Approval gate

Nothing consequential happens without a human. The pipeline's output is
at most one row in `pending_actions` with status `pending`;
`app/services/actions.py:approve()` is the only code path that calls
Nylas, and it refuses anything not `pending` (a repeat approve returns
409 from the API). Every transition writes an `audit_log` row, including
failed executions. Even then the blast radius is bounded: a draft is
created but never sent, and archive is a folder move, not a delete. The
`/actions` endpoints require `X-API-Key`, compared in constant time.

### Semantic search

`POST /search` (requires `X-API-Key`) embeds the query with the same
model used for emails and runs a pgvector cosine search (`<=>`) over
`email_embeddings`:

```sh
curl -X POST localhost:8000/search -H "X-API-Key: $APP_API_KEY" \
  -H "content-type: application/json" \
  -d '{"query": "interview next week", "top_k": 5, "category": "interview_invite"}'
```

It returns `[{email_id, score, subject, sender, received_at, snippet}]`,
best first, where `score` is cosine similarity (`1 - distance`).

- **Filters run in SQL**, in the same query as the vector search:
  `category` (the email's *latest* classification), `sender`
  (case-insensitive substring of address or name), and inclusive
  `date_from`/`date_to`. The query enables pgvector's
  `hnsw.iterative_scan`, so a selective filter still returns up to
  `top_k` matches instead of whatever survived the first index pass.
- **Only same-model vectors are compared.** Embeddings are stored per
  (email, model); a search only considers rows from the configured
  `EMBEDDING_MODEL`, so switching models never mixes incomparable
  vectors.
- **Idempotent embedding.** Each embedding stores a hash of its input
  text; an unchanged email is never re-sent to OpenAI.
- **`RAG_MIN_SCORE`** (default `0.0`) drops weak matches. It will also
  gate which emails a future `/ask` endpoint may use as context.

### Canonical schema

The goal is that provider data stops at `app/integrations/`. The mapper
validates a raw Nylas payload (`WebhookEvent` / `EmailObject`) before
turning it into the canonical Pydantic `Email`; the analyzer, routing,
and persistence work only with `Email` and `EmailAnalysis`. (Not fully
there yet: pending-action payloads are still built in Nylas's request
shapes inside `app/services/`.) The LLM's
output is parsed into `EmailAnalysis` via structured output, so a
malformed or out-of-range response (e.g. `confidence` outside 0–1, an
unknown category) fails validation instead of flowing onward.

### Prompt-injection handling

Every email is treated as attacker-controlled. Defences are layered so
that no single one has to hold:

1. **No tools.** The analyzer makes one structured-output call; the
   model has nothing to invoke, and its only output is an
   `EmailAnalysis` whose fields are enums, dates, bounded strings, and a
   number.
2. **Untrusted-data block.** The email is wrapped in
   `<untrusted_email>…</untrusted_email>`, and the system prompt says
   its contents are data, never instructions. Delimiter tags inside the
   email are neutralized so it can't close the block early; the subject
   and sender name are collapsed to one line, and every body line is
   quoted with `> `, so no email text can pass as prompt structure.
3. **Deterministic routing.** The action type is decided by `route()`
   from the category and confidence threshold — never by the model's
   own `suggested_action`, which a fooled model would set to whatever
   the attacker asked for.
4. **Fixed action shapes.** A draft reply only ever addresses the
   original sender (no cc/bcc); there is no send, forward, or delete
   action to be tricked into.
5. **Human approval** (above) for whatever gets through.

`tests/test_injection.py` runs five malicious emails through the
pipeline with a *fully fooled* model and asserts these properties hold:
no tools offered, content stays inside one untrusted block, at most one
pending action of the routed type, and no Nylas call or approval.

## Limitations

- **Confidence is model-reported, not calibrated.** `confidence` is the
  model's own self-assessment; 0.9 does not mean "right 90% of the
  time". It is used only as a relative routing signal (below 0.7 →
  human review). The eval's calibration table exists to measure how far
  off it is.
- **Tunnels are for development only.** Nylas must reach the webhook at
  a public URL. In development that means a tunnel to your machine;
  that tunnel exposes the *whole* app, including the dashboard at `/`,
  which has no authentication. Don't leave one running, and don't
  deploy this as-is.
- **Known open issues** from a security review, not yet fixed:
  concurrent approvals of the same action can both execute (no row
  lock); empty `WEBHOOK_SECRET`/`APP_API_KEY` values are accepted at
  startup; any `PROCESSING_MODE` other than `test` processes all mail;
  the webhook stores raw payloads after checking only the signature
  and `id`, not the full `WebhookEvent` schema; the `Dockerfile`
  doesn't start the API server and has no `.dockerignore`; archive's
  folder semantics are unverified for Gmail.
- Email bodies are passed to the model as HTML, unconverted.

## Evaluation

`eval/dataset.jsonl` holds 50 synthetic, labeled emails — 10 per
category, including 10 German, 10 deliberately ambiguous, and 5
adversarial (prompt-injection) examples; all addresses use the reserved
`.example` domain. Run it with:

```sh
uv run python -m eval.run --prompt-version analyze_v1 \
  --input-usd-per-mtok <price> --output-usd-per-mtok <price>
```

It reports accuracy, per-class precision/recall/F1, a confusion matrix,
a calibration table (confidence bins vs. actual accuracy), per-slice
accuracy, mean tokens, estimated cost, and latency per email, and writes
the full run to `eval/results/<timestamp>.json`.

**Results:** no eval run has been recorded yet. Numbers will be added
here copied verbatim from an `eval/results/*.json` file, together with
the model and prompt version that produced them.
