# InboxPilot

[![CI](https://github.com/Ehsan-Ghoreishi/inboxpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Ehsan-Ghoreishi/inboxpilot/actions/workflows/ci.yml)

An event-driven assistant that triages job-search email: it classifies
each incoming message, extracts the facts that matter (company, role,
deadline, urgency), and proposes a next step — a draft reply, a calendar
event, or archiving — that only happens after a human approves it.
Stored emails are embedded with pgvector, so the mailbox can be searched
by meaning, not just keywords, and questions like "which companies asked
for a take-home?" get an answer that cites the emails it came from.

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
    K["POST /ask (X-API-Key)"] -->|"retrieve"| V
    K -->|"answer from retrieved emails"| O["answerer (OpenAI)<br/>no tools"]
```

| Layer | Path | Responsibility |
|---|---|---|
| API | `app/api/` | Webhook (`/events`), approval API (`/actions`), semantic search (`/search`), question answering (`/ask`), dashboard (`/`) |
| Integrations | `app/integrations/nylas/`, `app/integrations/langfuse/` | Nylas: signature check, SDK client, payload → `Email` mapper, actions. Langfuse: tracing client, step helpers, email redaction |
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
the Postgres at `DATABASE_URL`, starting the Compose `db` service if it
isn't reachable, and are skipped if neither works — except under `CI`,
where they fail instead). Tests never call OpenAI: any real request
fails the test (`tests/conftest.py`).

### CI

`.github/workflows/ci.yml` runs on every pull request and on pushes to
`main`: `uv sync --frozen`, `uv run alembic upgrade head`, then
`uv run pytest -q`, against a `pgvector/pgvector:pg16` service with
dummy credentials. `eval/` is not run in CI (it calls the real OpenAI
API). To run the same steps locally with CI's environment (variables
set here override `.env`; optional ones not set here, like
`RAG_MIN_SCORE`, still come from `.env` if present):

```sh
docker compose up -d db
env -i HOME="$HOME" PATH="$PATH" CI=true \
  DATABASE_URL=postgresql+psycopg://inboxpilot:inboxpilot@localhost:5432/inboxpilot \
  OPENAI_API_KEY=sk-test-dummy OPENAI_MODEL=gpt-test \
  NYLAS_API_KEY=nylas-test NYLAS_CLIENT_ID=client-test NYLAS_GRANT_ID=grant-test \
  WEBHOOK_SECRET=ci-webhook-secret SERVER_URL=http://localhost:8000 \
  APP_API_KEY=ci-app-key PROCESSING_MODE=test DRY_RUN=false \
  sh -c 'uv sync --frozen && uv run alembic upgrade head && uv run pytest -q'
```

Or run the workflow itself in Docker with [act](https://github.com/nektos/act):
`act pull_request` (stop the Compose `db` first — both bind port 5432).

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

## Container image and Cloud Run

The `Dockerfile` builds a runtime-only image (Python 3.11 to match
`requires-python`, `uv sync --frozen --no-dev`, only `app/`, `alembic/`,
`alembic.ini` and `scripts/`), runs as a non-root user and serves on
`$PORT` (default 8080). `.dockerignore` keeps `.env*`, `.git`, tests,
`eval/results` and `requests/` out of the build context. The image sets
`STORE_RAW_EVENT_FILES=false`: its filesystem is ephemeral and not
writable by the app user, and the database holds every raw event.

Settings that matter when deployed:

- `ENVIRONMENT=production` disables `/docs`, `/redoc` and
  `/openapi.json`. Startup logs `ENVIRONMENT`, `PROCESSING_MODE`,
  `DRY_RUN` and `PROCESS_EVENTS_INLINE` (never secret values).
- `PROCESS_EVENTS_INLINE=true` triages each webhook event inside the
  request, within an 8 second budget, instead of as a background task:
  Cloud Run throttles CPU once a response is sent, which would stall
  background work. On timeout or error the response is still 200 and
  the raw event stays unprocessed.
- `python -m scripts.reprocess_raw_events [--limit N]` (default 50)
  triages stored events that have no email yet. It is idempotent, prints
  counts, and exits non-zero if any event failed, so it can run as a
  scheduled job to retry what inline processing left behind.

Probes: `/health` is liveness and never touches the database; `/ready`
runs `SELECT 1` and returns 503 when the database is unreachable. Each
process holds at most `DB_POOL_SIZE + DB_MAX_OVERFLOW` database
connections (default 5 + 0). Each Cloud Run instance and each run of the
reprocess job has its own pool, so keep (max instances + jobs) x that
well below the database's limit; Supabase's session pooler allows 15
client connections in total.

Logs go to stdout as one JSON object per line (`severity`, `message`,
and `event_id`/`outcome` where relevant), never with email content,
tracebacks or secrets.

To send a signed synthetic event to a deployment, use the deployment's
secret:
`WEBHOOK_SECRET=... uv run python scripts/send_fake_event.py --url https://<service-url>`
(`/events` is appended to a bare base URL).

## Deployment

How the image above runs on Google Cloud. Every identifier below
(`PROJECT_ID`, `REGION`, `SERVICE_ACCOUNT`, `SERVICE_URL`) is a
placeholder.

### Architecture

```mermaid
flowchart LR
    N[Nylas] -->|"signed webhook"| S["Cloud Run service<br/>inboxpilot (API)"]
    CS["Cloud Scheduler<br/>every 15 min"] -->|"jobs.run"| J["Cloud Run job<br/>inboxpilot-reprocess"]
    S --> DB[("Supabase Postgres<br/>+ pgvector<br/>session pooler")]
    J --> DB
    SM[Secret Manager] -.->|"secrets as env vars"| S
    SM -.-> J
```

- **Cloud Run service** `inboxpilot`: the FastAPI app (webhook,
  `/actions`, `/search`, `/ask`, dashboard, probes). Scales to zero,
  at most 2 instances.
- **Cloud Run job** `inboxpilot-reprocess`: the same image running
  `python -m scripts.reprocess_raw_events`, which triages stored raw
  events that have no email yet.
- **Cloud Scheduler** runs the job every 15 minutes.
- **Secret Manager** holds the four secrets; Cloud Run injects them as
  environment variables at instance start.
- **Supabase Postgres with pgvector**, reached through the session
  pooler (the only target `scripts/migrate_prod.py` accepts).
- A **dedicated service account** runs both the service and the job. Its
  only grants are `roles/secretmanager.secretAccessor` on the four
  secrets and `roles/run.invoker` on the job (so the scheduler can start
  it).

### Why `PROCESS_EVENTS_INLINE=true`

Locally, triage runs as a background task after `POST /events` returns.
Cloud Run throttles an instance's CPU once the response has been sent,
so that background work would stall or be lost when the instance scales
down. With `PROCESS_EVENTS_INLINE=true` triage runs inside the request,
within an 8 second budget.

Nothing depends on that budget being met: the raw event is written to
`raw_events` *before* triage starts, and a timeout or error still
returns 200. Anything left unprocessed is picked up by the reprocess
job, which acts as a sweeper. It is idempotent (already-processed
messages are skipped), so overlapping with inline processing is safe.

### Configuration

All names are fields of `app/config.py`.

| Variable | Purpose | Kind |
|---|---|---|
| `DATABASE_URL` | Supabase session-pooler URL, `postgresql+psycopg://` scheme | Secret |
| `OPENAI_API_KEY` | Classification, embeddings, `/ask` answers | Secret |
| `WEBHOOK_SECRET` | HMAC key for verifying `X-Nylas-Signature` | Secret |
| `APP_API_KEY` | Value expected in `X-API-Key` on `/actions`, `/search`, `/ask` | Secret |
| `ENVIRONMENT` | `production` disables `/docs`, `/redoc`, `/openapi.json` | Plain |
| `PROCESS_EVENTS_INLINE` | `true`: triage inside the webhook request (see above) | Plain |
| `PROCESSING_MODE` | `test`: only mail from `TEST_EMAIL` is processed | Plain |
| `DRY_RUN` | `true`: approvals never call Nylas | Plain |
| `DB_POOL_SIZE` | Connections kept per process (default 5) | Plain |
| `DB_MAX_OVERFLOW` | Extra connections per process above the pool (default 0) | Plain |
| `RAG_MIN_SCORE` | Minimum cosine similarity for `/search` and `/ask` (default 0.30) | Plain |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project key; tracing needs both keys (optional) | Plain |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret (optional) | Secret |
| `LANGFUSE_BASE_URL` | Langfuse host; default Langfuse Cloud (EU) (optional) | Plain |
| `LANGFUSE_TRACING_ENABLED` | `false` turns tracing off even with keys set (default `true`) | Plain |
| `TRACE_EMAIL_CONTENT` | `true` sends email text to Langfuse, only with `PROCESSING_MODE=test` (default `false`) | Plain |

`app/config.py` also requires `OPENAI_MODEL`, `NYLAS_API_KEY`,
`NYLAS_CLIENT_ID`, `NYLAS_GRANT_ID` and `SERVER_URL`; the app refuses to
start without them. While no mailbox is connected and `DRY_RUN=true`,
Nylas is never called, so the Nylas values can be placeholders. Move
`NYLAS_API_KEY` into Secret Manager before connecting a real grant.
`TEST_EMAIL` is optional, but with `PROCESSING_MODE=test` and no
`TEST_EMAIL`, nothing is processed (the filter fails closed).
`STORE_RAW_EVENT_FILES=false` is already set by the image.

### Deploy

```sh
PROJECT_ID=your-project-id
REGION=your-region
SERVICE_ACCOUNT=inboxpilot-runtime
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com \
  cloudscheduler.googleapis.com
```

1. **Create the secrets.** Each command reads the value from stdin
   (paste it, then Ctrl-D), so it never lands in shell history or a file:

   ```sh
   for name in DATABASE_URL OPENAI_API_KEY WEBHOOK_SECRET APP_API_KEY; do
     gcloud secrets create "$name" --replication-policy=automatic --data-file=-
   done
   ```

2. **Create the service account** and **grant it access to each secret**
   (per secret, not project-wide):

   ```sh
   gcloud iam service-accounts create "$SERVICE_ACCOUNT" \
     --display-name="InboxPilot runtime"
   for name in DATABASE_URL OPENAI_API_KEY WEBHOOK_SECRET APP_API_KEY; do
     gcloud secrets add-iam-policy-binding "$name" \
       --member="serviceAccount:${SA_EMAIL}" \
       --role=roles/secretmanager.secretAccessor
   done
   ```

3. **Run the migrations** from your machine. `scripts/migrate_prod.py`
   reads `DATABASE_URL` only from `.env.prod` (gitignored, excluded from
   the image), refuses anything that isn't the Supabase pooler, and
   masks the password when printing:

   ```sh
   uv run python scripts/migrate_prod.py check     # connection + vector extension
   uv run python scripts/migrate_prod.py upgrade   # alembic upgrade head
   uv run python scripts/migrate_prod.py report    # tables + alembic_version
   ```

4. **Deploy the service** from source (Cloud Build builds the
   `Dockerfile`). It must allow unauthenticated requests so Nylas can
   reach `/events`; the app does its own authentication (see below).

   ```sh
   gcloud run deploy inboxpilot --source . --region "$REGION" \
     --service-account "$SA_EMAIL" \
     --allow-unauthenticated --min-instances 0 --max-instances 2 \
     --set-secrets DATABASE_URL=DATABASE_URL:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,WEBHOOK_SECRET=WEBHOOK_SECRET:latest,APP_API_KEY=APP_API_KEY:latest \
     --set-env-vars ENVIRONMENT=production,PROCESS_EVENTS_INLINE=true,PROCESSING_MODE=test,DRY_RUN=true,DB_POOL_SIZE=5,DB_MAX_OVERFLOW=0,RAG_MIN_SCORE=0.30,OPENAI_MODEL=your-model,NYLAS_API_KEY=unused,NYLAS_CLIENT_ID=unused,NYLAS_GRANT_ID=unused,SERVER_URL=https://SERVICE_URL
   ```

5. **Create the reprocess job** from the image the service just
   deployed, so both always run the same code. It runs events one at a
   time, so a pool of 2 is enough:

   ```sh
   IMAGE=$(gcloud run services describe inboxpilot --region "$REGION" \
     --format='value(spec.template.spec.containers[0].image)')
   gcloud run jobs create inboxpilot-reprocess --image "$IMAGE" --region "$REGION" \
     --service-account "$SA_EMAIL" --max-retries 0 --task-timeout 10m \
     --command python --args=-m,scripts.reprocess_raw_events,--limit,50 \
     --set-secrets DATABASE_URL=DATABASE_URL:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,WEBHOOK_SECRET=WEBHOOK_SECRET:latest,APP_API_KEY=APP_API_KEY:latest \
     --set-env-vars ENVIRONMENT=production,PROCESSING_MODE=test,DRY_RUN=true,DB_POOL_SIZE=2,DB_MAX_OVERFLOW=0,OPENAI_MODEL=your-model,NYLAS_API_KEY=unused,NYLAS_CLIENT_ID=unused,NYLAS_GRANT_ID=unused,SERVER_URL=https://SERVICE_URL
   ```

   After a redeploy of the service, point the job at the new image with
   `gcloud run jobs update inboxpilot-reprocess --image "$IMAGE" --region "$REGION"`.

6. **Schedule the job** every 15 minutes, running as the same service
   account, which may invoke only this job:

   ```sh
   gcloud run jobs add-iam-policy-binding inboxpilot-reprocess --region "$REGION" \
     --member="serviceAccount:${SA_EMAIL}" --role=roles/run.invoker
   gcloud scheduler jobs create http inboxpilot-reprocess-every-15m \
     --location "$REGION" --schedule="*/15 * * * *" --http-method POST \
     --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/inboxpilot-reprocess:run" \
     --oauth-service-account-email="$SA_EMAIL"
   ```

Connection budget with these values: 2 service instances x (5 + 0) plus
one job run x (2 + 0) = 12 connections, under the session pooler's 15.
Raising `--max-instances` or the pool sizes needs that sum rechecked.

### Security model

- **Webhook:** `POST /events` recomputes HMAC-SHA256 of the exact raw
  body with `WEBHOOK_SECRET` and compares it to `X-Nylas-Signature` in
  constant time before parsing anything; a mismatch is a 401 and nothing
  is stored. A blank `WEBHOOK_SECRET` or `APP_API_KEY` fails startup.
- **API key:** `/actions`, `/search` and `/ask` require `X-API-Key`
  equal to `APP_API_KEY`.
- **No API docs in production:** `ENVIRONMENT=production` removes
  `/docs`, `/redoc` and `/openapi.json`.
- **Secrets only from Secret Manager:** `.dockerignore` keeps every
  `.env*` file except `.env.example` out of the build context, so no
  env file is baked into the image. Logs never contain secret values,
  email content or tracebacks.
- **`DRY_RUN=true` never calls the email provider:** approved actions
  get a fake `dry-run-<uuid>` id and audit rows flagged `dry_run`.
  Triage never calls Nylas in either mode.
- **Least privilege:** the runtime service account can read the four
  secrets and start the reprocess job, nothing else.
- **Not covered:** the dashboard at `/` has no authentication and shows
  sender addresses and subjects of processed emails. On a public
  service URL it is reachable by anyone who finds it (see
  [Known limitations](#known-limitations)).

### Operations

Run the sweeper now instead of waiting for the schedule:

```sh
gcloud run jobs execute inboxpilot-reprocess --region "$REGION" --wait
```

Read logs (one JSON object per line, so `severity`, `message`,
`event_id` and `outcome` are queryable fields):

```sh
# service: last hour
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="inboxpilot"' \
  --freshness=1h --limit=50
# webhook events that timed out inline (left for the job)
gcloud logging read \
  'resource.type="cloud_run_revision" AND jsonPayload.outcome="timeout"' --freshness=1d
# reprocess job runs
gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="inboxpilot-reprocess"' \
  --freshness=1d --limit=50
```

Check the probes, and the startup line that reports the safety flags:

```sh
SERVICE_URL=$(gcloud run services describe inboxpilot --region "$REGION" \
  --format='value(status.url)')
curl -fsS "$SERVICE_URL/health"   # {"status":"ok"}, no database access
curl -fsS "$SERVICE_URL/ready"    # {"status":"ready"}, or 503 if the database is unreachable
gcloud logging read \
  'resource.type="cloud_run_revision" AND jsonPayload.message:"Starting with ENVIRONMENT="' \
  --freshness=1d --limit=1
```

### Cost notes

- The service scales to zero (`--min-instances 0`) and is capped at 2
  instances, so idle time costs nothing and a burst of webhooks can't
  scale it up without limit.
- The deployment is sized to stay within the free tiers of Cloud Run,
  Cloud Scheduler, Secret Manager and Supabase. That is a dependency,
  not a guarantee: free-tier terms change, and leaving them (more
  instances, a paid Supabase plan) changes the bill.
- OpenAI usage is billed separately and is the main variable cost; it
  grows with processed email, not with uptime.
- Set a Cloud Billing budget with an alert on the project (for example
  at a few dollars), so unexpected usage is noticed early.

### Known limitations

- **Safe mode by default:** the deploy above sets `PROCESSING_MODE=test`
  and `DRY_RUN=true`. Only mail from `TEST_EMAIL` is triaged, and
  approvals never reach a mailbox.
- **No real mailbox connected:** the Nylas settings are placeholders and
  no webhook is registered against the service, so it only sees signed
  synthetic events from `scripts/send_fake_event.py`.
- **Session pooler connection limit:** Supabase's session pooler allows
  15 client connections in total, shared by every service instance and
  job run. The current settings use up to 12; more instances or larger
  pools can exhaust it, which shows up as `/ready` returning 503 and
  triage failures.
- **Unauthenticated dashboard:** `/` is public on the service URL (see
  the general [Limitations](#limitations)).

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
   action to be tricked into. A calendar event's description is built
   from the email's own subject and sender, never from the model's
   free-text `reasoning`.
5. **Human approval** (above) for whatever gets through.

`tests/test_injection.py` runs five malicious emails through the
pipeline with a *fully fooled* model and asserts these properties hold:
no tools offered, content stays inside one untrusted block, at most one
pending action of the routed type, and no Nylas call or approval.

### Meeting times

The analyzer extracts a meeting's `start_time` (ISO 8601 with offset),
the IANA `timezone` the email states, an optional `duration_minutes`,
and `needs_clarification` when a time of day or timezone is missing.
`app/services/event_time.py` then re-checks the model's offset against
that zone's rules on that date (`Europe/Berlin` is +01:00 in winter,
+02:00 in summer, even if the email says "CET"). A timed event is
proposed only if everything checks out. Otherwise (missing time or zone,
unknown zone, wrong offset, a local time skipped by a DST change) the
proposal is an all-day event titled "[Time TBC] …", never a guessed
time. Without a stated duration the event is a single point in time
rather than an invented length.

## Retrieval-augmented question answering

Stored emails are embedded so the mailbox can be searched by meaning
(`POST /search`) and asked questions whose answers cite the emails they
came from (`POST /ask`). Both are read-only: nothing here proposes or
takes an action, so neither needs the approval step.

```mermaid
flowchart LR
    I["ingest<br/>triage commits the email"] --> E["embed (OpenAI)<br/>subject + sender + date + body,<br/>text-embedding-3-small"]
    E --> V[("email_embeddings<br/>pgvector, HNSW index<br/>one row per email + model")]
    V --> RE["retrieve_emails()<br/>exact cosine search + SQL filters,<br/>content-hash dedupe,<br/>score ≥ RAG_MIN_SCORE"]
    S["POST /search"] --> RE
    RE --> SR["ranked results"]
    K["POST /ask"] --> RE
    RE --> T{"anything left?"}
    T -->|"no (threshold gate)"| R1["answerable: false<br/>model never called"]
    T -->|yes| M["answerer (OpenAI)<br/>numbered sources [1]..[k],<br/>structured output, no tools"]
    M --> G{"model says<br/>answerable?"}
    G -->|"no (model gate)"| R2["answerable: false"]
    G -->|yes| C{"server-side check:<br/>citations in 1..k,<br/>at least one"}
    C -->|"invalid, after one retry"| R3["answerable: false"]
    C -->|valid| OK["answer + citations + sources"]
```

`/search` and `/ask` share the same retrieval step
(`app/services/search.py: retrieve_emails`), so a threshold or filter
change affects both identically, and the evals below run that same
function.

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
  `date_from`/`date_to`.
- **Exact search, not ANN.** Emails whose embedded text is identical
  (same content hash) collapse to the best-scoring one, so duplicates
  don't use up `top_k` slots. That collapse is a window function, which
  rules out an HNSW index-ordered scan, so every candidate is scored
  exactly. The HNSW index exists but this query doesn't use it; that is
  fine at a single inbox's size, not at large scale.
- **Only same-model vectors are compared.** Embeddings are stored per
  (email, model); a search only considers rows from the configured
  `EMBEDDING_MODEL`, so switching models never mixes incomparable
  vectors.
- **Idempotent embedding.** Each embedding stores a hash of its input
  text; an unchanged email is never re-sent to OpenAI.
- **`RAG_MIN_SCORE`** (default `0.30`) drops weak matches, here and in
  `/ask`. It is specific to `text-embedding-3-small`; see
  [Choosing the threshold](#choosing-the-threshold).

### Asking questions

`POST /ask` (requires `X-API-Key`) answers a question from your stored
emails and says which ones it used:

```sh
curl -X POST localhost:8000/ask -H "X-API-Key: $APP_API_KEY" \
  -H "content-type: application/json" \
  -d '{"question": "Which companies asked for a take-home?"}'
```

It returns `{answerable, answer, citations, sources}`: `citations` are
the ids of the emails the answer relies on, and `sources` are those
emails in the same shape as `/search` results. It takes the same `top_k`
and filters as `/search`.

- **Retrieval is `/search`.** The question is embedded and the closest
  emails passing `RAG_MIN_SCORE` and the filters become the context,
  each body truncated to 3000 characters.
- **Two gates against answering without evidence.** If nothing passes
  `RAG_MIN_SCORE`, the response is a fixed refusal and the model is
  never called. Against the eval corpus:

  ```sh
  curl -X POST localhost:8000/ask -H "X-API-Key: $APP_API_KEY" \
    -H "content-type: application/json" \
    -d '{"question": "What did my landlord say about the rent increase?"}'
  # {"answerable":false,"answer":"Insufficient evidence in stored emails.","citations":[],"sources":[]}
  ```

  The threshold is only a cheap first filter: a question that is
  off-topic but *sounds* like job-search mail (e.g. "what time does my
  flight to the company offsite depart?", top-1 score 0.41) passes it,
  and the model then has to say `answerable: false` itself.
- **Emails are untrusted.** Each is numbered `[1]..[k]` and rendered in
  the same delimited, `> `-quoted `<untrusted_email>` block as triage
  uses, and the versioned prompt (`app/agents/prompts/ask_v1.md`) tells
  the model to answer only from them and never follow instructions
  inside them.
- **Citations are checked server-side.** The model returns structured
  output (`answerable`, `answer`, `citations` as numbers). Every
  citation must be in `1..k`, and an answerable result must cite at
  least one email; otherwise the call is retried once, then the
  response is `answerable: false`. Valid numbers are mapped back to
  email ids.
- **Read-only.** The answerer has no tools.

### Running the evals

Both evals need a configured `.env` and the Compose database. They load
the 50-email corpus (`eval/dataset.jsonl`) into a separate `<db>_eval`
database (created, migrated, and emptied on each run — never the dev
database) and embed it with the real configured model.

```sh
# retrieval only: recall@k, MRR, top-1 score ranges and a threshold sweep
uv run python -m eval.run_retrieval [--min-score X]

# end to end through ask_question(): real embedder and real model
uv run python -m eval.run_ask [--min-score X] \
  --input-usd-per-mtok <price> --output-usd-per-mtok <price>
```

The questions are in `eval/retrieval_dataset.jsonl`: 25 over the same
50 emails, 20 answerable, each labeled with the email ids that answer it
(4 have two right answers, several are English questions about German
emails), and 5 with no answer in the corpus. The questions and labels
were drafted with an AI assistant and have not yet been independently
reviewed.

`eval.run_retrieval` retrieves for every question without a threshold,
so every score is visible, and writes
`eval/results/retrieval-<timestamp>.json`. `eval.run_ask` scores
answerable questions on how often the model answers and on its
citations — *valid* (every cited email is an expected one), *precision*
(fraction of citations that are expected) and *recall* (fraction of
expected emails that were cited) — and unanswerable ones on refusals,
split into refused by the threshold (no model call) and by the model.
It writes `eval/results/ask-<timestamp>.json`. Cost counts model tokens
only, not the question's embedding.

### Results

Retrieval: `eval/results/retrieval-20261025T101312Z.json`. Ask:
`eval/results/ask-20261025T102347Z.json` (`gpt-4o`, prompt `ask_v1`,
`RAG_MIN_SCORE` 0.30, prices passed as $2.50 / $10.00 per 1M
input/output tokens). Both use `text-embedding-3-small` over the
50-email corpus.

| Metric | Result |
|---|---|
| Retrieval recall@3 (n=20) | 0.925 |
| Retrieval recall@5 (n=20) | 0.950 |
| Retrieval MRR, depth 20 (n=20) | 0.963 |
| Ask: answerable questions answered | 20/20 |
| Ask: answers with valid citations | 20/20 |
| Ask: mean citation precision | 1.000 |
| Ask: mean citation recall | 0.950 |
| Ask: unanswerable questions refused | 5/5 (1 by threshold, 4 by model) |
| Ask: mean latency | 1287 ms |
| Ask: estimated cost | ~$0.0033 / question |

Citation recall was not in the original run's summary; 0.950 is
computed from the cited and expected ids recorded in that same file,
the same way `eval.run_ask` now reports it.

19 of 20 questions ranked a correct email first. The imperfect ones are
both multi-answer questions, and they are also the two whose answers
cite only one of their two expected emails (citation recall 0.5 each):

- **q-12** ("Which German companies invited me to an interview?"):
  retrieval found only one of its two emails in the top 5, and the
  answer cites `int-06-de` but not `int-07-de`.
- **q-13** ("Which coding exercises are due on Friday, March 6?"):
  retrieval ranked a challenge due March 7 first, with the first
  correct email at rank 4; the answer cites `asm-06-de` but not
  `asm-01`.

Latency is the wall time of `ask_question()` (embedding, retrieval and
model call); the one threshold refusal took 145 ms because no model
call was made.

#### Choosing the threshold

`/ask` refuses without a model call exactly when the top-1 email is
below the threshold, so the retrieval eval sweeps each question's top-1
score:

| min_score | answerable kept | unanswerable refused by threshold |
|---|---|---|
| 0.00 | 20/20 | 0/5 |
| 0.25 | 20/20 | 1/5 |
| **0.30 (default)** | **20/20** | **1/5** |
| 0.35 | 20/20 | 4/5 |
| 0.40 | 19/20 | 4/5 |
| 0.45 | 19/20 | 5/5 |
| 0.50 | 18/20 | 5/5 |
| 0.55 | 15/20 | 5/5 |

The answerable and unanswerable top-1 scores overlap: the lowest
answerable question scores 0.353 (q-18, "Who offered to put in a
referral for me?", whose email *is* ranked first) and the highest
unanswerable one 0.410 (u-05, the offsite flight). No threshold
separates them. The eval's own suggestion, 0.446, refuses all five
unanswerable questions by threshold alone but also refuses q-18. The
default is 0.30 instead, below every answerable question, so the
threshold never costs a correct answer and the model's `answerable`
flag does the rest; the ask eval above measures that combination.

### RAG limitations

- **Synthetic corpus, semi-automated labels.** 50 short, clean,
  single-topic synthetic emails are much easier than a real inbox (long
  threads, quoted replies, HTML, near-duplicates), so expect real
  recall to be lower. The questions and expected-email labels were
  drafted with an AI assistant and not independently reviewed.
- **Small n.** 20 answerable and 5 unanswerable questions. 5/5 refusals
  says little about the real false-answer rate, and one question moves
  any rate by 5 or 20 points.
- **The threshold is sensitive to the corpus.** 0.30 was chosen from
  the sweep above on this corpus; the hard overlap (0.353–0.410) means
  any value trades refusals against kept answers, and a different inbox
  or `EMBEDDING_MODEL` will shift the scores. Re-run both evals after
  changing either.
- **Exact search, not ANN.** Every candidate is scored exactly (see
  above), so search cost grows with the number of stored emails.
- **Citation recall is below 1.0 on two multi-email questions** (q-12,
  q-13): when two emails answer a question, the answer can be built
  from one of them.
- **Single-embedding retrieval.** Each email is one vector over its
  first 8000 characters; there is no chunking, hybrid keyword search,
  or reranking.
- **No real-mailbox testing.** All numbers come from the synthetic
  corpus loaded directly into the database; neither eval has been run
  against a real Nylas mailbox.

## Observability

InboxPilot sends traces to [Langfuse](https://langfuse.com) so each
email and each question can be inspected step by step: what decided,
what it cost, how long it took. It is optional: without keys, or with
`LANGFUSE_TRACING_ENABLED=false`, the app starts and behaves exactly as
before, and no Langfuse client is created that could export anything.

**Traces**

| Trace | Steps (metadata) |
|---|---|
| `email`, one per processed `message.*` event | `triage` (`decided_by=sender_filter`, `decision`), `classify` (`decided_by=llm`, `category`, `confidence`, `model`, `prompt_version`) with the `classify_llm` generation inside it, `propose_action` (`decided_by=confidence_threshold` or `category_table`, `category`, `confidence`, `status`, `proposed`, `pending_action_id`), `embed` (`embedded`, `model`). Each step also carries `processing_mode` and `dry_run`; the trace carries `event_id` and `outcome`. |
| `ask`, one per `/ask` call | `retrieve` (`top_k`, `scores`, `threshold`) and the `answer_llm` generation. The trace carries `answerable`, `citations_valid` and, for a refusal, `refusal`. |

Model, token counts and cost come from the OpenAI client, created from
`langfuse.openai`, which records `responses.parse` calls (checked with a
real trace, including cost). Embedding calls appear as generations
inside the `embed` and `retrieve` steps. Events that are not messages
produce no trace. Eval runs produce standalone `classify_llm`, `answer_llm`
and embedding traces.

`answerable` and `citations_valid` are on the `ask` trace rather than on
`retrieve`: a span cannot be changed after it ends, and both are only
known once the model has answered.

**Privacy.** Function arguments and return values are never captured
(`capture_input=False`, `capture_output=False`); what a step knows is
attached explicitly as metadata, and it contains no email text. The one
place email content appears is the model and embedding calls the OpenAI
wrapper records. Email text and addresses are sent to Langfuse only when
`TRACE_EMAIL_CONTENT=true` **and** `PROCESSING_MODE=test`; startup logs
which mode is in effect and warns when the flag is set but ignored. In
every other case the client's `mask_otel_spans` hook
(`app/integrations/langfuse/masking.py`) rewrites each span before export:

- email addresses become `[email]` everywhere;
- an `<untrusted_email>` block in a prompt keeps only its opening tag and
  the length of its body (sender, recipients and subject are dropped);
- an embedding call's input becomes the length of each text;
- a model call's output becomes its length;
- status messages (exception text) become `[redacted]`.

Model name, token usage, cost, latency and step metadata are untouched.
A span that cannot be redacted loses its content attributes instead of
being exported as it is. After changing a prompt's layout, check that
`tests/test_trace_masking.py` still passes: the redaction relies on the
`<untrusted_email>` delimiter lines.

**Runtime.** Langfuse sends spans in the background. The webhook with
`PROCESS_EVENTS_INLINE=true` and `/ask` call `flush()` before returning
(waiting at most 2 seconds, so an unreachable Langfuse cannot hold up a
webhook), because Cloud Run throttles CPU once the response is sent.
`scripts/reprocess_raw_events.py`, `scripts/embed_emails.py` and the eval
scripts call `shutdown()` before exiting. Tests run with tracing off and
cannot build a client that exports (`tests/conftest.py`).

**Setup.** Create a project in Langfuse, then set `LANGFUSE_PUBLIC_KEY`
and `LANGFUSE_SECRET_KEY` (and `LANGFUSE_BASE_URL` for the US region,
`https://us.cloud.langfuse.com`, or a self-hosted instance). On Cloud
Run, keep the secret key in Secret Manager like the other secrets.

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
  any `PROCESSING_MODE` other than `test` processes all mail;
  the webhook stores raw payloads after checking only the signature
  and `id`, not the full `WebhookEvent` schema; archive's folder
  semantics are unverified for Gmail.
- Email bodies are passed to the model as HTML, unconverted.
- Retrieval and question-answering limitations are listed under
  [RAG limitations](#rag-limitations).

## Evaluation

### Classification and time extraction

`eval/dataset.jsonl` holds synthetic, labeled emails, all with
addresses on the reserved `.example` domain. Every item is tagged with a
`language` (`en`, `de`, `mixed`) and a `difficulty` (`easy`, `medium`,
`hard`) for slicing. The original 60 are the `base` suite:

- 50 for classification: 10 per category, including 10 German, 10
  deliberately ambiguous, and 5 adversarial (prompt-injection) examples.
- 10 interview emails (4 German) labeled with the expected start time
  (ISO 8601 with offset), duration, or that the time needs
  clarification: missing timezone, missing time of day, "send us your
  availability", and times on both sides of the 2026-03-29 CET → CEST
  switch and the 2026-03-08 US DST switch.

The `hard` suite adds 44 classification items that are hard on purpose
(each has a `note` giving the reason for its label): German-only and
mixed German/English mails, forwarded recruiter and agency mails,
reschedules versus new interviews, soft rejections ("we will keep your CV
on file"), automated "application received" confirmations (`other`),
job-board newsletters that mention interviews, assessment links with
relative deadlines, spam and phishing "recruiters" (`other`), and
multi-intent mails. Where one mail holds several intents, the actionable
interview or assessment wins over a rejection or an outreach. The
retrieval and ask evals use only the `base` suite as their corpus.

**Dev and test split.** `assign_splits` divides the dataset 50/50,
stratified by category, with a fixed seed (`SPLIT_SEED` in `eval/run.py`);
the split is not stored in the file, and adding or editing items re-draws
it. Only the dev split may be used for prompt tuning, so it is the
default. Run `--split test` for a final number only, and `--split all`
for the whole set.

Run it with:

```sh
uv run python -m eval.run --prompt-version analyze_v2 [--split dev|test|all] \
  --input-usd-per-mtok <price> --output-usd-per-mtok <price>
```

It reports classification accuracy and time-extraction accuracy (the
10 time-labeled items) separately, then per-class precision/recall/F1,
the recall of `interview_invite` and `assessment_deadline` on their own
(a missed one is the costly error), a confusion matrix, a calibration
table, accuracy per language, difficulty and id slice, mean tokens,
estimated cost, and latency per email, and writes the full run to
`eval/results/<timestamp>.json`.

Each email is its own Langfuse trace (`eval_item`, tag `eval`) with a
boolean `correct` score, sent through the same tracing client as the
app: email text stays redacted unless `TRACE_EMAIL_CONTENT=true` and
`PROCESSING_MODE=test`, and tracing is flushed and shut down when the run
ends. Trace metadata holds only the item id, labels, split and tags.

`uv run python -m eval.report` turns the newest `eval/results/<timestamp>.json`
into `docs/eval-report.md` (tables only, every number recomputed from the
results file). Time extraction is
scored on what would actually be scheduled, after the deterministic
timezone check in `app/services/event_time.py`: the start must match in
both instant and UTC offset, and a clarification item passes only if no
timed event would be created.

**Results** (`eval/results/20261025T193701Z.json`, `gpt-4o`,
`analyze_v2`, the 60 `base` items, run before the hard suite and the split
were added):

| Metric | Result |
|---|---|
| Classification accuracy | 0.983 (59/60) |
| Time-extraction accuracy | 0.900 (9/10) |
| Duration accuracy | 0.857 (6/7) |

- The one misclassification, `oth-09-amb` (a friend offering a
  referral), flips between `other` and `recruiter_outreach` from run to
  run under both `analyze_v1` and `analyze_v2` (2 of 3 repeats each).
- The one time miss, `time-05` ("Thursday, March 12 at 2pm ET"): the
  model gave `-05:00` (EST), but US daylight saving had started on
  March 8 (`-04:00`). The offset check rejected it, so the action
  became an all-day "[Time TBC]" event rather than a timed event an
  hour off. Its duration counts as a miss for the same reason.

### Retrieval and question answering

See [Running the evals](#running-the-evals) and [Results](#results)
under Retrieval-augmented question answering.
