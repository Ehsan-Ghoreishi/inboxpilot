# InboxPilot

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

### Retrieval pipeline

```mermaid
flowchart LR
    I["ingest<br/>triage commits the email"] --> E["embed<br/>subject + sender + date + body,<br/>text-embedding-3-small"]
    E --> V[(email_embeddings<br/>one row per email + model)]
    V --> S["search<br/>cosine top-k + SQL filters,<br/>score ≥ RAG_MIN_SCORE"]
    S --> A{"ask:<br/>anything left?"}
    A -->|no| R["answerable: false<br/>model never called"]
    A -->|yes| M["answerer (OpenAI)<br/>cites [1]..[k], citations checked"]
```

`/search` and `/ask` share the same retrieval step
(`app/services/search.py: retrieve_emails`), so a threshold or filter
change affects both identically. The eval below runs that same function.

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
- **`RAG_MIN_SCORE`** (default `0.45`) drops weak matches, here and in
  `/ask`. The value comes from the retrieval eval, not a guess (see
  [Evaluation](#retrieval)); it is specific to `text-embedding-3-small`.

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
- **No evidence, no model call.** If nothing passes `RAG_MIN_SCORE`,
  the response is a fixed refusal and the model is never called, so it
  has nothing to invent an answer from. Against the eval corpus:

  ```sh
  curl -X POST localhost:8000/ask -H "X-API-Key: $APP_API_KEY" \
    -H "content-type: application/json" \
    -d '{"question": "What did my landlord say about the rent increase?"}'
  # {"answerable":false,"answer":"Insufficient evidence in stored emails.","citations":[],"sources":[]}
  ```

  Only similarity decides this. A question that is off-topic but
  *sounds* like job-search mail (e.g. "what time does my flight to the
  company offsite depart?", top-1 score 0.41) gets close to the
  threshold; if it passes, the model sees the context and is expected
  to answer `answerable: false` itself.
- **Emails are untrusted.** Each is numbered `[1]..[k]` and rendered in
  the same delimited, `> `-quoted `<untrusted_email>` block as triage
  uses, and the versioned prompt (`app/agents/prompts/ask_v1.md`) tells
  the model to answer only from them and never follow instructions
  inside them.
- **Citations are checked.** The model returns structured output
  (`answerable`, `answer`, `citations` as numbers). Every citation must
  be in `1..k`, and an answerable result must cite at least one email;
  otherwise the call is retried once, then the response is
  `answerable: false`. Valid numbers are mapped back to email ids.
- **Read-only.** The answerer has no tools and nothing here proposes or
  takes an action, so `/ask` needs no approval step.

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
  any `PROCESSING_MODE` other than `test` processes all mail;
  the webhook stores raw payloads after checking only the signature
  and `id`, not the full `WebhookEvent` schema; the `Dockerfile`
  doesn't start the API server and has no `.dockerignore`; archive's
  folder semantics are unverified for Gmail.
- Email bodies are passed to the model as HTML, unconverted.
- **Retrieval numbers come from a tiny synthetic corpus.** 50 short,
  clean, single-topic emails are much easier than a real inbox (long
  threads, quoted replies, HTML, near-duplicates), so expect real
  recall to be lower.
- **`RAG_MIN_SCORE` is calibrated on 25 questions.** The answerable and
  unanswerable top-1 scores overlap (0.353–0.787 vs. 0.209–0.410), so
  no threshold is perfect: 0.45 refuses all 5 unanswerable questions
  but also refuses one answerable one. Five unanswerable questions is
  far too few to estimate a false-answer rate; treat 0.45 as a starting
  point, and re-run the eval after changing `EMBEDDING_MODEL` (scores
  aren't comparable across models).
- **Single-embedding retrieval.** Each email is one vector over its
  first 8000 characters; there is no chunking, hybrid keyword search,
  or reranking. Questions with several right answers suffer most (see
  q-12 and q-13 below).

## Evaluation

### Classification

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

### Retrieval

`eval/retrieval_dataset.jsonl` holds 25 questions over the same 50
emails: 20 answerable, each labeled with the email ids that answer it
(4 have two right answers, several are English questions about German
emails), and 5 with no answer in the corpus. The questions and labels
were drafted with an AI assistant and have not yet been independently
reviewed. Run it with:

```sh
uv run python -m eval.run_retrieval [--min-score X]
```

It loads the corpus into a separate `<db>_eval` database (created,
migrated, and emptied on each run — never the dev database), embeds it
with the real configured model, and retrieves for every question
without a threshold, so every score is visible. Results go to
`eval/results/retrieval-<timestamp>.json`.

**Results** (`eval/results/retrieval-20261025T093150Z.json`,
`text-embedding-3-small`, corpus of 50):

| Answerable (n=20) | |
|---|---|
| recall@3 | 0.925 |
| recall@5 | 0.950 |
| MRR (depth 20) | 0.963 |

19 of 20 questions ranked a correct email first. The two imperfect
ones are both multi-answer questions: q-13 ("coding exercises due Friday, March 6")
ranked a challenge due March 7 first, with the first correct email at
rank 4, and q-12 ("German companies that invited me to interview") found
only one of its two emails in the top 5.

To calibrate `RAG_MIN_SCORE`, the eval compares each question's top-1
score, since `/ask` refuses exactly when the top-1 email is below the
threshold:

| min_score | answerable kept | unanswerable refused |
|---|---|---|
| 0.00 (old default) | 20/20 | 0/5 |
| 0.35 | 20/20 | 4/5 |
| 0.40 | 19/20 | 4/5 |
| **0.45 (default)** | **19/20** | **5/5** |
| 0.50 | 18/20 | 5/5 |
| 0.55 | 15/20 | 5/5 |

The eval's suggested threshold is the cut that handles the most
questions correctly, placed in the middle of the widest gap between
scores: 0.446, between the highest unanswerable question (0.410) and
the next answerable one (0.482). The answerable question it gives up is
q-18 ("who offered to put in a referral for me?", 0.353): its email is
ranked first, just with a low score.
