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
- Retrieval and question-answering limitations are listed under
  [RAG limitations](#rag-limitations).

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

### Retrieval and question answering

See [Running the evals](#running-the-evals) and [Results](#results)
under Retrieval-augmented question answering.
