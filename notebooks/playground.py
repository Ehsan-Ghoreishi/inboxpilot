# %% [markdown]
# # InboxPilot playground: email_analyzer
#
# Runs `app.agents.email_analyzer.analyze()` on 3 synthetic emails against
# the real OpenAI API. Needs a real `OPENAI_API_KEY` and `OPENAI_MODEL` in
# `.env` — this makes live API calls and is not part of the automated test
# suite (see `tests/test_email_analyzer.py` for the mocked version).
#
# Cells are marked with `# %%` so this file opens as a notebook in VS Code
# / Jupytext without needing a `.ipynb`.

# %%
from datetime import datetime, timezone

from app.agents.email_analyzer import analyze
from app.schemas.email import Email, EmailParticipant

# %% [markdown]
# ## Synthetic emails

# %%
recruiter_outreach = Email(
    id="synthetic-1",
    thread_id="synthetic-thread-1",
    grant_id="synthetic-grant",
    sender=EmailParticipant(name="Priya Shah", email="priya.shah@brightforge.io"),
    recipients=[EmailParticipant(name="Candidate", email="candidate@example.com")],
    subject="Quick question about your background",
    body=(
        "Hi! I came across your profile and we have a Senior Backend "
        "Engineer opening at Brightforge. Would you be open to a quick "
        "chat sometime next week to see if it's a fit?"
    ),
    received_at=datetime(2026, 2, 3, 15, 10, tzinfo=timezone.utc),
)

interview_invite = Email(
    id="synthetic-2",
    thread_id="synthetic-thread-2",
    grant_id="synthetic-grant",
    sender=EmailParticipant(name="Recruiting", email="recruiting@nimbuscloud.com"),
    recipients=[EmailParticipant(name="Candidate", email="candidate@example.com")],
    subject="Interview confirmed: Thursday 2pm PT",
    body=(
        "This confirms your technical interview for the Platform Engineer "
        "role is scheduled for Thursday, February 7th at 2:00 PM PT via "
        "video call. A calendar invite will follow."
    ),
    received_at=datetime(2026, 2, 5, 9, 0, tzinfo=timezone.utc),
)

rejection = Email(
    id="synthetic-3",
    thread_id="synthetic-thread-3",
    grant_id="synthetic-grant",
    sender=EmailParticipant(name="Talent Team", email="talent@vectorworks.example"),
    recipients=[EmailParticipant(name="Candidate", email="candidate@example.com")],
    subject="Update on your application",
    body=(
        "Thank you for your interest in the Data Analyst position. After "
        "careful consideration, we've decided to move forward with other "
        "candidates at this time. We wish you the best in your search."
    ),
    received_at=datetime(2026, 2, 1, 18, 30, tzinfo=timezone.utc),
)

synthetic_emails = [recruiter_outreach, interview_invite, rejection]

# %% [markdown]
# ## Run analyze() on each

# %%
for synthetic_email in synthetic_emails:
    analysis = analyze(synthetic_email)
    print(f"--- {synthetic_email.subject} ---")
    print(analysis.model_dump_json(indent=2))
    print()
