"""Service layer: persist an email's LLM classification."""

from app.database.models import Classification
from app.database.repository import get_or_create_email, save_classification
from app.database.session import session_scope
from app.schemas.email import Email, EmailAnalysis


def record_classification(
    email: Email, analysis: EmailAnalysis, *, model: str, prompt_version: str
) -> Classification:
    with session_scope() as session:
        email_record = get_or_create_email(session, email)
        return save_classification(
            session,
            email_record.id,
            analysis,
            model=model,
            prompt_version=prompt_version,
        )
