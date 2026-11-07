"""Human approval endpoints for pending actions.

All routes require `X-API-Key`. The state machine itself (only `pending`
can be approved/rejected, Nylas called only on approve) lives in
`app.services.actions`; this module just maps its outcomes to HTTP.
"""

import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from sqlalchemy.exc import NoResultFound

from app.api.auth import require_api_key
from app.database.models import PendingAction, PendingActionStatus
from app.database.repository import list_audit_log, list_pending_actions
from app.database.session import session_scope
from app.schemas.actions import (
    AuditLogEntryOut,
    AuditLogList,
    PendingActionDetail,
    PendingActionList,
    PendingActionOut,
)
from app.services import actions as actions_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/actions", dependencies=[Depends(require_api_key)])


def _detail(action: PendingAction) -> PendingActionDetail:
    latest = max(
        action.email.classifications, key=lambda c: c.created_at, default=None
    )
    return PendingActionDetail(
        **PendingActionOut.model_validate(action).model_dump(),
        message_id=action.email.nylas_message_id,
        category=latest.category if latest else None,
        confidence=latest.confidence if latest else None,
    )


@router.get("")
def get_actions(
    status: PendingActionStatus = PendingActionStatus.PENDING,
) -> PendingActionList:
    with session_scope() as session:
        actions = [
            _detail(action) for action in list_pending_actions(session, status)
        ]
    return PendingActionList(actions=actions)


@router.get("/{pending_action_id}/audit")
def get_audit_log(pending_action_id: int) -> AuditLogList:
    """The audit trail of one action, oldest first."""
    with session_scope() as session:
        if session.get(PendingAction, pending_action_id) is None:
            raise HTTPException(
                http_status.HTTP_404_NOT_FOUND, "Pending action not found"
            )
        return AuditLogList(
            entries=[
                AuditLogEntryOut.model_validate(entry)
                for entry in list_audit_log(session, pending_action_id)
            ]
        )


def _decide(
    decide: Callable[[int], PendingAction], pending_action_id: int
) -> PendingActionOut:
    try:
        return PendingActionOut.model_validate(decide(pending_action_id))
    except NoResultFound:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, "Pending action not found")
    except actions_service.PendingActionError as exc:
        raise HTTPException(http_status.HTTP_409_CONFLICT, str(exc))


def _approve_or_502(pending_action_id: int) -> PendingAction:
    try:
        return actions_service.approve(pending_action_id)
    except (NoResultFound, actions_service.PendingActionError):
        raise
    except Exception:
        # The service has already marked the action `failed` and audited
        # it; don't leak provider error details to the client.
        logger.exception(
            "Executing pending action %s failed",
            pending_action_id,
            extra={"pending_action_id": pending_action_id, "outcome": "failed"},
        )
        raise HTTPException(
            http_status.HTTP_502_BAD_GATEWAY,
            "Executing the action failed; it has been marked 'failed'",
        )


@router.post("/{pending_action_id}/approve")
def approve_action(pending_action_id: int) -> PendingActionOut:
    return _decide(_approve_or_502, pending_action_id)


@router.post("/{pending_action_id}/reject")
def reject_action(pending_action_id: int) -> PendingActionOut:
    return _decide(actions_service.reject, pending_action_id)
