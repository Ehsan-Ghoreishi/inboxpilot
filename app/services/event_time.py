"""Deterministic check of a model-extracted meeting time.

The model reports a start time with an offset plus the timezone the
email named. Nothing here trusts that offset: it is recomputed from the
named zone's rules on that date (so "10:00 Europe/Berlin" is +01:00 in
winter and +02:00 in summer), and any gap — no time, no zone, an unknown
zone, an offset that doesn't match the zone, a local time skipped by a
DST change — means the meeting needs clarification and gets no timed
event, rather than a guessed one.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.schemas.email import EmailAnalysis

_FIXED_OFFSET = re.compile(r"^(?:UTC|GMT)?([+-])(\d{2}):(\d{2})$")


@dataclass(frozen=True)
class EventTiming:
    """A start is set only when it was fully and consistently stated."""

    start: datetime | None
    # The IANA name of `start`'s zone, or `None` for a fixed offset.
    timezone_name: str | None
    duration_minutes: int | None
    needs_clarification: bool
    # The day the email points at, timed or not (for an all-day event).
    local_date: date | None


def _zone(name: str | None) -> tzinfo | None:
    if name is None:
        return None
    match = _FIXED_OFFSET.match(name.strip())
    if match:
        sign, hours, minutes = match.groups()
        offset = timedelta(hours=int(hours), minutes=int(minutes))
        return timezone(-offset if sign == "-" else offset)
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError):
        return None


def localize(local: datetime, zone: tzinfo, offset: timedelta) -> datetime | None:
    """`local` (naive) in `zone`, disambiguated by `offset`; `None` if
    `offset` isn't valid for that wall-clock time there (wrong offset, or
    a time that doesn't exist because clocks skipped over it)."""
    for fold in (0, 1):
        candidate = local.replace(tzinfo=zone, fold=fold)
        if candidate.utcoffset() != offset:
            continue
        round_trip = candidate.astimezone(timezone.utc).astimezone(zone)
        if round_trip.replace(tzinfo=None) == local:
            return candidate
    return None


def resolve_timing(analysis: EmailAnalysis) -> EventTiming:
    """Turn the model's extraction into a start time only if it holds up."""
    start_time = analysis.start_time
    local_date = start_time.date() if start_time else analysis.deadline
    zone = _zone(analysis.timezone)
    start = None
    if (
        start_time is not None
        and zone is not None
        and not analysis.needs_clarification
    ):
        start = localize(
            start_time.replace(tzinfo=None), zone, start_time.utcoffset()
        )

    return EventTiming(
        start=start,
        timezone_name=getattr(zone, "key", None) if start else None,
        duration_minutes=analysis.duration_minutes if start else None,
        needs_clarification=start is None
        and (start_time is not None or analysis.needs_clarification),
        local_date=local_date,
    )
