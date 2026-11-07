"""Guided end-to-end demo of a running InboxPilot service, for live presentations.

Usage:
    WEBHOOK_SECRET=... APP_API_KEY=... uv run python -m scripts.demo \
        --url https://<host> [--pause]

Both secrets come only from the environment (never arguments, never
printed). The target should run with DRY_RUN=true: step 5 approves a real
pending action, and only DRY_RUN keeps that from touching a mailbox or
calendar. Uses only synthetic data. Exits non-zero if any step fails.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

from scripts.send_fake_event import build_event, resolve_url, sign

Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]

REQUEST_TIMEOUT_SECONDS = 60
POLL_ATTEMPTS = 10
POLL_INTERVAL_SECONDS = 2.0

SUBJECT = "Interview invitation - Data Scientist"
BODY = (
    "Dear candidate, we would like to invite you to an interview for the Data "
    "Scientist role at Demo Analytics GmbH on Tuesday at 10:00 CET. "
    "Please confirm your availability."
)
ANSWERABLE_QUESTION = "Which company invited me to a Data Scientist interview, and when?"
UNANSWERABLE_QUESTION = "What did my landlord say about the rent increase?"


@dataclass
class Outcome:
    ok: bool
    lines: list[str] = field(default_factory=list)
    skipped: bool = False


@dataclass
class Demo:
    base_url: str
    webhook_secret: str
    api_key: str
    transport: Transport
    sender: str = "test.sender@example.com"
    message_id: str = field(default_factory=lambda: f"demo-{uuid.uuid4().hex}")
    email_id: int | None = None
    action_id: int | None = None

    def call(
        self, method: str, path: str, *, body: bytes | None = None, headers=None
    ) -> tuple[int, Any]:
        return self.transport(method, urljoin(self.base_url, path), headers or {}, body)

    def authed(
        self, method: str, path: str, payload: dict | None = None
    ) -> tuple[int, Any]:
        headers = {"X-API-Key": self.api_key}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode()
        return self.call(method, path, body=body, headers=headers)

    def event_body(self) -> bytes:
        event = build_event(SUBJECT, BODY, self.sender)
        event["id"] = f"demo-evt-{uuid.uuid4().hex}"
        obj = event["data"]["object"]
        obj["id"] = self.message_id
        obj["thread_id"] = f"demo-thread-{uuid.uuid4().hex}"
        return json.dumps(event, separators=(",", ":")).encode()

    # --- steps ---------------------------------------------------------------

    def liveness(self) -> Outcome:
        lines, ok, seen = [], True, 0
        for path in ("/health", "/ready"):
            status, _ = self.call("GET", path)
            if status in (403, 404):
                lines.append(f"{path}: disabled ({status}), skipped")
                continue
            seen += 1
            lines.append(f"{path}: {status}")
            ok = ok and status == 200
        return Outcome(ok, lines, skipped=seen == 0)

    def bad_signature(self) -> Outcome:
        body = self.event_body()
        status, _ = self.call(
            "POST",
            "/events",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Nylas-Signature": sign(body, "not-the-real-secret"),
            },
        )
        return Outcome(status == 401, [f"POST /events -> {status}"])

    def signed_event(self) -> Outcome:
        body = self.event_body()
        status, _ = self.call(
            "POST",
            "/events",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Nylas-Signature": sign(body, self.webhook_secret),
            },
        )
        return Outcome(status == 200, [f"id {self.message_id}", f"POST /events -> {status}"])

    def pending_action(self) -> Outcome:
        for attempt in range(POLL_ATTEMPTS):
            status, data = self.authed("GET", "/actions?status=pending")
            if status != 200:
                return Outcome(False, [f"GET /actions -> {status}"])
            match = next(
                (a for a in data["actions"] if a["message_id"] == self.message_id), None
            )
            if match:
                self.action_id = match["id"]
                self.email_id = match["email_id"]
                confidence = match["confidence"]
                return Outcome(
                    True,
                    [
                        f"category:   {match['category']}",
                        "confidence: "
                        + (f"{confidence:.2f}" if confidence is not None else "n/a"),
                        f"proposed:   {match['action_type']} "
                        f"(action #{match['id']}, {match['status']})",
                    ],
                )
            if attempt < POLL_ATTEMPTS - 1:
                time.sleep(POLL_INTERVAL_SECONDS)
        return Outcome(False, [f"no pending action for {self.message_id}"])

    def approve(self) -> Outcome:
        if self.action_id is None:
            return Outcome(False, ["no pending action from step 4"])
        status, data = self.authed("POST", f"/actions/{self.action_id}/approve")
        if status != 200:
            return Outcome(False, [f"POST approve -> {status}"])
        status, audit = self.authed("GET", f"/actions/{self.action_id}/audit")
        if status != 200:
            return Outcome(False, [f"GET audit -> {status}"])
        entries = audit["entries"]
        executed = next((e for e in entries if e["action"] == "executed"), None)
        ok = data["status"] == "executed" and bool(executed and executed["dry_run"])
        lines = [f"status: {data['status']}"]
        lines += [
            f"audit_log: {e['actor']} {e['action']} dry_run={str(e['dry_run']).lower()}"
            for e in entries
        ]
        lines.append(
            "DRY_RUN is on: no mailbox or calendar call was made."
            if ok
            else "Expected an 'executed' audit entry with dry_run=true."
        )
        return Outcome(ok, lines)

    def ask_answerable(self) -> Outcome:
        status, data = self.authed("POST", "/ask", {"question": ANSWERABLE_QUESTION})
        if status != 200:
            return Outcome(False, [f"POST /ask -> {status}"])
        cited = self.email_id in data["citations"]
        lines = [f"Q: {ANSWERABLE_QUESTION}", f"A: {data['answer']}"]
        for number, source in enumerate(data["sources"], start=1):
            lines.append(f"[{number}] {source['subject']} (email #{source['email_id']})")
        return Outcome(bool(data["answerable"]) and cited, lines)

    def ask_unanswerable(self) -> Outcome:
        status, data = self.authed("POST", "/ask", {"question": UNANSWERABLE_QUESTION})
        if status != 200:
            return Outcome(False, [f"POST /ask -> {status}"])
        refused = not data["answerable"] and not data["citations"]
        return Outcome(
            refused,
            [f"Q: {UNANSWERABLE_QUESTION}", f"A: {data['answer']}", f"sources: {len(data['sources'])}"],
        )

    def steps(self) -> list[tuple[str, str, Callable[[], Outcome]]]:
        return [
            ("Liveness", "health endpoints answer 200 (or are disabled)", self.liveness),
            ("Bad signature", "an invalid HMAC is rejected with 401", self.bad_signature),
            ("Signed interview email", "a valid signed event is accepted with 200", self.signed_event),
            ("Pending action", "the email is classified and an action is proposed", self.pending_action),
            ("Approve", "a human approves; the audit log records a dry run", self.approve),
            ("Ask: answerable", "answer cites the interview email", self.ask_answerable),
            ("Ask: unanswerable", "a refusal, not an invented answer", self.ask_unanswerable),
        ]


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def run(demo: Demo, pause: bool, out=print) -> int:
    failures = 0
    steps = demo.steps()
    for number, (title, expect, step) in enumerate(steps, start=1):
        if pause and number > 1:
            input()
        out(f"\n== {number}/{len(steps)}  {title}")
        out(f"   expect: {expect}")
        try:
            outcome = step()
        except Exception as exc:  # noqa: BLE001 - report any failure, keep going
            outcome = Outcome(False, [f"{type(exc).__name__}: {str(exc)[:120]}"])
        for line in outcome.lines:
            out(f"   {line}")
        verdict = "SKIP" if outcome.skipped else "PASS" if outcome.ok else "FAIL"
        out(f"   {verdict}")
        failures += not outcome.ok
    out(f"\n{len(steps) - failures}/{len(steps)} steps passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None, transport: Transport = urllib_transport) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="Service base URL.")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter between steps.")
    args = parser.parse_args(argv)

    webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
    api_key = os.environ.get("APP_API_KEY", "")
    if not webhook_secret.strip() or not api_key.strip():
        print("WEBHOOK_SECRET and APP_API_KEY must be set in the environment", file=sys.stderr)
        return 2
    try:
        base_url = resolve_url(args.url)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    demo = Demo(
        base_url=base_url,
        webhook_secret=webhook_secret,
        api_key=api_key,
        transport=transport,
        sender=os.environ.get("TEST_EMAIL") or "test.sender@example.com",
    )
    return run(demo, args.pause)


if __name__ == "__main__":
    sys.exit(main())
