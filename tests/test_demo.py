"""`scripts/demo.py` against an in-memory fake of the service: no network."""

import json

import pytest

from app.integrations.nylas.webhooks import compute_signature
from scripts import demo

WEBHOOK_SECRET = "demo-webhook-secret"
API_KEY = "demo-api-key"


@pytest.fixture(autouse=True)
def pinned_env(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("APP_API_KEY", API_KEY)
    monkeypatch.setenv("TEST_EMAIL", "tester@example.com")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("PROCESSING_MODE", "test")


class FakeService:
    """Just enough of the service's HTTP behaviour to drive the demo.

    Knobs let a test break one behaviour; `requests` records everything
    sent, to check what leaves the process.
    """

    def __init__(self, **overrides):
        self.requests: list[tuple[str, str, dict, bytes | None]] = []
        self.message_ids: list[str] = []
        self.thread_ids: list[str] = []
        self.dry_run = True
        self.health_status = 200
        self.proposes_action = True
        self.polls_before_action = 0
        self.ask_answerable = True
        self.ask_cites_demo = True
        self.unanswerable_answers = False
        self.approved = False
        self.__dict__.update(overrides)

    def __call__(self, method, url, headers, body):
        self.requests.append((method, url, headers, body))
        path = url.split("//", 1)[1].split("/", 1)[1]
        path = "/" + path
        if path in ("/health", "/ready"):
            return self.health_status, {"status": "ok"}
        if path == "/events" and method == "POST":
            return self._event(headers, body)
        if headers.get("X-API-Key") != API_KEY:
            return 401, {"detail": "Invalid or missing API key"}
        if path.startswith("/actions?") and method == "GET":
            return self._actions()
        if path == "/actions/7/approve":
            self.approved = True
            return 200, {"id": 7, "status": "executed"}
        if path == "/actions/7/audit":
            return 200, {"entries": self._audit()}
        if path == "/ask":
            return self._ask(json.loads(body))
        return 404, None

    def _event(self, headers, body):
        if headers["X-Nylas-Signature"] != compute_signature(body, WEBHOOK_SECRET):
            return 401, {"detail": "Invalid signature"}
        obj = json.loads(body)["data"]["object"]
        self.message_ids.append(obj["id"])
        self.thread_ids.append(obj["thread_id"])
        return 200, {"status": "ok"}

    def _actions(self):
        if not self.proposes_action or not self.message_ids:
            return 200, {"actions": []}
        if self.polls_before_action > 0:
            self.polls_before_action -= 1
            return 200, {"actions": []}
        return 200, {
            "actions": [
                {
                    "id": 7,
                    "email_id": 42,
                    "action_type": "create_event",
                    "status": "pending",
                    "message_id": self.message_ids[-1],
                    "category": "interview_invite",
                    "confidence": 0.93,
                }
            ]
        }

    def _audit(self):
        return [
            {"actor": "system", "action": "created", "dry_run": False},
            {"actor": "human", "action": "approved", "dry_run": self.dry_run},
            {"actor": "human", "action": "executed", "dry_run": self.dry_run},
        ]

    def _ask(self, payload):
        if payload["question"] == demo.ANSWERABLE_QUESTION:
            if not self.ask_answerable:
                return 200, {"answerable": False, "answer": "No.", "citations": [], "sources": []}
            cited = 42 if self.ask_cites_demo else 99
            return 200, {
                "answerable": True,
                "answer": "Demo Analytics GmbH, Tuesday at 10:00 CET [1].",
                "citations": [cited],
                "sources": [{"email_id": cited, "subject": demo.SUBJECT}],
            }
        if self.unanswerable_answers:
            return 200, {
                "answerable": True,
                "answer": "Your landlord raised it 5%.",
                "citations": [42],
                "sources": [{"email_id": 42, "subject": demo.SUBJECT}],
            }
        return 200, {
            "answerable": False,
            "answer": "Insufficient evidence in stored emails.",
            "citations": [],
            "sources": [],
        }


def run_demo(service, capsys, extra=()):
    code = demo.main(["--url", "https://svc.example.test", *extra], transport=service)
    captured = capsys.readouterr()
    run_demo.err = captured.err
    return code, captured.out


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(demo.time, "sleep", lambda _: None)


def test_happy_path_passes_every_step(capsys):
    service = FakeService()

    code, out = run_demo(service, capsys)

    assert code == 0
    assert out.count("PASS") == 7
    assert "FAIL" not in out
    assert "interview_invite" in out and "0.93" in out and "create_event" in out
    assert "dry_run=true" in out
    assert "DRY_RUN is on: no mailbox or calendar call was made." in out
    assert "[1] " + demo.SUBJECT in out
    assert "Insufficient evidence" in out
    for number in range(1, 8):
        assert f"== {number}/7" in out


def test_each_run_uses_fresh_demo_ids(capsys):
    first, second = FakeService(), FakeService()

    run_demo(first, capsys)
    run_demo(second, capsys)

    ids = first.message_ids + second.message_ids + first.thread_ids + second.thread_ids
    assert all(i.startswith("demo-") for i in ids)
    assert len(set(ids)) == len(ids)


def test_bad_signature_step_sends_an_invalid_hmac(capsys):
    service = FakeService()

    run_demo(service, capsys)

    first_post, second_post = [r for r in service.requests if r[0] == "POST" and r[1].endswith("/events")]
    assert first_post[2]["X-Nylas-Signature"] != compute_signature(first_post[3], WEBHOOK_SECRET)
    assert second_post[2]["X-Nylas-Signature"] == compute_signature(second_post[3], WEBHOOK_SECRET)


def test_secrets_never_appear_in_output(capsys):
    service = FakeService()

    code, out = run_demo(service, capsys)

    err = run_demo.err
    assert code == 0
    for secret in (WEBHOOK_SECRET, API_KEY):
        assert secret not in out and secret not in err


def test_secrets_never_appear_in_output_on_failure(capsys):
    service = FakeService(dry_run=False, health_status=500)

    code, out = run_demo(service, capsys)

    assert code == 1
    for secret in (WEBHOOK_SECRET, API_KEY):
        assert secret not in out


def test_api_key_only_sent_as_header_to_authenticated_routes(capsys):
    service = FakeService()

    run_demo(service, capsys)

    for method, url, headers, body in service.requests:
        assert API_KEY not in url
        assert API_KEY.encode() not in (body or b"")
        if url.endswith(("/health", "/ready", "/events")):
            assert "X-API-Key" not in headers


def test_disabled_health_endpoints_are_skipped_not_failed(capsys):
    service = FakeService(health_status=404)

    code, out = run_demo(service, capsys)

    assert code == 0
    assert "disabled (404), skipped" in out
    assert "SKIP" in out


def test_unhealthy_service_fails_step_one_and_exit_code(capsys):
    service = FakeService(health_status=503)

    code, out = run_demo(service, capsys)

    assert code == 1
    assert "FAIL" in out


def test_non_dry_run_audit_entry_fails_the_approve_step(capsys):
    service = FakeService(dry_run=False)

    code, out = run_demo(service, capsys)

    assert code == 1
    assert "Expected an 'executed' audit entry with dry_run=true." in out
    assert "no mailbox or calendar call was made" not in out


def test_missing_pending_action_fails_dependent_steps(capsys):
    service = FakeService(proposes_action=False)

    code, out = run_demo(service, capsys)

    assert code == 1
    assert "no pending action for demo-" in out
    assert "no pending action from step 4" in out
    assert not service.approved


def test_pending_action_is_polled_until_it_appears(capsys):
    service = FakeService(polls_before_action=3)

    code, _ = run_demo(service, capsys)

    assert code == 0


def test_answer_not_citing_the_demo_email_fails(capsys):
    service = FakeService(ask_cites_demo=False)

    code, _ = run_demo(service, capsys)

    assert code == 1


def test_invented_answer_to_unanswerable_question_fails(capsys):
    service = FakeService(unanswerable_answers=True)

    code, out = run_demo(service, capsys)

    assert code == 1
    assert out.count("FAIL") == 1


def test_unreachable_service_fails_every_step_without_crashing(capsys):
    def down(*_):
        raise OSError("connection refused")

    code, out = run_demo(down, capsys)

    assert code == 1
    assert out.count("FAIL") == 7


def test_pause_waits_for_enter_between_steps(monkeypatch, capsys):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda *a: prompts.append(a))

    code, _ = run_demo(FakeService(), capsys, ["--pause"])

    assert code == 0
    assert len(prompts) == 6


def test_no_pause_never_reads_stdin(monkeypatch, capsys):
    def forbidden(*_):
        raise AssertionError("input() called without --pause")

    monkeypatch.setattr("builtins.input", forbidden)

    code, _ = run_demo(FakeService(), capsys)

    assert code == 0


@pytest.mark.parametrize("missing", ["WEBHOOK_SECRET", "APP_API_KEY"])
def test_missing_secret_exits_2_without_any_request(monkeypatch, capsys, missing):
    monkeypatch.delenv(missing)
    service = FakeService()

    code, _ = run_demo(service, capsys)

    assert code == 2
    assert service.requests == []
    assert missing in run_demo.err


def test_secrets_cannot_be_passed_as_arguments(capsys):
    with pytest.raises(SystemExit):
        demo.main(["--url", "https://x.test", "--api-key", "nope"], transport=FakeService())


def test_non_http_url_exits_2(capsys):
    assert demo.main(["--url", "ftp://x.test"], transport=FakeService()) == 2
