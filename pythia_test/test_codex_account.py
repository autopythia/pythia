"""Account services tested with local callbacks and fake HTTP/token data only."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import queue
import socket
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit

from pythia.interaction import cli, CodexAuth, load_codex_auth, ToolCall
from pythia.interaction import codex_login, codex_quota, user_tools
from pythia.interaction._account_http import AccountServiceError


def _tokens(account="account-one"):
    payload = base64.urlsafe_b64encode(json.dumps({
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }).encode()).rstrip(b"=").decode()
    return {"access_token": "FAKE_ACCESS_SECRET", "refresh_token": "FAKE_REFRESH_SECRET",
            "id_token": f"header.{payload}.signature"}


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


class LoginServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "auth.json"

    def _callback(self, query, *, state=None, error=None):
        redirect = urlsplit(query["redirect_uri"][0])
        params = {"state": query["state"][0] if state is None else state}
        params.update({"error": error} if error else {"code": "FAKE_CODE_SECRET"})
        connection = HTTPConnection("127.0.0.1", redirect.port, timeout=2)
        try:
            connection.request("GET", redirect.path + "?" + urlencode(params))
            response = connection.getresponse()
            body = response.read()
            self.assertNotIn(b"FAKE_CODE_SECRET", body)
            return response.status
        finally:
            connection.close()

    def _run(self, *, complete=True, denial=False, cancel_exchange=False, tokens=None,
             expected_account=None, fail_save=False, timeout_seconds=2):
        notices = queue.Queue()
        cancel = threading.Event()
        exchange_entered, exchange_release = threading.Event(), threading.Event()
        request_values = []
        responses = []

        def opener(request, *, timeout):
            self.assertEqual(request.full_url, "https://auth.openai.com/oauth/token")
            values = parse_qs(request.data.decode())
            request_values.append(values)
            if cancel_exchange:
                exchange_entered.set()
                if not exchange_release.wait(2):
                    raise AssertionError("exchange was not released")
            result = Response(tokens or _tokens())
            responses.append(result)
            return result

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(codex_login.login, self.path, notify=notices.put,
                                 cancel=cancel, callback_port=0, opener=opener,
                                 timeout_seconds=timeout_seconds, expected_account=expected_account)
            try:
                notice = notices.get(timeout=2)
                query = parse_qs(urlsplit(notice.splitlines()[-1]).query)
                self.assertEqual(query["code_challenge_method"], ["S256"])
                self.assertEqual(query["response_type"], ["code"])
                if complete:
                    self.assertEqual(self._callback(query, state="wrong"), 400)
                    if fail_save:
                        patch = mock.patch.object(codex_login.os, "replace", side_effect=OSError("FAKE_ACCESS_SECRET"))
                    else:
                        patch = mock.patch.object(codex_login.os, "replace", wraps=os.replace)
                    with patch:
                        self.assertEqual(self._callback(query, error="denied-secret" if denial else None), 200)
                        if cancel_exchange:
                            self.assertTrue(exchange_entered.wait(2))
                            cancel.set()
                            exchange_release.set()
                        future.result(timeout=3)
                elif timeout_seconds >= 1:
                    cancel.set()
                    future.result(timeout=3)
                else:
                    future.result(timeout=3)
            finally:
                cancel.set()
                exchange_release.set()
                # The worker closes the listener even on denial/timeout/write errors.
                try:
                    future.result(timeout=3)
                except AccountServiceError:
                    pass
                self.assertTrue(all(response.closed for response in responses))
                if "query" in locals():
                    port = urlsplit(query["redirect_uri"][0]).port
                    with socket.socket() as probe:
                        probe.settimeout(0.2)
                        self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)
        if request_values:
            verifier = request_values[0]["code_verifier"][0]
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            self.assertEqual(query["code_challenge"], [challenge])
            self.assertEqual(request_values[0]["redirect_uri"], query["redirect_uri"])
            self.assertEqual(request_values[0]["code"], ["FAKE_CODE_SECRET"])
            self.assertNotIn(verifier, notice)
        return request_values

    def test_login_pkce_callback_atomic_credentials_and_permissions(self):
        self.path.write_text('{"unrelated":"preserved","tokens":{"access_token":"old"}}')
        requests = self._run(expected_account="account-one")
        self.assertEqual(len(requests), 1)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["unrelated"], "preserved")
        self.assertEqual(saved["tokens"]["refresh_token"], "FAKE_REFRESH_SECRET")
        self.assertEqual(load_codex_auth(auth_file=self.path), CodexAuth("FAKE_ACCESS_SECRET", "account-one"))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.path.parent.glob(".auth.*.tmp")), [])

    def test_denial_cancel_timeout_account_mismatch_and_save_failure_preserve_old_file(self):
        cases = (
            ({"denial": True}, "declined"),
            ({"complete": False}, "cancelled"),
            ({"complete": False, "timeout_seconds": 0.3}, "timed out"),
            ({"cancel_exchange": True}, "cancelled"),
            ({"expected_account": "different-account"}, "mismatch"),
            ({"fail_save": True}, "could not be saved"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                original = '{"tokens":{"access_token":"old"}}'
                self.path.write_text(original)
                with self.assertRaisesRegex(AccountServiceError, message) as error:
                    self._run(**kwargs)
                self.assertNotIn("SECRET", str(error.exception))
                self.assertEqual(self.path.read_text(), original)
                self.assertEqual(list(self.path.parent.glob(".auth.*.tmp")), [])

    def test_port_conflict_and_pre_cancel_do_not_publish_challenge_or_write(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            notify = mock.Mock()
            with self.assertRaisesRegex(AccountServiceError, "listener"):
                codex_login.login(self.path, notify=notify, cancel=threading.Event(),
                                  callback_port=occupied.getsockname()[1])
            notify.assert_not_called()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(AccountServiceError, "cancelled"):
            codex_login.login(self.path, notify=notify, cancel=cancel)
        self.assertFalse(self.path.exists())


class QuotaServiceTests(unittest.TestCase):
    def test_plan_identifiers_are_preserved_without_a_tier_whitelist(self):
        payload = {"rate_limit": {"primary_window": {"used_percent": 25}},
                   "credits": {"has_credits": False, "unlimited": False, "balance": "0"}}
        baseline = codex_quota.format_quota(payload, queried_at="fixed").splitlines()
        cases = [(name, name) for name in (
            "free", "plus", "pro", "team", "business", "enterprise", "edu",
            "prolite", "future-plan_v2", "0", "p" * 64,
        )]
        cases.extend(((" ProLite ", "ProLite"), (" " + "p" * 64 + " ", "p" * 64)))
        for value, expected in cases:
            with self.subTest(value=value):
                lines = codex_quota.format_quota({**payload, "plan_type": value}, queried_at="fixed").splitlines()
                self.assertEqual(lines, [baseline[0], f"plan: {expected}", *baseline[2:]])

    def test_invalid_plan_labels_are_unavailable_without_discarding_quota_fields(self):
        payload = {"rate_limit": {"primary_window": {"used_percent": 25}},
                   "additional_rate_limits": [{"metered_feature": "spark", "rate_limit": {}}]}
        expected = codex_quota.format_quota(payload, queried_at="fixed")
        self.assertIn("\nplan: unavailable\n", expected)
        for value in (
            None, "", "   ", 0, False, [], {}, b"prolite", "p" * 65,
            "pro lite", "pro.lite", "pro/lite", "\tprolite", "prolite\t",
            "prolite\n", "\rprolite", "prolite\x00", "prolite\x1b[2J",
            "prolite\u0085", "\u00a0prolite", "prölite",
        ):
            with self.subTest(value=value):
                self.assertEqual(codex_quota.format_quota(
                    {**payload, "plan_type": value}, queried_at="fixed",
                ), expected)

    def test_bearer_reflected_as_valid_plan_identifier_is_still_redacted(self):
        token = "FAKE_BEARER"
        response = Response({"plan_type": token})
        output = codex_quota.query_quota(CodexAuth(token), opener=mock.Mock(return_value=response))
        self.assertIn("\nplan: [redacted]\n", output)
        self.assertNotIn(token, output)
        self.assertTrue(response.closed)

    def test_normalized_quota_request_and_historical_output(self):
        response = Response({"plan_type": "pro", "rate_limit": {
            "primary_window": {"used_percent": 20, "limit_window_seconds": 18000, "reset_at": 2000000000},
        }, "credits": {"has_credits": True, "unlimited": False, "balance": "12.5"},
            "additional_rate_limits": [{"metered_feature": "spark", "rate_limit": {
                "secondary_window": {"used_percent": 5},
            }}]})

        def opener(request, *, timeout):
            self.assertEqual(request.full_url, "https://chatgpt.com/backend-api/wham/usage")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("Authorization"), "Bearer FAKE_SECRET")
            self.assertEqual(request.get_header("Chatgpt-account-id"), "account")
            self.assertEqual(timeout, 17)
            return response

        text = codex_quota.query_quota(CodexAuth("FAKE_SECRET", "account"), timeout_seconds=17, opener=opener)
        self.assertTrue(response.closed)
        self.assertIn("Quota snapshot at", text)
        self.assertIn("\nplan: pro\n", text)
        self.assertIn("primary: used=20%", text)
        self.assertIn("secondary: unavailable", text)
        self.assertIn("credits.balance: 12.5", text)
        self.assertIn("spark.secondary: used=5%", text)
        self.assertNotIn("FAKE_SECRET", text)
        self.assertIn("unavailable", codex_quota.format_quota({}))

    def test_invalid_fields_and_http_errors_are_safe_and_closed(self):
        for value in (-1, 101, True, float("nan"), "FAKE_SECRET"):
            with self.subTest(value=value), self.assertRaises(AccountServiceError):
                codex_quota.format_quota({"rate_limit": {"primary_window": {"used_percent": value}}})
        for code in (302, 401, 403):
            body = io.BytesIO(b"FAKE_SECRET in provider response")
            error = HTTPError("https://example.invalid/FAKE_SECRET", code, "FAKE_SECRET", {}, body)
            with self.subTest(code=code):
                with self.assertRaises(AccountServiceError) as raised:
                    codex_quota.query_quota(CodexAuth("FAKE_SECRET"), opener=mock.Mock(side_effect=error))
                self.assertNotIn("FAKE_SECRET", str(raised.exception))
                self.assertTrue(body.closed)

    def test_unsupported_routes_unknown_account_and_exception_guard_do_not_expose_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ([], ["--model-api", "codex", "--model", "muse-spark-1.3"],
                            ["--model-api", "codex", "--model", "test", "--api-url", "https://example.org"]):
                args = cli._build_parser().parse_args(options)
                environment = user_tools.create_user_environment(args, notify=mock.Mock(), cancel=threading.Event())
                with mock.patch.object(user_tools, "login") as login, mock.patch.object(user_tools, "query_quota") as quota:
                    for name in ("login", "quota"):
                        self.assertFalse(environment.execute_tool_calls((ToolCall(name, "one", "{}"),)).items[0].success)
                    login.assert_not_called()
                    quota.assert_not_called()
            args = cli._build_parser().parse_args(["--model-api", "codex", "--model", "test", "--codex-home", directory])
            environment = user_tools.create_user_environment(args, notify=mock.Mock(), cancel=threading.Event(), provider_history=True)
            with mock.patch.object(user_tools, "login") as login:
                result = environment.execute_tool_calls((ToolCall("login", "one", "{}"),)).items[0]
                self.assertFalse(result.success)
                login.assert_not_called()
            environment = user_tools.create_user_environment(args, notify=mock.Mock(), cancel=threading.Event())
            with mock.patch.object(user_tools, "login", side_effect=RuntimeError("FAKE_SECRET")):
                result = environment.execute_tool_calls((ToolCall("login", "one", "{}"),)).items[0]
            self.assertNotIn("FAKE_SECRET", result.output)


if __name__ == "__main__":
    unittest.main()
