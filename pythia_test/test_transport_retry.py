from __future__ import annotations

from datetime import datetime
from datetime import timezone
from email.utils import format_datetime
import unittest

from pythia.interaction._transport_retry import DEFAULT_MAX_TRANSIENT_RETRIES
from pythia.interaction._transport_retry import MAX_RETRY_AFTER_SECONDS
from pythia.interaction._transport_retry import QUOTA_ERROR_CODES
from pythia.interaction._transport_retry import body_error_codes
from pythia.interaction._transport_retry import quota_error_code
from pythia.interaction._transport_retry import retry_after_seconds
from pythia.interaction._transport_retry import retry_delay_seconds


class TransportRetryPolicyTests(unittest.TestCase):
    def test_default_is_two_retries_with_bounded_exponential_backoff(self):
        self.assertEqual(DEFAULT_MAX_TRANSIENT_RETRIES, 2)
        self.assertEqual(retry_delay_seconds(1), 0.25)
        self.assertEqual(retry_delay_seconds(2), 0.5)
        self.assertEqual(retry_delay_seconds(10), 2.0)
        for value in (True, 0, -1, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    retry_delay_seconds(value)

    def test_retry_after_delta_is_case_insensitive_and_bounded(self):
        self.assertEqual(
            retry_after_seconds({"retry-after": "1.5"}),
            1.5,
        )
        self.assertEqual(
            retry_after_seconds({"Retry-After": "100000"}),
            MAX_RETRY_AFTER_SECONDS,
        )
        for value in ("", "-1", "nan", "not a date", "1\r\nInjected: x"):
            with self.subTest(value=value):
                self.assertIsNone(
                    retry_after_seconds({"Retry-After": value})
                )

    def test_retry_after_http_date_uses_supplied_clock(self):
        now = 1_800_000_000.0
        future = format_datetime(
            datetime.fromtimestamp(now + 12, timezone.utc),
            usegmt=True,
        )
        past = format_datetime(
            datetime.fromtimestamp(now - 12, timezone.utc),
            usegmt=True,
        )
        self.assertEqual(
            retry_after_seconds(
                {"Retry-After": future},
                now_seconds=now,
            ),
            12.0,
        )
        self.assertEqual(
            retry_after_seconds(
                {"Retry-After": past},
                now_seconds=now,
            ),
            0.0,
        )

    def test_invalid_retry_after_falls_back_to_exponential_delay(self):
        self.assertEqual(
            retry_delay_seconds(2, {"Retry-After": "invalid"}),
            0.5,
        )

    def test_a_rate_limit_error_gets_the_provider_s_whole_retry_after(self):
        # The adapter's own waits keep the cap; a rate limit's error reports
        # the whole delay, which the turn loop decides on.
        self.assertEqual(retry_after_seconds({"Retry-After": "7200"}, cap=None), 7200.0)
        self.assertEqual(retry_after_seconds({"Retry-After": "7200"}), MAX_RETRY_AFTER_SECONDS)
        self.assertIsNone(retry_after_seconds({"Retry-After": "soon"}, cap=None))
        self.assertIsNone(retry_after_seconds({}, cap=None))
        self.assertIsNone(retry_after_seconds(None, cap=None))


class RateLimitClassificationTests(unittest.TestCase):
    def test_body_error_codes_are_the_error_s_code_and_type_identifiers(self):
        self.assertEqual(body_error_codes(
            '{"type": "error", "error": {"type": "rate_limit_error", "message": "Slow down"}}'),
            ("rate_limit_error",))
        self.assertEqual(body_error_codes(
            '{"error": {"code": "rate_limit_exceeded", "type": "requests", "message": "x"}}'),
            ("rate_limit_exceeded", "requests"))
        # Free text is never taken for a code: it may hold a secret.
        for detail in ('{"error": {"code": "has spaces sk-abc"}}', '{"error": "busy"}',
                       '{"error": {"code": 429}}', "<html>Too Many Requests</html>", "", "[1]",
                       '{"error": {"code": "' + "x" * 65 + '"}}'):
            with self.subTest(detail=detail):
                self.assertEqual(body_error_codes(detail), ())

    def test_a_body_may_name_an_exhausted_quota_or_plan(self):
        # Such a 429 is retried like any other; only its failure's wording and
        # code differ.
        self.assertEqual(QUOTA_ERROR_CODES,
                         {"insufficient_quota", "usage_limit_reached", "usage_not_included"})
        # OpenAI's out-of-credit answer in error.code; Codex's plan limits in error.type.
        for body, code in (
            ('{"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}',
             "insufficient_quota"),
            ('{"error": {"type": "usage_limit_reached", "resets_in_seconds": 9000}}',
             "usage_limit_reached"),
            ('{"error": {"code": null, "type": "usage_not_included"}}', "usage_not_included"),
            ('{"error": {"code": "rate_limit_exceeded", "type": "insufficient_quota"}}',
             "insufficient_quota"),
        ):
            with self.subTest(body=body):
                self.assertEqual(quota_error_code(body), code)
        for body in ("", "<html>Too Many Requests</html>", '{"error": {"type": "rate_limit_error"}}',
                     '{"error": {"code": "rate_limit_exceeded", "type": "tokens"}}',
                     '{"error": {"message": "insufficient_quota"}}'):
            with self.subTest(body=body):
                self.assertIsNone(quota_error_code(body))


if __name__ == "__main__":
    unittest.main()
