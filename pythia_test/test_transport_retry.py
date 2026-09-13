from __future__ import annotations

from datetime import datetime
from datetime import timezone
from email.utils import format_datetime
import unittest

from pythia.interaction._transport_retry import DEFAULT_MAX_TRANSIENT_RETRIES
from pythia.interaction._transport_retry import MAX_RETRY_AFTER_SECONDS
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


if __name__ == "__main__":
    unittest.main()
