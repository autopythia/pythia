"""Payload-free diagnostics and bounded framing; no model or process launches."""
import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction.claude_relay._diagnostics import Diagnostics, event_identity
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.claude_relay._trace import RelayTrace
from pythia.interaction.model import ModelResponseError


class DiagnosticTests(unittest.TestCase):
    def test_nested_identity_codes_counts_timing_and_freeze(self):
        clock = SimpleNamespace(now=0.0)
        d = Diagnostics(clock=lambda: clock.now)
        for _ in range(100):
            d.record({'type': 'system', 'subtype': 'thinking_tokens'})
        clock.now = 2
        d.received()
        d.record({'type': 'stream_event', 'event': {'type': 'error',
                  'error': {'type': 'overloaded_error', 'message': 'PRIVATE_BODY'}}})
        d.set_phase('message_stream')
        clock.now = 3
        frozen = d.snapshot(freeze=True)
        self.assertEqual(frozen.event_count, 101)
        self.assertEqual(len(frozen.event_types), 64)
        self.assertEqual(frozen.error_code, 'overloaded_error')
        self.assertEqual(frozen.last_event_type, 'stream_event/error')
        self.assertEqual(frozen.last_byte_age, 1)
        self.assertEqual(frozen.elapsed_seconds, 3)
        self.assertNotIn('PRIVATE_BODY', json.dumps(asdict(frozen)))
        d.begin()
        d.set_phase('stopping')
        d.record({'type': 'result'})
        self.assertEqual(d.snapshot(), frozen)

    def test_scope_counts_reset_without_resetting_invocation_total(self):
        d = Diagnostics()
        d.record({'type': 'system', 'subtype': 'init'})
        d.begin()
        d.record({'type': 'stream_event', 'event': {'type': 'ping'}})
        self.assertEqual(d.snapshot().event_count, 1)
        self.assertEqual(d.snapshot().invocation_event_count, 2)
        self.assertEqual(d.snapshot().event_types, ('stream_event/ping',))

    def test_missing_invalid_types_never_include_repr_or_payload(self):
        for value in ('TOKEN\nsecret', 'x' * 65, {'secret': 'value'}, False):
            label, code = event_identity({'type': 'stream_event', 'event': {'type': value}})
            self.assertEqual(label, 'stream_event/<invalid>')
            self.assertIsNone(code)
        self.assertEqual(event_identity({'type': 'stream_event', 'event': {}})[0], 'stream_event/<missing>')
        self.assertIsNone(event_identity({'type': 'error', 'error': 'PRIVATE_BODY'})[1])
        self.assertEqual(event_identity({'type': 'stream_event', 'event': {'type': 'content_block_delta',
                         'delta': {'type': 'thinking_delta', 'thinking': 'SECRET'}}})[0],
                         'stream_event/content_block_delta/thinking_delta')

    def test_retry_error_is_observational_not_a_terminal_error_code(self):
        for error in ('authentication_failed', {'type': 'overloaded_error', 'message': 'PRIVATE_BODY'}):
            d = Diagnostics()
            record = {'type': 'system', 'subtype': 'api_retry', 'error': error}
            self.assertEqual(event_identity(record), ('system/api_retry', None))
            d.record(record)
            self.assertIsNone(d.snapshot().error_code)
            self.assertNotIn('PRIVATE_BODY', json.dumps(asdict(d.snapshot())))
            d.record({'type': 'result', 'error': 'authentication_failed'})
            self.assertEqual(d.snapshot().error_code, 'authentication_failed')


class FramingTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        rt = Runtime.__new__(Runtime)
        rt.proc = SimpleNamespace(stdout=asyncio.StreamReader())
        rt.diagnostics = Diagnostics()
        rt.trace = RelayTrace(None, 'test')
        return rt

    async def test_split_lines_and_multiple_records_preserve_exact_bytes(self):
        rt = self.runtime()
        async def collect():
            return [line async for line in rt._records()]
        task = asyncio.create_task(collect())
        for chunk in (b'abc', b'\n', b'xy\nz', b'\n'):
            rt.proc.stdout.feed_data(chunk)
            await asyncio.sleep(0)
        rt.proc.stdout.feed_eof()
        self.assertEqual(await task, [b'abc\n', b'xy\n', b'z\n'])

    async def test_partial_receive_activity_truncation_and_limit(self):
        rt = self.runtime()
        async def collect():
            return [line async for line in rt._records()]
        task = asyncio.create_task(collect())
        rt.proc.stdout.feed_data(b'{partial')
        await asyncio.sleep(0)
        self.assertIsNotNone(rt.diagnostics.snapshot().last_byte_age)
        self.assertIsNone(rt.diagnostics.snapshot().last_record_age)
        self.assertFalse(task.done())
        rt.proc.stdout.feed_eof()
        with self.assertRaisesRegex(ModelResponseError, 'Truncated'):
            await task
        for data in (b'x' * 17, b'x' * 16 + b'\n'):
            rt = self.runtime()
            rt.proc.stdout.feed_data(data)
            with mock.patch('pythia.interaction.claude_relay._runtime.MAX_RECORD', 16):
                with self.assertRaisesRegex(ModelResponseError, 'exceeds limit'):
                    await collect()
