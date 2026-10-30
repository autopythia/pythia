"""Native plugins (plugin.py): the C interface through ctypes, against a fake
library of ctypes callbacks, and against the UI plugin's crate when there is a
build of it for this machine (make -C _pythia_ui plugin-check)."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from pythia.interaction import plugin
from pythia.interaction.plugin import PLUGIN_ABI, Plugin, PluginError, PluginVersionError
from pythia.interaction.plugin import load_plugin


class FakeLibrary:
    """A plugin's three C functions, as ctypes callbacks around ``answer``,
    which takes a request's bytes and returns the reply's bytes, or None for
    no reply. A reply stays in ``buffers`` until it's freed; ``freed`` holds
    (the length given, the reply's length) for each."""

    def __init__(self, answer, *, abi=PLUGIN_ABI):
        self.answer = answer
        self.buffers = {}
        self.freed = []
        self.errors = []  # an exception in a callback would only be printed
        self.pythia_plugin_abi = plugin._ABI(lambda: abi)
        self.pythia_plugin_call = plugin._CALL(self._call)
        self.pythia_plugin_free = plugin._FREE(self._free)

    def _call(self, request, length, reply, reply_length):
        try:
            data = self.answer(request[:length])
            if data is None:
                return 1
            buffer = ctypes.create_string_buffer(data, len(data))
            self.buffers[ctypes.addressof(buffer)] = buffer
            reply[0] = ctypes.cast(buffer, plugin._REPLY)
            reply_length[0] = len(data)
            return 0
        except Exception as exc:
            self.errors.append(exc)
            return 2

    def _free(self, reply, length):
        try:
            buffer = self.buffers.pop(ctypes.cast(reply, ctypes.c_void_p).value)
            self.freed.append((length, len(buffer)))
        except Exception as exc:
            self.errors.append(exc)


def json_answer(handler):
    """A FakeLibrary answer: ``handler`` takes the request as an object and
    returns the reply's fields, which go under the request's id."""
    def answer(raw):
        request = json.loads(raw)
        return json.dumps({"id": request.get("id"), **handler(request)}).encode()
    return answer


def _crate_build():
    """This machine's build of the UI plugin's crate, if there is one."""
    target = Path(__file__).resolve().parents[1] / "_pythia_ui" / "target"
    suffix = "dylib" if sys.platform == "darwin" else "so"
    found = sorted(target.glob(f"*/release/libpythia_ui.{suffix}"))
    found += sorted(target.glob(f"release/libpythia_ui.{suffix}"))
    return found[0] if found else None


class PluginTests(unittest.TestCase):
    def test_a_call_sends_json_under_an_id_of_its_own_and_frees_each_reply(self):
        seen = []

        def handler(request):
            seen.append(request)
            return {"ok": True, "op": request["op"]}

        library = FakeLibrary(json_answer(handler))
        loaded = Plugin(library, path="fake.dylib")
        self.assertEqual(loaded.abi, PLUGIN_ABI)
        self.assertEqual(loaded.call({"op": "info"}), {"id": 1, "ok": True, "op": "info"})
        self.assertEqual(loaded.call({"op": "screenshot", "display": 2})["id"], 2)
        self.assertEqual(seen, [{"id": 1, "op": "info"},
                                {"id": 2, "op": "screenshot", "display": 2}])
        self.assertEqual(library.buffers, {})
        self.assertEqual([given == made for given, made in library.freed], [True, True])
        self.assertEqual(library.errors, [])

    def test_a_failure_reply_comes_back_as_it_is(self):
        reply = {"ok": False, "kind": "permission", "error": "screenshot: not allowed"}
        loaded = Plugin(FakeLibrary(json_answer(lambda request: reply)), path="fake.dylib")
        self.assertEqual(loaded.call({"op": "screenshot"}), {"id": 1, **reply})

    def test_no_usable_reply_is_a_plugin_error(self):
        cases = {
            "no reply": (lambda raw: None,
                         "the plugin made no reply (pythia_plugin_call returned 1)."),
            "not JSON": (lambda raw: b"{", "the plugin's reply isn't JSON: "),
            "not UTF-8": (lambda raw: b'{"id": 1, "ok": true, "x": "\xff"}',
                          "the plugin's reply isn't JSON: "),
            "not an object": (lambda raw: b"[1]", "the plugin's reply isn't a JSON object."),
            "another id": (lambda raw: b'{"id": 7, "ok": true}',
                           "the plugin's reply has id 7, not 1."),
            "a boolean id": (lambda raw: b'{"id": true, "ok": true}',
                             "the plugin's reply has id True, not 1."),
            "no ok": (lambda raw: b'{"id": 1}', "the plugin's reply has no ok."),
        }
        for name, (answer, expected) in cases.items():
            with self.subTest(name):
                library = FakeLibrary(answer)
                with self.assertRaises(PluginError) as caught:
                    Plugin(library, path="fake.dylib").call({"op": "info"})
                self.assertTrue(str(caught.exception).startswith(expected), caught.exception)
                self.assertEqual(library.buffers, {})
                self.assertEqual(library.errors, [])

    def test_a_request_must_leave_the_id_to_the_plugin_object(self):
        loaded = Plugin(FakeLibrary(json_answer(lambda request: {"ok": True})), path="x")
        with self.assertRaises(ValueError):
            loaded.call({"id": 3, "op": "info"})
        with self.assertRaises(TypeError):
            loaded.call_bytes('{"op": "info"}')

    def test_only_a_known_interface_version_loads(self):
        with self.assertRaises(PluginVersionError) as caught:
            Plugin(FakeLibrary(lambda raw: None, abi=2), path="/p/libx.dylib")
        self.assertEqual(str(caught.exception), "/p/libx.dylib has plugin interface version "
                                                "2, and this pythia knows version 1.")
        self.assertEqual(caught.exception.abi, 2)

    def test_a_library_without_the_three_functions_isnt_a_plugin(self):
        library = FakeLibrary(lambda raw: None)
        del library.pythia_plugin_free
        with self.assertRaises(PluginError) as caught:
            Plugin(library, path="/p/libx.dylib")
        self.assertEqual(str(caught.exception),
                         "/p/libx.dylib isn't a pythia plugin: it has no pythia_plugin_free().")

    def test_load_plugin_says_what_the_loader_said(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "libnot.dylib"
            path.write_text("not a library")
            with self.assertRaises(PluginError) as caught:
                load_plugin(path)
        self.assertTrue(str(caught.exception).startswith(f"can't load {path}: "),
                        caught.exception)

    def test_calls_from_several_threads_take_turns(self):
        active, seen = [0], []

        def handler(request):
            active[0] += 1
            seen.append(active[0])
            time.sleep(0.005)  # lets another thread run, if the calls could overlap
            active[0] -= 1
            return {"ok": True}

        loaded = Plugin(FakeLibrary(json_answer(handler)), path="fake.dylib")
        threads = [threading.Thread(target=lambda: [loaded.call({"op": "info"})
                                                    for _ in range(5)])
                   for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual((len(seen), max(seen)), (15, 1))


@unittest.skipIf(_crate_build() is None, "no build of _pythia_ui for this machine "
                                         "(make -C _pythia_ui plugin-check)")
class CrateTests(unittest.TestCase):
    def test_the_crates_build_answers_through_the_interface(self):
        loaded = load_plugin(_crate_build())
        self.assertEqual(loaded.abi, PLUGIN_ABI)
        info = loaded.call({"op": "info"})
        self.assertIs(info["ok"], True)
        self.assertEqual((info["name"], info["abi"]), ("pythia_ui", PLUGIN_ABI))
        self.assertIn("info", info["ops"])
        refused = json.loads(loaded.call_bytes(b'{"op":'))
        self.assertEqual((refused["ok"], refused["kind"]), (False, "request"))
        if "screenshot" not in info["ops"]:  # on a Mac, that would take one
            reply = loaded.call({"op": "screenshot"})
            self.assertEqual((reply["ok"], reply["kind"]), (False, "unsupported"))


if __name__ == "__main__":
    unittest.main()
