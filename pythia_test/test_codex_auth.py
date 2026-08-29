from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import CodexAuth
from pythia.interaction import CodexAuthError
from pythia.interaction import load_codex_auth


def _write_auth(path: Path, *, token: str, account_id: str = "account-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": token,
                    "account_id": account_id,
                }
            }
        ),
        encoding="utf-8",
    )


class CodexAuthTests(unittest.TestCase):
    def test_explicit_auth_file_loads_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "custom-auth.json"
            _write_auth(
                auth_file,
                token="secret-token",
                account_id="account-42",
            )

            auth = load_codex_auth(auth_file=auth_file)

        self.assertEqual(auth.access_token, "secret-token")
        self.assertEqual(auth.account_id, "account-42")
        self.assertNotIn("secret-token", repr(auth))

    def test_resolution_supports_codex_home_environment_and_home_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            explicit_home = root / "explicit"
            environment_home = root / "environment"
            default_home = root / "default-home"
            _write_auth(
                explicit_home / "auth.json",
                token="explicit-token",
            )
            _write_auth(
                environment_home / "auth.json",
                token="environment-token",
            )
            _write_auth(
                default_home / ".codex" / "auth.json",
                token="home-token",
            )

            with mock.patch.dict(
                os.environ,
                {"CODEX_HOME": str(environment_home)},
                clear=True,
            ):
                self.assertEqual(
                    load_codex_auth(codex_home=explicit_home).access_token,
                    "explicit-token",
                )
                self.assertEqual(
                    load_codex_auth().access_token,
                    "environment-token",
                )

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch(
                    "pythia.interaction.codex_auth.Path.home",
                    return_value=default_home,
                ):
                    self.assertEqual(
                        load_codex_auth().access_token,
                        "home-token",
                    )

    def test_explicit_auth_file_takes_precedence_over_codex_home(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            auth_file = root / "selected.json"
            codex_home = root / "ignored"
            _write_auth(auth_file, token="selected-token")
            _write_auth(
                codex_home / "auth.json",
                token="ignored-token",
            )

            auth = load_codex_auth(
                auth_file=auth_file,
                codex_home=codex_home,
            )

        self.assertEqual(auth.access_token, "selected-token")

    def test_missing_and_invalid_auth_are_actionable_without_token_leaks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing.json"
            with self.assertRaisesRegex(
                CodexAuthError,
                "codex login",
            ):
                load_codex_auth(auth_file=missing)

            malformed = root / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(CodexAuthError, "could not parse"):
                load_codex_auth(auth_file=malformed)

            missing_token = root / "missing-token.json"
            missing_token.write_text(
                json.dumps(
                    {
                        "OPENAI_API_KEY": "must-not-be-used",
                        "tokens": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CodexAuthError) as raised:
                load_codex_auth(auth_file=missing_token)
            self.assertIn("tokens.access_token", str(raised.exception))
            self.assertNotIn("must-not-be-used", str(raised.exception))

    def test_auth_value_validation(self):
        with self.assertRaisesRegex(CodexAuthError, "access_token"):
            CodexAuth(access_token=" ")
        with self.assertRaisesRegex(CodexAuthError, "whitespace"):
            CodexAuth(access_token="two words")
        with self.assertRaisesRegex(CodexAuthError, "account_id"):
            CodexAuth(access_token="token", account_id=" ")


if __name__ == "__main__":
    unittest.main()
