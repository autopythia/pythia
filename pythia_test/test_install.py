"""Installer tests use fake checkouts and never touch the user's installation."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "install.py"
COMMANDS = {"ipythia": "cli", "autopythia": "auto", "apply_patch": "apply_patch"}
HAS_SAFE_PATH = sys.version_info >= (3, 11)  # python3 -P


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "home with 'quotes'"
        self.home.mkdir()
        self.env = {**os.environ, "HOME": str(self.home), "PYTHONPATH": "ignored"}
        self.use_python3(safe_path=HAS_SAFE_PATH)
        self.checkout = self.make_checkout("checkout one")
        self.bin = self.home / ".local" / "bin"
        self.link = self.home / ".pythia" / "lib" / "pythia-dev"

    def use_python3(self, safe_path):
        """Make the wrappers' python3 this interpreter; without safe_path, reject -P like Python < 3.11."""
        shim_dir = self.root / f"python3 safe_path={safe_path}"
        shim_dir.mkdir(exist_ok=True)
        reject = "" if safe_path else (
            'for arg do case $arg in -P) echo "Unknown option: -P" >&2; exit 2;; -[cm]) break;; esac; done\n'
        )
        shim = shim_dir / "python3"
        shim.write_text(f'#!/bin/sh\n{reject}exec {shlex.quote(sys.executable)} "$@"\n')
        shim.chmod(0o755)
        self.safe_path = safe_path
        self.env["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', os.defpath)}"

    def make_checkout(self, name):
        checkout = self.root / name
        package = checkout / "pythia" / "interaction"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").touch()
        (package / "__init__.py").touch()
        for module in COMMANDS.values():
            (package / f"{module}.py").write_text(
                "import json, sys\n"
                f"print(json.dumps([{name!r}, {module!r}, sys.argv[1:], getattr(sys, 'orig_argv', None)]))\n"
            )
        return checkout

    def install(self, *args, checkout=None, success=True):
        result = subprocess.run(
            [sys.executable, str(INSTALLER), *args], cwd=checkout or self.checkout,
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def assert_launchers(self, bin_dir, checkout_name):
        launch = self.root / "launch"
        launch.mkdir(exist_ok=True)
        # Neither a local pythia module nor a local stdlib module may shadow imports.
        for module in ("pythia", "json", "runpy"):
            (launch / f"{module}.py").write_text("raise AssertionError('shadowed')\n")
        args = ["--help", "spaces and 'quotes'", "", "$HOME"]
        for command, module in COMMANDS.items():
            path = bin_dir / command
            self.assertEqual(path.stat().st_mode & 0o777, 0o755)
            result = subprocess.run(
                [str(path), *args], cwd=launch, env=self.env,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            *ran, orig_argv = json.loads(result.stdout)
            self.assertEqual(ran, [checkout_name, module, args])
            if orig_argv is None:  # Python < 3.10
                continue
            # The interpreter's command line, as `ps` lists it after argv[0].
            if self.safe_path:
                self.assertEqual(orig_argv[1:], ["-P", "-m", f"pythia.interaction.{module}", *args])
            else:
                self.assertEqual([orig_argv[1], *orig_argv[3:]], ["-c", *args])

    def test_each_command_and_checkout_switch(self):
        result = self.install("-e")
        for command in COMMANDS:
            self.assertIn(str(self.bin / command), result.stdout)
        self.assertEqual(self.link.resolve(), self.checkout)
        self.assert_launchers(self.bin, "checkout one")
        before = {name: (self.bin / name).read_bytes() for name in COMMANDS}
        self.install("-e")
        other = self.make_checkout("checkout two")
        self.install("-e", checkout=other)
        self.assertEqual(self.link.resolve(), other)
        self.assertEqual(before, {name: (self.bin / name).read_bytes() for name in COMMANDS})
        self.assert_launchers(self.bin, "checkout two")

    def test_python_without_safe_path_option_runs_equivalent_c_code(self):
        self.use_python3(safe_path=False)
        self.install("-e")
        self.assert_launchers(self.bin, "checkout one")

    def test_custom_prefix_replaces_legacy_wrapper_or_symlink(self):
        prefix = self.root / "prefix with 'quotes'"
        bin_dir = prefix / "bin"
        bin_dir.mkdir(parents=True)
        target = self.root / "legacy"
        target.write_text("legacy target must not change")
        (bin_dir / "autopythia").symlink_to(target)
        (bin_dir / "ipythia").write_text("old wrapper")
        self.install("-e", "--prefix", str(prefix))
        self.assertEqual(target.read_text(), "legacy target must not change")
        self.assertFalse((bin_dir / "autopythia").is_symlink())
        self.assert_launchers(bin_dir, "checkout one")

    def test_preflight_checks_every_module_and_destination(self):
        for module in COMMANDS.values():
            with self.subTest(module=module):
                entry = self.checkout / "pythia" / "interaction" / f"{module}.py"
                content = entry.read_text()
                entry.unlink()
                self.install("-e", success=False)
                self.assertFalse(self.link.exists())
                self.assertFalse(self.bin.exists())
                entry.write_text(content)
        (self.bin / "autopythia").mkdir(parents=True)
        (self.bin / "ipythia").write_text("unchanged")
        self.install("-e", success=False)
        self.assertFalse(self.link.exists())
        self.assertEqual((self.bin / "ipythia").read_text(), "unchanged")

    def test_a_foreign_apply_patch_is_left_in_place_and_reported(self):
        # Another tool may install a command of that name, so only a wrapper
        # that this installer wrote is replaced there.
        target = self.root / "other target"
        target.write_text("other target must not change")
        cases = {
            "file": lambda path: path.write_text("#!/bin/sh\necho other\n"),
            "symlink": lambda path: path.symlink_to(target),
            "directory": lambda path: path.mkdir(),
        }
        for kind, make in cases.items():
            with self.subTest(kind=kind):
                bin_dir = self.root / f"prefix {kind}" / "bin"
                bin_dir.mkdir(parents=True)
                foreign = bin_dir / "apply_patch"
                make(foreign)
                result = self.install("-e", "--prefix", str(bin_dir.parent))
                self.assertIn(f"Left in place       = {foreign} (not written by install.py; "
                              "remove it and re-run to install pythia's)", result.stdout)
                self.assertNotIn(f"Installed           = {foreign}", result.stdout)
                if kind == "file":
                    self.assertEqual(foreign.read_text(), "#!/bin/sh\necho other\n")
                self.assertEqual(foreign.is_symlink(), kind == "symlink")
                self.assertEqual(foreign.is_dir(), kind == "directory")
                self.assertEqual(target.read_text(), "other target must not change")
                for command in ("ipythia", "autopythia"):
                    self.assertIn(f"Installed           = {bin_dir / command}", result.stdout)
        # Its own wrapper, even edited, is rewritten.
        self.install("-e")
        wrapper = self.bin / "apply_patch"
        written = wrapper.read_text()
        wrapper.write_text(written + "echo edited\n")
        result = self.install("-e")
        self.assertIn(f"Installed           = {wrapper}", result.stdout)
        self.assertEqual(wrapper.read_text(), written)

    # The UI plugin (--enable-experimental-ui-plugin): built with cargo on a Mac,
    # which these tests stand in for (install_on_mac, use_cargo).

    def install_on_mac(self, *args, success=True):
        """Run the installer as on macOS, where sys.platform reads darwin."""
        code = ("import runpy, sys; sys.platform = 'darwin'; sys.argv = sys.argv[1:]; "
                "runpy.run_path(sys.argv[0], run_name='__main__')")
        result = subprocess.run(
            [sys.executable, "-c", code, str(INSTALLER), *args], cwd=self.checkout,
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def use_cargo(self, exit_code=0):
        """Put a fake cargo first on PATH: it logs where it ran and its arguments,
        and, exiting 0, writes the library where a build would."""
        cargo_dir = self.root / "cargo bin"
        cargo_dir.mkdir(exist_ok=True)
        self.cargo_log = self.root / "cargo.log"
        built = "target/aarch64-apple-darwin/release"
        build = (f'mkdir -p {built} && printf "built in %s" "$PWD" > {built}/libpythia_ui.dylib\n'
                 if exit_code == 0 else "")
        cargo = cargo_dir / "cargo"
        cargo.write_text(f'#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > '
                         f'{shlex.quote(str(self.cargo_log))}\n{build}exit {exit_code}\n')
        cargo.chmod(0o755)
        self.env["PATH"] = f"{cargo_dir}{os.pathsep}{self.env['PATH']}"

    def make_crate(self):
        crate = self.checkout / "_pythia_ui"
        crate.mkdir()
        (crate / "Cargo.toml").write_text('[package]\nname = "pythia_ui"\n')
        return crate

    def installed_plugin(self, content="old plugin"):
        plugin = self.home / ".pythia" / "plugin" / "libpythia_ui.dylib"
        plugin.parent.mkdir(parents=True)
        plugin.write_text(content)
        return plugin

    def test_the_ui_plugin_is_built_and_installed_by_renaming(self):
        crate = self.make_crate()
        self.use_cargo()
        plugin = self.installed_plugin()
        with open(plugin) as loaded:  # as a running ipythia holds the old one
            result = self.install_on_mac("-e", "--enable-experimental-ui-plugin")
            self.assertEqual(loaded.read(), "old plugin")
        self.assertEqual(self.cargo_log.read_text().splitlines(),
                         [str(crate), "build", "--offline", "--release", "--lib"])
        self.assertEqual(plugin.read_text(), f"built in {crate}")
        self.assertEqual(plugin.stat().st_mode & 0o777, 0o755)
        self.assertEqual(sorted(path.name for path in plugin.parent.iterdir()),
                         ["libpythia_ui.dylib"])
        self.assertIn(f"Installed           = {plugin} (UI plugin; running sessions keep the "
                      "one they loaded until they restart)", result.stdout)
        self.assertEqual(self.link.resolve(), self.checkout)
        self.assert_launchers(self.bin, "checkout one")

    def test_a_ui_plugin_that_cant_be_built_changes_nothing(self):
        plugin = self.installed_plugin()
        result = self.install_on_mac("-e", "--enable-experimental-ui-plugin", success=False)
        self.assertIn("_pythia_ui/Cargo.toml not found", result.stderr)
        self.make_crate()
        path = self.env["PATH"]
        self.env["PATH"] = str(self.root / "no cargo here")
        result = self.install_on_mac("-e", "--enable-experimental-ui-plugin", success=False)
        self.assertIn("--enable-experimental-ui-plugin: cargo is not on PATH; the plugin needs "
                      "a rustup toolchain", result.stderr)
        self.env["PATH"] = path
        self.use_cargo(exit_code=101)
        result = self.install_on_mac("-e", "--enable-experimental-ui-plugin", success=False)
        self.assertIn("--enable-experimental-ui-plugin: building the UI plugin failed (cargo "
                      "exited 101). Nothing was installed.", result.stderr)
        self.assertFalse(self.link.exists())
        self.assertFalse(self.bin.exists())
        self.assertEqual(plugin.read_text(), "old plugin")

    @unittest.skipIf(sys.platform == "darwin", "the UI plugin builds on macOS")
    def test_the_ui_plugin_option_is_refused_off_macos(self):
        self.make_crate()
        self.use_cargo()
        result = self.install("-e", "--enable-experimental-ui-plugin", success=False)
        self.assertIn(f"--enable-experimental-ui-plugin: the UI plugin is for macOS on Apple "
                      f"Silicon, and this is {sys.platform}", result.stderr)
        self.assertFalse(self.cargo_log.exists())
        self.assertFalse(self.link.exists())
        self.assertFalse((self.home / ".pythia" / "plugin").exists())

    def test_without_the_option_the_plugin_directory_is_left_alone(self):
        self.make_crate()
        self.use_cargo()
        self.install_on_mac("-e")
        self.assertFalse((self.home / ".pythia" / "plugin").exists())
        plugin = self.installed_plugin()
        before = plugin.stat()
        self.install_on_mac("-e")
        self.assertEqual(plugin.read_text(), "old plugin")
        self.assertEqual(plugin.stat().st_ino, before.st_ino)
        self.assertFalse(self.cargo_log.exists())

    def test_requires_editable_and_preserves_non_symlink_checkout_path(self):
        self.install(success=False)
        self.assertFalse(self.link.exists())
        self.link.mkdir(parents=True)
        self.install("-e", success=False)
        self.assertTrue(self.link.is_dir())
        self.assertFalse(self.bin.exists())
