"""Toolkit-level tests for runtime-mode (registration, dispatch, CLI)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry  # noqa: E402
from ucm_toolkit.cli import main  # noqa: E402
from ucm_toolkit.tools.runtime_mode import cli  # noqa: E402


class RuntimeModeToolkitTest(unittest.TestCase):
    """Toolkit integration: runtime-mode is registered, listed, and runnable."""

    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_runtime_mode_is_registered_top_level_tool(self):
        registry.init_builtin_tools()

        tool = registry.get("runtime-mode")

        self.assertEqual(tool.name, "runtime-mode")
        self.assertIn("runtime_mode", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_list_shows_runtime_mode(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            main(["list"])

        self.assertIn("runtime-mode", output.getvalue())

    def test_cli_can_dispatch_runtime_mode_help(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["run", "runtime-mode", "--help"])

        self.assertEqual(result, 0)
        self.assertIn("runtime-mode", output.getvalue())

    def test_doctor_advertises_runtime_mode(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["doctor", "runtime-mode"])

        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("runtime-mode", text)
        self.assertIn("control directory", text)

    def test_runtime_mode_has_readme(self):
        readme = ROOT.parent / "docs/docs-site/docs/en/toolkit/user/runtime-mode.md"
        self.assertTrue(readme.exists())


class RuntimeModeCliTest(unittest.TestCase):
    """CLI behavior of `ucm-toolkit run runtime-mode`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.control_file = os.path.join(self._tmp.name, "ucm_runtime_control_e0.json")

    def _run(self, argv):
        registry._TOOLS.clear()
        registry._ALIASES.clear()
        registry.init_builtin_tools()
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["run", "runtime-mode", *argv])
        return code, output.getvalue(), errors.getvalue()

    def test_set_writes_atomic_json_payload(self):
        code, out, _ = self._run(
            ["set", "disabled", "--control-file", self.control_file, "--wait", "0"]
        )

        self.assertEqual(code, 0)
        self.assertIn("new mode: disabled", out)
        with open(self.control_file, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"mode": "disabled"})
        self.assertFalse(
            [
                name
                for name in os.listdir(self._tmp.name)
                if name != os.path.basename(self.control_file)
            ]
        )

    def test_set_accepts_aliases_and_reports_old_mode(self):
        self._run(
            ["set", "enabled", "--control-file", self.control_file, "--wait", "0"]
        )

        code, out, _ = self._run(
            ["set", "off", "--control-file", self.control_file, "--wait", "0"]
        )

        self.assertEqual(code, 0)
        self.assertIn("old mode: enabled", out)
        self.assertIn("new mode: disabled", out)
        self.assertEqual(cli.read_mode(self.control_file), "disabled")

    def test_set_rejects_invalid_mode(self):
        code, _, err = self._run(["set", "bogus", "--control-file", self.control_file])

        self.assertEqual(code, 2)
        self.assertIn("invalid mode", err)
        self.assertFalse(os.path.exists(self.control_file))

    def test_get_reads_written_mode(self):
        self._run(["set", "lite", "--control-file", self.control_file])

        code, out, _ = self._run(["get", "--control-file", self.control_file])

        self.assertEqual(code, 0)
        self.assertIn("mode: lite", out)
        self.assertIn(self.control_file, out)

    def test_get_missing_file_fails(self):
        missing = os.path.join(self._tmp.name, "nope.json")

        code, _, err = self._run(["get", "--control-file", missing])

        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_get_invalid_file_fails(self):
        with open(self.control_file, "w", encoding="utf-8") as f:
            f.write("garbage")

        code, _, err = self._run(["get", "--control-file", self.control_file])

        self.assertEqual(code, 1)
        self.assertIn("unreadable", err)

    def test_default_path_matches_connector_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_tmpdir = tempfile.gettempdir
            tempfile.gettempdir = lambda: tmp
            try:
                expected = os.path.join(tmp, "ucm_runtime_control_engine-0.json")
                code, out, _ = self._run(
                    ["set", "lite", "--engine-id", "engine-0_dp3", "--wait", "0"]
                )
                self.assertEqual(code, 0)
                self.assertIn(expected, out)
                self.assertTrue(os.path.exists(expected))

                code, out, _ = self._run(["get", "--engine-id", "engine-0"])
                self.assertEqual(code, 0)
                self.assertIn("mode: lite", out)
            finally:
                tempfile.gettempdir = old_tmpdir

    def test_env_var_path_is_used_when_no_explicit_path(self):
        env_file = os.path.join(self._tmp.name, "env_control.json")
        old_env = os.environ.get(cli.ENV_CONTROL_FILE)
        os.environ[cli.ENV_CONTROL_FILE] = env_file
        try:
            code, out, _ = self._run(["set", "disabled", "--wait", "0"])
            self.assertEqual(code, 0)
            self.assertIn(env_file, out)

            code, out, _ = self._run(["get"])
            self.assertEqual(code, 0)
            self.assertIn("mode: disabled", out)
        finally:
            if old_env is None:
                del os.environ[cli.ENV_CONTROL_FILE]
            else:
                os.environ[cli.ENV_CONTROL_FILE] = old_env

    def test_list_discovers_control_files(self):
        cli.write_mode(
            os.path.join(tempfile.gettempdir(), "ucm_runtime_control_e0.json"), "lite"
        )
        self.addCleanup(
            lambda: os.path.exists(
                os.path.join(tempfile.gettempdir(), "ucm_runtime_control_e0.json")
            )
            and os.remove(
                os.path.join(tempfile.gettempdir(), "ucm_runtime_control_e0.json")
            )
        )

        code, out, _ = self._run(["list"])

        self.assertEqual(code, 0)
        self.assertIn("ucm_runtime_control_e0.json", out)
        self.assertIn("lite", out)
        self.assertIn("[no engine ack]", out)

    def test_parse_control_content_supports_bare_and_json(self):
        self.assertEqual(cli.parse_control_content("disabled"), "disabled")
        self.assertEqual(cli.parse_control_content('{"mode": "shadow"}'), "lite")
        self.assertEqual(cli.parse_control_content('{"mode": "bogus"}'), None)
        self.assertEqual(cli.parse_control_content(""), None)
        self.assertEqual(cli.parse_control_content("{broken"), None)


class RuntimeModeVerificationTest(unittest.TestCase):
    """`set` verification against the engine status file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.control_file = os.path.join(self._tmp.name, "ucm_runtime_control_e0.json")

    def _run(self, argv):
        registry._TOOLS.clear()
        registry._ALIASES.clear()
        registry.init_builtin_tools()
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["run", "runtime-mode", *argv])
        return code, output.getvalue(), errors.getvalue()

    def _write_status(self, **payload):
        status = {
            "mode": "enabled",
            "supports_runtime_mode_control": True,
            "connector": "UCMDirectConnector",
            "pid": 4242,
            "poll_interval_s": 1.0,
            "updated_at": 0.0,
        }
        status.update(payload)
        with open(cli.status_file(self.control_file), "w", encoding="utf-8") as f:
            json.dump(status, f)

    def test_set_fails_when_connector_unsupported(self):
        self._write_status(
            mode="enabled",
            supports_runtime_mode_control=False,
            connector="UCMFAWAConnector",
        )

        code, _, err = self._run(
            ["set", "disabled", "--control-file", self.control_file, "--wait", "2"]
        )

        self.assertEqual(code, 1)
        self.assertIn("UCMFAWAConnector", err)
        self.assertIn("does NOT support runtime-mode control", err)
        # The control file was still written; the engine simply ignores it.
        self.assertEqual(cli.read_mode(self.control_file), "disabled")

    def test_set_succeeds_when_engine_acknowledges(self):
        self._write_status(mode="disabled")

        code, out, _ = self._run(
            ["set", "disabled", "--control-file", self.control_file, "--wait", "2"]
        )

        self.assertEqual(code, 0)
        self.assertIn("verified: engine acknowledged", out)
        self.assertIn("UCMDirectConnector", out)
        self.assertIn("pid 4242", out)

    def test_set_succeeds_when_engine_updates_status_lazily(self):
        """A background 'engine' updates the status file after a delay."""
        self._write_status(mode="enabled")

        def fake_engine():
            time.sleep(0.4)
            self._write_status(mode="lite")

        worker = threading.Thread(target=fake_engine)
        worker.start()
        try:
            code, out, _ = self._run(
                ["set", "lite", "--control-file", self.control_file, "--wait", "5"]
            )
        finally:
            worker.join()

        self.assertEqual(code, 0)
        self.assertIn("verified: engine acknowledged", out)

    def test_set_fails_when_no_engine_acknowledges(self):
        code, _, err = self._run(
            ["set", "disabled", "--control-file", self.control_file, "--wait", "0.3"]
        )

        self.assertEqual(code, 1)
        self.assertIn("no engine acknowledged", err)
        self.assertIn("possible causes", err)

    def test_set_fails_when_engine_keeps_old_mode(self):
        self._write_status(mode="enabled")

        code, _, err = self._run(
            ["set", "disabled", "--control-file", self.control_file, "--wait", "0.3"]
        )

        self.assertEqual(code, 1)
        self.assertIn("still reports mode 'enabled'", err)

    def test_set_wait_zero_skips_verification(self):
        code, out, _ = self._run(
            ["set", "disabled", "--control-file", self.control_file, "--wait", "0"]
        )

        self.assertEqual(code, 0)
        self.assertIn("verification disabled", out)
        self.assertFalse(os.path.exists(cli.status_file(self.control_file)))

    def test_get_reports_engine_ack_and_support(self):
        cli.write_mode(self.control_file, "lite")
        self._write_status(mode="lite")

        code, out, err = self._run(["get", "--control-file", self.control_file])

        self.assertEqual(code, 0)
        self.assertIn("engine:  UCMDirectConnector (pid 4242)", out)
        self.assertIn("acknowledged mode: lite", out)
        self.assertEqual(err, "")

    def test_get_warns_on_unsupported_connector(self):
        cli.write_mode(self.control_file, "disabled")
        self._write_status(
            mode="disabled",
            supports_runtime_mode_control=False,
            connector="UCMHybridLinearAttentionConnector",
        )

        code, _, err = self._run(["get", "--control-file", self.control_file])

        self.assertEqual(code, 0)
        self.assertIn("does NOT support runtime-mode control", err)

    def test_get_notes_missing_status_file(self):
        cli.write_mode(self.control_file, "lite")

        code, out, _ = self._run(["get", "--control-file", self.control_file])

        self.assertEqual(code, 0)
        self.assertIn("no engine status file found", out)

    def test_list_marks_unsupported_engine(self):
        cli.write_mode(self.control_file, "disabled")
        self._write_status(
            supports_runtime_mode_control=False, connector="UCMFAWAConnector"
        )

        old_tmpdir = tempfile.gettempdir
        tempfile.gettempdir = lambda: self._tmp.name
        try:
            code, out, _ = self._run(["list"])
        finally:
            tempfile.gettempdir = old_tmpdir

        self.assertEqual(code, 0)
        self.assertIn("ucm_runtime_control_e0.json", out)
        self.assertIn("[engine: UCMFAWAConnector, UNSUPPORTED]", out)


if __name__ == "__main__":
    unittest.main()
