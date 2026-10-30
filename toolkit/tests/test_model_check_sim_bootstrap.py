"""Tests that the device-less declaration reaches a *fresh* interpreter.

vLLM inspects a model architecture by starting a new Python process
(``python -m vllm.model_executor.models.registry``) which loads plugins before
anything else, so a monkey patch applied in the parent never reaches it. On
vllm-ascend 0.30.0rc2 that child repeated the device query and failed with
``cann_get_soc_name failed`` / ``get_arch returned NULL``.

These tests really spawn interpreters, with a fake ``triton`` package on the
path, so the child-process path is covered rather than assumed. No triton, no
vLLM and no device are needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.model_check import sim_bootstrap  # noqa: E402

# A stand-in for triton's Ascend backend whose lookups fail the way the real
# ones do without a device, and whose owning class is named as on rc2.
FAKE_TRITON = {
    "triton/__init__.py": '__version__ = "3.6.0+dev-fake"\n',
    "triton/backends/__init__.py": "",
    "triton/backends/ascend/__init__.py": "",
    "triton/backends/ascend/driver.py": textwrap.dedent(
        """
        class NPUUtils:
            def get_arch(self):
                raise SystemError(
                    "<built-in function get_arch> returned NULL "
                    "without setting an exception"
                )

            def get_aicore_num(self):
                raise SystemError("cann_get_soc_name failed.")
        """
    ),
}

# What a child does: import triton's driver and report what the lookups give.
CHILD_PROBE = textwrap.dedent(
    """
    import json, os, sys
    from triton.backends.ascend.driver import NPUUtils

    utils = NPUUtils()
    try:
        arch = utils.get_arch()
    except Exception as exc:
        arch = f"{type(exc).__name__}: {exc}"
    try:
        cores = utils.get_aicore_num()
    except Exception as exc:
        cores = f"{type(exc).__name__}: {exc}"
    print(json.dumps({
        "arch": arch,
        "cores": cores,
        "applied": os.environ.get("UCM_MODEL_CHECK_SIM_APPLIED", ""),
    }))
    """
)


def _write_fake_triton(directory: Path) -> Path:
    for relative, body in FAKE_TRITON.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return directory


def _run_child(environment: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("UCM_MODEL_CHECK_SIM_APPLIED", None)
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", CHILD_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


class ChildInterpreterTest(unittest.TestCase):
    def setUp(self):
        self._temp = TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.fake_triton = _write_fake_triton(self.root / "fake-triton")
        self.addCleanup(self._temp.cleanup)

    def _environment(self, bootstrap: Path | None) -> dict[str, str]:
        parts = [str(self.fake_triton), str(ROOT)]
        if bootstrap is not None:
            parts.insert(0, str(bootstrap))
        return {
            "PYTHONPATH": os.pathsep.join(parts),
            "SOC_VERSION": "ascend910b1",
            sim_bootstrap.PLATFORM_ENV: "ascend",
        }

    def test_without_the_bootstrap_a_child_still_queries_the_device(self):
        """The failure this exists to fix, reproduced in a real child."""

        result = _run_child(self._environment(None))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("get_arch", payload["arch"])
        self.assertIn("SystemError", payload["arch"])

    def test_with_the_bootstrap_a_child_gets_the_declared_arch(self):
        bootstrap = sim_bootstrap.write_child_bootstrap(self.root)
        result = _run_child(self._environment(bootstrap))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["arch"], "Ascend910B1")
        self.assertEqual(payload["cores"], 1)
        self.assertEqual(payload["applied"], "ascend")

    def test_an_explicit_arch_reaches_the_child_too(self):
        bootstrap = sim_bootstrap.write_child_bootstrap(self.root)
        environment = self._environment(bootstrap)
        environment[sim_bootstrap.ARCH_ENV] = "Ascend910B4"
        payload = json.loads(_run_child(environment).stdout)
        self.assertEqual(payload["arch"], "Ascend910B4")

    def test_bootstrap_does_nothing_when_no_platform_is_declared(self):
        """It must not change interpreters outside a model-check run."""

        bootstrap = sim_bootstrap.write_child_bootstrap(self.root)
        environment = self._environment(bootstrap)
        environment.pop(sim_bootstrap.PLATFORM_ENV)
        payload = json.loads(_run_child(environment).stdout)
        self.assertIn("SystemError", payload["arch"])
        self.assertEqual(payload["applied"], "")

    def test_a_shadowed_sitecustomize_still_runs(self):
        """Ours is found first; the environment's own must not be lost."""

        other = self.root / "other-site"
        other.mkdir()
        (other / "sitecustomize.py").write_text(
            "import os\nos.environ['OTHER_SITECUSTOMIZE'] = 'ran'\n",
            encoding="utf-8",
        )
        bootstrap = sim_bootstrap.write_child_bootstrap(self.root)
        environment = self._environment(bootstrap)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(bootstrap), str(other), str(self.fake_triton), str(ROOT)]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('OTHER_SITECUSTOMIZE', 'missing'))",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, **environment},
            timeout=120,
        )
        self.assertEqual(result.stdout.strip(), "ran", result.stderr)

    def test_a_child_without_a_shadowed_sitecustomize_stays_quiet(self):
        """Chaining must not make ``site`` report a KeyError in every child.

        The generated file used to chain by re-importing the name, which took
        the running module out of ``sys.modules`` while ``site`` was still
        inside that import: every interpreter printed
        "Error in sitecustomize ... KeyError: 'sitecustomize'" on stderr even
        though the declaration itself worked.
        """

        bootstrap = sim_bootstrap.write_child_bootstrap(self.root)
        result = _run_child(self._environment(bootstrap))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["arch"], "Ascend910B1")
        self.assertEqual(result.stderr, "")

    def test_a_broken_bootstrap_warns_but_does_not_kill_the_child(self):
        """Replacing a clear error with an unrelated crash would be worse."""

        bootstrap = sim_bootstrap.write_child_bootstrap(self.root)
        environment = self._environment(bootstrap)
        # No arch to declare: SOC_VERSION removed, no explicit override.
        environment.pop("SOC_VERSION")
        result = _run_child(environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("could not declare the simulated device", result.stderr)


class InstallBootstrapTest(unittest.TestCase):
    def test_prepends_the_directory_and_keeps_the_existing_pythonpath(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(["/keep/me", "/and/me"])
        try:
            directory = sim_bootstrap.install_child_bootstrap("ascend", temp.name)
            entries = os.environ["PYTHONPATH"].split(os.pathsep)
        finally:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous
            os.environ.pop(sim_bootstrap.PLATFORM_ENV, None)

        self.assertEqual(entries[0], str(directory))
        self.assertIn("/keep/me", entries)
        self.assertIn("/and/me", entries)
        self.assertTrue((directory / "sitecustomize.py").is_file())

    def test_the_generated_file_is_importable_on_its_own(self):
        """A syntax or name error here fails silently at interpreter startup."""

        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        generated = sim_bootstrap.write_child_bootstrap(temp.name) / "sitecustomize.py"
        compile(generated.read_text(encoding="utf-8"), str(generated), "exec")


if __name__ == "__main__":
    unittest.main()
