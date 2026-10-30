"""Tests for the device-less Ascend launcher's declared triton arch.

Covers the blocker found on vllm-ascend v0.30.0rc2: its triton logs
"host environment overrides are ignored" and queries the device anyway, so
setting ``TRITON_ASCEND_ARCH`` is not enough and the driver's own arch lookup
has to be declared too. Fake triton modules stand in for both versions, so
these run with no triton, no vLLM and no device.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.model_check import ascend_sim  # noqa: E402
from ucm_toolkit.tools.model_check.common import UnsupportedEnvironment  # noqa: E402

ARCH_ENV = "TRITON_ASCEND_ARCH"
SOC_ENV = "SOC_VERSION"


def _fake_triton(*, with_arch_lookup: bool = True, version: str = "3.6.0+dev"):
    """Build the triton modules the launcher touches, in sys.modules form."""

    triton = types.ModuleType("triton")
    triton.__version__ = version

    driver = types.ModuleType("triton.backends.ascend.driver")

    class Utils:  # the class that owns the device-reaching lookups
        def get_arch(self):
            raise SystemError("<built-in function get_arch> returned NULL")

        def get_aicore_num(self):
            raise SystemError("cann_get_soc_name failed")

    if with_arch_lookup:
        driver.Utils = Utils
    driver.SomethingElse = int  # a non-owning attribute, must be left alone

    utils = types.ModuleType("triton.backends.ascend.utils")
    # Newer triton dropped the env-override helper; omit it on purpose.

    return {
        "triton": triton,
        "triton.backends": types.ModuleType("triton.backends"),
        "triton.backends.ascend": types.ModuleType("triton.backends.ascend"),
        "triton.backends.ascend.driver": driver,
        "triton.backends.ascend.utils": utils,
    }


class DeclaredArchTest(unittest.TestCase):
    def test_explicit_override_wins(self):
        with (
            patch.dict(sys.modules, {}, clear=False),
            patch.dict("os.environ", {ARCH_ENV: "Ascend910B4", SOC_ENV: "ascend910b1"}),
        ):
            self.assertEqual(ascend_sim._declared_ascend_arch(), "Ascend910B4")

    def test_derived_from_the_image_declaration(self):
        with patch.dict("os.environ", {SOC_ENV: "ascend910b1"}, clear=True):
            self.assertEqual(ascend_sim._declared_ascend_arch(), "Ascend910B1")

    def test_refuses_to_read_an_arch_from_nowhere(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(UnsupportedEnvironment) as caught:
                ascend_sim._declared_ascend_arch()
        self.assertIn("TRITON_ASCEND_ARCH", str(caught.exception))


class PinTritonArchTest(unittest.TestCase):
    def test_replaces_the_lookup_that_reaches_the_device(self):
        """rc2 ignores the env var, so the lookup itself must be declared."""

        modules = _fake_triton()
        with (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {SOC_ENV: "ascend910b1"}, clear=True),
        ):
            ascend_sim._pin_triton_arch()
            utils = modules["triton.backends.ascend.driver"].Utils()
            # Before the fix these raised SystemError, which is what surfaced
            # as "get_arch returned NULL" on rc2.
            self.assertEqual(utils.get_arch(), "Ascend910B1")
            self.assertEqual(utils.get_aicore_num(), 1)

    def test_still_sets_the_environment_override_for_older_triton(self):
        modules = _fake_triton(version="3.2.2")
        environment = {SOC_ENV: "ascend910b1"}
        with (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", environment, clear=True),
        ):
            ascend_sim._pin_triton_arch()
            import os

            self.assertEqual(os.environ[ARCH_ENV], "Ascend910B1")

    def test_fails_loudly_when_the_lookup_moved(self):
        """A future triton that renames it must not silently query the device."""

        modules = _fake_triton(with_arch_lookup=False, version="9.9.9")
        with (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {SOC_ENV: "ascend910b1"}, clear=True),
        ):
            with self.assertRaises(UnsupportedEnvironment) as caught:
                ascend_sim._pin_triton_arch()
        message = str(caught.exception)
        self.assertIn("9.9.9", message)
        self.assertIn("cann_get_soc_name failed", message)

    def test_missing_triton_backend_is_not_fatal(self):
        """Without triton's Ascend backend the env override is all there is."""

        with (
            patch.dict(sys.modules, {"triton": types.ModuleType("triton")}),
            patch.dict("os.environ", {SOC_ENV: "ascend910b1"}, clear=True),
        ):
            with patch.object(
                ascend_sim.importlib, "import_module", side_effect=ImportError("absent")
            ):
                ascend_sim._pin_triton_arch()  # logs and returns


if __name__ == "__main__":
    unittest.main()
