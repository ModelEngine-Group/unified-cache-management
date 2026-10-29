"""Tests for the device-less CUDA launcher's declared-capability handling."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.model_check import cuda_sim  # noqa: E402
from ucm_toolkit.tools.model_check.common import UnsupportedEnvironment  # noqa: E402


def _parse(capability: str | None):
    """Run the parser with this declaration, without reimporting the module."""

    declared = replace(cuda_sim.config, cuda_capability=capability)
    with patch.object(cuda_sim, "config", declared):
        return cuda_sim._parse_capability()


class CudaCapabilityTest(unittest.TestCase):
    def test_accepts_dotted_and_nvcc_forms(self):
        self.assertEqual(_parse("9.0"), (9, 0))
        self.assertEqual(_parse("90"), (9, 0))
        self.assertEqual(_parse("8.9"), (8, 9))
        self.assertEqual(_parse("10.0"), (10, 0))
        self.assertEqual(_parse("100"), (10, 0))

    def test_refuses_to_invent_a_capability(self):
        for missing in (None, ""):
            with self.assertRaises(UnsupportedEnvironment) as caught:
                _parse(missing)
            self.assertIn("--cuda-capability", str(caught.exception))

    def test_rejects_unreadable_or_impossible_values(self):
        for invalid in ("abc", "9.x", "0.0"):
            with self.assertRaises(UnsupportedEnvironment):
                _parse(invalid)


class CudaSimContractTest(unittest.TestCase):
    def test_declares_cuda_and_keeps_tensors_on_the_host(self):
        self.assertEqual(cuda_sim.HOST.type, "cpu")
        self.assertEqual(
            cuda_sim.CUDA_PLATFORM_QUALNAME, "vllm.platforms.cuda.CudaPlatform"
        )

    def test_refuses_a_build_without_the_cuda_attention_kernels(self):
        """A CPU or "+empty" build has no vllm.vllm_flash_attn: say so clearly."""

        with patch.object(
            cuda_sim.importlib, "import_module", side_effect=ImportError("absent")
        ):
            with self.assertRaises(UnsupportedEnvironment) as caught:
                cuda_sim._require_cuda_flash_attn()
        self.assertIn("CUDA build of vLLM", str(caught.exception))

    def test_refuses_to_declare_cuda_after_the_platform_resolved(self):
        """Declaring the platform too late would silently check the wrong one."""

        stub = type(sys)("vllm.platforms")
        stub._current_platform = object()
        stub.builtin_platform_plugins = {}
        with patch.object(cuda_sim.importlib, "import_module", return_value=stub):
            with self.assertRaises(UnsupportedEnvironment) as caught:
                cuda_sim._force_cuda_platform()
        self.assertIn("already resolved", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
