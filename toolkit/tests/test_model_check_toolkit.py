"""Toolkit-level tests for the model compatibility checker."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry  # noqa: E402
from ucm_toolkit.cli import main  # noqa: E402
from ucm_toolkit.tools.model_check.adapter import ModelCheckTool  # noqa: E402
from ucm_toolkit.tools.model_check.config import load_config  # noqa: E402


class ModelCheckToolkitTest(unittest.TestCase):
    """Verify registration and subprocess dispatch without importing vLLM."""

    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_model_check_is_registered(self):
        registry.init_builtin_tools()

        tool = registry.get("model-check")

        self.assertEqual(tool.name, "model-check")
        self.assertIn("model_check", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_list_shows_model_check(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["list"])

        self.assertEqual(result, 0)
        self.assertIn("model-check", output.getvalue())

    def test_help_does_not_import_runtime_modules(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["run", "model-check", "--help"])

        self.assertEqual(result, 0)
        self.assertIn("--model", output.getvalue())
        self.assertIn("--device-id", output.getvalue())
        self.assertIn("--tokens", output.getvalue())
        self.assertIn("--additional-config", output.getvalue())
        self.assertNotIn("ucm_toolkit.tools.model_check.cuda", sys.modules)
        self.assertNotIn("ucm_toolkit.tools.model_check.ascend", sys.modules)

    def test_child_configuration_reads_all_overrides(self):
        values = {
            "UCM_MODEL_CHECK_MODEL": "org/model",
            "UCM_MODEL_CHECK_TOKENS": "2048",
            "UCM_MODEL_CHECK_BLOCK_SIZE": "32",
            "UCM_MODEL_CHECK_USE_LAYERWISE": "false",
            "UCM_MODEL_CHECK_ADDITIONAL_CONFIG": '{"feature": true}',
            "UCM_MODEL_CHECK_STORE_PIPELINE": "Cache|Fake",
            "UCM_MODEL_CHECK_STORAGE_BACKENDS": "/cache/0:/cache/1",
            "UCM_MODEL_CHECK_DEVICE_ID": "5",
            "UCM_MODEL_CHECK_DTYPE": "float16",
            "UCM_MODEL_CHECK_KV_CACHE_DTYPE": "auto",
        }
        with patch.dict(os.environ, values, clear=True):
            config = load_config()

        self.assertEqual(config.model, "org/model")
        self.assertEqual(config.tokens, 2048)
        self.assertEqual(config.block_size, 32)
        self.assertFalse(config.use_layerwise)
        self.assertEqual(config.additional_config, {"feature": True})
        self.assertEqual(config.store_pipeline, "Cache|Fake")
        self.assertEqual(config.storage_backends, "/cache/0:/cache/1")
        self.assertEqual(config.visible_devices, "5")
        self.assertEqual(config.dtype, "float16")
        self.assertEqual(config.kv_cache_dtype, "auto")

    def test_cuda_runs_as_child_module(self):
        tool = ModelCheckTool()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: object() if name == "vllm" else None,
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=7
            ) as run,
        ):
            result = tool.run(
                [
                    "--model",
                    "/models/example",
                    "--tokens",
                    "8192",
                    "--block-size",
                    "128",
                    "--no-layerwise",
                    "--additional-config",
                    '{"enable_sparse_sfa_c8": true}',
                    "--store-pipeline",
                    "Cache|Posix",
                    "--storage-backends",
                    "/data/0:/data/1",
                    "--device-id",
                    "7",
                    "--dtype",
                    "bfloat16",
                    "--kv-cache-dtype",
                    "fp8",
                ]
            )

        self.assertEqual(result, 7)
        run.assert_called_once_with(
            [sys.executable, "-m", "ucm_toolkit.tools.model_check.cuda"],
            env={
                "UCM_MODEL_CHECK_MODEL": "/models/example",
                "UCM_MODEL_CHECK_TOKENS": "8192",
                "UCM_MODEL_CHECK_BLOCK_SIZE": "128",
                "UCM_MODEL_CHECK_USE_LAYERWISE": "false",
                "UCM_MODEL_CHECK_ADDITIONAL_CONFIG": (
                    '{"enable_sparse_sfa_c8": true}'
                ),
                "UCM_MODEL_CHECK_STORE_PIPELINE": "Cache|Posix",
                "UCM_MODEL_CHECK_STORAGE_BACKENDS": "/data/0:/data/1",
                "UCM_MODEL_CHECK_DEVICE_ID": "7",
                "UCM_MODEL_CHECK_DTYPE": "bfloat16",
                "UCM_MODEL_CHECK_KV_CACHE_DTYPE": "fp8",
                "CUDA_VISIBLE_DEVICES": "7",
            },
        )

    def test_ascend_runs_as_child_module(self):
        tool = ModelCheckTool()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: (
                    object() if name == "vllm_ascend" else None
                ),
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=8
            ) as run,
        ):
            result = tool.run(
                ["--model", "org/model", "--device-id", "3"]
            )

        self.assertEqual(result, 8)
        run.assert_called_once_with(
            [sys.executable, "-m", "ucm_toolkit.tools.model_check.ascend"],
            env={
                "UCM_MODEL_CHECK_MODEL": "org/model",
                "UCM_MODEL_CHECK_DEVICE_ID": "3",
                "ASCEND_RT_VISIBLE_DEVICES": "3",
            },
        )


if __name__ == "__main__":
    unittest.main()
