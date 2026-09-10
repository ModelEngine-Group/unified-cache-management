"""Tests for the SGLang model compatibility checker."""

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
from ucm_toolkit.tools.sglang_model_check.compatibility import (  # noqa: E402
    find_missing_capabilities,
    infer_capabilities,
    requirements_from_facts,
)
from ucm_toolkit.tools.sglang_model_check.config import CONFIG_ENV  # noqa: E402
from ucm_toolkit.tools.sglang_model_check.runner import (  # noqa: E402
    CheckFailure,
    _require_page_results,
    _result_for_pool,
    _roundtrip,
)
from ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector import (  # noqa: E402
    _linear_attention,
    supported_platforms_for_quantization,
)


class SglangModelCheckToolkitTest(unittest.TestCase):
    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_registered(self):
        registry.init_builtin_tools()
        tool = registry.get("sglang-model-check")
        self.assertEqual(tool.name, "sglang-model-check")
        self.assertEqual(registry.get("sglang_model_check"), tool)
        self.assertFalse(tool.buildable)

    def test_help_is_lazy(self):
        registry.init_builtin_tools()
        output = io.StringIO()
        with patch(
            "ucm_toolkit.tools.sglang_model_check.adapter.importlib.util.find_spec"
        ) as find_spec, redirect_stdout(output):
            result = main(["run", "sglang-model-check", "--help"])
        self.assertEqual(result, 0)
        self.assertIn("without starting a service", output.getvalue())
        find_spec.assert_not_called()

    @patch("ucm_toolkit.tools.sglang_model_check.adapter.run_command")
    @patch("ucm_toolkit.tools.sglang_model_check.adapter.importlib.util.find_spec")
    def test_dispatches_isolated_child(self, find_spec, run_command):
        find_spec.return_value = object()
        run_command.return_value = 0
        registry.init_builtin_tools()
        result = main(
            [
                "run",
                "sglang-model-check",
                "--model",
                "/models/qwen",
                "--skip-meta-model",
            ]
        )
        self.assertEqual(result, 0)
        command = run_command.call_args.args[0]
        env = run_command.call_args.kwargs["env"]
        self.assertEqual(command[:2], [sys.executable, "-m"])
        self.assertTrue(command[2].endswith("sglang_model_check.runner"))
        self.assertIn('"model": "/models/qwen"', env[CONFIG_ENV])
        self.assertNotIn(CONFIG_ENV, os.environ)

    def test_single_pool_mla_requires_v1(self):
        requirements = requirements_from_facts({"attention_arch": "MLA"})
        self.assertEqual(requirements.pool_names, ("kv",))
        self.assertEqual(requirements.storage_api, "v1")
        self.assertTrue(requirements.is_mla)

    def test_linear_model_requires_v2_mamba_pool(self):
        requirements = requirements_from_facts(
            {"attention_arch": "MHA", "has_linear_attention": True}
        )
        self.assertEqual(requirements.pool_names, ("kv", "mamba"))
        self.assertEqual(requirements.storage_api, "v2")

    def test_combined_mamba_swa_model_keeps_both_side_pools(self):
        requirements = requirements_from_facts(
            {
                "attention_arch": "MHA",
                "has_linear_attention": True,
                "is_hybrid_swa": True,
            }
        )
        self.assertEqual(requirements.pool_names, ("kv", "mamba", "swa"))

    def test_inherited_v2_stubs_are_not_capabilities(self):
        class Base:
            def batch_get_v1(self):
                raise NotImplementedError

            def batch_set_v1(self):
                raise NotImplementedError

            def batch_exists_v2(self):
                raise NotImplementedError

            def batch_get_v2(self):
                raise NotImplementedError

            def batch_set_v2(self):
                raise NotImplementedError

        class Storage(Base):
            def batch_get_v1(self):
                return []

            def batch_set_v1(self):
                return []

        capabilities = infer_capabilities(Storage, Base)
        self.assertEqual(capabilities["api_versions"], ["v1"])
        self.assertEqual(capabilities["detection"], "method_override")

    def test_does_not_depend_on_business_capability_declaration(self):
        class Base:
            def batch_get_v1(self):
                raise NotImplementedError

            def batch_set_v1(self):
                raise NotImplementedError

        class Storage(Base):
            @classmethod
            def get_capabilities(cls):
                raise AssertionError("checker must not call business declarations")

            def batch_get_v1(self):
                return []

            def batch_set_v1(self):
                return []

        capabilities = infer_capabilities(Storage, Base)
        self.assertEqual(capabilities["api_versions"], ["v1"])

    def test_missing_capabilities_are_actionable(self):
        requirements = requirements_from_facts(
            {"attention_arch": "MLA", "is_dsa": True}
        )
        capabilities = {
            "api_versions": ["v1"],
            "layouts": ["page_first"],
            "pool_names": ["kv"],
            "attention_archs": ["MLA"],
            "supports_multi_pool": False,
        }
        missing = find_missing_capabilities(requirements, capabilities)
        self.assertIn("storage API v2", missing)

    def test_hybrid_contract_requires_v1_anchor_and_v2_side_pools(self):
        requirements = requirements_from_facts(
            {"attention_arch": "MHA", "has_linear_attention": True}
        )
        missing = find_missing_capabilities(
            requirements, {"api_versions": ["v2"]}
        )
        self.assertEqual(missing, ["storage API v1"])

    def test_v2_results_accept_enum_like_and_string_pool_keys(self):
        class Pool(str):
            value = "mamba"

            def __str__(self):
                return self.value

        pool = Pool("mamba")
        self.assertEqual(_result_for_pool({"mamba": [True]}, pool), [True])
        _require_page_results(
            {"mamba": [True, True]}, [pool], 2, "LOAD_FAILED", "load"
        )

    def test_v2_page_failure_reports_pool_name(self):
        with self.assertRaises(CheckFailure) as raised:
            _require_page_results(
                {"swa": [True, False]},
                ["swa"],
                2,
                "LOAD_FAILED",
                "roundtrip_load_v2",
            )
        self.assertEqual(raised.exception.code, "LOAD_FAILED")
        self.assertIn("swa", raised.exception.reason)

    def test_requested_layout_is_preserved_for_runtime_probe(self):
        requirements = requirements_from_facts(
            {"attention_arch": "MHA"}, host_layout="layer_first"
        )
        self.assertEqual(requirements.host_layout, "layer_first")

    def test_deepseek_v4_pool_names_follow_installed_sglang(self):
        requirements = requirements_from_facts(
            {
                "attention_arch": "MLA",
                "is_deepseek_v4": True,
                "sglang_pool_names": [
                    "kv",
                    "deepseek_v4_c4",
                    "deepseek_v4_c4_indexer_scale",
                ],
            }
        )
        self.assertEqual(
            requirements.pool_names,
            ("deepseek_v4_c4", "deepseek_v4_c4_indexer_scale"),
        )
        self.assertEqual(requirements.probe_fidelity, "runtime_pool_required")

    def test_hybrid_roundtrip_never_uses_synthetic_kv_pools(self):
        requirements = requirements_from_facts(
            {"attention_arch": "MLA", "is_dsa": True}
        )
        with self.assertRaises(CheckFailure) as raised:
            _roundtrip(object(), object(), requirements)
        self.assertEqual(
            raised.exception.code, "CHECKER_RUNTIME_POOL_PROBE_REQUIRED"
        )

    def test_linear_detection_does_not_require_uses_kda_attention(self):
        config = type(
            "Config",
            (),
            {"linear_attn_registry_result": None, "hf_text_config": object()},
        )()

        def optional(module, name):
            if name == "mambaish_config":
                return lambda _: object()
            return None

        with patch(
            "ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector._optional_callable",
            side_effect=optional,
        ):
            value, source, error = _linear_attention(config)
        self.assertTrue(value)
        self.assertEqual(source, "mambaish_config")
        self.assertIsNone(error)

    def test_nvfp4_is_reported_as_cuda_specific(self):
        self.assertEqual(
            supported_platforms_for_quantization("modelopt_fp4"), ["cuda"]
        )
        self.assertEqual(supported_platforms_for_quantization("NVFP4"), ["cuda"])
        self.assertIsNone(supported_platforms_for_quantization("awq"))


if __name__ == "__main__":
    unittest.main()
