"""Tests for the SGLang model compatibility checker."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
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
from ucm_toolkit.tools.sglang_model_check.result import CheckResult  # noqa: E402
from ucm_toolkit.tools.sglang_model_check.runner import (  # noqa: E402
    CheckFailure,
    _require_page_results,
    _result_for_pool,
    _roundtrip,
)
from ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector import (  # noqa: E402
    _linear_attention,
    server_args_guard,
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

    def test_roundtrip_requires_explicit_storage_backend(self):
        from ucm_toolkit.errors import ToolkitError
        from ucm_toolkit.tools.sglang_model_check.adapter import SglangModelCheckTool

        with self.assertRaisesRegex(ToolkitError, "--storage-backends is required"):
            SglangModelCheckTool().run(["--model", "model", "--mode", "roundtrip"])

    def test_help_separates_advanced_options(self):
        from ucm_toolkit.tools.sglang_model_check.adapter import SglangModelCheckTool

        for flag, visible in (("--help", False), ("--help-all", True)):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(SglangModelCheckTool().run([flag]), 0)
            self.assertEqual("--c128-page-size" in output.getvalue(), visible)

    @patch("ucm_toolkit.tools.sglang_model_check.adapter.run_command", return_value=0)
    @patch("ucm_toolkit.tools.sglang_model_check.adapter.importlib.util.find_spec", return_value=object())
    def test_default_inspect_and_explicit_roundtrip(self, find_spec, run_command):
        import json
        from ucm_toolkit.tools.sglang_model_check.adapter import SglangModelCheckTool

        for options, expected in (([], "inspect"),
                                  (["--storage-backends", "/tmp/ucm"], "inspect"),
                                  (["--storage-backends", "/tmp/ucm", "--mode", "roundtrip"], "roundtrip"),
                                  (["--storage-backends", "/tmp/ucm", "--mode", "inspect"], "inspect")):
            self.assertEqual(SglangModelCheckTool().run(["--model", "model"] + options), 0)
            config = json.loads(run_command.call_args.kwargs["env"][CONFIG_ENV])
            self.assertEqual(config["mode"], expected)
            self.assertEqual(config["page_size"], 0)

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

    def test_deepseek_v4_pools_are_discovered_not_all_enum_members(self):
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
            ("kv", "deepseek_v4_runtime_pools"),
        )
        self.assertEqual(requirements.probe_fidelity, "sglang_native_pool_provider")

    def test_indexer_routes_to_native_physical_probe(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import native_family

        requirements = requirements_from_facts(
            {"attention_arch": "MLA", "is_dsa": True}
        )
        self.assertEqual(native_family(requirements), "dsa")

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

    def test_invalid_host_layout_rejected_before_io(self):
        requirements = requirements_from_facts({"attention_arch": "MHA"}, "layer_first")
        missing = find_missing_capabilities(requirements, {"api_versions": ["v1"]})
        self.assertTrue(any("host layout" in item for item in missing))

    def test_non_callable_override_is_not_a_capability(self):
        class Base:
            def batch_get_v1(self):
                pass

            def batch_set_v1(self):
                pass

        class Storage(Base):
            batch_get_v1 = None
            batch_set_v1 = None

        self.assertEqual(infer_capabilities(Storage, Base)["api_versions"], [])

    def test_environment_and_probe_failures_are_inconclusive(self):
        for code in ("ENVIRONMENT_UNSUPPORTED", "META_MODEL_INIT_UNSUPPORTED",
                     "CHECKER_HOST_POOL_API_UNSUPPORTED", "CHECKER_MEMORY_LIMIT",
                     "CACHE_REQUIREMENTS_UNDETERMINED", "CHECKER_CONFIG_ONLY"):
            result = CheckResult(model="model", mode="inspect")
            result.fail(code, "probe", "unable to validate")
            self.assertEqual(result.status, "inconclusive", code)

    def test_guard_does_not_swallow_model_server_args_exception(self):
        runtime = SimpleNamespace(get_server_args=lambda: object())
        with patch(
            "ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
            return_value=runtime,
        ):
            with self.assertRaisesRegex(ValueError, "server args constructor failure"):
                with server_args_guard():
                    raise ValueError("server args constructor failure")

    def test_meta_module_detection_keeps_shortconv_and_does_not_mutate_facts(self):
        from ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector import refine_facts_from_model

        shortconv = type("ShortConv", (), {})()
        model = SimpleNamespace(named_modules=lambda: [("state", shortconv)])
        facts = {"has_linear_attention": None, "detection_sources": {}}
        updated = refine_facts_from_model(facts, model)
        self.assertTrue(updated["has_linear_attention"])
        self.assertIsNone(facts["has_linear_attention"])
        self.assertEqual(facts["detection_sources"], {})

    def test_nvfp4_is_reported_as_cuda_specific(self):
        self.assertEqual(
            supported_platforms_for_quantization("modelopt_fp4"), ["cuda"]
        )
        self.assertEqual(supported_platforms_for_quantization("NVFP4"), ["cuda"])
        self.assertIsNone(supported_platforms_for_quantization("awq"))

    def test_server_args_guard_reuses_published_context(self):
        runtime = SimpleNamespace(get_server_args=lambda: object())
        with patch(
            "ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
            return_value=runtime,
        ):
            with server_args_guard():
                pass

    def test_result_defaults_to_summary_and_verbose_preserves_details(self):
        result = CheckResult(model="/models/test", mode="inspect")
        result.reason = "ImportError: missing helper\nTraceback: details"
        result.model_info = {
            "architectures": ["TestForCausalLM"],
            "model_type": "test",
            "detection_sources": {"linear_attention": "mambaish_config"},
        }
        result.roundtrip = {}
        summary = result.to_dict()
        self.assertEqual(summary["detail_level"], "summary")
        self.assertEqual(summary["reason"], "ImportError: missing helper")
        self.assertNotIn("detection_sources", summary["model_info"])
        self.assertNotIn("roundtrip", summary)

        verbose = result.to_dict(verbose=True)
        self.assertEqual(verbose["detail_level"], "verbose")
        self.assertIn("Traceback", verbose["reason"])
        self.assertIn("detection_sources", verbose["model_info"])

    def test_result_exposes_compatible_and_inconclusive_status(self):
        result = CheckResult(model="model", mode="inspect")
        result.succeed("inspect", "passed")
        self.assertTrue(result.supported)
        self.assertEqual(result.status, "compatible")

        result.fail(
            "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
            "roundtrip_prepare",
            "runtime pool required",
        )
        self.assertFalse(result.supported)
        self.assertEqual(result.status, "inconclusive")

        result.fail("UCM_STORAGE_API_UNSUPPORTED", "capability_check", "v2")
        self.assertEqual(result.status, "incompatible")

    def test_summary_hides_large_pool_details_but_verbose_preserves_them(self):
        result = CheckResult(model="hybrid", mode="roundtrip")
        specs = {"swa": {"layer_ids": list(range(43))}}
        native = {"compress_ratios": [4, 128] * 22}
        result.cache_requirements = {"pool_names": ["kv", "swa"],
                                     "storage_api": "v2", "pool_specs": specs,
                                     "native_provider": native}
        result.roundtrip = {"bytes_compared": 1000, "storage_api": "v2",
                            "logical_anchor": True, "pool_specs": specs,
                            "native_provider": native, "pages_written": 0,
                            "persisted_pages": {"swa": [1]},
                            "checks": ["component_data_compare"]}
        summary = result.to_dict()
        self.assertEqual(summary["roundtrip"], {"bytes_compared": 1000,
                         "storage_api": "v2", "logical_anchor": True})
        self.assertNotIn("pool_specs", summary["cache_requirements"])
        self.assertNotIn("native_provider", summary["cache_requirements"])
        verbose = result.to_dict(verbose=True)
        self.assertEqual(verbose["roundtrip"], result.roundtrip)
        self.assertEqual(verbose["cache_requirements"], result.cache_requirements)
        self.assertEqual(result.roundtrip["pool_specs"], specs)


if __name__ == "__main__":
    unittest.main()
