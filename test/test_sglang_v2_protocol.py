"""Tests for the thin Store v2 assembler interface, without native dependencies."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

path = Path(__file__).resolve().parents[1] / "ucm/integration/sglang/v2/runtime.py"
spec = importlib.util.spec_from_file_location("sglang_v2_runtime_test", path)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.assembler = Mock()
        self.native = types.SimpleNamespace(
            StoreAssembler=Mock(return_value=self.assembler)
        )
        self.patch = patch.dict(sys.modules, {"ucm_proxy": self.native})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.config = {
            "branches": [
                {
                    "pipe": [
                        {"store": "test"},
                        {"store": "test", "config": {"llm_role": "worker"}},
                    ]
                }
            ]
        }
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config_file = Path(directory.name) / "ucm.yaml"
        self.config_file.write_text(
            yaml.safe_dump({"ucm_connectors": [{"ucm_connector_config": self.config}]}),
            encoding="utf-8",
        )
        self.runtime = runtime.Runtime({"UCM_CONFIG_FILE": str(self.config_file)})

    def test_partitioned_dump_rejects_multiple_store_branches(self):
        config = {"branches": [{"pipe": []}, {"pipe": []}]}
        self.config_file.write_text(
            yaml.safe_dump({"ucm_connectors": [{"ucm_connector_config": config}]}),
            encoding="utf-8",
        )
        self.native.StoreAssembler.reset_mock()
        with self.assertRaisesRegex(ValueError, "single shared Store branch"):
            runtime.Runtime({"UCM_CONFIG_FILE": str(self.config_file)}, dump_partitions=2)
        self.native.StoreAssembler.assert_not_called()

    def test_one_assembler_provides_both_roles(self):
        self.native.StoreAssembler.assert_called_once_with(
            {
                "branches": [
                    {
                        "pipe": [
                            {"store": "test", "config": {"llm_role": "both"}},
                            {"store": "test", "config": {"llm_role": "worker"}},
                        ]
                    }
                ]
            }
        )
        self.assertIs(self.runtime.scheduler, self.assembler.get_scheduler.return_value)
        self.assertIs(self.runtime.worker, self.assembler.get_worker.return_value)

    def test_scheduler_calls_preserve_arguments_and_results(self):
        context, keys, token = object(), b"x" * 16, object()
        calls = {
            "on_new_request": (context,),
            "on_request_finished": ("r",),
            "lookup": ("r", keys),
            "lookup_on_prefix": ("r", keys),
            "lookup_on_reverse": ("r", keys),
            "touch": ("r", keys),
            "prepare_get": ("r", keys),
            "complete_get": ("r", token),
            "prepare_put": ("r", keys),
            "complete_put": ("r", token, False),
            "prepare_offload": (),
            "complete_offload": (context,),
            "prepare_prefetch": ("r",),
            "complete_prefetch": ("r", True),
        }
        for name, args in calls.items():
            with self.subTest(name=name):
                target = getattr(self.runtime.scheduler, name)
                self.assertIs(getattr(self.runtime, name)(*args), target.return_value)
                target.assert_called_once_with(*args)

    def test_worker_calls_do_not_poll_or_consume_extra_results(self):
        payloads, token, context = [object()], object(), object()
        calls = {
            "submit_get": (1, payloads, token, context),
            "submit_put": (2, payloads, token, context),
            "submit_offload": (4, token),
            "submit_prefetch": (3, token),
            "wait": (1,),
            "get_finished": (),
        }
        for name, args in calls.items():
            with self.subTest(name=name):
                target = getattr(self.runtime.worker, name)
                self.assertIs(getattr(self.runtime, name)(*args), target.return_value)
                target.assert_called_once_with(*args)
        self.assertEqual(len(self.runtime.worker.mock_calls), len(calls))

    def test_native_prefetch_signal_and_error_are_not_rewritten(self):
        self.runtime.scheduler.lookup_on_prefix.return_value = None
        self.assertIsNone(self.runtime.lookup_on_prefix("r", b"x" * 16))
        error = RuntimeError("native failure")
        self.runtime.worker.wait.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.runtime.wait(1)
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
