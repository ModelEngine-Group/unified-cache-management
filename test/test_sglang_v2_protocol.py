"""Tests for the thin Store v2 assembler interface, without native dependencies."""

import importlib.util
import logging
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
with patch.dict(
    sys.modules, {"ucm.logger": types.SimpleNamespace(init_logger=logging.getLogger)}
):
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
                        {
                            "store": "test",
                            "config": {
                                "unique_id": "old",
                                "block_size": 1,
                                "device_id": 0,
                            },
                        },
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
        self.runtime_args = {
            "unique_id": "sglang-test",
            "block_size": 8192,
            "device_id": 3,
            "worker_index": 1,
            "worker_number": 2,
        }
        self.runtime = runtime.Runtime(
            {"UCM_CONFIG_FILE": str(self.config_file)}, **self.runtime_args
        )

    def test_partitioned_dump_accepts_multiple_store_branches(self):
        config = {"branches": [{"pipe": []}, {"pipe": []}]}
        self.config_file.write_text(
            yaml.safe_dump({"ucm_connectors": [{"ucm_connector_config": config}]}),
            encoding="utf-8",
        )
        self.native.StoreAssembler.reset_mock()
        runtime.Runtime(
            {"UCM_CONFIG_FILE": str(self.config_file)},
            dump_partitions=2,
            **self.runtime_args,
        )
        self.native.StoreAssembler.assert_called_once_with(config, self.runtime_args)

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
            },
            self.runtime_args,
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
            "prepare_prefetch": ("r", b"x" * 16),
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

    def test_array_submission_preserves_buffers_and_old_worker_detection(self):
        self.runtime.worker = types.SimpleNamespace()
        self.assertFalse(self.runtime.supports_get_arrays)
        submit = Mock()
        self.runtime.worker.submit_get_arrays = submit
        self.assertTrue(self.runtime.supports_get_arrays)
        pointers, sizes, offsets, token, context = (object() for _ in range(5))
        self.runtime.submit_get_arrays(7, (pointers, sizes, offsets), token, context)
        submit.assert_called_once_with(7, pointers, sizes, offsets, token, context)


if __name__ == "__main__":
    unittest.main()
