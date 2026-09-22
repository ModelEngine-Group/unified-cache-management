import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch


class HealthThreadBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger_module = ModuleType("vllm.logger")
        logger_module.logger = logging.getLogger(__name__)
        utils_module = ModuleType("ucm.integration.vllm.patch.utils")
        utils_module.patch_or_inject = setattr
        source = (
            Path(__file__).resolve().parents[1]
            / "ucm/integration/vllm/patch/cpu_binding_affinity_patch.py"
        )
        spec = importlib.util.spec_from_file_location("cpu_binding_under_test", source)
        cls.binding = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "psutil": ModuleType("psutil"),
                "vllm.logger": logger_module,
                "ucm.integration.vllm.patch.utils": utils_module,
            },
        ):
            spec.loader.exec_module(cls.binding)

    def test_binds_all_health_roles_to_current_npu_health_core(self):
        allocator = SimpleNamespace(
            current_npu=3,
            assign_ucm={0: [4, 5], 3: [10, 11]},
            assign_ucm_health={0: [6], 3: [12]},
            bind=Mock(),
        )
        tasks = [
            (101, "ucm_health_mon"),
            (102, "ucm_health_pmon"),
            (103, "ucm_health_io"),
            (104, "ucm_posix_aio"),
            (105, "python"),
        ]
        with (
            patch.dict(os.environ, {"VLLM_CPU_AFFINITY": "1"}),
            patch.object(self.binding, "_task_snapshot", return_value=tasks),
            patch.object(self.binding, "_logger", return_value=Mock()) as get_logger,
        ):
            self.binding._bind_ucm_threads(allocator)

        self.assertEqual(
            allocator.bind.call_args_list,
            [
                call("101", [12], False),
                call("102", [12], False),
                call("103", [12], False),
                call("104", [10, 11], False),
            ],
        )
        get_logger.return_value.info.assert_any_call(
            "[UCM CPU Affinity] vllm-ascend bound %s health tasks to cores %s",
            3,
            [12],
        )

    def test_legacy_device_mapping_without_reserved_core(self):
        allocator = SimpleNamespace(
            rank_id=1,
            device_info=SimpleNamespace(running_npu_list=[0, 3]),
            assign_ucm={3: [10]},
            assign_ucm_health={3: []},
            bind=Mock(),
        )
        with (
            patch.dict(os.environ, {"VLLM_CPU_AFFINITY": "1"}),
            patch.object(
                self.binding, "_task_snapshot", return_value=[(101, "ucm_health_io")]
            ),
            patch.object(self.binding, "_logger", return_value=Mock()),
        ):
            self.binding._bind_ucm_threads(allocator)
        allocator.bind.assert_called_once_with("101", [10], False)

    def test_disabled_affinity_leaves_threads_unchanged(self):
        allocator = SimpleNamespace(
            current_npu=0,
            assign_ucm={0: [10, 11]},
            assign_ucm_health={0: [12]},
            bind=Mock(),
        )
        with patch.dict(os.environ, {"VLLM_CPU_AFFINITY": "0"}):
            self.binding._bind_ucm_threads(allocator)
        allocator.bind.assert_not_called()


if __name__ == "__main__":
    unittest.main()
