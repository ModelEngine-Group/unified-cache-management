import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = (
    REPO_ROOT
    / "ucm"
    / "integration"
    / "vllm"
    / "patch"
    / "cpu_binding_affinity_patch.py"
)


def _load_symbols():
    source = PATCH_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    selected_names = {
        "_ucm_affinity_enabled",
        "_split_contiguous_halves",
        "_split_health_cores",
        "_ucm_thread_cores",
        "assign_cpu_roles",
    }
    selected_nodes = [
        node for node in tree.body if getattr(node, "name", None) in selected_names
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_UCM_HEALTH_THREAD_PREFIX": "ucm_health_",
        "os": os,
    }
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return namespace


SYMBOLS = _load_symbols()
assign_cpu_roles = SYMBOLS["assign_cpu_roles"]
ucm_thread_cores = SYMBOLS["_ucm_thread_cores"]


class CpuBindingAffinityPatchTest(unittest.TestCase):
    def test_reserves_one_health_core_from_ucm_pool(self):
        allocator = SimpleNamespace(
            assign_main={},
            assign_ucm={},
            assign_ucm_health={},
            assign_acl={},
            assign_rel={},
        )

        with patch.dict(os.environ, {"VLLM_CPU_AFFINITY": "1"}):
            assign_cpu_roles(allocator, 0, [2, 3, 4, 5, 6, 7], [8], [9])

        self.assertEqual(allocator.assign_main[0], [2, 3, 4])
        self.assertEqual(allocator.assign_ucm[0], [5, 6])
        self.assertEqual(allocator.assign_ucm_health[0], [7])

    def test_single_ucm_core_is_shared_with_health_monitor(self):
        allocator = SimpleNamespace(
            assign_main={},
            assign_ucm={},
            assign_ucm_health={},
            assign_acl={},
            assign_rel={},
        )

        with patch.dict(os.environ, {"VLLM_CPU_AFFINITY": "1"}):
            assign_cpu_roles(allocator, 0, [2, 3], [4], [5])

        self.assertEqual(allocator.assign_main[0], [2])
        self.assertEqual(allocator.assign_ucm[0], [3])
        self.assertEqual(allocator.assign_ucm_health[0], [3])

    def test_only_health_threads_use_health_cores(self):
        self.assertEqual(ucm_thread_cores("ucm_health_mon", [5, 6], [7]), [7])
        self.assertEqual(ucm_thread_cores("ucm_health_exec", [5, 6], [7]), [7])
        self.assertEqual(ucm_thread_cores("ucm_load_disp", [5, 6], [7]), [5, 6])


if __name__ == "__main__":
    unittest.main()
