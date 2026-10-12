import importlib.util
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cache_numa_affinity():
    module_name = "_test_cache_numa_affinity"
    path = (
        REPO_ROOT
        / "ucm"
        / "integration"
        / "vllm"
        / "cache_numa_affinity.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cache_numa_affinity = _load_cache_numa_affinity()
CacheNumaPlacement = cache_numa_affinity.CacheNumaPlacement
available_cache_numa_nodes = cache_numa_affinity.available_cache_numa_nodes
select_numa_local_ucm_bundle = cache_numa_affinity.select_numa_local_ucm_bundle


def _load_cpu_affinity_patch():
    fake_psutil = ModuleType("psutil")
    fake_psutil.Process = lambda: SimpleNamespace(pid=123)
    fake_vllm = ModuleType("vllm")
    fake_vllm_logger = ModuleType("vllm.logger")
    fake_vllm_logger.logger = logging.getLogger("test-vllm")
    fake_utils = ModuleType("ucm.integration.vllm.patch.utils")
    fake_utils.patch_or_inject = lambda target, name, replacement: setattr(
        target, name, replacement
    )

    module_name = "_test_cpu_binding_affinity_patch"
    path = (
        REPO_ROOT
        / "ucm"
        / "integration"
        / "vllm"
        / "patch"
        / "cpu_binding_affinity_patch.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "psutil": fake_psutil,
            "vllm": fake_vllm,
            "vllm.logger": fake_vllm_logger,
            "ucm.integration.vllm.cache_numa_affinity": cache_numa_affinity,
            "ucm.integration.vllm.patch.utils": fake_utils,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class CacheNumaAffinityTest(unittest.TestCase):
    def test_available_nodes_match_cache_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            online = root / "online"
            memory = root / "has_memory"
            status = root / "status"
            online.write_text("0-9\n", encoding="utf-8")
            memory.write_text("0-8\n", encoding="utf-8")
            status.write_text(
                "Name:\tpython\nMems_allowed_list:\t0-7,9\n", encoding="utf-8"
            )
            self.assertEqual(
                available_cache_numa_nodes(online, memory, status), list(range(8))
            )

    def test_role_split_starts_at_actual_allowed_cpu(self):
        module = _load_cpu_affinity_patch()
        owner = SimpleNamespace(
            assign_ucm={},
            assign_ucm_health={},
            assign_main={},
            assign_acl={},
            assign_rel={},
        )
        with patch.dict("os.environ", {"VLLM_CPU_AFFINITY": "1"}, clear=False):
            module.assign_cpu_roles(
                owner,
                0,
                list(range(3, 39)),
                [39],
                [40],
            )
        self.assertEqual(owner.assign_main[0], list(range(3, 21)))
        self.assertEqual(owner.assign_ucm[0], list(range(21, 38)))
        self.assertEqual(owner.assign_ucm_health[0], [38])
        self.assertEqual(owner.assign_acl[0], [39])
        self.assertEqual(owner.assign_rel[0], [40])

    def test_16_rank_8_numa_bundle_permutation(self):
        assign_ucm = {}
        assign_health = {}
        cpu_to_node = {}
        for node in range(8):
            for core in range(node * 80 + 1, node * 80 + 81):
                cpu_to_node[core] = node
        for rank in range(16):
            base = rank * 40 + 1
            assign_ucm[rank] = list(range(base + 20, base + 37))
            assign_health[rank] = [base + 37]

        nodes = tuple(range(8))
        expected_first_cores = {
            0: 21,
            8: 61,
            1: 101,
            9: 141,
            2: 181,
            10: 221,
            3: 261,
            11: 301,
            4: 341,
            12: 381,
            5: 421,
            13: 461,
            6: 501,
            14: 541,
            7: 581,
            15: 621,
        }
        selected = []
        for fallback_rank, expected_first in expected_first_cores.items():
            placement = CacheNumaPlacement(
                node=nodes[fallback_rank % len(nodes)],
                fallback_rank=fallback_rank,
                available_nodes=nodes,
            )
            bundle = select_numa_local_ucm_bundle(
                placement, assign_ucm, assign_health, cpu_to_node
            )
            self.assertIsNotNone(bundle)
            assert bundle is not None
            self.assertEqual(bundle.cores[0], expected_first)
            self.assertEqual(
                {cpu_to_node[core] for core in bundle.cores},
                {placement.node},
            )
            selected.append((bundle.cores, bundle.health_cores))

        self.assertEqual(len(selected), len(set(selected)))

    def test_alignment_changes_only_current_ucm_bundle(self):
        module = _load_cpu_affinity_patch()
        owner = SimpleNamespace(
            current_npu=1,
            assign_main={0: [3], 1: [43], 2: [83], 3: [123]},
            assign_acl={0: [39], 1: [79], 2: [119], 3: [159]},
            assign_rel={0: [40], 1: [80], 2: [120], 3: [160]},
            assign_ucm={0: [21], 1: [61], 2: [101], 3: [141]},
            assign_ucm_health={0: [38], 1: [78], 2: [118], 3: [158]},
            cpu_node={
                21: 0,
                38: 0,
                61: 0,
                78: 0,
                101: 1,
                118: 1,
                141: 1,
                158: 1,
            },
        )
        placement = CacheNumaPlacement(
            node=1,
            fallback_rank=1,
            available_nodes=(0, 1),
        )
        main_before = dict(owner.assign_main)
        acl_before = dict(owner.assign_acl)
        rel_before = dict(owner.assign_rel)
        with (
            patch.dict("os.environ", {"VLLM_CPU_AFFINITY": "1"}, clear=False),
            patch.object(
                module, "get_cache_numa_placement", return_value=placement
            ),
            patch.object(
                module,
                "_logger",
                return_value=logging.getLogger("test-vllm"),
            ),
        ):
            module.align_current_ucm_cores_to_cache_numa(owner)

        self.assertEqual(owner.assign_ucm[1], [101])
        self.assertEqual(owner.assign_ucm_health[1], [118])
        self.assertEqual(owner.assign_main, main_before)
        self.assertEqual(owner.assign_acl, acl_before)
        self.assertEqual(owner.assign_rel, rel_before)


if __name__ == "__main__":
    unittest.main()
