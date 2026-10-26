"""Native-family routing and constructor guards; no accelerator dependencies."""

from __future__ import annotations

import sys
import ctypes
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm_toolkit.tools.sglang_model_check.compatibility import requirements_from_facts
from ucm_toolkit.tools.sglang_model_check.native_probe import (
    NativePools, ProbeMemoryLimit, cpu_allocation_budget, native_family,
    _build_dsv4,
)
from ucm_toolkit.tools.sglang_model_check.config import CheckConfig
from ucm_toolkit.tools.sglang_model_check.runner import _pool_page_views, _fill_pattern
from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolProbeUnavailable
from ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector import server_args_guard


class NativeRoutingTest(unittest.TestCase):
    def test_dsv4_ratios_select_only_main_layers(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _dsv4_main_layer_ratios
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolProbeUnavailable

        ratios = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]
        model = SimpleNamespace(num_hidden_layers=43, compress_ratios=ratios)
        self.assertEqual(_dsv4_main_layer_ratios(model), ratios[:43])
        self.assertEqual(len(model.compress_ratios), 46)
        self.assertEqual(_dsv4_main_layer_ratios(
            SimpleNamespace(num_hidden_layers=2, compress_ratios=[4, 128])), [4, 128])
        with self.assertRaisesRegex(PoolProbeUnavailable, "1 entries.*2 main model layers"):
            _dsv4_main_layer_ratios(
                SimpleNamespace(num_hidden_layers=2, compress_ratios=[4]))

    def test_registered_families(self):
        for facts, family in (
            ({"attention_arch": "MLA", "is_deepseek_v4": True}, "deepseek_v4"),
            ({"attention_arch": "MLA", "is_dsa": True}, "dsa"),
            ({"attention_arch": "MHA", "is_minimax_sparse": True}, "minimax_sparse"),
            ({"attention_arch": "MHA", "is_hybrid_swa": True}, None),
        ):
            with self.subTest(family=family):
                self.assertEqual(native_family(requirements_from_facts(facts)), family)

    def test_dsv4_activation_is_not_derived_from_enum(self):
        facts = {"attention_arch": "MLA", "is_deepseek_v4": True,
                 "sglang_pool_names": ["deepseek_v4_c4", "deepseek_v4_unused"]}
        req = requirements_from_facts(facts)
        self.assertNotIn("deepseek_v4_unused", req.pool_names)
        self.assertEqual(req.pool_names, ("kv", "deepseek_v4_runtime_pools"))

    def fake_torch(self):
        dtype = SimpleNamespace(itemsize=4)
        called = []

        def allocate(*args, **kwargs):
            called.append((args, kwargs))
            return "allocated"

        torch = SimpleNamespace(get_default_dtype=lambda: dtype)
        for name in ("empty", "zeros", "ones", "full", "rand", "randn",
                     "empty_like", "zeros_like", "ones_like"):
            setattr(torch, name, allocate)
        return torch, called, allocate

    def test_cumulative_budget_checks_before_allocation_and_restores_functions(self):
        torch, called, original = self.fake_torch()
        with self.assertRaises(ProbeMemoryLimit):
            with cpu_allocation_budget(torch, 32) as used:
                torch.empty((2, 3), device="cpu")
                self.assertEqual(used[0], 24)
                torch.zeros((2, 3), device="cpu")
        self.assertEqual(len(called), 1)
        self.assertIs(torch.empty, original)

    def test_hidden_device_allocation_is_rejected_not_redirected(self):
        torch, called, original = self.fake_torch()
        with cpu_allocation_budget(torch, 4096):
            with self.assertRaisesRegex(PoolProbeUnavailable, "real device"):
                torch.ones((2, 3), device="cuda:0")
        self.assertFalse(called)
        self.assertIs(torch.ones, original)

    def test_like_factory_uses_source_shape_and_dtype(self):
        torch, called, _ = self.fake_torch()
        source = SimpleNamespace(shape=(4, 3), dtype=SimpleNamespace(itemsize=2), device="cpu")
        with cpu_allocation_budget(torch, 24) as used:
            torch.empty_like(source)
            self.assertEqual(used[0], 24)
        self.assertEqual(len(called), 1)

    def test_native_bundle_retains_producer_until_host_release(self):
        device = object()
        closed = []
        bundle = NativePools(keepalive=[device])
        bundle.pools["kv"] = SimpleNamespace(destroy=lambda: closed.append(len(bundle.keepalive)))
        bundle.close()
        self.assertEqual(closed, [1])
        self.assertFalse(bundle.keepalive)
        self.assertFalse(bundle.pools)

    def test_explicit_cache_context_does_not_reuse_unrelated_published_args(self):
        entered = []

        class Override:
            def __enter__(self):
                entered.append("enter")

            def __exit__(self, *args):
                entered.append("exit")

        def override(**fields):
            entered.append(fields)
            return Override()

        runtime = SimpleNamespace(get_server_args=lambda: object(),
                                  get_context=lambda: SimpleNamespace(override_server_args=override))
        with patch("ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
                   return_value=runtime):
            with server_args_guard(page_size=256, hicache_ratio=1.0):
                self.assertEqual(entered[-1], "enter")
        self.assertEqual(entered, [{"page_size": 256, "hicache_ratio": 1.0}, "enter", "exit"])

    def test_byte_payload_pattern_is_nonzero(self):
        calls = []
        tensor = SimpleNamespace(device="cpu", is_floating_point=lambda: False,
                                 random_=lambda low, high, **kwargs: calls.append((low, high)))
        torch = SimpleNamespace(Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: None))
        with patch.dict(sys.modules, {"torch": torch}):
            _fill_pattern(tensor)
        self.assertEqual(calls, [(1, 127)])

    def test_rank_context_is_complete_scoped_and_does_not_read_unset_properties(self):
        from contextlib import contextmanager

        class Parallel:
            def __init__(self):
                self.values = {}

            @contextmanager
            def override(self, **values):
                previous = self.values.copy()
                self.values.update(values)
                try:
                    yield
                finally:
                    self.values = previous

        prefixes = ("tp", "pp", "attn_tp", "attn_cp", "moe_ep",
                    "moe_dp", "moe_tp", "dcp", "attn_dcp")
        for prefix in prefixes:
            name = f"{prefix}_rank"
            setattr(Parallel, name, property(lambda self, name=name: self.values[name]))
        parallel = Parallel()
        runtime = SimpleNamespace(get_server_args=lambda: object(),
                                  get_parallel=lambda: parallel)
        with patch("ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
                   return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                with server_args_guard():
                    for prefix in prefixes:
                        self.assertEqual(getattr(parallel, f"{prefix}_rank"), 0)
                        self.assertEqual(parallel.values[f"{prefix}_size"], 1)
                    with server_args_guard():
                        self.assertEqual(parallel.tp_rank, 0)
                    self.assertEqual(parallel.tp_rank, 0)
                    raise RuntimeError("constructor failed")
        self.assertEqual(parallel.values, {})

    def test_page_byte_views_reference_real_host_memory(self):
        backing = (ctypes.c_ubyte * 16)()
        pool = SimpleNamespace(
            device="cpu", page_size=2,
            get_page_buffer_meta=lambda indices: ([ctypes.addressof(backing) + indices[0] * 4], [8]),
        )
        fake_torch = SimpleNamespace(
            int64="int64", uint8="uint8", arange=lambda start, stop, **kwargs: list(range(start, stop)),
            frombuffer=lambda array, **kwargs: memoryview(array).cast("B"),
        )
        connector = SimpleNamespace(_flatten_page_meta=lambda ptrs, sizes, count: ([ptrs], [sizes]))
        with patch.dict(sys.modules, {"torch": fake_torch,
             "ucm.integration.sglang.ucm_connector": SimpleNamespace(SglangUcmConnector=connector)}):
            views = _pool_page_views(pool, "packed", 1, True)
            views[0][0] = 79
            views[0][-1] = 127
        self.assertEqual(list(backing), [0] * 8 + [79] + [0] * 6 + [127])


def _module_tree(leaves):
    modules = {}
    for name, attributes in leaves.items():
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            path = ".".join(parts[:count])
            module = modules.setdefault(path, ModuleType(path))
            if count > 1:
                setattr(modules[".".join(parts[:count - 1])], parts[count - 1], module)
        modules[name].__dict__.update(attributes)
    return modules


class NativeAssemblyTest(unittest.TestCase):
    """Check provider orchestration, not native kernel/allocator execution."""

    def test_dsv4_discovers_active_regions_and_preserves_native_page_units(self):
        calls = []

        class Host:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                calls.append(kwargs)

        swa = SimpleNamespace(kv_buffer=["swa"], bytes_per_page_padded=64, page_size=256)
        cache = SimpleNamespace(
            swa_kv_pool=swa, _unified_kv=False, swa_page_size=256,
            kv_pools={4: SimpleNamespace(), 128: SimpleNamespace(kernel_page_size=16)},
            compress_state_pools=["state", None], indexer_compress_state_pools=["index-state", None],
        )
        constructor_calls = []

        def create(**kwargs):
            constructor_calls.append(kwargs)
            return cache

        mapping = SimpleNamespace(c4={0: 0}, c128={1: 0}, c4_state_global_layers=[0])
        modules = _module_tree({
            "sglang.srt.mem_cache.deepseek_v4_memory_pool": {
                "DeepSeekV4TokenToKVPool": create,
                "select_dsv4_kv_layout": lambda: ("v4", "v4"),
                "resolve_compress_state_ring_size": lambda ratio: 8 if ratio == 4 else 128,
            },
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler": {
                "_resolve_deepseek_v4_layer_mappings": lambda cache: mapping,
                "_require_single_row_dsv4_swa_pages": lambda **kwargs: None,
                "_dsv4_page_aligned_only": lambda pool: False,
                "_dsv4_compressed_region_buffers": lambda cache, ratio: ([f"c{ratio}"], ratio * 64),
                "_dsv4_rope_sibling": lambda cache, ratio: ("deepseek_v4_c4_rope", ["rope"], 128) if ratio == 4 else None,
                "_dsv4_indexer_regions": lambda cache, size: [
                    SimpleNamespace(name=name, device_buffers=[name], item_bytes=16,
                                    slot_page_size=size, page_aligned_only=True)
                    for name in ("deepseek_v4_c4_indexer", "deepseek_v4_c4_indexer_scale")],
                "_dsv4_low_ratio_regions": lambda cache, size: [],
            },
            "sglang.srt.mem_cache.kv_cache_configurator": {
                "_get_dsv4_compress_state_dtypes": lambda: ("fp32", "fp32"),
            },
            "sglang.srt.mem_cache.memory_pool_host": {
                "LogicalHostPool": Host, "DeepSeekV4PagedHostPool": Host,
                "DeepSeekV4StateHostPool": Host,
            },
        })
        model = SimpleNamespace(num_hidden_layers=2, compress_ratios=[4, 128, 0, 0, 0],
                                qk_nope_head_dim=448, qk_rope_head_dim=64,
                                index_head_dim=128, window_size=128, hf_config=SimpleNamespace())
        bundle = NativePools()
        with patch.dict(sys.modules, modules):
            _build_dsv4(bundle, model, CheckConfig(model="test", page_size=256), "cuda", "fp8")
        self.assertEqual(set(bundle.pools), {
            "kv", "swa", "deepseek_v4_c4", "deepseek_v4_c128", "deepseek_v4_c4_rope",
            "deepseek_v4_c4_indexer", "deepseek_v4_c4_indexer_scale",
            "deepseek_v4_c4_state", "deepseek_v4_c4_indexer_state",
        })
        self.assertNotIn("deepseek_v4_c128_state", bundle.pools)
        self.assertEqual(bundle.pools["deepseek_v4_c128"].kwargs["slot_page_size"], 16)
        self.assertEqual(bundle.pools["deepseek_v4_c4"].kwargs["slot_page_size"], 256)
        self.assertTrue(all(not kwargs["pin_memory"] for kwargs in calls if "pin_memory" in kwargs))
        self.assertEqual(constructor_calls[0]["device"], "cpu")
        self.assertEqual(constructor_calls[0]["compression_ratios"], [4, 128])


if __name__ == "__main__":
    unittest.main()
