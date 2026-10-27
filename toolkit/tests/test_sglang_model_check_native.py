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
    def test_native_constructor_receives_half_of_host_budget(self):
        from contextlib import nullcontext
        from ucm_toolkit.tools.sglang_model_check.native_probe import build_native_pools

        requirements = requirements_from_facts({"attention_arch": "MLA", "is_dsa": True})

        def build(bundle, *args):
            bundle.pools.update(kv=object(), indexer=object())

        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), patch(
            "ucm_toolkit.tools.sglang_model_check.native_probe.cpu_allocation_budget",
            return_value=nullcontext([123]),
        ) as guard, patch(
            "ucm_toolkit.tools.sglang_model_check.native_probe._build_dsa", side_effect=build
        ), patch("ucm_toolkit.tools.sglang_model_check.native_probe._describe"):
            bundle = build_native_pools(object(), requirements,
                CheckConfig(model="test", max_host_mib=2048), "cuda", "bf16")
        self.assertEqual(guard.call_args.args[1], 1024 * 1024 * 1024)
        self.assertEqual(bundle.report["constructor_factory_bytes"], 123)

    def test_combined_dsa_state_uses_compact_attention_and_state_descriptors(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _build_dsa_recurrent
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolSpec

        bundle = NativePools()
        model = object()
        config = CheckConfig(model="hybrid", page_size=256)
        plan = [PoolSpec("kv", "native", (0, 3)),
                PoolSpec("mamba", "mamba", (1, 2), window_pages=1)]
        state_host = object()
        with patch("ucm_toolkit.tools.sglang_model_check.pool_probe.build_pool_specs", return_value=plan) as planner, patch(
            "ucm_toolkit.tools.sglang_model_check.native_probe._build_dsa"
        ) as dsa, patch("ucm_toolkit.tools.sglang_model_check.runner._make_mamba_host_pool", return_value=state_host) as state:
            _build_dsa_recurrent(bundle, model, config, "cuda", "bf16")
        planner.assert_called_once_with(model, ("kv", "mamba"), 256, "cuda", external_attention=True)
        dsa.assert_called_once_with(bundle, model, config, "cuda", "bf16", full_layers=(0, 3))
        state.assert_called_once_with(plan[1], config)
        self.assertIs(bundle.pools["mamba"], state_host)
        self.assertEqual(bundle.specs, [plan[1]])
        self.assertEqual(bundle.layer_ids["mamba"], (1, 2))

    def test_low_ratio_discovery_compatibility(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _dsv4_low_ratio_regions_compat
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolProbeUnavailable

        old = SimpleNamespace()
        pool = SimpleNamespace()
        self.assertEqual(_dsv4_low_ratio_regions_compat(old, pool, 256, [0, 4, 128]), [])
        for ratios, physical in (([1, 4], pool), ([2], pool),
                                 ([4], SimpleNamespace(sources_by_ratio={1: [0]})),
                                 ([4], SimpleNamespace(kv_pools={2: object()})),
                                 ([4], SimpleNamespace(index_pools={1: object()}))):
            with self.assertRaisesRegex(PoolProbeUnavailable, "active C1/C2 payloads"):
                _dsv4_low_ratio_regions_compat(old, physical, 256, ratios)
        discover = unittest.mock.Mock(return_value=["native region"])
        self.assertEqual(_dsv4_low_ratio_regions_compat(
            SimpleNamespace(_dsv4_low_ratio_regions=discover), pool, 256, [1]), ["native region"])
        discover.assert_called_once_with(pool, 256)
        discover.side_effect = RuntimeError("native discovery failed")
        with self.assertRaisesRegex(RuntimeError, "native discovery failed"):
            _dsv4_low_ratio_regions_compat(
                SimpleNamespace(_dsv4_low_ratio_regions=discover), pool, 256, [1])

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
    def test_meta_constructor_reads_real_model_path_and_revision(self):
        from contextlib import contextmanager, nullcontext
        from ucm_toolkit.tools.sglang_model_check.runner import _check_meta_model

        current = {"model_path": "dummy", "revision": None}
        model_config = SimpleNamespace(model_path="/models/qwen", revision="local-rev", dtype="bf16")

        @contextmanager
        def guard(**fields):
            saved = current.copy()
            current.update(fields)
            try:
                yield
            finally:
                current.clear()
                current.update(saved)

        def initialize(*args):
            self.assertEqual(current["model_path"], model_config.model_path)
            self.assertEqual(current["revision"], model_config.revision)
            return "meta-model"

        modules = _module_tree({
            "torch": {"get_default_dtype": lambda: "fp32", "set_default_dtype": lambda value: None,
                      "device": lambda value: nullcontext()},
            "sglang.srt.configs.load_config": {"LoadConfig": lambda: object()},
            "sglang.srt.model_loader.loader": {
                "_get_quantization_config": lambda *args: None, "_initialize_model": initialize},
            "sglang.srt.model_executor.model_runner_components.layer_setup": {
                "resolve_layer_indices": lambda model, model_config, is_draft_worker: None},
            "sglang.srt.speculative.spec_info": {"SpeculativeAlgorithm": SimpleNamespace(NONE=None)},
        })
        with patch.dict(sys.modules, modules), patch(
            "ucm_toolkit.tools.sglang_model_check.sglang_compat.server_args_guard", side_effect=guard
        ), patch("ucm_toolkit.tools.sglang_model_check.runner._single_rank_model_parallel", side_effect=nullcontext), patch(
            "ucm_toolkit.tools.sglang_model_check.runner._meta_device_guard", side_effect=lambda torch: nullcontext()
        ):
            self.assertEqual(_check_meta_model(model_config), "meta-model")
        self.assertEqual(current, {"model_path": "dummy", "revision": None})

    def test_legacy_hosts_receive_native_cpu_pools(self):
        from ucm_toolkit.tools.sglang_model_check.runner import _make_host_pool

        allocations = []

        class DevicePool:
            def __init__(self, **kwargs):
                allocations.append(kwargs)
                self.row_dim = kwargs.get("head_num", 0) * kwargs.get("head_dim", 0)
                self.k_data_ptrs, self.v_data_ptrs = object(), object()
                self.kv_buffer = object()

        class Host:
            def __init__(self, device_pool, **kwargs):
                self.device_pool = device_pool
                self.head_num = device_pool.row_dim // 128
                self.pointers = (device_pool.k_data_ptrs, device_pool.v_data_ptrs)

        class MLAHost:
            def __init__(self, device_pool, **kwargs):
                self.device_pool = device_pool
                self.kv_buffer = device_pool.kv_buffer

        modules = _module_tree({
            "torch": {},
            "sglang.srt.mem_cache.memory_pool": {"MHATokenToKVPool": DevicePool,
                                                 "MLATokenToKVPool": DevicePool},
            "sglang.srt.mem_cache.pool_host.mha": {"MHATokenToKVPoolHost": Host},
            "sglang.srt.mem_cache.pool_host.mla": {"MLATokenToKVPoolHost": MLAHost},
        })
        model = SimpleNamespace(num_hidden_layers=62, num_key_value_heads=2,
                                head_dim=128, dtype=SimpleNamespace(itemsize=2),
                                kv_lora_rank=192, qk_rope_head_dim=64)
        with patch.dict(sys.modules, modules):
            for arch in ("MHA", "MLA"):
                for layout in ("page_first", "page_first_direct"):
                    with self.subTest(arch=arch, layout=layout):
                        host = _make_host_pool(model, requirements_from_facts({"attention_arch": arch}),
                            CheckConfig(model="test", page_size=64, layout=layout))
                        self.assertIsInstance(host.device_pool, DevicePool)
                        if arch == "MHA":
                            self.assertEqual(host.head_num, 2)
                            self.assertFalse(allocations[-1]["enable_alt_stream"])
                        else:
                            self.assertIs(host.kv_buffer, host.device_pool.kv_buffer)
                            self.assertEqual(allocations[-1]["kv_lora_rank"], 192)
                        self.assertEqual(allocations[-1]["layer_num"], 62)
                        self.assertEqual(allocations[-1]["device"], "cpu")

    def test_native_mla_host_supports_old_and_new_constructor_apis(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _host_from_device

        calls = []

        class LegacyHost:
            def __init__(self, device_pool, **kwargs):
                calls.append((device_pool, kwargs))

        class ModernHost:
            def __init__(self, buffer_info, **kwargs):
                calls.append((buffer_info, kwargs))

        pool = SimpleNamespace(size=1024, start_layer=0, end_layer=2,
                               layer_num=2, kv_cache_dim=256)
        modules = _module_tree({
            "sglang.srt.mem_cache.pool_host.mha": {"get_mha_host_pool_cls": lambda info: None},
            "sglang.srt.mem_cache.pool_host.mla": {"MLATokenToKVPoolHost": LegacyHost},
        })
        with patch.dict(sys.modules, modules):
            _host_from_device(pool, 256, "page_first", is_mla=True)
        self.assertIs(calls[-1][0], pool)
        self.assertFalse(calls[-1][1]["pin_memory"])
        self.assertNotIn("device_capacity_tokens", calls[-1][1])
        modules["sglang.srt.mem_cache.pool_host.mla"].MLATokenToKVPoolHost = ModernHost
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(PoolProbeUnavailable, "no compatible device-buffer descriptor"):
                _host_from_device(pool, 256, "page_first", is_mla=True)
            info = object()
            pool.get_device_pool_infos = lambda: [SimpleNamespace(buffer_info=info)]
            _host_from_device(pool, 256, "page_first", is_mla=True)
        self.assertIs(calls[-1][0], info)
        self.assertEqual(calls[-1][1]["target_layer_num"], 2)
        self.assertEqual(calls[-1][1]["device_capacity_tokens"], 1024)

    def test_dsa_child_allocates_only_attention_layers_and_maps_topk_flags(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _build_dsa

        calls, queried_layers = [], []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(host_pool_decls=lambda: [])

        modules = _module_tree({
            "sglang.srt.configs.model_config": {
                "get_dsa_index_head_dim": lambda config: 64,
                "get_dsa_index_kpool": lambda config: 4,
                "get_dsa_index_kpool_compress": lambda config: False,
                "dsa_layer_skips_topk": lambda config, layer: queried_layers.append(layer) or layer == 3,
            },
            "sglang.srt.mem_cache.kv_cache_configurator": {
                "calculate_mla_kv_cache_dim": lambda **kwargs: 256,
            },
            "sglang.srt.mem_cache.memory_pool": {"DSATokenToKVPool": create},
            "sglang.srt.mem_cache.pool_host.dsa": {"DSAIndexerPoolHost": type("Host", (), {})},
            "sglang.srt.platforms": {"current_platform": SimpleNamespace(get_kv_pool_cls=lambda **kwargs: create)},
        })
        model = SimpleNamespace(num_hidden_layers=4, hf_config=object(),
                                kv_lora_rank=192, qk_rope_head_dim=64)
        host = SimpleNamespace(index_k_buffer=object())
        bundle = NativePools()
        with patch.dict(sys.modules, modules), patch(
            "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device", return_value=host
        ):
            _build_dsa(bundle, model, CheckConfig(model="test", page_size=256),
                       "cuda", "bf16", full_layers=(0, 3))
        self.assertEqual(calls[0]["layer_num"], 2)
        self.assertEqual(calls[0]["end_layer"], 2)
        self.assertEqual(queried_layers, [0, 3])
        self.assertEqual(calls[0]["skip_topk_layers"], [False, True])
        self.assertEqual(bundle.layer_ids["kv"], (0, 3))

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
