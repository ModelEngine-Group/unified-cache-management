"""Native-family routing and constructor guards; no accelerator dependencies."""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm_toolkit.tools.sglang_model_check.compatibility import requirements_from_facts
from ucm_toolkit.tools.sglang_model_check.config import CheckConfig
from ucm_toolkit.tools.sglang_model_check.native_probe import (
    NativePools,
    ProbeMemoryLimit,
    _build_dsv4,
    cpu_allocation_budget,
    native_family,
)
from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolProbeUnavailable
from ucm_toolkit.tools.sglang_model_check.runner import _pool_page_views
from ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector import (
    server_args_guard,
)


class NativeRoutingTest(unittest.TestCase):
    def test_combined_dsa_state_uses_compact_attention_and_state_descriptors(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import (
            _build_dsa_recurrent,
        )
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolSpec

        bundle = NativePools()
        model = object()
        config = CheckConfig(model="hybrid", page_size=256)
        plan = [
            PoolSpec("kv", "native", (0, 3)),
            PoolSpec("mamba", "mamba", (1, 2), window_pages=1),
        ]
        state_host = object()
        with (
            patch(
                "ucm_toolkit.tools.sglang_model_check.pool_probe.build_pool_specs",
                return_value=plan,
            ) as planner,
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._build_dsa"
            ) as dsa,
            patch(
                "ucm_toolkit.tools.sglang_model_check.runner._make_mamba_host_pool",
                return_value=state_host,
            ) as state,
        ):
            _build_dsa_recurrent(bundle, model, config, "cuda", "bf16")
        planner.assert_called_once_with(
            model, ("kv", "mamba"), 256, "cuda", external_attention=True
        )
        dsa.assert_called_once_with(
            bundle, model, config, "cuda", "bf16", full_layers=(0, 3)
        )
        state.assert_called_once_with(plan[1], config, model)
        self.assertIs(bundle.pools["mamba"], state_host)
        self.assertEqual(bundle.specs, [plan[1]])
        self.assertEqual(bundle.layer_ids["mamba"], (1, 2))
        # A PP stage can have no recurrent layers. Reuse the same DSA child
        # rather than requiring an artificial state pool on every stage.
        with (
            patch(
                "ucm_toolkit.tools.sglang_model_check.pool_probe.build_pool_specs",
                return_value=[plan[0]],
            ) as planner,
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._build_dsa"
            ) as dsa,
            patch(
                "ucm_toolkit.tools.sglang_model_check.runner._make_mamba_host_pool"
            ) as state,
        ):
            stage = NativePools()
            _build_dsa_recurrent(
                stage, model, config, "cuda", "bf16", layer_range=(0, 4), tp_size=2
            )
        planner.assert_called_once_with(
            model,
            ("kv", "mamba"),
            256,
            "cuda",
            external_attention=True,
            layer_range=(0, 4),
            tp_size=2,
        )
        state.assert_not_called()
        dsa.assert_called_once_with(
            stage,
            model,
            config,
            "cuda",
            "bf16",
            full_layers=(0, 3),
            layer_range=(0, 4),
            tp_size=2,
        )
        self.assertEqual(stage.report["hybrid_layer_mapping"]["state_model_layers"], [])

    def test_low_ratio_discovery_compatibility(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import (
            _dsv4_low_ratio_regions_compat,
        )
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolProbeUnavailable

        old = SimpleNamespace()
        pool = SimpleNamespace()
        self.assertEqual(
            _dsv4_low_ratio_regions_compat(old, pool, 256, [0, 4, 128]), []
        )
        for ratios, physical in (
            ([1, 4], pool),
            ([2], pool),
            ([4], SimpleNamespace(sources_by_ratio={1: [0]})),
            ([4], SimpleNamespace(kv_pools={2: object()})),
            ([4], SimpleNamespace(index_pools={1: object()})),
        ):
            with self.assertRaisesRegex(PoolProbeUnavailable, "active C1/C2 payloads"):
                _dsv4_low_ratio_regions_compat(old, physical, 256, ratios)
        discover = unittest.mock.Mock(return_value=["native region"])
        self.assertEqual(
            _dsv4_low_ratio_regions_compat(
                SimpleNamespace(_dsv4_low_ratio_regions=discover), pool, 256, [1]
            ),
            ["native region"],
        )
        discover.assert_called_once_with(pool, 256)
        discover.side_effect = RuntimeError("native discovery failed")
        with self.assertRaisesRegex(RuntimeError, "native discovery failed"):
            _dsv4_low_ratio_regions_compat(
                SimpleNamespace(_dsv4_low_ratio_regions=discover), pool, 256, [1]
            )

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
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "rand",
            "randn",
            "empty_like",
            "zeros_like",
            "ones_like",
        ):
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
        source = SimpleNamespace(
            shape=(4, 3), dtype=SimpleNamespace(itemsize=2), device="cpu"
        )
        with cpu_allocation_budget(torch, 24) as used:
            torch.empty_like(source)
            self.assertEqual(used[0], 24)

    def test_hidden_device_allocation_is_rejected_not_redirected(self):
        torch, called, original = self.fake_torch()
        with cpu_allocation_budget(torch, 4096):
            with self.assertRaisesRegex(PoolProbeUnavailable, "real device"):
                torch.ones((2, 3), device="cuda:0")
        self.assertFalse(called)
        self.assertIs(torch.ones, original)

    def test_native_bundle_retains_producer_until_host_release(self):
        device = object()
        closed = []
        bundle = NativePools(keepalive=[device])
        bundle.pools["kv"] = SimpleNamespace(
            destroy=lambda: closed.append(len(bundle.keepalive))
        )
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

        runtime = SimpleNamespace(
            get_server_args=lambda: object(),
            get_context=lambda: SimpleNamespace(override_server_args=override),
        )
        with patch(
            "ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
            return_value=runtime,
        ):
            with server_args_guard(page_size=256, hicache_ratio=1.0):
                self.assertEqual(entered[-1], "enter")
        self.assertEqual(
            entered, [{"page_size": 256, "hicache_ratio": 1.0}, "enter", "exit"]
        )

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

        prefixes = (
            "tp",
            "pp",
            "attn_tp",
            "attn_cp",
            "moe_ep",
            "moe_dp",
            "moe_tp",
            "dcp",
            "attn_dcp",
        )
        for prefix in prefixes:
            name = f"{prefix}_rank"
            setattr(Parallel, name, property(lambda self, name=name: self.values[name]))
        parallel = Parallel()
        runtime = SimpleNamespace(
            get_server_args=lambda: object(), get_parallel=lambda: parallel
        )
        with patch(
            "ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
            return_value=runtime,
        ):
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
            device="cpu",
            page_size=2,
            get_page_buffer_meta=lambda indices: (
                [ctypes.addressof(backing) + indices[0] * 4],
                [8],
            ),
        )
        fake_torch = SimpleNamespace(
            int64="int64",
            uint8="uint8",
            arange=lambda start, stop, **kwargs: list(range(start, stop)),
            frombuffer=lambda array, **kwargs: memoryview(array).cast("B"),
        )
        connector = SimpleNamespace(
            _flatten_page_meta=lambda ptrs, sizes, count: ([ptrs], [sizes])
        )
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "ucm.integration.sglang.ucm_connector": SimpleNamespace(
                    SglangUcmConnector=connector
                ),
            },
        ):
            views = _pool_page_views(pool, "packed", 1)
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
                setattr(modules[".".join(parts[: count - 1])], parts[count - 1], module)
        modules[name].__dict__.update(attributes)
    return modules


class NativeAssemblyTest(unittest.TestCase):
    def test_meta_constructor_reads_real_model_path_and_revision(self):
        from contextlib import contextmanager, nullcontext

        from ucm_toolkit.tools.sglang_model_check.runner import _check_meta_model

        current = {"model_path": "dummy", "revision": None}
        model_config = SimpleNamespace(
            model_path="/models/qwen", revision="local-rev", dtype="bf16"
        )

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

        modules = _module_tree(
            {
                "torch": {
                    "get_default_dtype": lambda: "fp32",
                    "set_default_dtype": lambda value: None,
                    "device": lambda value: nullcontext(),
                },
                "sglang.srt.configs.load_config": {"LoadConfig": lambda: object()},
                "sglang.srt.model_loader.loader": {
                    "_get_quantization_config": lambda *args: None,
                    "_initialize_model": initialize,
                },
                "sglang.srt.model_executor.model_runner_components.layer_setup": {
                    "resolve_layer_indices": lambda model, model_config, is_draft_worker: None
                },
                "sglang.srt.speculative.spec_info": {
                    "SpeculativeAlgorithm": SimpleNamespace(NONE=None)
                },
            }
        )
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.sglang_compat.server_args_guard",
                side_effect=guard,
            ),
            patch(
                "ucm_toolkit.tools.sglang_model_check.runner._single_rank_model_parallel",
                side_effect=nullcontext,
            ),
            patch(
                "ucm_toolkit.tools.sglang_model_check.runner._meta_device_guard",
                side_effect=lambda torch: nullcontext(),
            ),
        ):
            self.assertEqual(_check_meta_model(model_config), "meta-model")
        self.assertEqual(current, {"model_path": "dummy", "revision": None})

    def test_legacy_hosts_receive_native_cpu_pools(self):
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolSpec
        from ucm_toolkit.tools.sglang_model_check.runner import _make_host_pool

        allocations = []

        class DevicePool:
            def __init__(self, **kwargs):
                allocations.append(kwargs)
                self.row_dim = kwargs.get("head_num", 0) * kwargs.get("head_dim", 0)
                self.k_data_ptrs, self.v_data_ptrs = object(), object()
                self.kv_buffer = object()
                self.kv_cache_dim = kwargs.get("kv_lora_rank", 0) + kwargs.get(
                    "qk_rope_head_dim", 0
                )

        class Host:
            def __init__(self, device_pool, **kwargs):
                self.device_pool = device_pool
                self.head_num = device_pool.row_dim // 128
                self.pointers = (device_pool.k_data_ptrs, device_pool.v_data_ptrs)

        class MLAHost:
            def __init__(self, device_pool, **kwargs):
                self.device_pool = device_pool
                self.kv_buffer = device_pool.kv_buffer

        class Configurator:
            __dataclass_fields__ = {}

            def _build_mla_kv_pool(self, *, max_total_num_tokens):
                return DevicePool(
                    size=max_total_num_tokens,
                    layer_num=self.layer_info.num_effective_layers,
                    device=self.device,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                )

            def _build_mha_kv_pool(self, *, max_total_num_tokens, mha_pool_class):
                return mha_pool_class(
                    size=max_total_num_tokens,
                    layer_num=self.layer_info.num_effective_layers,
                    device=self.device,
                    head_num=self.model_config.get_num_kv_heads(1),
                    head_dim=self.model_config.head_dim,
                )

        modules = _module_tree(
            {
                "torch": {},
                "sglang.srt.mem_cache.memory_pool": {
                    "MHATokenToKVPool": DevicePool,
                    "MLATokenToKVPool": DevicePool,
                },
                "sglang.srt.mem_cache.pool_host.mha": {
                    "MHATokenToKVPoolHost": Host,
                    "get_mha_host_pool_cls": lambda pool: Host,
                },
                "sglang.srt.mem_cache.pool_host.mla": {"MLATokenToKVPoolHost": MLAHost},
                "sglang.srt.mem_cache.kv_cache_configurator": {
                    "KVCacheConfigurator": Configurator
                },
                "sglang.srt.model_executor.model_runner_components.layer_setup": {
                    "ModelLayerInfo": SimpleNamespace
                },
            }
        )
        model = SimpleNamespace(
            num_hidden_layers=62,
            num_key_value_heads=2,
            get_num_kv_heads=lambda *args: 2,
            head_dim=128,
            dtype=SimpleNamespace(itemsize=2),
            kv_lora_rank=192,
            qk_rope_head_dim=64,
        )
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.sglang_compat.environment_info",
                return_value={"platform": "cuda"},
            ),
        ):
            for arch, spec in (
                ("MHA", None),
                (
                    "MHA",
                    PoolSpec(
                        "swa", "mha", (1, 3), head_num=1, head_dim=128, v_head_dim=128
                    ),
                ),
                ("MLA", PoolSpec("kv", "mla", (3, 7), kv_cache_dim=256)),
            ):
                for layout in ("page_first", "page_first_direct"):
                    with self.subTest(arch=arch, layout=layout):
                        host = _make_host_pool(
                            model,
                            requirements_from_facts({"attention_arch": arch}),
                            CheckConfig(model="test", page_size=64, layout=layout),
                            spec,
                        )
                        self.assertIsInstance(host.device_pool, DevicePool)
                        if arch == "MHA":
                            self.assertEqual(
                                host.head_num, spec.head_num if spec else 2
                            )
                            self.assertFalse(allocations[-1]["enable_alt_stream"])
                        else:
                            self.assertIs(host.kv_buffer, host.device_pool.kv_buffer)
                            self.assertEqual(allocations[-1]["kv_lora_rank"], 192)
                        self.assertEqual(
                            allocations[-1]["layer_num"], 2 if spec else 62
                        )
                        self.assertEqual(allocations[-1]["device"], "cpu")
                        self.assertEqual(model.get_num_kv_heads(1), 2)

    def test_native_state_buffers_support_both_host_apis(self):
        from ucm_toolkit.tools.sglang_model_check.pool_probe import PoolSpec
        from ucm_toolkit.tools.sglang_model_check.runner import _make_mamba_host_pool

        calls, params, info = [], object(), object()

        class StatePool:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def get_device_pool_infos(self):
                return [SimpleNamespace(buffer_info=info)]

        class LegacyHost:
            def __init__(self, device_pool, **kwargs):
                self.source = device_pool

        class ModernHost:
            def __init__(self, buffer_info, **kwargs):
                self.source = buffer_info

        modules = _module_tree(
            {
                "sglang.srt.configs.hybrid_arch": {
                    "mambaish_config": lambda model: SimpleNamespace(
                        mamba2_cache_params=params
                    )
                },
                "sglang.srt.mem_cache.memory_pool": {"MambaPool": StatePool},
                "sglang.srt.mem_cache.pool_host.mamba": {},
            }
        )
        for host in (LegacyHost, ModernHost):
            modules["sglang.srt.mem_cache.pool_host.mamba"].MambaPoolHost = host
            with self.subTest(host=host), patch.dict(sys.modules, modules):
                result = _make_mamba_host_pool(
                    PoolSpec("mamba", "mamba", (1, 3)),
                    CheckConfig(model="test"),
                    object(),
                )
                self.assertIs(calls[-1]["cache_params"], params)
                self.assertEqual(calls[-1]["mamba_layer_ids"], [1, 3])
                self.assertEqual(calls[-1]["device"], "cpu")
                if host is LegacyHost:
                    self.assertIsInstance(result.source, StatePool)
                else:
                    self.assertIs(result.source, info)

    def test_minimax_native_factory_keeps_stage_and_platform(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _build_minimax

        native = SimpleNamespace(main_pool=object(), index_k_pool=object())
        main, index = object(), object()
        calls = []

        class Configurator:
            __dataclass_fields__ = {}

            def _build_minimax_sparse_kv_pool(self, *, max_total_num_tokens):
                calls.append(
                    (
                        self.layer_info.start_layer,
                        self.layer_info.end_layer,
                        self.device,
                        max_total_num_tokens,
                    )
                )
                return native

            _build_ascend_minimax_sparse_kv_pool = _build_minimax_sparse_kv_pool

        modules = _module_tree(
            {
                "sglang.srt.configs.model_config": {
                    "get_minimax_sparse_attention_config": lambda cfg: {},
                    "get_minimax_sparse_layer_ids": lambda cfg: ([0, 2], [1, 3]),
                    "get_minimax_sparse_disable_value_layer_ids": lambda cfg: [1, 3],
                },
                "sglang.srt.mem_cache.kv_cache_configurator": {
                    "KVCacheConfigurator": Configurator
                },
                "sglang.srt.mem_cache.pool_host.mha": {
                    "MHATokenToKOnlyPoolHost": lambda *args, **kwargs: index
                },
                "sglang.srt.mem_cache.memory_pool": {
                    "MHATokenToKVPool": type("MHA", (), {})
                },
                "sglang.srt.model_executor.model_runner_components.layer_setup": {
                    "ModelLayerInfo": SimpleNamespace
                },
                "sglang.srt.runtime_context": {"get_server_args": object},
            }
        )
        model = SimpleNamespace(num_hidden_layers=4, hf_config=object(), dtype="bf16")
        for platform in ("cuda", "ascend"):
            with (
                self.subTest(platform=platform),
                patch.dict(sys.modules, modules),
                patch(
                    "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                    return_value=main,
                ),
            ):
                bundle = NativePools()
                _build_minimax(
                    bundle,
                    model,
                    CheckConfig(model="test", page_size=64),
                    platform,
                    "bf16",
                    layer_range=(2, 4),
                    tp_size=2,
                )
            self.assertEqual(calls[-1], (2, 4, "cpu", 384))
            self.assertEqual(bundle.pools, {"kv": main, "indexer": index})
            self.assertEqual(bundle.layer_ids, {"kv": (2, 3), "indexer": (3,)})

    def test_native_mla_host_supports_old_and_new_constructor_apis(self):
        from ucm_toolkit.tools.sglang_model_check.native_probe import _host_from_device

        calls = []

        class LegacyHost:
            def __init__(self, device_pool, **kwargs):
                calls.append((device_pool, kwargs))

        class ModernHost:
            def __init__(self, buffer_info, **kwargs):
                calls.append((buffer_info, kwargs))

        pool = SimpleNamespace(
            size=1024, start_layer=0, end_layer=2, layer_num=2, kv_cache_dim=256
        )
        modules = _module_tree(
            {
                "sglang.srt.mem_cache.pool_host.mha": {
                    "get_mha_host_pool_cls": lambda info: None
                },
                "sglang.srt.mem_cache.pool_host.mla": {
                    "MLATokenToKVPoolHost": LegacyHost
                },
            }
        )
        with patch.dict(sys.modules, modules):
            _host_from_device(pool, 256, "page_first", is_mla=True)
        self.assertIs(calls[-1][0], pool)
        self.assertFalse(calls[-1][1]["pin_memory"])
        self.assertNotIn("device_capacity_tokens", calls[-1][1])
        modules["sglang.srt.mem_cache.pool_host.mla"].MLATokenToKVPoolHost = ModernHost
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                PoolProbeUnavailable, "no compatible device-buffer descriptor"
            ):
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
        declarations = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                host_pool_decls=lambda: declarations,
                skip_topk_layers=kwargs["skip_topk_layers"],
            )

        class Configurator:
            __dataclass_fields__ = {}

            def _build_dsa_kv_pool(self, *, max_total_num_tokens, max_running_requests):
                layers = range(self.layer_info.start_layer, self.layer_info.end_layer)
                return self.create(
                    layers, self.layer_info.start_layer, max_total_num_tokens
                )

            def _build_hybrid_linear_kv_pool(
                self, *, max_total_num_tokens, req_to_token_pool, mha_pool_class
            ):
                layers = self.mambaish_config.full_attention_layer_ids
                layers = [
                    i
                    for i in layers
                    if self.layer_info.start_layer <= i < self.layer_info.end_layer
                ]
                return SimpleNamespace(
                    full_attention_layer_id_mapping={
                        layer: i for i, layer in enumerate(layers)
                    },
                    full_kv_pool=self.create(layers, 0, max_total_num_tokens),
                )

            def create(self, layers, start, size):
                layers = list(layers)
                queried_layers.extend(layers)
                return create(
                    size=size,
                    layer_num=len(layers),
                    start_layer=start,
                    end_layer=start + len(layers),
                    skip_topk_layers=[i == 3 for i in layers],
                )

        modules = _module_tree(
            {
                "sglang.srt.mem_cache.kv_cache_configurator": {
                    "KVCacheConfigurator": Configurator,
                },
                "sglang.srt.configs.hybrid_arch": {
                    "mambaish_config": lambda model: SimpleNamespace(
                        full_attention_layer_ids=(0, 3)
                    ),
                },
                "sglang.srt.model_executor.model_runner_components.layer_setup": {
                    "ModelLayerInfo": SimpleNamespace,
                },
                "sglang.srt.mem_cache.memory_pool": {
                    "DSATokenToKVPool": create,
                    "MHATokenToKVPool": create,
                    "MLATokenToKVPool": create,
                },
                "sglang.srt.mem_cache.pool_host.dsa": {
                    "DSAIndexerPoolHost": type("Host", (), {})
                },
            }
        )
        model = SimpleNamespace(
            num_hidden_layers=4,
            hf_config=object(),
            kv_lora_rank=192,
            qk_rope_head_dim=64,
            dtype="bf16",
        )
        host = SimpleNamespace(index_k_buffer=object())
        bundle = NativePools()
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                return_value=host,
            ),
        ):
            _build_dsa(
                bundle,
                model,
                CheckConfig(model="test", page_size=256),
                "cuda",
                "bf16",
                full_layers=(0, 3),
            )
        self.assertEqual(calls[0]["layer_num"], 2)
        self.assertEqual(calls[0]["end_layer"], 2)
        self.assertEqual(queried_layers, [0, 3])
        self.assertEqual(calls[0]["skip_topk_layers"], [False, True])
        self.assertEqual(bundle.layer_ids["kv"], (0, 3))
        # Ascend hybrids must use the same native hybrid path, not NPU MLA.
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                return_value=host,
            ),
        ):
            hybrid = NativePools()
            _build_dsa(
                hybrid,
                model,
                CheckConfig(model="test", page_size=256),
                "ascend",
                "bf16",
                full_layers=(3,),
                layer_range=(2, 4),
                tp_size=2,
            )
        self.assertEqual(hybrid.layer_ids["kv"], (3,))
        self.assertEqual(
            hybrid.report["device_pool_builder"],
            "KVCacheConfigurator._build_hybrid_linear_kv_pool",
        )

        # The installed native builder, not the checker, owns NPU arguments.
        # This strict legacy constructor deliberately has no index_size argument.
        def npu_legacy(size, *, page_size, layer_num):
            self.assertEqual((size, page_size, layer_num), (1536, 256, 4))
            return SimpleNamespace(host_pool_decls=lambda: [])

        def native_npu_builder(self, *, max_total_num_tokens, is_dsa_model):
            return npu_legacy(
                max_total_num_tokens,
                page_size=self.page_size,
                layer_num=self.layer_info.num_effective_layers,
            )

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                Configurator,
                "_build_ascend_mla_kv_pool",
                native_npu_builder,
                create=True,
            ),
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                return_value=host,
            ),
        ):
            npu = NativePools()
            _build_dsa(
                npu, model, CheckConfig(model="test", page_size=256), "ascend", "bf16"
            )
        self.assertEqual(
            npu.report["device_pool_builder"],
            "KVCacheConfigurator._build_ascend_mla_kv_pool",
        )
        calls.clear()
        queried_layers.clear()
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                return_value=host,
            ),
        ):
            _build_dsa(
                NativePools(),
                model,
                CheckConfig(model="test", page_size=256),
                "cuda",
                "bf16",
                layer_range=(2, 4),
                tp_size=2,
            )
        self.assertEqual((calls[0]["start_layer"], calls[0]["end_layer"]), (2, 4))
        self.assertEqual(calls[0]["layer_num"], 2)
        self.assertEqual(queried_layers, [2, 3])
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                return_value=host,
            ),
        ):
            elided = NativePools()
            _build_dsa(
                elided,
                model,
                CheckConfig(model="test", page_size=256),
                "cuda",
                "bf16",
                layer_range=(3, 4),
                tp_size=2,
            )
        self.assertEqual(elided.report["elided_payloads"], ["indexer"])

        # Modern SGLang owns validation and group assembly. The offline probe
        # changes allocation policy, not the declared side-pool selection.
        declarations.extend(
            [
                SimpleNamespace(
                    pool_name="kv", is_primary=True, owned_device_layers=None
                ),
                SimpleNamespace(
                    pool_name="indexer", is_primary=False, owned_device_layers=(0,)
                ),
            ]
        )
        prepared = object()
        indexer = object()

        def prepare(**kwargs):
            self.assertIs(kwargs["decls"], declarations)
            self.assertEqual(kwargs["full_layer_mapping"], {0: 0, 1: 1})
            self.assertEqual(kwargs["transfer_layer_id_max"], 2)
            self.assertEqual(kwargs["transfer_page_size"], 256)
            return prepared

        def build_group(*, config):
            self.assertIs(config, prepared)
            self.assertIs(assembler.build_kv_host_pool(), host)
            self.assertEqual(assembler._get_allocator_type(), "default")
            return SimpleNamespace(
                entries=[
                    SimpleNamespace(name="kv", host_pool=host),
                    SimpleNamespace(name="indexer", host_pool=indexer),
                ]
            )

        assembler = ModuleType(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"
        )
        assembler.build_host_pool_group = build_group
        assembler.build_kv_host_pool = assembler._get_allocator_type = None
        host_config = ModuleType("sglang.srt.mem_cache.hybrid_cache.host_pool_config")
        host_config.prepare_host_pool_config = prepare
        modules[assembler.__name__] = assembler
        modules[host_config.__name__] = host_config
        with (
            patch.dict(sys.modules, modules),
            patch(
                "ucm_toolkit.tools.sglang_model_check.native_probe._host_from_device",
                return_value=host,
            ),
        ):
            declared = NativePools()
            _build_dsa(
                declared,
                model,
                CheckConfig(model="test", page_size=256),
                "cuda",
                "bf16",
                full_layers=(0, 3),
            )
        self.assertIs(declared.pools["indexer"], indexer)
        self.assertEqual(declared.layer_ids["indexer"], (0,))
        self.assertEqual(
            declared.report["host_pool_assembly"], "sglang_host_pool_group"
        )

    """Check provider orchestration, not native kernel/allocator execution."""

    def test_dsv4_discovers_active_regions_and_preserves_native_page_units(self):
        calls = []

        class Host:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                calls.append(kwargs)

        swa = SimpleNamespace(
            kv_buffer=["swa"], bytes_per_page_padded=64, page_size=256
        )
        cache = SimpleNamespace(
            swa_kv_pool=swa,
            _unified_kv=False,
            swa_page_size=256,
            kv_pools={4: SimpleNamespace(), 128: SimpleNamespace(kernel_page_size=16)},
            compress_state_pools=["state", None],
            indexer_compress_state_pools=["index-state", None],
        )
        constructor_calls = []

        def create(**kwargs):
            constructor_calls.append(kwargs)
            return cache

        class Configurator:
            __dataclass_fields__ = {}

            def _build_dsv4_kv_pool(
                self,
                *,
                max_running_requests,
                swa_max_total_num_tokens,
                c4_max_total_num_tokens,
                c128_max_total_num_tokens,
                c4_state_pool_size,
                c128_state_pool_size,
                c4_state_dtype,
                c128_state_dtype,
                req_to_token_pool,
            ):
                return create(
                    device=self.device,
                    compression_ratios=self.model_config.compress_ratios,
                    start_layer=self.layer_info.start_layer,
                    end_layer=self.layer_info.end_layer,
                    layer_num=self.layer_info.num_effective_layers,
                )

        mapping = SimpleNamespace(c4={0: 0}, c128={1: 0}, c4_state_global_layers=[0])
        modules = _module_tree(
            {
                "sglang.srt.mem_cache.deepseek_v4_memory_pool": {
                    "DeepSeekV4TokenToKVPool": create,
                    "select_dsv4_kv_layout": lambda: ("v4", "v4"),
                    "resolve_compress_state_ring_size": lambda ratio: (
                        8 if ratio == 4 else 128
                    ),
                },
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler": {
                    "_resolve_deepseek_v4_layer_mappings": lambda cache: mapping,
                    "_require_single_row_dsv4_swa_pages": lambda **kwargs: None,
                    "_dsv4_page_aligned_only": lambda pool: False,
                    "_dsv4_compressed_region_buffers": lambda cache, ratio: (
                        [f"c{ratio}"],
                        ratio * 64,
                    ),
                    "_dsv4_rope_sibling": lambda cache, ratio: (
                        ("deepseek_v4_c4_rope", ["rope"], 128) if ratio == 4 else None
                    ),
                    "_dsv4_indexer_regions": lambda cache, size: [
                        SimpleNamespace(
                            name=name,
                            device_buffers=[name],
                            item_bytes=16,
                            slot_page_size=size,
                            page_aligned_only=True,
                        )
                        for name in (
                            "deepseek_v4_c4_indexer",
                            "deepseek_v4_c4_indexer_scale",
                        )
                    ],
                    "_dsv4_low_ratio_regions": lambda cache, size: [],
                },
                "sglang.srt.mem_cache.kv_cache_configurator": {
                    "_get_dsv4_compress_state_dtypes": lambda: ("fp32", "fp32"),
                    "KVCacheConfigurator": Configurator,
                },
                "sglang.srt.model_executor.model_runner_components.layer_setup": {
                    "ModelLayerInfo": SimpleNamespace
                },
                "sglang.srt.speculative.spec_info": {
                    "SpeculativeAlgorithm": SimpleNamespace(NONE=None)
                },
                "sglang.srt.mem_cache.memory_pool_host": {
                    "LogicalHostPool": Host,
                    "DeepSeekV4PagedHostPool": Host,
                    "DeepSeekV4StateHostPool": Host,
                },
            }
        )
        model = SimpleNamespace(
            num_hidden_layers=2,
            compress_ratios=[4, 128, 0, 0, 0],
            dtype="bf16",
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
            window_size=128,
            hf_config=SimpleNamespace(),
        )
        bundle = NativePools()
        with patch.dict(sys.modules, modules):
            _build_dsv4(
                bundle, model, CheckConfig(model="test", page_size=256), "cuda", "fp8"
            )
        self.assertEqual(
            set(bundle.pools),
            {
                "kv",
                "swa",
                "deepseek_v4_c4",
                "deepseek_v4_c128",
                "deepseek_v4_c4_rope",
                "deepseek_v4_c4_indexer",
                "deepseek_v4_c4_indexer_scale",
                "deepseek_v4_c4_state",
                "deepseek_v4_c4_indexer_state",
            },
        )
        self.assertNotIn("deepseek_v4_c128_state", bundle.pools)
        self.assertEqual(bundle.pools["deepseek_v4_c128"].kwargs["slot_page_size"], 16)
        self.assertEqual(bundle.pools["deepseek_v4_c4"].kwargs["slot_page_size"], 256)
        self.assertTrue(
            all(not kwargs["pin_memory"] for kwargs in calls if "pin_memory" in kwargs)
        )
        self.assertEqual(constructor_calls[0]["device"], "cpu")
        self.assertEqual(constructor_calls[0]["compression_ratios"], [4, 128])
        calls.clear()
        constructor_calls.clear()
        # Use the same native mapping stub for a second PP stage. Global
        # compression configuration is retained; allocation range is local.
        mapping.c4, mapping.c128, mapping.c4_state_global_layers = {}, {0: 0}, []
        stage = NativePools()
        with patch.dict(sys.modules, modules):
            _build_dsv4(
                stage,
                model,
                CheckConfig(model="test", page_size=256),
                "cuda",
                "fp8",
                layer_range=(1, 2),
                tp_size=2,
            )
        self.assertEqual(constructor_calls[0]["compression_ratios"], [4, 128])
        self.assertEqual(
            (constructor_calls[0]["start_layer"], constructor_calls[0]["end_layer"]),
            (1, 2),
        )
        self.assertEqual(constructor_calls[0]["layer_num"], 1)
        self.assertNotIn("deepseek_v4_c4", stage.pools)
        self.assertNotIn("deepseek_v4_c4_state", stage.pools)
        self.assertEqual(stage.layer_ids["deepseek_v4_c128"], (1,))
        model.compress_ratios = [4]
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(PoolProbeUnavailable, "main model layers"),
        ):
            _build_dsv4(
                NativePools(),
                model,
                CheckConfig(model="test", page_size=256),
                "cuda",
                "fp8",
            )


if __name__ == "__main__":
    unittest.main()
