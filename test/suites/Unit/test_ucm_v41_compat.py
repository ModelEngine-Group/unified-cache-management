"""Small dependency-free regression tests for the vLLM KV-cache adapters."""

import ast
import copy
import ctypes
import hashlib
import math
import pickle
import types
import unittest
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
HMA_PATH = ROOT / "ucm/integration/vllm/hma_connector.py"
UCM_PATH = ROOT / "ucm/integration/vllm/ucm_connector.py"
HASHER_PATH = ROOT / "ucm/integration/vllm/request_hasher.py"


def _load_hash_helpers():
    namespace = {
        "hashlib": hashlib,
        "pickle": pickle,
        "Callable": Callable,
        "request_hasher_module": types.SimpleNamespace(
            _layout_source_digest=lambda: "test-layout"
        ),
    }
    for path, name in (
        (HASHER_PATH, "RequestHasher"),
        (UCM_PATH, "_get_sparse_c8_flags"),
        (UCM_PATH, "build_kv_hash_meta"),
        (HMA_PATH, "_build_v41_hash_meta"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if getattr(n, "name", None) == name)
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _hash_config(ratios=(1, 2), cache_dtype="fp8_ds_mla", block_size=64, **hf_fields):
    return types.SimpleNamespace(
        speculative_config=None,
        additional_config={},
        cache_config=types.SimpleNamespace(
            cache_dtype=cache_dtype, block_size=block_size
        ),
        model_config=types.SimpleNamespace(
            model="deepseek",
            dtype="bfloat16",
            hf_config=types.SimpleNamespace(compress_ratios=ratios, **hf_fields),
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
    )


def _load_hma_methods(names):
    tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", None) == "UCMFAWAConnector")
    nodes = [n for n in cls.body if getattr(n, "name", None) in names]
    helpers = [
        n
        for n in tree.body
        if getattr(n, "name", None)
        in (
            "_v41_whole_page_sizes",
            "_ascend_v41_whole_page_sizes",
            "select_transfer_views",
        )
    ]
    module = ast.Module(body=helpers + nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    class GroupMeta:
        def __init__(self, group_id, token_block_size, tail_blocks, tail_tokens):
            self.group_id = group_id
            self.token_block_size = token_block_size
            self.tail_blocks = tail_blocks
            self.tail_tokens = tail_tokens

    namespace = {
        "math": math,
        "torch": types.SimpleNamespace(Tensor=object),
        "np": __import__("numpy"),
        "Optional": Optional,
        "uses_partitioned_cache": lambda: False,
        "cache_core_version": lambda: 0,
        "UcmKVStoreBaseV1": object,
        "round_up": lambda n, a: (n + a - 1) // a * a,
        "KVCacheGroupMeta": GroupMeta,
        "BlockGeometry": namedtuple("BlockGeometry", "logical physical"),
        "KVCacheGroupLayout": object,
        "FAWARequestMeta": types.SimpleNamespace,
        "FAWARequestDispatchMeta": types.SimpleNamespace,
        "UCMFAWAConnectorMetadata": object,
        "SchedulerOutput": object,
        "extract_layer_index": lambda name: int(name.split(".")[1]),
        "logger": types.SimpleNamespace(
            info_once=lambda *_args: None, info=lambda *_args: None
        ),
    }
    store_helpers = [
        n
        for n in ast.parse(UCM_PATH.read_text(encoding="utf-8-sig")).body
        if getattr(n, "name", None) == "_get_store_gc_block_size"
    ]
    store_module = ast.Module(body=store_helpers, type_ignores=[])
    exec(compile(store_module, str(UCM_PATH), "exec"), namespace)
    exec(compile(module, str(HMA_PATH), "exec"), namespace)
    for name in names:
        value = namespace[name]
        if isinstance(value, (staticmethod, classmethod)):
            namespace[name] = value.__func__
    return namespace


def _init_fake_connector(methods, *, ratios, features, resolver):
    spec_full = types.SimpleNamespace(
        block_size=32,
        sliding_window=None,
        tokens_per_state=1,
        page_size_bytes=640,
        prefix_cacheable=True,
    )
    spec_swa = types.SimpleNamespace(
        block_size=32,
        sliding_window=128,
        tokens_per_state=1,
        page_size_bytes=640,
        prefix_cacheable=True,
    )
    spec_transient = types.SimpleNamespace(
        block_size=8,
        sliding_window=None,
        tokens_per_state=1,
        page_size_bytes=128,
        prefix_cacheable=False,
    )
    groups = [
        types.SimpleNamespace(kv_cache_spec=spec_full, layer_names=("l0",)),
        types.SimpleNamespace(kv_cache_spec=spec_swa, layer_names=("swa_cache",)),
        types.SimpleNamespace(kv_cache_spec=spec_transient, layer_names=("ring",)),
    ]
    hf_config = types.SimpleNamespace(compress_ratios=ratios, **features)
    config = types.SimpleNamespace(
        kv_cache_groups=groups,
        model_config=types.SimpleNamespace(hf_config=hf_config),
        speculative_config=None,
    )
    self_obj = types.SimpleNamespace(
        _kv_cache_config=config,
        _vllm_config=config,
        is_ascend_layout=False,
        ascend_base_block_size=None,
        hash_block_size=256,
        transient_group_ids=set(),
        fa_group_ids=[],
        window_group_ids=[],
        group_metas={},
        file_size={},
        max_token_block_size=0,
    )
    if "_derive_v41_file_sizes" in methods:
        self_obj._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], self_obj
        )
    self_obj.can_handle_ascend_kv_cache_config = lambda _config: False
    methods_ns = {"resolve_kv_cache_block_sizes": resolver}
    return self_obj, groups, methods_ns


class UCMV41CompatibilityTest(unittest.TestCase):
    @staticmethod
    def _ascend_v41_connector():
        # Match the 0.30.0rc1 layer-outermost planner: MLA + indexer share
        # each page, while ring and SWA groups alias it with different IDs.
        ns = types.SimpleNamespace
        mla_type = type("AscendMLAAttentionSpec", (ns,), {})
        swa_type = type("AscendSlidingWindowMLASpec", (ns,), {})
        ring_type = type("CircularBufferSpec", (ns,), {})
        full, windows, tensors = {}, {}, []
        for layer, ratio in ((0, 1), (1, 2)):
            prefix = f"model.layers.{layer}.self_attn"
            kv, index, swa = (
                prefix + suffix
                for suffix in (".long_kv_cache", ".indexer.k_cache", ".swa_cache")
            )
            for name in (kv, index):
                full[name] = mla_type(
                    block_size=128,
                    tokens_per_state=ratio,
                    model_version="deepseek_v41",
                    prefix_cacheable=True,
                )
            windows[swa] = swa_type(
                block_size=128,
                sliding_window=256,
                tokens_per_state=1,
                model_version="deepseek_v41",
                prefix_cacheable=True,
            )
            tensors.append(
                ns(
                    layers=[kv, index, swa],
                    layer_stride=0,
                    offset=0,
                    block_stride=262144,
                )
            )
        ring_name = "model.layers.1.self_attn.compressor.state_cache"
        ring = ring_type(block_size=32, prefix_cacheable=False)
        tensors[1].layers.append(ring_name)
        groups = [
            ns(
                layer_names=list(full),
                kv_cache_spec=ns(
                    block_size=128,
                    prefix_cacheable=True,
                    kv_cache_specs=full,
                ),
            ),
            ns(layer_names=[ring_name], kv_cache_spec=ring),
            ns(
                layer_names=list(windows),
                kv_cache_spec=ns(
                    block_size=128,
                    prefix_cacheable=True,
                    kv_cache_specs=windows,
                ),
            ),
        ]
        config = ns(
            kv_cache_groups=groups,
            kv_cache_tensors=tensors,
            # Default upstream placement labels do not describe Ascend views.
            kv_cache_layout="BLHNC",
        )
        methods = _load_hma_methods(
            {
                "_init_group_metas",
                "_specs_for_group",
                "_derive_v41_file_sizes",
            }
        )
        connector = ns(
            _kv_cache_config=config,
            _vllm_config=ns(model_config=ns(hf_config=ns(compress_ratios=[1, 2]))),
            is_ascend_layout=False,
            ascend_base_block_size=None,
            hash_block_size=256,
            group_metas={},
            max_token_block_size=0,
            can_handle_ascend_kv_cache_config=lambda _: True,
            _specs_for_group=methods["_specs_for_group"],
        )
        connector._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], connector
        )
        return methods, connector

    def test_ascend_v41_page_metadata_survives_scheduler_spec_flattening(self):
        methods, worker = self._ascend_v41_connector()
        scheduler = copy.deepcopy(worker)
        for group in scheduler._kv_cache_config.kv_cache_groups:
            nested = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            if nested:
                group.kv_cache_spec = next(iter(nested.values()))
        for connector in (worker, scheduler):
            methods["_init_group_metas"](connector)
            self.assertEqual(connector.hash_block_size, 128)  # Not C4 * 128.
            self.assertEqual(connector.resume_token_alignment, 2)
            self.assertEqual(connector.fa_group_ids, [0])
            self.assertEqual(connector.window_group_ids, [2])
            self.assertEqual(connector.transient_group_ids, {1})
            self.assertEqual(connector.file_size, {"FA": 524288, "WA": 1048576})
            self.assertEqual(len(connector.whole_page_sizes), 4)
        self.assertEqual(worker.whole_page_sizes, scheduler.whole_page_sizes)

    def test_ascend_v41_rounds_window_snapshots_to_complete_pages(self):
        methods, connector = self._ascend_v41_connector()
        group = connector._kv_cache_config.kv_cache_groups[2]
        for spec in group.kv_cache_spec.kv_cache_specs.values():
            spec.sliding_window = 129
        methods["_init_group_metas"](connector)
        self.assertEqual(connector.group_metas[2].tail_tokens, 256)
        self.assertEqual(connector.group_metas[2].tail_blocks, 2)

    def test_ascend_v41_rejects_unsupported_replay_or_page_ownership(self):
        for change in (
            "replay",
            "disabled_transfer",
            "missing_page",
            "folded",
            "layer_stride",
            "context_parallel",
        ):
            with self.subTest(change=change):
                methods, connector = self._ascend_v41_connector()
                cfg = connector._kv_cache_config
                if change == "replay":
                    cfg.kv_cache_groups[2].kv_cache_spec.prefix_replay_tokens = 256
                elif change == "disabled_transfer":
                    cfg.kv_cache_groups[2].enable_kv_transfer = False
                elif change == "missing_page":
                    cfg.kv_cache_tensors.pop()
                elif change == "folded":
                    folded = "model.layers.0.self_attn.indexer.k_cache_folded"
                    cfg.kv_cache_groups[0].layer_names.append(folded)
                    cfg.kv_cache_tensors[0].layers.append(folded)
                    cfg.kv_cache_groups[0].kv_cache_spec.kv_cache_specs[folded] = next(
                        iter(
                            cfg.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values()
                        )
                    )
                elif change == "layer_stride":
                    cfg.kv_cache_tensors[0].layer_stride = 128
                else:
                    connector._vllm_config.parallel_config = types.SimpleNamespace(
                        decode_context_parallel_size=2
                    )
                with self.assertRaises(ValueError):
                    methods["_init_group_metas"](connector)

    def test_ascend_v41_registers_one_page_owner_and_roundtrips_scales(self):
        methods, connector = self._ascend_v41_connector()
        methods["_init_group_metas"](connector)
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        layout_node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            dtype = "bfloat16"

            def __init__(self, page_bytes):
                self.storage = (ctypes.c_ubyte * (3 * page_bytes))()
                self.shape = (3, 64, 1, 512)
                self.page_bytes = page_bytes

            def dim(self):
                return 4

            def __getitem__(self, _):
                return self

            def data_ptr(self):
                return ctypes.addressof(self.storage)

            def element_size(self):
                return 2

            def stride(self, index=None):
                strides = (self.page_bytes // 2, 512, 512, 1)
                return strides if index is None else strides[index]

        namespace = dict(
            methods,
            torch=types.SimpleNamespace(Tensor=Tensor),
            Sequence=Sequence,
            Tuple=tuple,
            extract_layer_index=lambda name: int(name.split(".")[2]),
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[layout_node], type_ignores=[])
        )
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        register = _load_hma_methods({"register_kv_caches"})
        register.update(namespace)
        register["create_device"] = lambda: object()
        register["_use_ucm_connector_cpu_affinity"] = lambda: False
        connector.group_layouts = {}
        connector._create_fa_store = lambda *_: None
        connector._create_wa_store = lambda *_: None
        # The indexer and ring need no separate transfer view. If registration
        # tries to treat them as owners, these sentinels fail immediately.
        caches = {
            name: None
            for group in connector._kv_cache_config.kv_cache_groups
            for name in group.layer_names
        }
        for name, size in connector.whole_page_sizes.items():
            caches[name] = Tensor(size)
        register["register_kv_caches"](connector, caches)
        self.assertEqual(set(connector.group_layouts), {0, 2})
        layout = connector.group_layouts[0]
        self.assertEqual(len(layout.base_ptrs), 2)
        self.assertEqual(layout.segment_tensor_size_list(128, 128), [262144] * 2)
        np = __import__("numpy")
        src = layout.extract_addrs(np.array([1]))
        dst = layout.extract_addrs(np.array([2]))
        for col, size in enumerate(layout.segment_tensor_size_list(128, 128)):
            # Cover main KV, the separate indexer/scales, and page padding.
            pattern = bytes(range(256)) * (size // 256)
            ctypes.memmove(int(src[0, col]), pattern, size)
            ctypes.memmove(int(dst[0, col]), int(src[0, col]), size)
            self.assertEqual(ctypes.string_at(int(dst[0, col]), size), pattern)
            self.assertEqual(
                ctypes.string_at(int(layout.base_ptrs[col]), size), bytes(size)
            )
        with self.assertRaises(ValueError):
            layout.segment_tensor_size_list(64, 128)
        with self.assertRaises(ValueError):
            layout.extract_addrs_with_offsets(np.array([1]), 128, np.array([64]))
        bad_view = Tensor(131072)
        with self.assertRaisesRegex(ValueError, "page stride mismatch"):
            namespace["KVCacheGroupLayout"](
                {"model.layers.0.self_attn.long_kv_cache": bad_view},
                is_ascend_layout=True,
                whole_page_sizes_by_layer={
                    "model.layers.0.self_attn.long_kv_cache": 262144
                },
            )

    def test_v41_external_load_waits_for_compute_before_submitting_copies(self):
        methods = _load_hma_methods(["start_load_kv"])
        methods["UCMFAWAConnectorMetadata"] = types.SimpleNamespace
        for is_v41, has_hit, has_window in (
            (True, True, True),
            (True, False, True),
            (False, True, True),
            (True, True, False),
        ):
            with self.subTest(is_v41=is_v41, has_hit=has_hit, has_window=has_window):
                events = []
                request = types.SimpleNamespace(
                    load_keys=[b"key"] if has_hit else [],
                    load_hash_start=0,
                    load_hash_end=1,
                    load_vllm_block_ids=[[1]],
                )
                metadata = types.SimpleNamespace(request_meta={"req": request})

                def submit(_request_id, kind, *_args):
                    events.append(kind)
                    return kind

                connector = types.SimpleNamespace(
                    is_v41_layout=is_v41,
                    device=types.SimpleNamespace(
                        synchronize=lambda: events.append("sync")
                    ),
                    _get_connector_metadata=lambda: metadata,
                    fa_store=object(),
                    window_group_ids=[1] if has_window else [],
                    wa_store=object() if has_window else None,
                    _extract_fa_ptr=lambda *_args: [],
                    _extract_wa_ptr=lambda *_args: [],
                    _submit_load_task=submit,
                    _wait_all_load_task=lambda tasks: events.append(tuple(tasks)),
                )
                methods["start_load_kv"](connector, None)
                expected = ["sync"] if is_v41 and has_hit else []
                if has_hit:
                    labels = ["FA", "WA"] if has_window else ["FA"]
                    expected.extend([*labels, tuple(labels)])
                else:
                    expected.append(())
                self.assertEqual(events, expected)

    def test_spec_formats_select_packed_value_scale_descriptors(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if getattr(n, "name", None) == "_group_physical_layout_by_layer"
        )
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        group = types.SimpleNamespace(
            layer_names=(
                "model.layers.0.self_attn",
                "model.layers.1.self_attn.indexer.k_cache",
            ),
            kv_cache_spec=types.SimpleNamespace(
                kv_cache_specs={
                    "model.layers.0.self_attn": types.SimpleNamespace(
                        state_content_size_bytes=584,
                        cache_dtype_str="fp8_ds_mla",
                    ),
                    "model.layers.1.self_attn.indexer.k_cache": types.SimpleNamespace(
                        state_content_size_bytes=68,
                        cache_dtype_str=None,
                    ),
                }
            ),
        )
        self.assertEqual(
            namespace["_group_physical_layout_by_layer"](group),
            {
                "model.layers.0.self_attn": (576, 8),
                "model.layers.1.self_attn.indexer.k_cache": (64, 4),
            },
        )
        main_spec = group.kv_cache_spec.kv_cache_specs["model.layers.0.self_attn"]
        for cache_dtype, width, expected in (
            ("fp8_ds_mla", 528, (512, 16)),
            ("nvfp4_ds_mla", 288, (256, 32)),
        ):
            main_spec.cache_dtype_str = cache_dtype
            main_spec.state_content_size_bytes = width
            self.assertEqual(
                namespace["_group_physical_layout_by_layer"](group)[
                    "model.layers.0.self_attn"
                ],
                expected,
            )

    def test_packed_v41_page_roundtrip_preserves_values_and_scales(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            dtype = "uint8"

            def __init__(self, storage, states, page_bytes, c_bytes=584):
                self.storage = storage
                self.shape = (2, 1, states, c_bytes)
                self.page_bytes = page_bytes

            def __getitem__(self, _index):
                return self

            def data_ptr(self):
                return ctypes.addressof(self.storage)

            def dim(self):
                return 4

            def element_size(self):
                return 1

            def stride(self, index=None):
                values = (self.page_bytes, self.page_bytes, self.shape[3], 1)
                return values if index is None else values[index]

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "torch": types.SimpleNamespace(Tensor=Tensor),
            "Tuple": tuple,
            "Optional": Optional,
            "Sequence": Sequence,
            "extract_layer_index": lambda name: int(name.split(".")[2]),
            "logger": types.SimpleNamespace(info=lambda *_args: None),
        }
        exec(compile(module, str(HMA_PATH), "exec"), namespace)

        for value_bytes, scale_bytes, c_bytes in (
            (576, 8, 584),  # Main fp8_ds_mla.
            (512, 16, 528),  # SM100 MXFP8.
            (256, 32, 288),  # SM100 NVFP4 compressed KV.
            (128, 4, 132),  # Indexer FP8.
            (64, 4, 68),  # Indexer MXFP4.
        ):
            with self.subTest(value_bytes=value_bytes, scale_bytes=scale_bytes):
                page_strides = {states: states * c_bytes + 128 for states in (64, 32)}
                source_storages = {
                    states: (ctypes.c_ubyte * (2 * stride))()
                    for states, stride in page_strides.items()
                }
                dest_storages = {
                    states: (ctypes.c_ubyte * (2 * stride))()
                    for states, stride in page_strides.items()
                }
                for states, storage in source_storages.items():
                    for i in range(len(storage)):
                        storage[i] = (i * (states + 11) + 7) % 251
                for storage in dest_storages.values():
                    for i in range(len(storage)):
                        storage[i] = 0xCD

                source = {
                    "model.layers.0.self_attn": Tensor(
                        source_storages[64], 64, page_strides[64], c_bytes
                    ),
                    "model.layers.1.self_attn": Tensor(
                        source_storages[32], 32, page_strides[32], c_bytes
                    ),
                }
                destination = {
                    "model.layers.0.self_attn": Tensor(
                        dest_storages[64], 64, page_strides[64], c_bytes
                    ),
                    "model.layers.1.self_attn": Tensor(
                        dest_storages[32], 32, page_strides[32], c_bytes
                    ),
                }
                physical_layout = {name: (value_bytes, scale_bytes) for name in source}
                common_kwargs = dict(
                    expected_block_size=64,
                    standardized_layout=True,
                    expected_tokens_per_state_by_layer={
                        "model.layers.0.self_attn": 1,
                        "model.layers.1.self_attn": 2,
                    },
                    physical_layout_by_layer=physical_layout,
                )
                source_layout = namespace["KVCacheGroupLayout"](source, **common_kwargs)
                destination_layout = namespace["KVCacheGroupLayout"](
                    destination, **common_kwargs
                )
                block_ids = __import__("numpy").array([1, 1])
                offsets = __import__("numpy").array([0, 32])
                source_addresses = source_layout.extract_addrs_with_offsets(
                    block_ids, 64, offsets
                )
                destination_addresses = destination_layout.extract_addrs_with_offsets(
                    block_ids, 64, offsets
                )
                sizes = source_layout.segment_tensor_size_list(32, 64)
                self.assertEqual(
                    sizes,
                    [
                        32 * value_bytes,
                        32 * scale_bytes,
                        16 * value_bytes,
                        16 * scale_bytes,
                    ],
                )

                # Transfer one hash half at a time.  The other half must stay
                # untouched in the destination page.
                for row, offset in enumerate((0, 32)):
                    for storage in dest_storages.values():
                        for i in range(len(storage)):
                            storage[i] = 0xCD
                    for col, size in enumerate(sizes):
                        saved_bytes = ctypes.string_at(
                            int(source_addresses[row, col]), size
                        )
                        ctypes.memmove(
                            int(destination_addresses[row, col]),
                            saved_bytes,
                            size,
                        )

                    for states, source_storage, destination_storage in (
                        (64, source_storages[64], dest_storages[64]),
                        (32, source_storages[32], dest_storages[32]),
                    ):
                        stride = page_strides[states]
                        block_base = stride
                        scale_base = block_base + states * value_bytes
                        state_offset = offset * states // 64
                        selected_states = states // 2
                        value_start = block_base + state_offset * value_bytes
                        value_end = value_start + selected_states * value_bytes
                        scale_start = scale_base + state_offset * scale_bytes
                        scale_end = scale_start + selected_states * scale_bytes
                        expected = bytearray([0xCD]) * len(destination_storage)
                        expected[value_start:value_end] = bytes(
                            source_storage[value_start:value_end]
                        )
                        expected[scale_start:scale_end] = bytes(
                            source_storage[scale_start:scale_end]
                        )
                        self.assertEqual(bytes(destination_storage), bytes(expected))

                # The 0.31 adapter copies padded pages as opaque byte ranges.
                whole_kwargs = dict(
                    common_kwargs,
                    whole_page_sizes_by_layer={
                        name: page_strides[tensor.shape[2]]
                        for name, tensor in source.items()
                    },
                )
                src = namespace["KVCacheGroupLayout"](source, **whole_kwargs)
                dst = namespace["KVCacheGroupLayout"](destination, **whole_kwargs)
                ids = __import__("numpy").array([1])
                sizes = src.segment_tensor_size_list(64, 64)
                self.assertEqual(sizes, [page_strides[64], page_strides[32]])
                for storage in dest_storages.values():
                    for i in range(len(storage)):
                        storage[i] = 0xCD
                for col, size in enumerate(sizes):
                    ctypes.memmove(
                        int(dst.extract_addrs(ids)[0, col]),
                        int(src.extract_addrs(ids)[0, col]),
                        size,
                    )
                for states, stride in page_strides.items():
                    self.assertEqual(
                        bytes(dest_storages[states][:stride]), b"\xcd" * stride
                    )
                    self.assertEqual(
                        bytes(dest_storages[states][stride:]),
                        bytes(source_storages[states][stride:]),
                    )
                with self.assertRaises(ValueError):
                    src.segment_tensor_size_list(32, 64)
                with self.assertRaises(ValueError):
                    src.extract_addrs_with_offsets(
                        ids, 64, __import__("numpy").array([32])
                    )

    def test_v41_hash_meta_isolates_physical_layout(self):
        namespace = _load_hash_helpers()
        empty_layout = types.SimpleNamespace(kv_cache_groups=[])

        def hasher(config, kv_config):
            return namespace["RequestHasher"](
                namespace["_build_v41_hash_meta"](config, 0, kv_config)
            )

        methods, ascend = self._ascend_v41_connector()
        # Ascend V4.1 can retain legacy-looking ratios. The resolved FAWA
        # layout, rather than a second model-config heuristic, selects hashing.
        ascend._vllm_config = _hash_config((4, 128))
        methods["_init_group_metas"](ascend)
        self.assertTrue(ascend.is_v41_layout)
        explicit = hasher(ascend._vllm_config, ascend._kv_cache_config)
        for marker in (
            b"kv_layout=v41_packed_v3",
            b"ascend_v41_page_transfer=v1",
            b"transfer_layout=v4",
            b"placement=",
        ):
            self.assertIn(marker, explicit.meta_bytes)
        self.assertNotEqual(
            explicit.meta_bytes,
            namespace["build_kv_hash_meta"](ascend._vllm_config, 0).encode(),
        )
        v41 = hasher(_hash_config(), empty_layout)
        base_config = _hash_config()
        base = hasher(base_config, empty_layout)
        self.assertIn(b":dcp=1:layout=test-layout", base.meta_bytes)
        base_config.parallel_config.decode_context_parallel_size = 2
        self.assertNotEqual(base.seed, hasher(base_config, empty_layout).seed)
        base_config.parallel_config.decode_context_parallel_size = 1
        namespace["request_hasher_module"]._layout_source_digest = (
            lambda: "changed-layout"
        )
        try:
            self.assertNotEqual(base.seed, hasher(base_config, empty_layout).seed)
        finally:
            namespace["request_hasher_module"]._layout_source_digest = (
                lambda: "test-layout"
            )
        self.assertNotEqual(
            v41.meta_bytes,
            hasher(_hash_config(cache_dtype="bfloat16"), empty_layout).meta_bytes,
        )
        self.assertNotEqual(
            v41.meta_bytes,
            hasher(_hash_config(block_size=256), empty_layout).meta_bytes,
        )
        self.assertNotEqual(
            v41.meta_bytes,
            hasher(_hash_config((1,)), empty_layout).meta_bytes,
        )

        spec = types.SimpleNamespace(
            block_size=64,
            tokens_per_state=1,
            state_content_size_bytes=584,
            prefix_replay_tokens=0,
        )
        group = types.SimpleNamespace(kv_cache_spec=spec, layer_names=("l0",))
        kv_config = types.SimpleNamespace(kv_cache_groups=[group])
        fingerprint = hasher(_hash_config(), kv_config).meta_bytes
        self.assertNotEqual(
            fingerprint, hasher(_hash_config(), empty_layout).meta_bytes
        )
        spec.state_content_size_bytes = 528
        self.assertNotEqual(fingerprint, hasher(_hash_config(), kv_config).meta_bytes)
        spec.state_content_size_bytes = 584
        spec.prefix_replay_tokens = 128
        self.assertNotEqual(fingerprint, hasher(_hash_config(), kv_config).meta_bytes)
        spec.prefix_replay_tokens = 0
        kv_config.kv_cache_tensors = [
            types.SimpleNamespace(
                layers=["l0"], layer_stride=37440, block_stride=37440, offset=0
            )
        ]
        group.kv_cache_spec = types.SimpleNamespace(
            block_size=64, kv_cache_specs={"l0": spec}
        )
        worker = hasher(_hash_config(), kv_config)
        group.kv_cache_spec = spec
        scheduler = hasher(_hash_config(), kv_config)
        self.assertEqual(worker.meta_bytes, scheduler.meta_bytes)
        self.assertEqual(worker.seed, scheduler.seed)
        ascend_spec = type("AscendMLAAttentionSpec", (types.SimpleNamespace,), {})(
            **vars(spec), model_version="deepseek_v41"
        )
        group.kv_cache_spec = ascend_spec
        ascend_hasher = hasher(_hash_config(), kv_config)
        self.assertIn(b"ascend_v41_page_transfer=v1", ascend_hasher.meta_bytes)
        self.assertNotEqual(worker.meta_bytes, ascend_hasher.meta_bytes)
        group.kv_cache_spec = types.SimpleNamespace(
            block_size=64, kv_cache_specs={"l0": ascend_spec}
        )
        self.assertEqual(
            ascend_hasher.meta_bytes, hasher(_hash_config(), kv_config).meta_bytes
        )
        self.assertEqual(ascend_hasher.seed, hasher(_hash_config(), kv_config).seed)

    def test_v41_meta_preserves_packed_layout_encoding(self):
        namespace = _load_hash_helpers()
        config = _hash_config()
        kv_config = types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    layer_names=("l0",),
                    kv_cache_spec=types.SimpleNamespace(
                        block_size=64,
                        tokens_per_state=2,
                        state_content_size_bytes=584,
                        dtype="uint8",
                        kv_quant_mode="fp8",
                    ),
                )
            ]
        )
        # Preserve the V4.1 namespace encoding from #1507 across the refactor.
        expected = (
            "deepseek:1:bfloat16:0:sfa_c8=0:li_c8=0:dcp=1:layout=test-layout"
            ":kv_layout=v41_packed_v3:cache_dtype=fp8_ds_mla:block_size=64"
            ":compress_ratios=1,2"
            ":transfer_layout=v4:((64, True, 0, True, "
            "(('l0', 2, 584, 'uint8', 'fp8'),)),)"
        )
        meta = namespace["_build_v41_hash_meta"](config, 0, kv_config)
        self.assertEqual(meta, expected)
        self.assertEqual(
            namespace["RequestHasher"](meta).seed,
            namespace["RequestHasher"](expected).seed,
        )

    def test_v41_meta_requires_cache_config(self):
        namespace = _load_hash_helpers()
        config = _hash_config()
        del config.cache_config
        kv_config = types.SimpleNamespace(kv_cache_groups=[])
        with self.assertRaisesRegex(ValueError, "needs cache_config"):
            namespace["_build_v41_hash_meta"](config, 0, kv_config)
        # The generic KV builder remains usable without V4.1-specific settings.
        self.assertNotIn("kv_layout=", namespace["build_kv_hash_meta"](config, 0))

    def test_v41_store_uses_per_segment_sizes_even_with_legacy_tensor_size(self):
        method = _load_hma_methods({"_create_store"})["_create_store"]
        method.__globals__["KVConnectorRole"] = types.SimpleNamespace(WORKER="worker")
        method.__globals__["UcmConnectorFactoryV1"] = types.SimpleNamespace(
            create_connector=lambda _name, config, _path: config
        )
        published_sizes = []
        connector = types.SimpleNamespace(
            _role="worker",
            _base_store_config=lambda _suffix: (
                "fake",
                None,
                {"tensor_size": 584},
            ),
            _set_default_shm_buffer_capacity=lambda _config: None,
            _summarize_store_config=lambda config: config,
            _publish_block_size=lambda size, store_suffix: published_sizes.append(
                (size, store_suffix)
            ),
            device_id=0,
            file_size={"FA": 4096},
            is_v41_layout=True,
            is_mla=False,
            tp_size=1,
        )
        config = method(connector, "FA", "fa", [576, 8])
        self.assertNotIn("tensor_size", config)
        self.assertEqual(config["tensor_size_list"], [576, 8])
        self.assertEqual(published_sizes, [(4096, "fa")])

    def test_wa_load_reserve_default_preserves_fa_and_explicit_configuration(self):
        method = _load_hma_methods({"_create_store"})["_create_store"]
        method.__globals__["KVConnectorRole"] = types.SimpleNamespace(WORKER="worker")
        method.__globals__["UcmConnectorFactoryV1"] = types.SimpleNamespace(
            create_connector=lambda _name, config, _path: config
        )
        cases = (
            ("FA", 2, None, None),
            ("WA", 2, None, 512),
            ("WA", 0, None, None),
            ("WA", 2, 256, 256),
            ("WA", 2, 2048, 2048),
            ("FA", 2, 256, 256),
            ("FA", 1, None, None),
            ("WA", 1, None, 512),
            ("WA", 1, 2048, 2048),
        )
        for label, version, explicit, expected in cases:
            with self.subTest(label=label, version=version, explicit=explicit):
                method.__globals__["uses_partitioned_cache"] = lambda: version == 2
                method.__globals__["cache_core_version"] = lambda: version
                supplied = {
                    "cache_buffer_capacity_gb": 10,
                    "io_direct": False,
                    "share_buffer_enable": False,
                }
                if explicit is not None:
                    supplied["cache_load_exclusive_buffer_number"] = explicit
                shard_size = (3 if label == "FA" else 6) << 20
                connector = types.SimpleNamespace(
                    _role="worker",
                    _base_store_config=lambda _suffix: ("fake", None, dict(supplied)),
                    _get_world_size=lambda: 1,
                    _configure_numa_placement=lambda _config: None,
                    _set_default_shm_buffer_capacity=lambda _config: None,
                    _summarize_store_config=lambda config: config,
                    _publish_block_size=lambda *_args, **_kwargs: None,
                    device_id=0,
                    file_size={label: shard_size},
                    is_v41_layout=False,
                    is_mla=True,
                    tp_size=1,
                )
                config = method(connector, label, label.lower(), [shard_size])
                self.assertEqual(
                    config.get("cache_load_exclusive_buffer_number"), expected
                )
                self.assertEqual(config["cache_buffer_capacity_gb"], 10)
                self.assertFalse(config["io_direct"])
                self.assertFalse(config["share_buffer_enable"])

    def test_route_checks_fawa_before_hla(self):
        connector = UCM_PATH.read_text(encoding="utf-8")
        # The lite branch runs its own FAWA/HLA probe, so scope the check to the
        # main path; otherwise that earlier branch satisfies it by accident.
        main_path = connector[connector.index("use_inference_duration_monitor") :]
        self.assertLess(
            main_path.index("is_fawa = UCMFAWAConnector.can_handle_kv_cache_config"),
            main_path.index(
                "UCMHybridLinearAttentionConnector.supports_kv_cache_layout"
            ),
        )

    def test_ascend_dsv4_registration_keeps_native_token_dimension(self):
        """Descriptor.layers does not turn native Ascend [B,N,H,C] into [B,H,N,C]."""
        methods = _load_hma_methods({"register_kv_caches"})
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            dtype = "test"
            device = "npu:0"

            def __init__(self, width, element_bytes, page_bytes, offset=0):
                self.shape = (2, 128, 1, width)
                self.element_bytes = element_bytes
                self.strides = (page_bytes // element_bytes, width, width, 1)
                self.offset = offset

            def __getitem__(self, _index):
                return types.SimpleNamespace(
                    data_ptr=self.data_ptr,
                    numel=lambda: math.prod(self.shape[1:]),
                    is_contiguous=lambda: True,
                )

            def data_ptr(self):
                return 4096 + self.offset

            def untyped_storage(self):
                return types.SimpleNamespace(data_ptr=lambda: 4096)

            def dim(self):
                return 4

            def element_size(self):
                return self.element_bytes

            def stride(self, index=None):
                return self.strides if index is None else self.strides[index]

        def reject_gpu_layout(_):
            raise AssertionError(
                "Native DSV4 indexer must not use V4.1 packed descriptors"
            )

        methods.update(
            torch=types.SimpleNamespace(Tensor=Tensor),
            Tuple=tuple,
            Sequence=Sequence,
            create_device=lambda: object(),
            extract_layer_index=lambda name: int(name.split(".")[2]),
            _use_ucm_connector_cpu_affinity=lambda: False,
            _group_tokens_per_state_by_layer=reject_gpu_layout,
            _group_physical_layout_by_layer=reject_gpu_layout,
        )
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(HMA_PATH), "exec"), methods)
        main = "model.layers.2.attn"
        indexer = "model.layers.2.indexer.k_cache"
        group = types.SimpleNamespace(
            layer_names=[main, indexer],
            kv_cache_spec=types.SimpleNamespace(block_size=512),
        )
        connector = types.SimpleNamespace(
            _kv_cache_config=types.SimpleNamespace(
                kv_cache_groups=[group],
                kv_cache_tensors=[types.SimpleNamespace(layers=group.layer_names)],
            ),
            is_ascend_layout=True,
            is_ascend_v41_layout=False,
            transient_group_ids=set(),
            whole_page_sizes={},
            group_layouts={},
            ascend_block_geometry=lambda _: types.SimpleNamespace(
                logical=512, physical=128
            ),
            _create_fa_store=lambda *_: None,
            _create_wa_store=lambda *_: None,
        )
        for include_full_view in (False, True):
            # rc2 A2 exposes the first two views; older compressed-cache
            # profiles may also expose the overlapping complete byte page.
            views = [Tensor(128, 1, 16640), Tensor(1, 2, 16640, offset=16384)]
            expected = [131072, 16384, 256]
            if include_full_view:
                views.append(Tensor(130, 1, 16640))
                expected = [131072, 16640]
            methods["register_kv_caches"](
                connector, {main: Tensor(512, 2, 131072), indexer: tuple(views)}
            )
            layout = connector.group_layouts[0]
            self.assertFalse(layout.standardized_layout)
            self.assertEqual(layout.tensor_block_size, 128)
            self.assertEqual(layout.segment_tensor_size_list(512, 512), expected)

    def test_standardized_layout_uses_state_dimension(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            shape = (2, 1, 32, 4)
            dtype = "bf16"

            def __getitem__(self, _index):
                return self

            def data_ptr(self):
                return 4096

            def dim(self):
                return 4

            def element_size(self):
                return 1

            def stride(self, index=None):
                values = (128, 128, 4, 1)
                return values if index is None else values[index]

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "torch": types.SimpleNamespace(Tensor=Tensor),
            "Tuple": tuple,
            "Optional": Optional,
            "Sequence": Sequence,
            "extract_layer_index": lambda name: int(name.split(".")[2]),
            "logger": types.SimpleNamespace(info=lambda *_args: None),
        }
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        layout = namespace["KVCacheGroupLayout"](
            {"model.layers.0.swa": Tensor()},
            expected_block_size=32,
            standardized_layout=True,
            expected_tokens_per_state=1,
        )
        self.assertEqual(layout.block_strides.tolist(), [128])
        self.assertEqual(layout.segment_tensor_size_list(32, 32), [128])

    def test_ratio_two_standardized_segment_scales_physical_states(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            shape = (2, 1, 32, 4)
            dtype = "bf16"

            def __getitem__(self, _index):
                return self

            def data_ptr(self):
                return 4096

            def dim(self):
                return 4

            def element_size(self):
                return 1

            def stride(self, index=None):
                values = (128, 128, 4, 1)
                return values if index is None else values[index]

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "torch": types.SimpleNamespace(Tensor=Tensor),
            "Tuple": tuple,
            "Optional": Optional,
            "Sequence": Sequence,
            "extract_layer_index": lambda name: int(name.split(".")[2]),
            "logger": types.SimpleNamespace(info=lambda *_args: None),
        }
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        layout = namespace["KVCacheGroupLayout"](
            {"model.layers.0.mla": Tensor()},
            expected_block_size=64,
            standardized_layout=True,
            expected_tokens_per_state=2,
        )
        self.assertEqual(layout.segment_tensor_size_list(32, 64), [64])

    def test_mixed_tokens_per_state_views_scale_offsets_independently(self):
        tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body if getattr(n, "name", None) == "KVCacheGroupLayout"
        )

        class Tensor:
            dtype = "bf16"

            def __init__(self, states, ptr):
                self.shape = (2, 1, states, 4)
                self.ptr = ptr

            def __getitem__(self, _index):
                return self

            def data_ptr(self):
                return self.ptr

            def dim(self):
                return 4

            def element_size(self):
                return 1

            def stride(self, index=None):
                values = (self.shape[1] * self.shape[2] * 4, self.shape[2] * 4, 4, 1)
                return values if index is None else values[index]

        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "torch": types.SimpleNamespace(Tensor=Tensor),
            "Tuple": tuple,
            "Optional": Optional,
            "Sequence": Sequence,
            "extract_layer_index": lambda name: int(name.split(".")[2]),
            "logger": types.SimpleNamespace(info=lambda *_args: None),
        }
        exec(compile(module, str(HMA_PATH), "exec"), namespace)
        layout = namespace["KVCacheGroupLayout"](
            {
                "model.layers.0.full": Tensor(64, 4096),
                "model.layers.1.compressed": Tensor(32, 8192),
            },
            expected_block_size=64,
            standardized_layout=True,
            expected_tokens_per_state_by_layer={
                "model.layers.0.full": 1,
                "model.layers.1.compressed": 2,
            },
        )
        addresses = layout.extract_addrs_with_offsets(
            __import__("numpy").array([0, 0]),
            64,
            __import__("numpy").array([0, 32]),
        )
        self.assertEqual(layout.segment_tensor_size_list(32, 64), [128, 64])
        self.assertEqual(addresses.tolist(), [[4096, 8192], [4224, 8256]])

    def test_transient_slice_keeps_original_group_index(self):
        methods = _load_hma_methods({"_slice_group_block_ids"})
        self_obj = types.SimpleNamespace(
            transient_group_ids={2},
            window_group_ids=[1],
            group_metas={2: types.SimpleNamespace(token_block_size=8)},
        )
        self_obj._slice_group_block_ids = types.MethodType(
            methods["_slice_group_block_ids"], self_obj
        )
        self.assertEqual(
            self_obj._slice_group_block_ids(
                2, [10, 11], __import__("numpy").array([31]), False
            ),
            [],
        )
        dispatch = [[], [], [], [], []]
        dispatch[2] = self_obj._slice_group_block_ids(
            2, [10, 11], __import__("numpy").array([31]), False
        )
        self.assertEqual(len(dispatch), 5)
        self.assertEqual(dispatch[2], [])

    def test_fawa_full_prompt_lookup_leaves_final_hash_block_to_compute(self):
        methods = _load_hma_methods(
            {"get_num_new_matched_tokens", "_lookup_external_hit_blocks"}
        )
        methods["get_num_new_matched_tokens"].__globals__[
            "FAWARequestMeta"
        ] = types.SimpleNamespace

        def run(
            num_tokens,
            is_v41,
            load_threshold,
            num_computed,
            prefix,
            reverse,
        ):
            class Rank:
                @staticmethod
                def lookup_on_prefix(_store, keys):
                    return min(prefix, len(keys) - 1)

                @staticmethod
                def lookup_on_reverse(_store, keys):
                    return min(reverse, len(keys) - 1)

            request_meta = {}
            self_obj = types.SimpleNamespace(
                hash_block_size=32,
                is_v41_layout=is_v41,
                persist_token_threshold=0,
                load_tokens_threshold=load_threshold,
                _seed=b"seed",
                requests_meta=request_meta,
                fa_store=object(),
                wa_store=object(),
                window_group_ids=[1],
                _rank_consistency=Rank(),
                group_metas={},
                _record_counter=lambda *_args: None,
                _prefetch_hit_key_hotness=lambda *_args: None,
                request_block_hasher=lambda request: [
                    bytes([index]) for index in range(len(request.all_token_ids) // 32)
                ],
            )
            request = types.SimpleNamespace(
                request_id="req",
                num_tokens=num_tokens,
                all_token_ids=list(range(num_tokens)),
            )
            method = types.MethodType(methods["get_num_new_matched_tokens"], self_obj)
            self_obj._lookup_external_hit_blocks = types.MethodType(
                methods["_lookup_external_hit_blocks"], self_obj
            )
            matched, need_load = method(request, num_computed)
            return matched, need_load, request_meta["req"]

        matched, need_load, meta = run(128, True, 0, 0, 3, 3)
        self.assertFalse(need_load)
        self.assertEqual(matched, 96)
        self.assertEqual(meta.total_hit_block_num, 3)
        self.assertEqual(meta.token_processed, 96)

        matched, need_load, meta = run(128, True, 0, 0, 3, 1)
        self.assertFalse(need_load)
        self.assertEqual(matched, 64)  # WA at 64 exists; WA at 96 misses.
        self.assertEqual(meta.total_hit_block_num, 2)
        self.assertEqual(meta.token_processed, 64)

        matched, _, meta = run(129, True, 0, 0, 3, 3)
        self.assertEqual(matched, 128)
        self.assertEqual(meta.total_hit_block_num, 4)

        matched, _, meta = run(128, False, 0, 0, 3, 3)
        self.assertEqual(matched, 127)
        self.assertEqual(meta.total_hit_block_num, 4)

        matched, _, meta = run(129, True, 96, 32, 2, 2)
        self.assertEqual(matched, 0)
        self.assertEqual(meta.token_processed, 32)

    def test_fawa_short_window_slice_clips_before_first_block(self):
        methods = _load_hma_methods({"_slice_group_block_ids"})
        self_obj = types.SimpleNamespace(
            transient_group_ids=set(),
            window_group_ids={0},
            group_metas={
                0: types.SimpleNamespace(
                    token_block_size=32, tail_blocks=4, tail_tokens=128
                )
            },
        )
        method = types.MethodType(methods["_slice_group_block_ids"], self_obj)
        self.assertEqual(
            method(0, [10, 11], __import__("numpy").array([31]), False),
            [10, 10, 10, 10],
        )
        self.assertEqual(
            method(0, [10, 11, 12, 13], __import__("numpy").array([31, 63, 127]), True),
            [10, 10, 10, 10, 10, 10, 10, 11, 10, 11, 12, 13],
        )
        self.assertEqual(
            method(0, [10, 11, 12, 13], __import__("numpy").array([63]), False),
            [10, 10, 10, 11],
        )
        with self.assertRaises(RuntimeError):
            method(0, [10, 11, 12, 13], __import__("numpy").array([159]), False)

    def test_dynamic_size_scales_physical_group_block(self):
        methods = _load_hma_methods({"_specs_for_group", "_derive_v41_file_sizes"})
        spec = types.SimpleNamespace(block_size=64, page_size_bytes=640)
        group = types.SimpleNamespace(kv_cache_spec=spec, layer_names=("l0",))
        self_obj = types.SimpleNamespace(
            transient_group_ids=set(),
            fa_group_ids=[0],
            window_group_ids=[],
            group_metas={0: types.SimpleNamespace(tail_tokens=128, tail_blocks=2)},
        )
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], self_obj
        )
        self.assertEqual(
            self_obj._derive_v41_file_sizes([group]), {"FA": 4096, "WA": 0}
        )

    def test_v41_model_features_select_resolved_hash_without_ratio_two(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        resolver = lambda _cache_config, _vllm_config: (64, 32)
        methods["_init_group_metas"].__globals__[
            "resolve_kv_cache_block_sizes"
        ] = resolver
        self_obj, groups, _ = _init_fake_connector(
            methods,
            ratios=[0, 1, 1],
            features={"kv_source_layer_ids": (1,), "index_source_layer_ids": (1,)},
            resolver=resolver,
        )
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        self_obj._init_group_metas()
        self.assertTrue(self_obj.is_v41_layout)
        self.assertEqual(self_obj.hash_block_size, 32)
        self.assertEqual(sorted(self_obj.transient_group_ids), [2])
        self.assertEqual(self_obj.group_metas[2].tail_blocks, 0)

    def test_bounded_replay_excludes_window_snapshots_and_keeps_group_indices(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        methods["resolve_kv_cache_block_sizes"] = lambda *_args: (32, 32)
        connector, groups, _ = _init_fake_connector(
            methods, ratios=[0, 1, 2], features={}, resolver=None
        )
        groups[1].kv_cache_spec.prefix_cacheable = False
        groups[1].kv_cache_spec.prefix_replay_tokens = 128
        connector._specs_for_group = methods["_specs_for_group"]
        methods["_init_group_metas"](connector)
        self.assertEqual(connector.fa_group_ids, [0])
        self.assertEqual(connector.window_group_ids, [])
        self.assertEqual(connector.transient_group_ids, {1, 2})
        self.assertEqual(connector.prefix_replay_tokens, 128)
        self.assertEqual(connector.file_size["WA"], 0)
        self.assertEqual(set(connector.group_metas), {0, 1, 2})

    def test_padded_page_sizes_survive_scheduler_flattening_of_mixed_specs(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        methods["resolve_kv_cache_block_sizes"] = lambda *_args: (64, 32)
        main = types.SimpleNamespace(block_size=64, tokens_per_state=1)
        indexer = types.SimpleNamespace(block_size=64, tokens_per_state=2)
        names = ("l0", "l1.indexer.k_cache")
        worker_spec = types.SimpleNamespace(
            block_size=64, kv_cache_specs=dict(zip(names, (main, indexer)))
        )
        for spec in (worker_spec, main):
            connector, groups, _ = _init_fake_connector(
                methods, ratios=[0, 1, 2], features={}, resolver=None
            )
            groups[0].kv_cache_spec = spec
            groups[0].layer_names = names
            groups[1].kv_cache_spec.prefix_cacheable = False
            groups[1].kv_cache_spec.prefix_replay_tokens = 128
            connector._kv_cache_config.kv_cache_layout = "BLHNC"
            connector._kv_cache_config.kv_cache_tensors = [
                types.SimpleNamespace(layers=[name], layer_stride=size)
                for name, size in zip(names, (37440, 4608))
            ]
            connector._specs_for_group = methods["_specs_for_group"]
            methods["_init_group_metas"](connector)
            self.assertEqual(connector.hash_block_size, 64)
            self.assertEqual(connector.file_size, {"FA": 45056, "WA": 0})

    def test_striped_fa_uses_absolute_hash_positions_for_token_offsets(self):
        import numpy as np

        methods = _load_hma_methods({"_extract_fa_ptr"})
        layout = types.SimpleNamespace(
            extract_addrs_with_offsets=lambda ids, _size, offsets: np.column_stack(
                (ids, offsets)
            )
        )
        connector = types.SimpleNamespace(
            fa_group_ids=[0],
            group_layouts={0: layout},
            group_metas={0: types.SimpleNamespace(token_block_size=128)},
            hash_block_size=32,
        )
        result = methods["_extract_fa_ptr"](
            connector,
            [b"a", b"b"],
            3,
            11,
            ([101, 102],),
            hash_positions=np.array([6, 10]),
        )
        self.assertEqual(result.tolist(), [[101, 64], [102, 64]])

    def test_fa_only_prefix_can_be_found_and_saved_without_a_window_store(self):
        methods = _load_hma_methods(
            {"_create_wa_store", "_lookup_external_hit_blocks", "wait_for_save"}
        )
        methods["UCMFAWAConnectorMetadata"] = types.SimpleNamespace
        methods["ucmmetrics"] = types.SimpleNamespace(update_stats=lambda _stats: None)
        keys = [bytes([i]) for i in range(8)]
        for tp_size, rank in ((1, 0), (8, 0), (8, 7)):
            with self.subTest(tp_size=tp_size, rank=rank):
                saved = []

                def submit(label, _store, row_keys, *_args):
                    saved.append((label, row_keys))
                    return types.SimpleNamespace(key_count=len(row_keys))

                request = types.SimpleNamespace(
                    dump_keys=keys,
                    dump_hash_start=0,
                    dump_hash_end=8,
                    dump_vllm_block_ids=(list(range(8)), []),
                )
                connector = types.SimpleNamespace(
                    window_group_ids=[],
                    fa_group_ids=[0],
                    fa_store=object(),
                    wa_store=None,
                    _rank_consistency=types.SimpleNamespace(
                        lookup_on_prefix=lambda _store, keys: len(keys) - 1
                    ),
                    _get_connector_metadata=lambda: types.SimpleNamespace(
                        request_meta={"req": request}
                    ),
                    _poll_completed_dump_tasks=lambda: None,
                    _get_mla_dump_slice=lambda start, rank=rank, size=tp_size: slice(
                        (rank - start) % size, None, size
                    ),
                    _extract_fa_ptr=lambda keys, *_args, **_kwargs: __import__(
                        "numpy"
                    ).zeros((len(keys), 2), dtype="uint64"),
                    _get_dump_event_handle=lambda: 1,
                    _submit_dump_task=submit,
                    tp_size=tp_size,
                    tp_rank=rank,
                    tp_dump_tasks={},
                    file_size={"FA": 4096, "WA": 0},
                )
                self.assertIsNone(methods["_create_wa_store"](connector, None))
                self.assertEqual(
                    methods["_lookup_external_hit_blocks"](connector, keys), 8
                )
                methods["wait_for_save"](connector)
                self.assertEqual(saved, [("FA", keys[rank::tp_size])])

    def test_replay_chunks_do_not_dump_past_the_scheduler_computed_position(self):
        methods = _load_hma_methods(
            {
                "build_connector_meta",
                "_generate_dispatch_meta",
                "_slice_group_block_ids",
            }
        )
        methods["UCMFAWAConnectorMetadata"] = lambda requests, preempted: (
            types.SimpleNamespace(request_meta=requests, preempted_req_ids=preempted)
        )
        request = types.SimpleNamespace(
            ucm_block_ids=[bytes([i]) for i in range(8)],
            total_hit_block_num=4,
            hbm_hit_block_num=0,
            token_processed=256,
            num_token_ids=512,
            vllm_block_ids=(),
        )
        connector = types.SimpleNamespace(
            prefix_replay_tokens=128,
            hash_block_size=64,
            transient_group_ids={1},
            window_group_ids=[],
            group_metas={0: types.SimpleNamespace(token_block_size=64), 1: object()},
            requests_meta={"req": request},
            wa_dump_block_wise=True,
        )
        for name in ("_generate_dispatch_meta", "_slice_group_block_ids"):
            setattr(connector, name, types.MethodType(methods[name], connector))
        output = types.SimpleNamespace(
            scheduled_new_reqs=[
                types.SimpleNamespace(
                    req_id="req",
                    block_ids=(list(range(8)), []),
                    num_computed_tokens=128,
                )
            ],
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={"req": 64},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
        dispatch = methods["build_connector_meta"](connector, output).request_meta[
            "req"
        ]
        self.assertEqual(dispatch.load_keys, request.ucm_block_ids[:4])
        self.assertEqual(dispatch.dump_keys, [])
        self.assertEqual(request.token_processed, 192)
        output.scheduled_new_reqs = []
        for computed, expected in ((192, []), (256, [bytes([4])])):
            output.scheduled_cached_reqs = types.SimpleNamespace(
                req_ids=["req"],
                new_block_ids=[None],
                resumed_req_ids=set(),
                num_computed_tokens=[computed],
            )
            dispatch = methods["build_connector_meta"](connector, output).request_meta[
                "req"
            ]
            self.assertEqual(dispatch.dump_keys, expected)
            self.assertEqual(request.token_processed, computed + 64)

    def test_invalid_resolver_result_fails_closed(self):
        methods = _load_hma_methods({"_init_group_metas"})
        methods["_init_group_metas"].__globals__["resolve_kv_cache_block_sizes"] = (
            lambda _cache_config, _vllm_config: [32, 32]
        )
        self_obj, _groups, _ = _init_fake_connector(
            methods,
            ratios=[0, 1],
            features={"kv_source_layer_ids": (1,)},
            resolver=lambda *_args: [32, 32],
        )
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        with self.assertRaisesRegex(RuntimeError, "Unable to resolve"):
            self_obj._init_group_metas()

    def test_v40_topology_keeps_legacy_hash_and_file_sizes(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        self_obj, groups, _ = _init_fake_connector(
            methods, ratios=[0, 4, 128], features={}, resolver=None
        )
        groups[2].kv_cache_spec.prefix_cacheable = True
        for group in groups:
            group.kv_cache_spec.block_size = 256
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        self_obj._init_group_metas()
        self.assertFalse(self_obj.is_v41_layout)
        self.assertEqual(self_obj.hash_block_size, 256)
        expected_fa = ((37376 + 8448) * 21 + 1168 * 20 + 4095) // 4096 * 4096
        self.assertEqual(self_obj.file_size["FA"], expected_fa)

    def test_transient_groups_do_not_select_v41_transfer(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        methods["_init_group_metas"].__globals__["resolve_kv_cache_block_sizes"] = (
            lambda *_: (256, 256)
        )
        for flag in ("prefix_cacheable", "enable_kv_transfer"):
            with self.subTest(flag=flag):
                connector, groups, _ = _init_fake_connector(
                    methods, ratios=[0, 4, 128], features={}, resolver=None
                )
                for group in groups:
                    group.kv_cache_spec.block_size = 256
                    group.kv_cache_spec.prefix_cacheable = True
                target = (
                    groups[2].kv_cache_spec if flag == "prefix_cacheable" else groups[2]
                )
                setattr(target, flag, False)
                connector._specs_for_group = methods["_specs_for_group"]
                methods["_init_group_metas"](connector)
                self.assertFalse(connector.is_v41_layout)
                self.assertEqual(connector.transient_group_ids, {2})
                self.assertEqual(connector.hash_block_size, 256)
                self.assertNotIn(2, connector.fa_group_ids)

    def test_nested_ascend_detection_is_independent_of_member_order(self):
        methods = _load_hma_methods(
            {
                "can_handle_kv_cache_config",
                "can_handle_ascend_kv_cache_config",
                "_specs_for_group",
            }
        )
        cls = types.SimpleNamespace(_specs_for_group=methods["_specs_for_group"])
        cls.can_handle_ascend_kv_cache_config = types.MethodType(
            methods["can_handle_ascend_kv_cache_config"], cls
        )
        swa = type("AscendSlidingWindowMLASpec", (types.SimpleNamespace,), {})()
        for members in ((object(), swa), (swa, object())):
            group = types.SimpleNamespace(
                layer_names=[0, 1],
                kv_cache_spec=types.SimpleNamespace(
                    kv_cache_specs=dict(enumerate(members))
                ),
            )
            config = types.SimpleNamespace(kv_cache_groups=[group])
            self.assertTrue(cls.can_handle_ascend_kv_cache_config(config))
            self.assertTrue(methods["can_handle_kv_cache_config"](cls, config))

    def test_nested_c4_detection_validates_every_members_geometry(self):
        methods = _load_hma_methods(
            {
                "_get_ascend_base_block_size",
                "_specs_for_group",
                "ascend_block_geometry",
                "ascend_compression_ratio",
            }
        )
        methods["ascend_block_geometry"].__globals__.update(
            _ascend_block_size_is_logical=lambda: False,
        )
        cls = types.SimpleNamespace(
            _specs_for_group=methods["_specs_for_group"],
            ascend_compression_ratio=methods["ascend_compression_ratio"],
            ASCEND_C4_COMPRESS_RATIO=4,
            ASCEND_SUPPORTED_VLLM_BLOCK_SIZES={32, 64, 128},
        )
        cls.ascend_block_geometry = types.MethodType(
            methods["ascend_block_geometry"], cls
        )
        members = {
            "first": types.SimpleNamespace(block_size=128, compress_ratio=128),
            "second": types.SimpleNamespace(block_size=128, compress_ratio=4),
        }
        group = types.SimpleNamespace(
            layer_names=list(members),
            kv_cache_spec=types.SimpleNamespace(block_size=128, kv_cache_specs=members),
        )
        config = types.SimpleNamespace(kv_cache_groups=[group])
        with self.assertRaisesRegex(ValueError, "must share one block geometry"):
            methods["_get_ascend_base_block_size"](cls, config)
        members["first"].compress_ratio = 4
        self.assertEqual(methods["_get_ascend_base_block_size"](cls, config), 128)

    def test_real_v41_flash_topology_keeps_nine_groups(self):
        methods = _load_hma_methods(
            {"_init_group_metas", "_specs_for_group", "_derive_v41_file_sizes"}
        )
        groups = []
        swa_layers = []
        for group_id, layer_count in enumerate((6, 6, 6, 6, 6, 5, 5)):
            names = tuple(
                f"model.layers.{group_id}_{i}.swa_cache" for i in range(layer_count)
            )
            swa_layers.extend(names)
            spec = types.SimpleNamespace(
                block_size=32,
                sliding_window=128,
                tokens_per_state=1,
                page_size_bytes=19008,
                prefix_cacheable=True,
            )
            groups.append(types.SimpleNamespace(kv_cache_spec=spec, layer_names=names))
        csa_names = tuple(f"model.layers.source_{i}" for i in range(4))
        index_names = tuple(f"model.layers.index_{i}" for i in range(4))
        all_fa_names = csa_names + index_names
        inner_specs = {
            name: types.SimpleNamespace(
                block_size=64,
                tokens_per_state=(1 if i < 4 else 2),
                page_size_bytes=(2000 if i < 4 else 1000),
                prefix_cacheable=True,
            )
            for i, name in enumerate(all_fa_names)
        }
        fa_spec = types.SimpleNamespace(
            block_size=64,
            kv_cache_specs=inner_specs,
            prefix_cacheable=True,
        )
        groups.append(
            types.SimpleNamespace(kv_cache_spec=fa_spec, layer_names=all_fa_names)
        )
        ring_spec = types.SimpleNamespace(
            block_size=8, tokens_per_state=1, prefix_cacheable=False
        )
        groups.append(
            types.SimpleNamespace(kv_cache_spec=ring_spec, layer_names=("ring",))
        )
        hf_config = types.SimpleNamespace(
            compress_ratios=[0] * 40,
            kv_source_layer_ids=(2, 8, 14, 20),
            index_source_layer_ids=(2, 8, 14, 20),
        )
        config = types.SimpleNamespace(
            kv_cache_groups=groups,
            model_config=types.SimpleNamespace(hf_config=hf_config),
            speculative_config=None,
        )
        self_obj = types.SimpleNamespace(
            _kv_cache_config=config,
            _vllm_config=config,
            is_ascend_layout=False,
            ascend_base_block_size=None,
            hash_block_size=256,
            transient_group_ids=set(),
            fa_group_ids=[],
            window_group_ids=[],
            group_metas={},
            file_size={},
            max_token_block_size=0,
        )
        methods["_init_group_metas"].__globals__["resolve_kv_cache_block_sizes"] = (
            lambda *_args: (64, 32)
        )
        self_obj._specs_for_group = methods["_specs_for_group"]
        self_obj._derive_v41_file_sizes = types.MethodType(
            methods["_derive_v41_file_sizes"], self_obj
        )
        self_obj._init_group_metas = types.MethodType(
            methods["_init_group_metas"], self_obj
        )
        self_obj.can_handle_ascend_kv_cache_config = lambda _config: False
        self_obj._init_group_metas()
        self.assertEqual(len(self_obj.group_metas), 9)
        self.assertEqual(self_obj.transient_group_ids, {8})
        self.assertEqual(self_obj.fa_group_ids, [7])
        self.assertEqual(self_obj.window_group_ids, list(range(7)))
        self.assertEqual(self_obj.hash_block_size, 32)
        self.assertEqual(self_obj.group_metas[8].token_block_size, 8)
        self.assertEqual(self_obj.group_metas[8].tail_blocks, 0)
        expected_fa = (4 * 2000 + 4 * 1000) * 32 // 64
        expected_fa = (expected_fa + 4095) // 4096 * 4096
        expected_wa = (40 * 19008 * 4 + 4095) // 4096 * 4096
        self.assertEqual(self_obj.file_size, {"FA": expected_fa, "WA": expected_wa})


if __name__ == "__main__":
    unittest.main()
