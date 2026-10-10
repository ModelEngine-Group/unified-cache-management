import ast
import math
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "ucm_connector.py"


class FakeTensor:
    def __init__(
        self,
        ptr: int,
        block_stride: int,
        num_blocks: int = 2,
        element_size: int = 1,
        dimensions: int = 4,
        *,
        shape: tuple[int, ...] | None = None,
        strides: tuple[int, ...] | None = None,
    ):
        self._ptr = ptr
        self._element_size = element_size
        if shape is None:
            inner_shape = (
                (1, block_stride // element_size)
                if dimensions == 3
                else (1, 1, block_stride // element_size)
            )
            shape = (num_blocks, *inner_shape)
        self.shape = shape
        if strides is None:
            running_stride = 1
            reverse_strides = []
            for size in reversed(shape):
                reverse_strides.append(running_stride)
                running_stride *= size
            strides = tuple(reversed(reverse_strides))
        self._strides = strides

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(f"Unsupported fake tensor index: {index!r}")
        return FakeTensor(
            self._ptr + index * self._strides[0] * self._element_size,
            block_stride=1,
            element_size=self._element_size,
            shape=self.shape[1:],
            strides=self._strides[1:],
        )

    def data_ptr(self):
        return self._ptr

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return self._element_size

    def stride(self, dimension):
        return self._strides[dimension]


class FakeCombinedTensor(FakeTensor):
    def __init__(
        self,
        ptr: int,
        block_stride: int,
        num_blocks: int = 3,
        element_size: int = 1,
        *,
        block_major: bool = False,
        kv_stride: int | None = None,
    ):
        component_elements = block_stride // element_size
        if block_major:
            shape = (num_blocks, 2, 1, 1, component_elements)
            if kv_stride is None:
                strides = None
            else:
                strides = (
                    component_elements,
                    kv_stride // element_size,
                    component_elements,
                    component_elements,
                    1,
                )
        else:
            shape = (2, num_blocks, 1, 1, component_elements)
            strides = None
        super().__init__(
            ptr,
            block_stride,
            num_blocks=num_blocks,
            element_size=element_size,
            shape=shape,
            strides=strides,
        )


class FakeTorch:
    Tensor = (FakeTensor, FakeCombinedTensor)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, *_args, **_kwargs):
        self.messages.append(_args[0] % _args[1:] if len(_args) > 1 else _args[0])


def _extract_layer_index(name: str) -> int:
    match = re.search(r"layers\.(\d+)", name)
    assert match is not None
    return int(match.group(1))


def _load_layout_symbols():
    source = CONNECTOR_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    selected_names = {
        "_get_store_gc_block_size",
        "_get_store_io_sizes",
        "_is_minimax_m3",
        "_has_shared_indexer_layers",
        "KVCacheSegment",
        "KVCacheTensorInfo",
        "SharedIndexerLayerInfo",
        "KVCacheLayout",
        "MiniMaxM3KVCacheLayout",
        "SharedIndexerKVCacheLayout",
    }
    selected_nodes = [
        node for node in tree.body if getattr(node, "name", None) in selected_names
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "dataclass": dataclass,
        "extract_layer_index": _extract_layer_index,
        "logger": FakeLogger(),
        "math": math,
        "np": np,
        "re": re,
        "torch": FakeTorch,
        "current_platform": SimpleNamespace(device_type="npu"),
    }
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace


LAYOUT_SYMBOLS = _load_layout_symbols()
get_store_gc_block_size = LAYOUT_SYMBOLS["_get_store_gc_block_size"]
get_store_io_sizes = LAYOUT_SYMBOLS["_get_store_io_sizes"]
KVCacheLayout = LAYOUT_SYMBOLS["KVCacheLayout"]
MiniMaxM3KVCacheLayout = LAYOUT_SYMBOLS["MiniMaxM3KVCacheLayout"]
SharedIndexerKVCacheLayout = LAYOUT_SYMBOLS["SharedIndexerKVCacheLayout"]


def _build_layout(
    row_strides: list[list[int]],
    *,
    use_layerwise: bool,
    shared_indexer: bool = False,
    enable_sparse_sfa_c8: bool = False,
    enable_sparse_li_c8: bool = False,
):
    next_ptr = 0x100000
    kvcaches = {}
    for layer_id, strides in enumerate(row_strides):
        tensors = []
        for stride in strides:
            tensors.append(FakeTensor(next_ptr, stride))
            next_ptr += 0x100000
        kvcaches[f"model.layers.{layer_id}.self_attn"] = tuple(tensors)

    hf_text_config = SimpleNamespace(num_hidden_layers=len(row_strides))
    if shared_indexer:
        hf_text_config.indexer_types = ["full", "shared"]
    vllm_config = SimpleNamespace(
        additional_config={
            "enable_sparse_sfa_c8": enable_sparse_sfa_c8,
            "enable_sparse_li_c8": enable_sparse_li_c8,
        },
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
    )
    kv_cache_config = SimpleNamespace(num_blocks=2)
    ucm_config = {"use_layerwise": use_layerwise}
    layout_cls = (
        SharedIndexerKVCacheLayout
        if SharedIndexerKVCacheLayout.supports(vllm_config, ucm_config)
        else KVCacheLayout
    )
    return layout_cls(kvcaches, ucm_config, vllm_config, kv_cache_config)


def _build_cuda_shared_layout(
    entries: list[tuple[str, int, int] | tuple[str, int, int, int]],
):
    num_blocks = 3
    next_ptr = 0x100000
    kvcaches = {}
    tensor_ptrs = {}
    for entry in entries:
        layer_name, block_stride, element_size, *dimension_values = entry
        dimensions = dimension_values[0] if dimension_values else 3
        tensor_ptrs[layer_name] = next_ptr
        if dimensions == 5:
            kvcaches[layer_name] = FakeCombinedTensor(
                next_ptr,
                block_stride,
                num_blocks=num_blocks,
                element_size=element_size,
            )
        else:
            kvcaches[layer_name] = FakeTensor(
                next_ptr,
                block_stride,
                num_blocks=num_blocks,
                element_size=element_size,
                dimensions=dimensions,
            )
        next_ptr += 0x100000

    return _create_cuda_shared_layout(kvcaches), tensor_ptrs


def _create_cuda_shared_layout(
    kvcaches,
    *,
    num_blocks: int = 3,
    hf_text_config=None,
):
    layer_ids = {_extract_layer_index(name) for name in kvcaches}
    if hf_text_config is None:
        hf_text_config = SimpleNamespace(
            num_hidden_layers=len(layer_ids),
            indexer_types=["full", "shared"],
        )
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
    )
    kv_cache_config = SimpleNamespace(num_blocks=num_blocks)
    ucm_config = {"use_layerwise": True}

    layout_globals = SharedIndexerKVCacheLayout.supports.__func__.__globals__
    original_platform = layout_globals["current_platform"]
    layout_globals["current_platform"] = SimpleNamespace(device_type="cuda")
    try:
        return SharedIndexerKVCacheLayout(
            kvcaches,
            ucm_config,
            vllm_config,
            kv_cache_config,
        )
    finally:
        layout_globals["current_platform"] = original_platform


def _create_minimax_layout(
    kvcaches,
    *,
    use_layerwise: bool = True,
    num_blocks: int = 3,
    platform: str = "cuda",
):
    layer_ids = {_extract_layer_index(name) for name in kvcaches}
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="minimax_m3_text",
                num_hidden_layers=len(layer_ids),
            )
        ),
    )
    layout_globals = MiniMaxM3KVCacheLayout.supports.__func__.__globals__
    original_platform = layout_globals["current_platform"]
    layout_globals["current_platform"] = SimpleNamespace(device_type=platform)
    try:
        return MiniMaxM3KVCacheLayout(
            kvcaches,
            {"use_layerwise": use_layerwise},
            vllm_config,
            SimpleNamespace(num_blocks=num_blocks),
        )
    finally:
        layout_globals["current_platform"] = original_platform


def _create_direct_layout(
    kvcaches,
    *,
    num_blocks: int = 3,
    platform: str = "cuda",
):
    return _create_minimax_layout(
        kvcaches,
        use_layerwise=False,
        num_blocks=num_blocks,
        platform=platform,
    )


class KVCacheLayoutTest(unittest.TestCase):
    def test_store_io_sizes_align_unconditionally(self):
        shard_size, block_size = get_store_io_sizes(
            116992,
            9242368,
        )

        self.assertEqual(shard_size, 118784)
        self.assertEqual(block_size, 9383936)

    def test_store_io_sizes_preserve_aligned_layout(self):
        shard_size, block_size = get_store_io_sizes(
            118784,
            9383936,
        )

        self.assertEqual(shard_size, 118784)
        self.assertEqual(block_size, 9383936)

    def test_yuanrong_posix_gc_uses_compact_persisted_block_size(self):
        block_size = get_store_gc_block_size(
            "YuanRong|Posix",
            [60000, 56992],
            118784,
            9383936,
        )

        self.assertEqual(block_size, 9242368)

    def test_other_pipeline_gc_keeps_aligned_store_block_size(self):
        block_size = get_store_gc_block_size(
            "Cache|Posix",
            [60000, 56992],
            118784,
            9383936,
        )

        self.assertEqual(block_size, 9383936)

    def test_direct_layout_flattens_only_real_tensors(self):
        layout = _build_layout(
            [[8, 4, 2], [8]],
            use_layerwise=False,
            shared_indexer=True,
        )

        self.assertIs(type(layout), KVCacheLayout)
        self.assertEqual(layout.base_ptrs.shape, (4,))
        self.assertEqual(layout.tensor_size_list, [8, 4, 2, 8])
        self.assertEqual(layout.buffer_sizes.tolist(), [16, 8, 4, 16])
        self.assertEqual(layout.shard_size, 22)
        self.assertEqual(layout.block_size, 22)

        addrs = layout.extract_block_addrs([0, 1])
        self.assertEqual(addrs.shape, (2, 4))
        np.testing.assert_array_equal(
            addrs[1], layout.base_ptrs + layout.block_stride_lists
        )

    def test_direct_layout_handles_minimax_m3_three_dimensional_index_cache(self):
        kvcaches = {
            "model.layers.0.self_attn.attn": (
                FakeTensor(0x100000, 32768),
                FakeTensor(0x200000, 32768),
            ),
            "model.layers.0.self_attn.attn.index_cache": (
                FakeTensor(0x300000, 32768, dimensions=3),
            ),
        }
        layout = _create_direct_layout(kvcaches, num_blocks=2, platform="npu")

        self.assertEqual(layout.tensor_size_list, [32768, 32768, 32768])
        self.assertEqual(layout.buffer_sizes.tolist(), [65536, 65536, 65536])

    def test_direct_layout_supports_vllm_024_block_first_5d_cache(self):
        ptr = 0x200000
        layout = _create_direct_layout(
            {
                "model.layers.0.self_attn.attn": FakeCombinedTensor(
                    ptr,
                    4096,
                    num_blocks=4,
                    block_major=True,
                )
            },
            num_blocks=3,
        )

        self.assertEqual(layout.tensor_size_list, [8192])
        self.assertEqual(layout.block_stride_lists.tolist(), [8192])
        self.assertEqual(layout.base_ptrs.tolist(), [ptr])
        self.assertEqual(layout.buffer_sizes.tolist(), [4 * 8192])
        self.assertEqual(layout.extract_block_addrs([2])[0, 0], ptr + 2 * 8192)

    def test_direct_layout_rejects_pre_m3_cuda_kv_major_cache(self):
        with self.assertRaisesRegex(ValueError, r"must use \[num_blocks, 2, \.\.\.\]"):
            _create_direct_layout(
                {
                    "model.layers.0.self_attn.attn": FakeCombinedTensor(
                        0x300000,
                        4096,
                    )
                }
            )

    def test_direct_layout_supports_new_packed_hnd_and_nhd_cache(self):
        num_blocks = 3
        num_heads = 8
        block_size = 16
        packed_head_size = 256
        page_elements = num_heads * block_size * packed_head_size
        shape = (num_blocks, num_heads, block_size, packed_head_size)
        layouts = {
            "hnd": None,
            "nhd": (
                page_elements,
                packed_head_size,
                num_heads * packed_head_size,
                1,
            ),
        }

        for layout_name, strides in layouts.items():
            with self.subTest(layout=layout_name):
                ptr = 0x400000
                tensor = FakeTensor(
                    ptr,
                    block_stride=1,
                    element_size=2,
                    shape=shape,
                    strides=strides,
                )
                layout = _create_direct_layout(
                    {"model.layers.0.self_attn.attn": tensor}
                )
                page_size = page_elements * 2

                self.assertEqual(layout.tensor_size_list, [page_size])
                self.assertEqual(layout.block_stride_lists.tolist(), [page_size])
                self.assertEqual(layout.buffer_sizes.tolist(), [num_blocks * page_size])
                self.assertEqual(
                    layout.extract_block_addrs([2])[0, 0], ptr + 2 * page_size
                )

    def test_direct_layout_allows_more_physical_blocks_than_configured(self):
        ptr = 0x580000
        tensor = FakeTensor(ptr, 1024, num_blocks=5)

        layout = _create_direct_layout(
            {"model.layers.0.self_attn.attn": tensor},
            num_blocks=3,
        )

        self.assertEqual(layout.tensor_size_list, [1024])
        self.assertEqual(layout.block_stride_lists.tolist(), [1024])
        self.assertEqual(layout.buffer_sizes.tolist(), [5 * 1024])
        self.assertEqual(layout.extract_block_addrs([2])[0, 0], ptr + 2 * 1024)

    def test_direct_layout_rejects_fewer_physical_blocks_than_configured(self):
        with self.assertRaisesRegex(
            ValueError,
            r"fewer physical blocks than configured.*minimum=3.*actual=2",
        ):
            _create_direct_layout(
                {
                    "model.layers.0.self_attn.attn": FakeTensor(
                        0x590000,
                        1024,
                        num_blocks=2,
                    )
                },
                num_blocks=3,
            )

    def test_minimax_layerwise_layout_pads_missing_indexer(self):
        kvcaches = {
            "model.layers.0.self_attn.attn": FakeTensor(
                0x700000,
                block_stride=1024,
                num_blocks=3,
            ),
            "model.layers.1.self_attn.attn": FakeTensor(
                0x800000,
                block_stride=1024,
                num_blocks=3,
            ),
            "model.layers.1.self_attn.attn.index_cache": FakeTensor(
                0x900000,
                block_stride=512,
                num_blocks=3,
                dimensions=3,
            ),
        }
        layout = _create_minimax_layout(kvcaches)

        self.assertEqual(layout.tensor_size_lists.tolist(), [[1024, 512], [1024, 512]])
        self.assertEqual(layout.block_stride_lists.tolist(), [[1024, 0], [1024, 512]])
        self.assertEqual(layout.buffer_sizes.tolist(), [[3072, 0], [3072, 1536]])

    def test_generic_layerwise_layout_accepts_regular_matrix(self):
        layout = _build_layout(
            [[131072, 16384, 32768], [131072, 16384, 32768]],
            use_layerwise=True,
        )

        self.assertIs(type(layout), KVCacheLayout)
        self.assertEqual(layout.tensor_size_list, [131072, 16384, 32768])
        self.assertEqual(layout.base_ptrs.shape, (2, 3))

    def test_generic_layerwise_layout_rejects_ragged_rows(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid generic KV cache layout.*every layer must have the same "
            r"tensor count.*SharedIndexerKVCacheLayout",
        ):
            _build_layout(
                [[131072, 16384, 32768], [131072, 16384]],
                use_layerwise=True,
            )

    def test_shared_indexer_config_selects_dedicated_layout(self):
        glm51_layout = _build_layout(
            [[8, 4], [8, 4]],
            use_layerwise=True,
            shared_indexer=False,
        )
        glm52_layout = _build_layout(
            [[8, 4, 2], [8, 4]],
            use_layerwise=True,
            shared_indexer=True,
        )

        self.assertIs(type(glm51_layout), KVCacheLayout)
        self.assertIs(type(glm52_layout), SharedIndexerKVCacheLayout)

    def test_minimax_m3_sparse_config_selects_dedicated_layout(self):
        sparse_config = {
            "use_sparse_attention": True,
            "sparse_attention_freq": [0, 0, 0, *([1] * 57)],
        }
        vllm_config = SimpleNamespace(
            additional_config={},
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    model_type="minimax_m3_text",
                    num_hidden_layers=60,
                    sparse_attention_config=sparse_config,
                )
            ),
        )

        self.assertTrue(MiniMaxM3KVCacheLayout.supports(vllm_config))
        self.assertFalse(
            SharedIndexerKVCacheLayout.supports(vllm_config, {"use_layerwise": True})
        )

    def test_minimax_m3_ascend_layout_handles_three_dimensional_index_cache(self):
        next_ptr = 0x100000
        kvcaches = {}
        indexer_ptrs = {}
        for layer_id in range(60):
            attention_name = f"model.layers.{layer_id}.self_attn.attn"
            kvcaches[attention_name] = (
                FakeTensor(next_ptr, 32768),
                FakeTensor(next_ptr + 0x100000, 32768),
            )
            next_ptr += 0x200000
            if layer_id >= 3:
                indexer_name = f"{attention_name}.index_cache"
                indexer_ptrs[layer_id] = next_ptr
                kvcaches[indexer_name] = (FakeTensor(next_ptr, 32768, dimensions=3),)
                next_ptr += 0x100000
        layout = _create_minimax_layout(kvcaches, num_blocks=2, platform="npu")

        self.assertEqual(layout.tensor_size_list, [32768, 32768, 32768])
        self.assertEqual(layout.base_ptrs.shape, (60, 3))
        np.testing.assert_array_equal(layout.base_ptrs[:3, 2], [0, 0, 0])
        self.assertEqual(layout.base_ptrs[3, 2], indexer_ptrs[3])
        self.assertEqual(layout.base_ptrs[59, 2], indexer_ptrs[59])
        np.testing.assert_array_equal(layout.block_stride_lists[:3, 2], [0, 0, 0])
        self.assertTrue(np.all(layout.block_stride_lists[3:, 2] == 32768))
        self.assertTrue(np.all(layout.buffer_sizes[3:, 2] == 65536))

    def test_minimax_m3_ascend_layout_requires_separate_k_and_v(self):
        with self.assertRaisesRegex(ValueError, r"separate K and V tensors"):
            _create_minimax_layout(
                {
                    "model.layers.0.self_attn.attn": FakeTensor(
                        0x100000,
                        32768,
                    )
                },
                num_blocks=2,
                platform="npu",
            )

    def test_shared_indexer_layout_supports_cuda(self):
        supports_globals = SharedIndexerKVCacheLayout.supports.__func__.__globals__
        original_platform = supports_globals["current_platform"]
        supports_globals["current_platform"] = SimpleNamespace(device_type="cuda")
        try:
            vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(indexer_types=["full", "shared"])
                )
            )
            supported = SharedIndexerKVCacheLayout.supports(
                vllm_config, {"use_layerwise": True}
            )
        finally:
            supports_globals["current_platform"] = original_platform

        self.assertTrue(supported)

    def test_tensor_role_mapping(self):
        self.assertEqual(
            SharedIndexerKVCacheLayout._cache_role(
                "model.layers.0.self_attn.indexer.k_cache"
            ),
            "indexer",
        )
        self.assertEqual(
            MiniMaxM3KVCacheLayout._is_indexer(
                "model.layers.0.self_attn.attn.index_cache"
            ),
            True,
        )
        self.assertEqual(
            SharedIndexerKVCacheLayout._cache_role("model.layers.0.self_attn.attn"),
            "attention",
        )

    def test_ascend_separate_indexer_uses_semantic_order(self):
        indexer = FakeTensor(0x100000, 32768)
        attention_0 = (
            FakeTensor(0x200000, 131072),
            FakeTensor(0x300000, 16384),
        )
        attention_1 = (
            FakeTensor(0x400000, 131072),
            FakeTensor(0x500000, 16384),
        )
        kvcaches = {
            "model.layers.0.self_attn.indexer.k_cache": (indexer,),
            "model.layers.0.self_attn": attention_0,
            "model.layers.1.self_attn": attention_1,
        }
        vllm_config = SimpleNamespace(
            additional_config={},
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=2),
                hf_config=SimpleNamespace(
                    text_config=SimpleNamespace(indexer_types=["full", "shared"])
                ),
            ),
        )

        self.assertTrue(
            SharedIndexerKVCacheLayout.supports(vllm_config, {"use_layerwise": True})
        )
        layout = SharedIndexerKVCacheLayout(
            kvcaches,
            {"use_layerwise": True},
            vllm_config,
            SimpleNamespace(num_blocks=2),
        )

        self.assertEqual(layout.tensor_size_list, [131072, 16384, 32768])
        self.assertEqual(
            layout.base_ptrs[0].tolist(),
            [0x200000, 0x300000, 0x100000],
        )
        self.assertEqual(layout.base_ptrs[1, 2], 0)

    def test_cuda_shared_indexer_uses_semantic_order_and_padding(self):
        layer_2_indexer = "model.layers.2.router.indexer.cache_storage"
        layer_0_indexer = "model.layers.0.self_attn.indexer.k_cache"
        layer_0_attention = "model.layers.0.self_attn.attn"
        layer_1_attention = "model.layers.1.mla.kv_storage"
        layer_2_attention = "model.layers.2.mla.kv_storage"
        entries = [
            (layer_2_indexer, 8448, 1),
            (layer_0_indexer, 8448, 1),
            (layer_0_attention, 73728, 2),
            (layer_1_attention, 73728, 2),
            (layer_2_attention, 73728, 2),
        ]
        layout, tensor_ptrs = _build_cuda_shared_layout(entries)

        self.assertEqual(layout.first_layer_id, 0)
        self.assertEqual(layout.tensor_size_list, [73728, 8448])
        self.assertEqual(layout.shard_size, 82176)
        self.assertEqual(layout.base_ptrs.shape, (3, 2))
        self.assertEqual(
            layout.tensor_size_lists.tolist(),
            [[73728, 8448], [73728, 8448], [73728, 8448]],
        )

        self.assertEqual(
            int(layout.base_ptrs[0, 0]),
            tensor_ptrs[layer_0_attention],
        )
        self.assertEqual(
            int(layout.base_ptrs[0, 1]),
            tensor_ptrs[layer_0_indexer],
        )
        self.assertEqual(
            int(layout.base_ptrs[2, 0]),
            tensor_ptrs[layer_2_attention],
        )
        self.assertEqual(
            int(layout.base_ptrs[2, 1]),
            tensor_ptrs[layer_2_indexer],
        )

        self.assertEqual(layout.base_ptrs[1, 1], 0)
        self.assertEqual(layout.block_stride_lists[1].tolist(), [73728, 0])
        self.assertEqual(layout.buffer_sizes[1].tolist(), [221184, 0])

        block_one_addrs = layout.extract_block_addrs([1], layer_first=True)
        self.assertEqual(
            int(block_one_addrs[0, 0, 0]),
            tensor_ptrs[layer_0_attention] + 73728,
        )
        self.assertEqual(
            int(block_one_addrs[0, 0, 1]),
            tensor_ptrs[layer_0_indexer] + 8448,
        )
        self.assertEqual(block_one_addrs[1, 0, 1], 0)

    def test_cuda_legacy_kv_major_5d_attention_splits_kv_segments(self):
        layer_0_indexer = "model.layers.0.self_attn.indexer.k_cache"
        layer_0_attention = "model.layers.0.self_attn.attn"
        layer_1_attention = "model.layers.1.self_attn.attn"
        entries = [
            (layer_0_indexer, 1024, 1),
            (layer_0_attention, 4096, 2, 5),
            (layer_1_attention, 4096, 2, 5),
        ]
        layout, tensor_ptrs = _build_cuda_shared_layout(entries)

        expected_sizes = [4096, 4096, 1024]
        self.assertEqual(layout.tensor_size_list, expected_sizes)
        self.assertEqual(layout.shard_size, sum(expected_sizes))
        self.assertEqual(layout.base_ptrs.shape, (2, 3))
        self.assertEqual(
            layout.tensor_size_lists.tolist(),
            [expected_sizes, expected_sizes],
        )

        layer_0_attention_ptr = tensor_ptrs[layer_0_attention]
        layer_0_value_ptr = layer_0_attention_ptr + 3 * 4096
        self.assertEqual(
            layout.base_ptrs[0].tolist(),
            [
                layer_0_attention_ptr,
                layer_0_value_ptr,
                tensor_ptrs[layer_0_indexer],
            ],
        )
        self.assertEqual(layout.block_stride_lists[0].tolist(), expected_sizes)
        self.assertEqual(
            layout.buffer_sizes[0].tolist(),
            [3 * 4096, 3 * 4096, 3 * 1024],
        )
        self.assertEqual(layout.base_ptrs[1, 2], 0)
        self.assertEqual(layout.block_stride_lists[1, 2], 0)

        block_one_addrs = layout.extract_block_addrs([1], layer_first=True)
        self.assertEqual(
            int(block_one_addrs[0, 0, 0]),
            layer_0_attention_ptr + 4096,
        )
        self.assertEqual(
            int(block_one_addrs[0, 0, 1]),
            layer_0_value_ptr + 4096,
        )
        self.assertEqual(
            int(block_one_addrs[0, 0, 2]),
            tensor_ptrs[layer_0_indexer] + 1024,
        )
        self.assertEqual(block_one_addrs[1, 0, 2], 0)

    def test_cuda_old_block_major_5d_attention_uses_one_page_segment(self):
        attention_0 = "model.layers.0.self_attn.attn"
        attention_1 = "model.layers.1.self_attn.attn"
        indexer_1 = "model.layers.1.self_attn.attn.index_cache"
        kvcaches = {
            attention_0: FakeCombinedTensor(
                0x100000,
                4096,
                block_major=True,
            ),
            attention_1: FakeCombinedTensor(
                0x200000,
                4096,
                block_major=True,
            ),
            indexer_1: FakeTensor(
                0x300000,
                1024,
                num_blocks=3,
                dimensions=3,
            ),
        }

        layout = _create_minimax_layout(kvcaches)

        self.assertEqual(layout.tensor_size_list, [8192, 1024])
        self.assertEqual(
            layout.tensor_size_lists.tolist(),
            [[8192, 1024], [8192, 1024]],
        )
        self.assertEqual(layout.block_stride_lists[0].tolist(), [8192, 0])
        self.assertEqual(layout.block_stride_lists[1].tolist(), [8192, 1024])
        self.assertEqual(layout.buffer_sizes[0].tolist(), [3 * 8192, 0])
        self.assertEqual(layout.buffer_sizes[1].tolist(), [3 * 8192, 3 * 1024])

    def test_cuda_minimax_m3_packed_hnd_nhd_and_sparse_layer_padding(self):
        num_blocks = 3
        num_heads = 8
        block_size = 16
        packed_head_size = 256
        page_elements = num_heads * block_size * packed_head_size
        page_size = page_elements * 2
        shape = (num_blocks, num_heads, block_size, packed_head_size)
        nhd_strides = (
            page_elements,
            packed_head_size,
            num_heads * packed_head_size,
            1,
        )
        kvcaches = {}
        indexer_ptrs = {}
        for layer_id in range(6):
            attention_name = f"model.layers.{layer_id}.self_attn.attn"
            kvcaches[attention_name] = FakeTensor(
                0x100000 + layer_id * 0x100000,
                block_stride=1,
                element_size=2,
                shape=shape,
                strides=nhd_strides if layer_id % 2 else None,
            )
            if layer_id >= 3:
                indexer_name = f"{attention_name}.index_cache"
                indexer_ptrs[layer_id] = 0x1000000 + layer_id * 0x100000
                kvcaches[indexer_name] = FakeTensor(
                    indexer_ptrs[layer_id],
                    32768,
                    num_blocks=num_blocks,
                    element_size=2,
                    dimensions=3,
                )

        layout = _create_minimax_layout(kvcaches)

        self.assertEqual(layout.tensor_size_list, [page_size, 32768])
        self.assertEqual(layout.base_ptrs.shape, (6, 2))
        self.assertTrue(np.all(layout.block_stride_lists[:, 0] == page_size))
        self.assertTrue(np.all(layout.buffer_sizes[:, 0] == num_blocks * page_size))
        self.assertTrue(np.all(layout.base_ptrs[:3, 1] == 0))
        self.assertTrue(np.all(layout.block_stride_lists[:3, 1] == 0))
        self.assertEqual(layout.base_ptrs[3, 1], indexer_ptrs[3])
        self.assertTrue(np.all(layout.block_stride_lists[3:, 1] == 32768))

    def test_cuda_shared_indexer_rejects_attention_size_mismatch(self):
        entries = [
            ("model.layers.0.self_attn.indexer.k_cache", 8448, 1),
            ("model.layers.0.self_attn.attn", 73728, 2),
            ("model.layers.1.self_attn.attn", 65536, 2),
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"same Attention slot count and per-block sizes.*"
            r"expected=\[73728\] from layer 0.*"
            r"incompatible_layers=\{1: \[65536\]\}",
        ):
            _build_cuda_shared_layout(entries)

    def test_cuda_shared_indexer_rejects_attention_slot_count_mismatch(self):
        entries = [
            ("model.layers.0.self_attn.indexer.k_cache", 1024, 1),
            ("model.layers.0.self_attn.attn", 4096, 2, 5),
            ("model.layers.1.self_attn.attn", 4096, 2),
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"same Attention slot count and per-block sizes.*"
            r"expected=\[4096, 4096\] from layer 0.*"
            r"incompatible_layers=\{1: \[4096\]\}",
        ):
            _build_cuda_shared_layout(entries)

    def test_shared_indexer_li_c8_disabled_uses_padding_without_mask(self):
        layout = _build_layout(
            [[131072, 16384, 32768], [131072, 16384]],
            use_layerwise=True,
            shared_indexer=True,
        )

        self.assertEqual(layout.tensor_size_list, [131072, 16384, 32768])
        self.assertEqual(layout.base_ptrs.shape, (2, 3))
        self.assertEqual(layout.base_ptrs[1, 2], 0)
        self.assertEqual(layout.block_stride_lists[1, 2], 0)
        self.assertEqual(layout.buffer_sizes[1, 2], 0)
        self.assertEqual(layout.extract_block_addrs([1], layer_first=True)[1, 0, 2], 0)

    def test_shared_indexer_sfa_c8_li_c8_disabled_uses_padding(self):
        layout = _build_layout(
            [[83968, 32768], [83968]],
            use_layerwise=True,
            shared_indexer=True,
            enable_sparse_sfa_c8=True,
        )

        self.assertEqual(layout.tensor_size_list, [83968, 32768])
        self.assertEqual(layout.base_ptrs[1, 1], 0)
        self.assertEqual(layout.block_stride_lists[1, 1], 0)

    def test_shared_indexer_all_li_c8_uses_compact_w8a8_layout(self):
        layout = _build_layout(
            [
                [83968, 16384, 256],
                [83968],
                [83968, 16384, 256],
            ],
            use_layerwise=True,
            shared_indexer=True,
            enable_sparse_sfa_c8=True,
            enable_sparse_li_c8=True,
        )

        expected_sizes = [83968, 16384, 256]
        self.assertEqual(layout.tensor_size_list, expected_sizes)
        self.assertEqual(layout.base_ptrs.shape, (3, 3))
        self.assertEqual(
            layout.tensor_size_lists.tolist(),
            [expected_sizes, expected_sizes, expected_sizes],
        )

        self.assertTrue(np.all(layout.base_ptrs[1, 1:] == 0))
        self.assertTrue(np.all(layout.block_stride_lists[1, 1:] == 0))
        self.assertTrue(np.all(layout.buffer_sizes[1, 1:] == 0))

        indexer_ptr = int(layout.base_ptrs[0, 1])
        scale_ptr = int(layout.base_ptrs[0, 2])
        block_one_addrs = layout.extract_block_addrs([1], layer_first=True)
        self.assertEqual(block_one_addrs[0, 0, 1], indexer_ptr + 16384)
        self.assertEqual(block_one_addrs[0, 0, 2], scale_ptr + 256)

    def test_shared_indexer_li_c8_splits_bf16_indexer(self):
        layout = _build_layout(
            [
                [131072, 16384, 16384, 256],
                [131072, 16384, 32768],
                [131072, 16384],
            ],
            use_layerwise=True,
            shared_indexer=True,
            enable_sparse_li_c8=True,
        )

        expected_sizes = [131072, 16384, 16384, 16384, 256]
        self.assertEqual(layout.tensor_size_list, expected_sizes)
        self.assertEqual(
            layout.tensor_size_lists.tolist(),
            [expected_sizes, expected_sizes, expected_sizes],
        )

        bf16_ptr = int(layout.base_ptrs[1, 2])
        self.assertEqual(int(layout.base_ptrs[1, 3]), bf16_ptr + 16384)
        self.assertEqual(layout.block_stride_lists[1, 2:4].tolist(), [32768, 32768])
        self.assertEqual(layout.buffer_sizes[1, 3], 0)

        self.assertEqual(layout.base_ptrs[0, 3], 0)
        self.assertEqual(layout.block_stride_lists[0, 3], 0)
        self.assertEqual(layout.base_ptrs[1, 4], 0)
        self.assertTrue(np.all(layout.base_ptrs[2, 2:] == 0))
        self.assertTrue(np.all(layout.block_stride_lists[2, 2:] == 0))

        block_one_addrs = layout.extract_block_addrs([1], layer_first=True)
        self.assertEqual(block_one_addrs[1, 0, 2], bf16_ptr + 32768)
        self.assertEqual(block_one_addrs[1, 0, 3], bf16_ptr + 16384 + 32768)
        self.assertTrue(np.all(block_one_addrs[2, 0, 2:] == 0))

    def test_shared_indexer_sfa_c8_supports_a5_fp32_scale(self):
        layout = _build_layout(
            [
                [83968, 16384, 512],
                [83968, 32768],
                [83968],
            ],
            use_layerwise=True,
            shared_indexer=True,
            enable_sparse_sfa_c8=True,
            enable_sparse_li_c8=True,
        )

        self.assertEqual(layout.tensor_size_list, [83968, 16384, 16384, 512])
        self.assertEqual(layout.block_stride_lists[1, 1:3].tolist(), [32768, 32768])
        self.assertEqual(layout.block_stride_lists[0, 3], 512)
        self.assertTrue(np.all(layout.base_ptrs[2, 1:] == 0))

    def test_shared_indexer_rejects_non_two_to_one_bf16_indexer(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Cannot split BF16 Indexer tensor.*bf16_size=24576, c8_size=16384",
        ):
            _build_layout(
                [[83968, 16384, 256], [83968, 24576], [83968]],
                use_layerwise=True,
                shared_indexer=True,
                enable_sparse_sfa_c8=True,
                enable_sparse_li_c8=True,
            )


if __name__ == "__main__":
    unittest.main()
