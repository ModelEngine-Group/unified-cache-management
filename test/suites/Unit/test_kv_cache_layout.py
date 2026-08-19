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
    ):
        self._ptr = ptr
        self._element_size = element_size
        inner_shape = (
            (1, block_stride // element_size)
            if dimensions == 3
            else (1, 1, block_stride // element_size)
        )
        self.shape = (num_blocks, *inner_shape)

    def __getitem__(self, _index):
        return self

    def data_ptr(self):
        return self._ptr

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return self._element_size

    def stride(self, dimension):
        return math.prod(self.shape[dimension + 1 :])


class FakeCombinedTensor:
    def __init__(
        self,
        ptr: int,
        block_stride: int,
        num_blocks: int = 2,
        element_size: int = 1,
    ):
        component_buffer_size = num_blocks * block_stride
        self._components = tuple(
            FakeTensor(
                ptr + component_id * component_buffer_size,
                block_stride,
                num_blocks=num_blocks,
                element_size=element_size,
            )
            for component_id in range(2)
        )
        self.shape = (2, *self._components[0].shape)

    def __getitem__(self, index):
        return self._components[index]

    def dim(self):
        return len(self.shape)


class FakeTorch:
    Tensor = FakeTensor


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
        "_has_shared_indexer_layers",
        "KVCacheSegment",
        "KVCacheTensorInfo",
        "SharedIndexerLayerInfo",
        "KVCacheLayout",
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
                element_size=element_size,
            )
        else:
            kvcaches[layer_name] = FakeTensor(
                next_ptr,
                block_stride,
                element_size=element_size,
                dimensions=dimensions,
            )
        next_ptr += 0x100000

    layer_ids = {_extract_layer_index(name) for name in kvcaches}
    hf_text_config = SimpleNamespace(
        num_hidden_layers=len(layer_ids),
        indexer_types=["full", "shared"],
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
    )
    kv_cache_config = SimpleNamespace(num_blocks=2)
    ucm_config = {"use_layerwise": True}

    layout_globals = SharedIndexerKVCacheLayout.supports.__func__.__globals__
    original_platform = layout_globals["current_platform"]
    layout_globals["current_platform"] = SimpleNamespace(device_type="cuda")
    try:
        layout = SharedIndexerKVCacheLayout(
            kvcaches,
            ucm_config,
            vllm_config,
            kv_cache_config,
        )
    finally:
        layout_globals["current_platform"] = original_platform
    return layout, tensor_ptrs


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

    def test_cuda_tensor_role_pattern_mapping_is_extensible(self):
        class ExtendedCUDAKVCacheLayout(SharedIndexerKVCacheLayout):
            CUDA_TENSOR_ROLE_PATTERNS = {
                "indexer": (
                    *SharedIndexerKVCacheLayout.CUDA_TENSOR_ROLE_PATTERNS["indexer"],
                    re.compile(
                        r"(?:^|\.)selector\.cache(?:\.|$)",
                        re.IGNORECASE,
                    ),
                ),
            }

        self.assertEqual(
            SharedIndexerKVCacheLayout._cuda_cache_role(
                "model.layers.0.self_attn.indexer.k_cache"
            ),
            "indexer",
        )
        self.assertEqual(
            SharedIndexerKVCacheLayout._cuda_cache_role(
                "model.layers.0.self_attn.attn"
            ),
            "attention",
        )

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
        self.assertEqual(layout.buffer_sizes[1].tolist(), [147456, 0])

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

    def test_cuda_combined_5d_attention_splits_kv_segments(self):
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
        layer_0_value_ptr = layer_0_attention_ptr + 2 * 4096
        self.assertEqual(
            layout.base_ptrs[0].tolist(),
            [
                layer_0_attention_ptr,
                layer_0_value_ptr,
                tensor_ptrs[layer_0_indexer],
            ],
        )
        self.assertEqual(layout.block_stride_lists[0].tolist(), expected_sizes)
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
