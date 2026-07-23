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
    ):
        self._ptr = ptr
        self._element_size = element_size
        self.shape = (num_blocks, 1, 1, block_stride // element_size)

    def __getitem__(self, _index):
        return self

    def data_ptr(self):
        return self._ptr

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return self._element_size


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
        "_has_shared_indexer_layers",
        "_supports_ascend_shared_indexer_layout",
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
        "torch": FakeTorch,
        "current_platform": SimpleNamespace(device_type="npu"),
    }
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace


LAYOUT_SYMBOLS = _load_layout_symbols()
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


class KVCacheLayoutTest(unittest.TestCase):
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

    def test_shared_indexer_layout_is_restricted_to_ascend(self):
        supports_globals = SharedIndexerKVCacheLayout.supports.__func__.__globals__
        original_platform = supports_globals["current_platform"]
        supports_globals["current_platform"] = SimpleNamespace(device_type="cuda")
        try:
            layout = _build_layout(
                [[8, 4, 2], [8, 4, 2]],
                use_layerwise=True,
                shared_indexer=True,
            )
        finally:
            supports_globals["current_platform"] = original_platform

        self.assertIs(type(layout), KVCacheLayout)

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
