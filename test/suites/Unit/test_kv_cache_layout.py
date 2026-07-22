import ast
import math
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "ucm_connector.py"


class FakeTensor:
    def __init__(self, ptr: int, block_stride: int, num_blocks: int = 2):
        self._ptr = ptr
        self.shape = (num_blocks, 1, 1, block_stride)

    def __getitem__(self, _index):
        return self

    def data_ptr(self):
        return self._ptr

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return 1


class FakeTorch:
    Tensor = FakeTensor


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, *_args, **_kwargs):
        self.messages.append(_args[0] % _args[1:] if len(_args) > 1 else _args[0])

    def debug(self, *_args, **_kwargs):
        self.messages.append(_args[0] % _args[1:] if len(_args) > 1 else _args[0])


def _extract_layer_index(name: str) -> int:
    match = re.search(r"layers\.(\d+)", name)
    assert match is not None
    return int(match.group(1))


def _load_kv_cache_layout_class():
    source = CONNECTOR_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheLayout"
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "List": List,
        "Tuple": Tuple,
        "extract_layer_index": _extract_layer_index,
        "logger": FakeLogger(),
        "math": math,
        "np": np,
        "torch": FakeTorch,
    }
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace["KVCacheLayout"]


KVCacheLayout = _load_kv_cache_layout_class()


def _build_layout(
    row_strides: list[list[int]],
    use_layerwise: bool,
    kv_cache_config=None,
):
    next_ptr = 0x1000
    kvcaches = {}
    for layer_id, strides in enumerate(row_strides):
        tensors = []
        for stride in strides:
            tensors.append(FakeTensor(next_ptr, stride))
            next_ptr += 0x1000
        kvcaches[f"model.layers.{layer_id}.self_attn"] = tuple(tensors)

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=len(row_strides))
        ),
    )
    kv_cache_config = kv_cache_config or SimpleNamespace(num_blocks=2)
    return KVCacheLayout(
        kvcaches,
        {"use_layerwise": use_layerwise},
        vllm_config,
        kv_cache_config,
    )


class KVCacheLayoutTest(unittest.TestCase):
    def test_direct_layout_flattens_only_real_tensors_without_padding(self):
        layout = _build_layout([[8, 4, 2], [8]], use_layerwise=False)

        self.assertEqual(layout.base_ptrs.shape, (4,))
        self.assertEqual(layout.tensor_size_list, [8, 4, 2, 8])
        self.assertEqual(layout.buffer_sizes.tolist(), [16, 8, 4, 16])
        self.assertTrue(layout.valid_mask.all())
        self.assertEqual(layout.shard_size, 22)
        self.assertEqual(layout.block_size, 22)

        addrs = layout.extract_block_addrs([0, 1])
        self.assertEqual(addrs.shape, (2, 4))
        np.testing.assert_array_equal(
            addrs[1], layout.base_ptrs + layout.block_stride_lists
        )
        with self.assertRaisesRegex(ValueError, "layer_first=True"):
            layout.extract_block_addrs([0], layer_first=True)

    def test_layerwise_layout_accepts_consistent_columns(self):
        cases = [
            (
                [[131072, 16384, 32768], [131072, 16384]],
                [131072, 16384, 32768],
            ),
            (
                [[131072, 16384, 16384, 256], [131072, 16384]],
                [131072, 16384, 16384, 256],
            ),
            ([[83968, 32768], [83968]], [83968, 32768]),
            ([[83968, 16384, 256], [83968]], [83968, 16384, 256]),
        ]

        for row_strides, expected_sizes in cases:
            with self.subTest(row_strides=row_strides):
                layout = _build_layout(row_strides, use_layerwise=True)

                self.assertEqual(layout.base_ptrs.shape, (2, len(expected_sizes)))
                self.assertEqual(layout.tensor_size_list, expected_sizes)
                self.assertEqual(
                    layout.tensor_size_lists.tolist(),
                    [expected_sizes, expected_sizes],
                )

                addrs = layout.extract_block_addrs([0, 1], layer_first=True)
                self.assertEqual(addrs.shape, (2, 2, len(expected_sizes)))
                self.assertTrue(np.all(addrs[1, :, len(row_strides[1]) :] == 0))

    def test_layerwise_layout_rejects_inconsistent_real_strides_in_a_column(self):
        with self.assertRaisesRegex(
            ValueError,
            r"`use_layerwise: true` in ucm_config.yaml:.*"
            r"column 1 must have an identical block stride, but got "
            r"\[16384, 32768\]; stride_to_layer_ids="
            r"\{16384: \[0\], 32768: \[1\]\}.*`use_layerwise: false`",
        ):
            _build_layout(
                [[83968, 16384, 256], [83968, 32768], [83968]],
                use_layerwise=True,
            )

    def test_layerwise_layout_logs_padding_details(self):
        logger = FakeLogger()
        previous_logger = KVCacheLayout.__init__.__globals__["logger"]
        KVCacheLayout.__init__.__globals__["logger"] = logger
        try:
            _build_layout([[8, 4, 2], [8]], use_layerwise=True)
        finally:
            KVCacheLayout.__init__.__globals__["logger"] = previous_logger

        padding_log = next(
            message for message in logger.messages if "uses padding" in message
        )
        self.assertIn("max_tensors_per_layer=3", padding_log)
        self.assertIn("padded_layers=1", padding_log)
        self.assertIn("ghost_slots=2", padding_log)
        self.assertIn("tensor_counts_per_layer=[3, 1]", padding_log)

    def test_logs_effective_sparse_c8_config_per_indexer_layer(self):
        sfa_c8_li_c8_spec = SimpleNamespace(
            sparse_head_dim=(656, 0, 128),
            cache_sparse_sfa_c8=True,
            cache_sparse_li_c8=True,
        )
        sfa_c8_bf16_li_spec = SimpleNamespace(
            sparse_head_dim=(656, 0, 128),
            cache_sparse_sfa_c8=True,
            cache_sparse_li_c8=False,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=2,
            kv_cache_tensors=[SimpleNamespace(size=1024)],
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["model.layers.0.self_attn"],
                    kv_cache_spec=sfa_c8_bf16_li_spec,
                ),
                SimpleNamespace(
                    layer_names=["model.layers.1.self_attn"],
                    kv_cache_spec=sfa_c8_li_c8_spec,
                ),
            ],
        )
        logger = FakeLogger()
        previous_logger = KVCacheLayout.__init__.__globals__["logger"]
        KVCacheLayout.__init__.__globals__["logger"] = logger
        try:
            _build_layout(
                [[8, 4], [8, 4]],
                use_layerwise=True,
                kv_cache_config=kv_cache_config,
            )
        finally:
            KVCacheLayout.__init__.__globals__["logger"] = previous_logger

        summary_log = next(
            message for message in logger.messages if "config summary" in message
        )
        self.assertIn("tensor_sizes=[1024]", summary_log)
        self.assertIn("effective_sfa_c8_counts={False: 0, True: 2}", summary_log)
        self.assertIn("li_c8_enabled_layer_ids=[1]", summary_log)
        self.assertIn("li_c8_disabled_layer_ids=[0]", summary_log)


if __name__ == "__main__":
    unittest.main()
