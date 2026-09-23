"""Full-view transfer tests; CPU parts run without the UCM/vLLM runtime.

    python -m unittest discover -s test/suites/Unit/connector -p test_ucm_fawa_fullview.py -v
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[4]

# Execute the real helper from hma_connector without importing the
# GPU/vLLM connector module.
_hma_tree = ast.parse(
    (ROOT / "ucm/integration/vllm/hma_connector.py").read_text(encoding="utf-8-sig")
)
_hma_ns = {}
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in _hma_tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "select_transfer_views"
            ],
            type_ignores=[],
        ),
        "hma_select_transfer_views",
        "exec",
    ),
    _hma_ns,
)
layout = NS(select_transfer_views=_hma_ns["select_transfer_views"])


class FullViewTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy needed for byte-level alias tests")
        self.np = np

    def views(self, block_size, padding=0):
        np = self.np
        stride = block_size * 132 + padding
        root = np.arange(stride * 3 + 16, dtype=np.int64).astype(np.uint8)

        class Tensor:
            device = "cpu"

            def __init__(self, array):
                self.array = array
                self.shape = array.shape

            def dim(self):
                return self.array.ndim

            def stride(self, dim=None):
                strides = tuple(
                    value // self.element_size() for value in self.array.strides
                )
                return strides if dim is None else strides[dim]

            @property
            def dtype(self):
                return self.array.dtype

            def element_size(self):
                return self.array.itemsize

            def data_ptr(self):
                return self.array.ctypes.data

            def untyped_storage(self):
                return NS(data_ptr=lambda: root.ctypes.data)

            def __getitem__(self, item):
                return Tensor(self.array[item])

            def numel(self):
                return self.array.size

            def is_contiguous(self):
                return self.array.flags.c_contiguous

        key = Tensor(
            np.ndarray(
                (3, block_size, 1, 128),
                dtype=np.uint8,
                buffer=root,
                offset=16,
                strides=(stride, 128, 128, 1),
            )
        )
        scale = Tensor(
            np.ndarray(
                (3, block_size, 1, 1),
                dtype=np.float32,
                buffer=root,
                offset=16 + block_size * 128,
                strides=(stride, 4, 4, 4),
            )
        )
        full = Tensor(
            np.ndarray(
                (3, block_size, 1, 132),
                dtype=np.uint8,
                buffer=root,
                offset=16,
                strides=(stride, 132, 132, 1),
            )
        )
        return key, scale, full

    def test_full_page_round_trip_restores_both_aliases(self):
        for size in (32, 64, 128):
            for padding in (0, 256):
                key, scale, full = self.views(size, padding)
                selected = layout.select_transfer_views((key, scale, full))
                self.assertEqual(selected, (full,))
                for block in (2, 0):
                    key_bytes = key.array[block].tobytes()
                    scale_bytes = scale.array[block].tobytes()
                    saved = selected[0].array[block].tobytes()
                    self.assertEqual(saved, key_bytes + scale_bytes)
                    selected[0].array[block].fill(0)
                    selected[0].array[block].flat[:] = self.np.frombuffer(
                        saved, dtype=self.np.uint8
                    )
                    self.assertEqual(key.array[block].tobytes(), key_bytes)
                    self.assertEqual(scale.array[block].tobytes(), scale_bytes)

    def test_split_views_preserved_and_bad_overlap_rejected(self):
        key, scale, full = self.views(128)
        self.assertEqual(layout.select_transfer_views((key, scale)), (key, scale))
        scale.device = "different-device"
        with self.assertRaises(ValueError):
            layout.select_transfer_views((key, scale, full))


class FullAliasLayoutTests(unittest.TestCase):
    """End-to-end layout wiring; requires torch + vLLM (CPU tensors, no NPU)."""

    def test_full_alias_layout_addresses_and_restore(self):
        try:
            import numpy as np
            import torch

            from ucm.integration.vllm.hma_connector import KVCacheGroupLayout
        except ImportError:
            self.skipTest("torch/vLLM required for KVCacheGroupLayout wiring test")

        for block_size in (32, 64, 128):
            for padding in (0, 256):
                with self.subTest(block_size=block_size, padding=padding):
                    page_stride = block_size * 132 + padding
                    raw = torch.arange(page_stride * 3, dtype=torch.int64).to(
                        torch.uint8
                    )
                    key = torch.as_strided(
                        raw, (3, block_size, 1, 128), (page_stride, 128, 128, 1)
                    )
                    scale = torch.as_strided(
                        raw.view(torch.float32),
                        (3, block_size, 1, 1),
                        (page_stride // 4, 1, 1, 1),
                        storage_offset=block_size * 32,
                    )
                    full = torch.as_strided(
                        raw, (3, block_size, 1, 132), (page_stride, 132, 132, 1)
                    )
                    layout_obj = KVCacheGroupLayout(
                        {"model.layers.0.indexer": (key, scale, full)},
                        is_ascend_layout=True,
                        expected_block_size=block_size,
                    )
                    self.assertEqual(layout_obj.base_ptrs.tolist(), [full.data_ptr()])
                    self.assertEqual(
                        layout_obj.segment_tensor_size_list(
                            block_size * 4, block_size * 4
                        ),
                        [block_size * 132],
                    )
                    self.assertEqual(
                        layout_obj.extract_addrs(np.array([2, 0])).tolist(),
                        [
                            [full.data_ptr() + 2 * page_stride],
                            [full.data_ptr()],
                        ],
                    )
                    saved = full[2].clone()
                    saved_key = key[2].clone()
                    saved_scale = scale[2].view(torch.uint8).clone()
                    full[2].zero_()
                    full[2].copy_(saved)
                    self.assertTrue(torch.equal(key[2], saved_key))
                    self.assertTrue(
                        torch.equal(scale[2].view(torch.uint8), saved_scale)
                    )
                    # Partial-page segments of the packed full view are
                    # semantically invalid (see select_transfer_views); the
                    # caller contract keeps this group whole-page only.


if __name__ == "__main__":
    unittest.main()
