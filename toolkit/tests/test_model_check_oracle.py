"""Tests for the connector-independent model-check byte oracles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.model_check.oracle import (  # noqa: E402
    POISON_BYTE,
    StorageSnapshot,
    byte_view,
    fill_pattern,
    first_mismatch,
    pattern_bytes,
    poison,
    snapshot_bytes,
)


class ByteViewTest(unittest.TestCase):
    def test_aliases_permuted_low_precision_tensor(self):
        base = torch.zeros(4, 3, 2, dtype=torch.bfloat16)
        view = base.permute(1, 0, 2)  # last stride is 1, middle is not
        bytes_ = byte_view(view)
        self.assertEqual(tuple(bytes_.shape), (3, 4, 2, 2))
        bytes_.fill_(0x7F)
        self.assertTrue(torch.all(base.view(torch.uint8) == 0x7F))

    def test_non_unit_last_stride(self):
        base = torch.zeros(6, 4, dtype=torch.float16)
        view = base.t()  # last stride 4, view(torch.uint8) would reject it
        fill_pattern(view, "x")
        self.assertTrue(
            torch.equal(snapshot_bytes(view), pattern_bytes((4, 6, 2), "x"))
        )


class PatternTest(unittest.TestCase):
    def test_range_excludes_poison(self):
        pattern = pattern_bytes((4096,), "layer", 0, 3)
        self.assertLessEqual(int(pattern.max()), 250)
        self.assertNotIn(POISON_BYTE, pattern.tolist())

    def test_labels_and_positions_are_distinguished(self):
        a = pattern_bytes((512,), "layer.0", 1, 5)
        b = pattern_bytes((512,), "layer.1", 0, 5)
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.equal(a[1:], a[:-1]))

    def test_nan_bit_patterns_compare_equal_as_bytes(self):
        source = torch.empty(64, dtype=torch.bfloat16)
        fill_pattern(source, "nan")
        target = torch.empty_like(source)
        target.copy_(source)
        self.assertIsNone(
            first_mismatch(snapshot_bytes(target), snapshot_bytes(source))
        )


class MismatchTest(unittest.TestCase):
    def test_poisoned_target_is_detected(self):
        source = torch.empty(2, 8, dtype=torch.float16)
        target = torch.empty_like(source)
        fill_pattern(source, "s")
        target.copy_(source)
        poison(target[1])
        index = first_mismatch(snapshot_bytes(target), snapshot_bytes(source))
        self.assertEqual(index, 16)

    def test_swapped_components_are_detected(self):
        kv = torch.empty(2, 4, 8, dtype=torch.float16)
        fill_pattern(kv[0], "layer", 0)
        fill_pattern(kv[1], "layer", 1)
        expected_k = snapshot_bytes(kv[0])
        swapped = kv.flip(0).clone()
        self.assertIsNotNone(first_mismatch(snapshot_bytes(swapped[0]), expected_k))


class StorageSnapshotTest(unittest.TestCase):
    def test_writes_inside_allowed_view_are_not_stray(self):
        pool = torch.zeros(4, 2, 8, dtype=torch.float16)
        snapshot = StorageSnapshot([pool[:, 0], pool[:, 1]])
        snapshot.allow_tensor(pool[2, 1])
        pool[2, 1].fill_(3)
        self.assertEqual(snapshot.stray_writes(), [])

    def test_write_outside_allowed_view_is_reported(self):
        pool = torch.zeros(4, 2, 8, dtype=torch.float16)
        snapshot = StorageSnapshot([pool])
        snapshot.allow_tensor(pool[2, 1])
        pool[3, 0, 5] = 1.0
        stray = snapshot.stray_writes()
        self.assertEqual(len(stray), 1)
        _base, offset, count = stray[0]
        # fp16 1.0 is 0x3c00: only the high (second, little-endian) byte changes.
        self.assertEqual(offset, (3 * 16 + 5) * 2 + 1)
        self.assertEqual(count, 1)

    def test_allow_range_uses_absolute_addresses(self):
        pool = torch.zeros(64, dtype=torch.uint8)
        snapshot = StorageSnapshot([pool])
        snapshot.allow_range(pool.data_ptr() + 8, 4)
        pool[8:12] = 9
        self.assertEqual(snapshot.stray_writes(), [])
        pool[12] = 9
        self.assertEqual(snapshot.stray_writes()[0][1:], (12, 1))

    def test_range_outside_storage_is_rejected(self):
        pool = torch.zeros(16, dtype=torch.uint8)
        snapshot = StorageSnapshot([pool])
        with self.assertRaises(AssertionError):
            snapshot.allow_range(pool.data_ptr() + 12, 8)


if __name__ == "__main__":
    unittest.main()
