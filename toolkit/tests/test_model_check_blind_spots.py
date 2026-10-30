"""Regression tests for the blind spots found in the 2026-10-29 review.

Each test fails on the behaviour as it was before the fix, so the fixes stay
fixed. They use the checker's own functions on small synthetic tensors -- no
vLLM, no model, no device.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.model_check.common import (  # noqa: E402
    restoration_verdicts,
)
from ucm_toolkit.tools.model_check.oracle import (  # noqa: E402
    PATTERN_PERIOD,
    POISON_BYTE,
    StorageSnapshot,
    pattern_bytes,
)


class PatternPositionTest(unittest.TestCase):
    """A single sequence modulo 251 repeated every 251 bytes."""

    def test_shifts_that_used_to_be_invisible_are_visible(self):
        pattern = pattern_bytes((200_000,), "layer.0", 0, 3)
        for shift in (1, 250, 251, 252, 502, 63_001, 126_002):
            with self.subTest(shift=shift):
                self.assertFalse(
                    torch.equal(pattern[:5_000], pattern[shift : shift + 5_000]),
                    f"a shift of {shift} bytes left the pattern unchanged",
                )

    def test_period_exceeds_any_plausible_block(self):
        # 251**3; a KV block in these checks is orders of magnitude smaller.
        self.assertEqual(PATTERN_PERIOD, 251**3)
        self.assertGreater(PATTERN_PERIOD, 8 << 20)

    def test_values_never_collide_with_the_poison_byte(self):
        pattern = pattern_bytes((100_000,), "layer.7", 1, 2)
        self.assertLessEqual(int(pattern.max()), 250)
        self.assertNotIn(POISON_BYTE, pattern.unique().tolist())


class AllowedRangeTest(unittest.TestCase):
    """An address the component under test reports must be checked, not trusted."""

    def _snapshot(self):
        pool = torch.zeros(4, 32, dtype=torch.uint8)
        snapshot = StorageSnapshot([pool])
        snapshot.allow_tensor(pool[2])  # the one block vLLM handed the request
        return pool, snapshot

    def test_range_inside_the_allowed_block_is_accepted(self):
        pool, snapshot = self._snapshot()
        self.assertTrue(snapshot.contains(pool[2].data_ptr(), 32))

    def test_range_in_another_block_is_rejected(self):
        pool, snapshot = self._snapshot()
        # Writing block 3 while only block 2 was allocated to this request.
        self.assertFalse(snapshot.contains(pool[3].data_ptr(), 32))

    def test_range_crossing_out_of_the_allowed_block_is_rejected(self):
        pool, snapshot = self._snapshot()
        self.assertFalse(snapshot.contains(pool[2].data_ptr() + 16, 32))

    def test_range_outside_every_registered_storage_is_rejected(self):
        pool, snapshot = self._snapshot()
        self.assertFalse(snapshot.contains(pool.data_ptr() + (1 << 20), 8))


class RestorationVerdictTest(unittest.TestCase):
    """A group with no required-restore set must not read as complete."""

    def test_group_without_an_asserted_set_is_labelled_partial(self):
        verdicts = restoration_verdicts({3: (0, 16)}, asserted_groups=[])
        self.assertEqual(len(verdicts), 1)
        self.assertIn("PARTIALLY VERIFIED", verdicts[0])
        self.assertIn("0/16", verdicts[0])

    def test_group_with_an_asserted_set_says_so(self):
        verdicts = restoration_verdicts({0: (128, 128)}, asserted_groups=[0])
        self.assertIn("required set asserted", verdicts[0])
        self.assertNotIn("PARTIALLY VERIFIED", verdicts[0])

    def test_every_group_gets_exactly_one_verdict(self):
        coverage = {0: (10, 10), 2: (5, 10), 5: (0, 10)}
        verdicts = restoration_verdicts(coverage, asserted_groups=[0, 2])
        self.assertEqual(len(verdicts), 3)
        self.assertEqual(sum("PARTIALLY VERIFIED" in v for v in verdicts), 1)


if __name__ == "__main__":
    unittest.main()
