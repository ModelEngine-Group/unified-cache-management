"""CPU tests for the optional KV non-finite probe.

The configuration/sampling layer runs without torch; the bit-pattern tests
need torch (CPU tensors only) and are skipped when torch is unavailable.

The module under test is loaded from its file directly so these tests do not
require the compiled ``ucmlogger`` native extension (same approach as
``test_ucm_fawa_fullview.py``: no UCM/vLLM runtime needed):

    python -m unittest discover -s test/suites/Unit/connector -p test_kv_nan_check.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = _REPO_ROOT / "ucm" / "integration" / "vllm" / "kv_nan_check.py"

_spec = importlib.util.spec_from_file_location("ucm_kv_nan_check", _MODULE_PATH)
kv_nan_check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ucm_kv_nan_check", kv_nan_check)
_spec.loader.exec_module(kv_nan_check)

from ucm_kv_nan_check import (  # noqa: E402
    KVCheckSelection,
    KVNaNCheckConfig,
    KVNaNChecker,
    sample_indices,
)

try:
    import torch
except ImportError:
    torch = None

_TORCH_REASON = "torch needed for bit-pattern tests"

_TORCH_SKIP = unittest.skip(_TORCH_REASON) if torch is None else (lambda cls: cls)


class ConfigTests(unittest.TestCase):
    def test_default_off(self):
        config = KVNaNCheckConfig.from_env({})
        self.assertEqual(config.phases, frozenset())
        self.assertEqual(config.groups, frozenset({"fa", "wa"}))
        self.assertEqual(config.max_blocks, 64)
        self.assertEqual(config.stride, 1)
        self.assertFalse(config.phase_enabled("dump"))
        self.assertFalse(config.phase_enabled("load"))

    def test_all_token(self):
        for raw in ("1", "all", "true", "BOTH"):
            config = KVNaNCheckConfig.from_env({"UCM_KV_NAN_CHECK": raw})
            self.assertEqual(config.phases, frozenset({"dump", "load"}), raw)

    def test_single_phase(self):
        config = KVNaNCheckConfig.from_env({"UCM_KV_NAN_CHECK": "dump"})
        self.assertEqual(config.phases, frozenset({"dump"}))
        config = KVNaNCheckConfig.from_env({"UCM_KV_NAN_CHECK": "LOAD,"})
        self.assertEqual(config.phases, frozenset({"load"}))

    def test_unknown_token_is_off(self):
        config = KVNaNCheckConfig.from_env({"UCM_KV_NAN_CHECK": "bogus"})
        self.assertEqual(config.phases, frozenset())

    def test_group_filter_and_bounds(self):
        config = KVNaNCheckConfig.from_env(
            {
                "UCM_KV_NAN_CHECK": "dump",
                "UCM_KV_NAN_CHECK_GROUPS": "fa",
                "UCM_KV_NAN_CHECK_MAX_BLOCKS": "0",
                "UCM_KV_NAN_CHECK_STRIDE": "junk",
            }
        )
        self.assertTrue(config.group_enabled("FA"))
        self.assertFalse(config.group_enabled("WA"))
        # max_blocks clamps to >= 1, stride falls back to 1.
        self.assertEqual(config.max_blocks, 1)
        self.assertEqual(config.stride, 1)


class SampleTests(unittest.TestCase):
    def test_small_and_edges(self):
        self.assertEqual(sample_indices(0, 8), [])
        self.assertEqual(sample_indices(5, 8), [0, 1, 2, 3, 4])
        self.assertEqual(sample_indices(3, 3), [0, 1, 2])

    def test_sampling_keeps_first_last_and_bounds(self):
        picks = sample_indices(100, 8)
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[-1], 99)
        self.assertLessEqual(len(picks), 8)
        self.assertEqual(picks, sorted(set(picks)))

    def test_single_limit_keeps_first(self):
        self.assertEqual(sample_indices(100, 1), [0])


class CheckerGatingTests(unittest.TestCase):
    def test_stride_counts_per_phase(self):
        checker = KVNaNChecker(
            KVNaNCheckConfig(phases=frozenset({"dump"}), stride=2)
        )
        self.assertEqual(
            [checker.should_run("dump") for _ in range(4)], [True, False, True, False]
        )
        # load phase is disabled entirely; must not raise nor count.
        self.assertEqual([checker.should_run("load") for _ in range(3)], [False] * 3)

    def test_run_without_selections_is_noop(self):
        checker = KVNaNChecker(KVNaNCheckConfig(phases=frozenset({"dump"})))
        checker.run([], {}, "dump")  # must not raise
        checker.run(None, None, "dump")  # defensive: None input

    def test_disabled_checker_never_scans(self):
        scanned = []

        class Recorder:
            def __getattr__(self, name):
                scanned.append(name)
                raise AssertionError("must not be touched")

        checker = KVNaNChecker(KVNaNCheckConfig())
        checker.run([NS(block_ids=[0])], {0: Recorder()}, "dump")
        self.assertEqual(scanned, [])


@_TORCH_SKIP
class BitPatternTests(unittest.TestCase):
    """bf16/fp16 detection via integer views, on synthetic tensors."""

    def _bf16(self, bits):
        # uint16 -> bit-identical int16 view -> bf16 view.
        return torch.tensor([bits], dtype=torch.uint16).view(torch.int16).view(
            torch.bfloat16
        )

    def test_bf16_nonfinite_patterns(self):
        # inf, -inf, quiet NaN (0x7FC0), negative NaN, signaling NaN.
        for bits in (0x7F80, 0xFF80, 0x7FC0, 0x7FFF, 0xFFC1, 0x7F81):
            with self.subTest(hex(bits)):
                self.assertTrue(
                    torch.isnan(self._bf16(bits)) or torch.isinf(self._bf16(bits))
                )

    def test_bf16_finite_patterns(self):
        # Max normal 0x7F7F, min normal, tiny subnormal, zero, sign bit only.
        for bits in (0x7F7F, 0x0080, 0x0001, 0x0000, 0x8000, 0x3F80):
            with self.subTest(hex(bits)):
                value = self._bf16(bits)
                self.assertFalse(
                    torch.isnan(value) or torch.isinf(value),
                    f"bits 0x{bits:04X} must be finite",
                )

    def test_fp16_patterns(self):
        tensor = (
            torch.tensor([0x7BFF, 0x7C00, 0x7E00, 0xFFFF], dtype=torch.uint16)
            .view(torch.int16)
            .view(torch.float16)
        )
        self.assertFalse(
            torch.isnan(tensor[0]) or torch.isinf(tensor[0])
        )  # 65504 = max finite
        self.assertTrue(torch.isinf(tensor[1]))  # 0x7C00 inf
        self.assertTrue(torch.isnan(tensor[2]))  # 0x7E00 NaN
        self.assertTrue(torch.isnan(tensor[3]))  # 0xFFFF -NaN

    def test_end_to_end_probe_reports_and_maps_keys(self):
        tensor = torch.zeros(6, 4, 3, dtype=torch.bfloat16)
        # Block 2: one non-finite element; block 5: whole-block NaN.
        tensor[2, 0, 1] = float("nan")
        tensor[5] = float("inf")
        int8_tensor = torch.full((6, 2), 127, dtype=torch.int8)

        layout = NS(
            block_views=[
                ("model.layers.3.long_kv_cache", tensor),
                ("model.layers.3.indexer.k_cache", int8_tensor),  # must be skipped
            ]
        )
        keys = [bytes([i]) * 16 for i in range(6)]
        selection = KVCheckSelection(
            group_id=0, label="FA", keys=keys, block_ids=list(range(6))
        )
        errors, metrics = [], []

        checker = KVNaNChecker(
            KVNaNCheckConfig(phases=frozenset({"dump"}), max_blocks=64),
            logger=NS(
                error=errors.append,
                warning=lambda *a, **k: None,
            ),
            metrics=lambda stats: metrics.append(stats),
        )
        checker.run([selection], {0: layout}, "dump")

        self.assertEqual(len(errors), 1)
        message = errors[0]
        self.assertIn("phase=dump", message)
        self.assertIn("group=0", message)
        self.assertIn("layer=model.layers.3.long_kv_cache", message)
        self.assertIn("dtype=torch.bfloat16", message)
        self.assertIn("checked=6/6", message)
        self.assertIn("bad_blocks=[2,5]", message)
        # Keys of the two bad blocks, first 8 bytes as hex.
        self.assertIn(keys[2][:8].hex(), message)
        self.assertIn(keys[5][:8].hex(), message)
        # int8 view must not appear in the report.
        self.assertNotIn("indexer", message)
        self.assertEqual(metrics, [{"kv_nan_bad_blocks": 2.0}])

    def test_end_to_end_clean_and_group_filter(self):
        tensor = torch.ones(4, 2, 2, dtype=torch.bfloat16)
        layout = NS(block_views=[("l", tensor)])
        selection = KVCheckSelection(
            group_id=0, label="WA", keys=[b"k" * 16] * 4, block_ids=list(range(4))
        )
        errors = []
        checker = KVNaNChecker(
            KVNaNCheckConfig(
                phases=frozenset({"load"}), groups=frozenset({"fa"})
            ),
            logger=NS(error=errors.append, warning=lambda *a: None),
        )
        checker.run([selection], {0: layout}, "load")
        self.assertEqual(errors, [])  # WA filtered out by UCM_KV_NAN_CHECK_GROUPS

    def test_sampling_limits_checked_blocks(self):
        tensor = torch.zeros(100, 2, dtype=torch.bfloat16)
        tensor[99] = float("nan")
        layout = NS(block_views=[("l", tensor)])
        selection = KVCheckSelection(
            group_id=0, label="FA", keys=[b"k" * 16] * 100, block_ids=list(range(100))
        )
        errors = []
        checker = KVNaNChecker(
            KVNaNCheckConfig(phases=frozenset({"dump"}), max_blocks=8),
            logger=NS(error=errors.append, warning=lambda *a: None),
        )
        checker.run([selection], {0: layout}, "dump")
        # sample_indices(100, 8) always keeps the last block.
        self.assertEqual(len(errors), 1)
        self.assertIn("checked=8/100", errors[0])
        self.assertIn("bad_blocks=[99]", errors[0])

    def test_probe_never_raises_on_bad_layout(self):
        selection = KVCheckSelection(
            group_id=0, label="FA", keys=None, block_ids=[0, 1]
        )
        warnings = []
        checker = KVNaNChecker(
            KVNaNCheckConfig(phases=frozenset({"load"})),
            logger=NS(error=lambda *a: None, warning=warnings.append),
        )
        # Layout missing, layout without views, non-tensor views: all no-ops.
        checker.run([selection], {}, "load")
        checker.run([selection], {0: NS(block_views=[])}, "load")
        checker.run(
            [selection], {0: NS(block_views=[("l", object())])}, "load"
        )
        self.assertEqual(warnings, [])

    def _checker(self, errors):
        return KVNaNChecker(
            KVNaNCheckConfig(phases=frozenset({"dump"}), max_blocks=64),
            logger=NS(
                error=errors.append, warning=lambda *a, **k: None
            ),
        )

    def test_partial_ranges_exclude_invalid_regions(self):
        # Blocks whose invalid region (outside the transferred range) is all
        # NaN must not trigger: this is the recycled-page pattern.
        tensor = torch.zeros(6, 64, 2, dtype=torch.bfloat16)
        tensor[:, :48, :] = float("nan")  # invalid head of every block
        layout = NS(block_views=[("swa", tensor, 1, 64, False)])
        selection = KVCheckSelection(
            group_id=0,
            label="WA",
            keys=[bytes([i]) * 16 for i in range(6)],
            block_ids=list(range(6)),
            token_offsets=[48] * 6,  # short tail: only last 16 tokens move
            token_counts=[16] * 6,
            token_block_size=64,
        )
        errors = []
        self._checker(errors).run([selection], {0: layout}, "dump")
        self.assertEqual(errors, [])

        # Poison inside the valid range must trigger with range hints.
        tensor[3, 48:64, :] = float("inf")
        self._checker(errors).run([selection], {0: layout}, "dump")
        self.assertEqual(len(errors), 1)
        self.assertIn("bad_blocks=[3]", errors[0])
        self.assertIn("tokens=16@48", errors[0])
        self.assertIn(bytes([3] * 16)[:8].hex(), errors[0])

    def test_varying_offsets_use_validity_mask(self):
        # Packed hash blocks: each row transfers a different 64-token slice
        # of a 512-token block; the rest of the block is NaN.
        tensor = torch.zeros(8, 512, 2, dtype=torch.bfloat16)
        for row in range(8):
            offset = (row * 64) % 512
            valid = torch.zeros(512, dtype=torch.bool)
            valid[offset : offset + 64] = True
            tensor[row][~valid.unsqueeze(1).expand(512, 2)] = float("nan")
        layout = NS(block_views=[("long_kv", tensor, 1, 512, False)])
        selection = KVCheckSelection(
            group_id=0,
            label="FA",
            keys=[bytes([i]) * 16 for i in range(8)],
            block_ids=list(range(8)),
            token_offsets=[(i * 64) % 512 for i in range(8)],
            token_counts=[64] * 8,
            token_block_size=512,
        )
        errors = []
        self._checker(errors).run([selection], {0: layout}, "dump")
        self.assertEqual(errors, [])

        tensor[5, 350, 0] = float("nan")  # inside row 5's valid slice [320,384)
        self._checker(errors).run([selection], {0: layout}, "dump")
        self.assertEqual(len(errors), 1)
        self.assertIn("bad_blocks=[5]", errors[0])

    def test_whole_only_views_skip_partial_ranges(self):
        # Whole-page layouts cannot express partial ranges: they are skipped
        # with a debug note rather than mis-sliced.
        tensor = torch.zeros(4, 8, 2, dtype=torch.bfloat16)
        tensor[:, :4, :] = float("nan")
        layout = NS(block_views=[("page", tensor, 1, 8, True)])
        selection = KVCheckSelection(
            group_id=0,
            label="FA",
            keys=[b"k" * 16] * 4,
            block_ids=list(range(4)),
            token_offsets=[4] * 4,
            token_counts=[4] * 4,
            token_block_size=8,
        )
        errors = []
        self._checker(errors).run([selection], {0: layout}, "dump")
        self.assertEqual(errors, [])

        # Whole-block transfers still check whole-only views.
        whole = KVCheckSelection(
            group_id=0,
            label="FA",
            keys=[b"k" * 16] * 4,
            block_ids=list(range(4)),
        )
        self._checker(errors).run([whole], {0: layout}, "dump")
        self.assertEqual(len(errors), 1)

    def test_compressed_axis_conversion(self):
        # A view whose token axis is states (axis_size=8) while the group's
        # token block is 64: ranges must scale with the pointer math
        # (offset * axis // group_block).
        tensor = torch.zeros(4, 8, 2, dtype=torch.bfloat16)
        tensor[:, 0, :] = float("nan")  # states[0] invalid
        layout = NS(block_views=[("v41", tensor, 2, 8, False)])
        selection = KVCheckSelection(
            group_id=0,
            label="FA",
            keys=[b"k" * 16] * 4,
            block_ids=list(range(4)),
            token_offsets=[8] * 4,   # tokens 8..64 => states 1..8
            token_counts=[56] * 4,
            token_block_size=64,
        )
        errors = []
        self._checker(errors).run([selection], {0: layout}, "dump")
        self.assertEqual(errors, [])

        selection_partial = KVCheckSelection(
            group_id=0,
            label="FA",
            keys=[b"k" * 16] * 4,
            block_ids=list(range(4)),
            token_offsets=[0] * 4,   # tokens 0..8 => state 0 only
            token_counts=[8] * 4,
            token_block_size=64,
        )
        self._checker(errors).run([selection_partial], {0: layout}, "dump")
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
