from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.kv_calc import presets  # noqa: E402
from ucm_toolkit.tools.kv_calc.detect import (  # noqa: E402
    classify,
    infer_attention_class,
    standard_variant,
)
from ucm_toolkit.tools.kv_calc.formulas import (  # noqa: E402
    compute_seq_cache,
    default_precision,
)

GIB = 1024**3


class KvCalcFormulaTest(unittest.TestCase):
    """Per-layer KV profile math, parity with calculator.js口径."""

    def test_standard_qwen3_32b_one_gib_per_seq(self):
        cfg = presets.get_preset("qwen3-32b")["fields"]
        seq = compute_seq_cache(cfg, "standard", 4096, default_precision("standard"))
        expected = 2 * 64 * 8 * 128 * 4096 * 2  # layers, kv, head_dim, tokens, bf16
        self.assertAlmostEqual(seq.bytes_per_seq, expected)
        self.assertAlmostEqual(seq.bytes_per_seq / GIB, 1.0)
        self.assertEqual(standard_variant(cfg), "GQA")

    def test_standard_mha_and_mqa_variants(self):
        self.assertEqual(
            standard_variant({"num_attention_heads": 32, "num_key_value_heads": 32}),
            "MHA",
        )
        self.assertEqual(
            standard_variant({"num_attention_heads": 32, "num_key_value_heads": 1}),
            "MQA",
        )

    def test_mla_deepseek_v3_no_factor_two(self):
        cfg = presets.get_preset("deepseek-v3")["fields"]
        seq = compute_seq_cache(cfg, "mla", 4096, default_precision("mla"))
        # MLA: layers * (kv_lora_rank + qk_rope_head_dim) * tokens * dtype (no x2).
        self.assertAlmostEqual(seq.bytes_per_seq, 61 * (512 + 64) * 4096 * 2)

    def test_dsa_glm5_ml_plus_indexer(self):
        cfg = presets.get_preset("glm-5")["fields"]
        prec = default_precision("dsa")
        seq = compute_seq_cache(cfg, "dsa", 4096, prec)
        ml = 78 * (512 + 64) * 4096 * prec.kv
        indexer = 78 * 128 * 4096 * prec.indexer
        self.assertAlmostEqual(seq.bytes_per_seq, ml + indexer)

    def test_v4_flash_paper_formula(self):
        cfg = presets.get_preset("deepseek-v4-flash")["fields"]
        prec = default_precision("deepseek_v4")
        seq = compute_seq_cache(cfg, "deepseek_v4", 4096, prec)
        ratios = cfg["compress_ratios"]
        entry = (512 - 64) * 1 + 64 * 2  # nope*FP8 + rope*BF16
        compressed = sum((4096 // r) * entry for r in ratios if r and r > 0)
        sliding = 43 * 128 * entry
        indexer = sum(1 for r in ratios if r == 4) * (4096 // 4) * 128 * 0.5
        self.assertAlmostEqual(seq.bytes_per_seq, compressed + sliding + indexer)

    def test_v4_measured_constants_present(self):
        for name in ("deepseek-v4-pro", "deepseek-v4-flash"):
            entry = presets.get_preset(name)
            dm = entry["deployment_measured"]
            self.assertGreater(dm["vllm"]["bytes_per_token"], 0)
            self.assertGreater(dm["vllm-ascend"]["bytes_per_token"], 0)

    def test_mixed_gemma4_shared_layer_scaling(self):
        # E2B: 35 layers, 7 full + 28 sliding, num_kv_shared_layers=20.
        # Stored = 35 - 20 = 15; full/sliding scaled by 15/35 -> 3 / 12.
        cfg = presets.get_preset("gemma-4-e2b-it")["fields"]
        seq = compute_seq_cache(
            cfg, "mixed_full_sliding", 4096, default_precision("mixed_full_sliding")
        )
        full = 3 * 1 * (512 + 512) * 4096 * 2
        swa = 512 * 12 * 1 * (256 + 256) * 2  # window=512
        self.assertAlmostEqual(seq.bytes_per_seq, full + swa)

    def test_gqa_copy_when_kv_not_divisible_by_tp(self):
        cfg = presets.get_preset("qwen3-32b")["fields"]
        seq = compute_seq_cache(cfg, "standard", 4096, default_precision("standard"))
        # kv_heads=8, tp=16 -> not divisible; without the flag it must error.
        with self.assertRaises(ValueError):
            seq.per_rank_bytes(16, False)
        # With the flag: ceil(8/16)=1 group per rank -> per_rank = c * 1/8.
        per_rank = seq.per_rank_bytes(16, True)
        self.assertAlmostEqual(per_rank, seq.bytes_per_seq / 8)

    def test_dp_tp_split_standard(self):
        cfg = presets.get_preset("qwen3-32b")["fields"]
        seq = compute_seq_cache(cfg, "standard", 4096, default_precision("standard"))
        n, tp, dp = 1000, 2, 4
        per_seq_per_gpu = seq.per_rank_bytes(tp, False)
        total = n * per_seq_per_gpu * tp  # = N * c
        self.assertAlmostEqual(total, 1000 * seq.bytes_per_seq)
        self.assertAlmostEqual(total / dp, 1000 * seq.bytes_per_seq / 4)  # instance
        self.assertAlmostEqual(
            total / (dp * tp), 1000 * seq.bytes_per_seq / 8
        )  # per-GPU

    def test_registry_classifies_known_architectures(self):
        self.assertEqual(
            classify(["DeepseekV32ForCausalLM"], {}).attention_class, "dsa"
        )
        self.assertEqual(classify(["GlmMoeDsaForCausalLM"], {}).attention_class, "dsa")
        self.assertEqual(
            classify(["Qwen3_5ForConditionalGeneration"], {}).attention_class,
            "qwen_linear_full",
        )
        self.assertEqual(
            classify(["MiniMaxM3SparseForConditionalGeneration"], {}).attention_class,
            "minimax_msa",
        )

    def test_inference_sliding_window_alone_is_not_hybrid(self):
        # Regression for the web tool bug: a bare sliding_window on a plain
        # GQA model must NOT trip the mixed/hybrid branch.
        cfg = {
            "head_dim": 128,
            "num_key_value_heads": 8,
            "num_attention_heads": 64,
            "sliding_window": 4096,
            "num_hidden_layers": 32,
        }
        cls, _ = infer_attention_class(cfg)
        self.assertEqual(cls, "standard")


if __name__ == "__main__":
    unittest.main()
