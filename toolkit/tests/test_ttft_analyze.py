from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry
from ucm_toolkit.cli import main
from ucm_toolkit.tools.ttft_analyze import kv_size, model


def make_arch(**overrides):
    base = dict(
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        index_head_dim=None,
        dtype="bfloat16",
    )
    base.update(overrides)
    return kv_size.ModelArchitecture(**base)


class KvSizeTest(unittest.TestCase):
    def test_gqa_cache_bytes(self):
        arch = make_arch()
        expected = 2 * 48 * 2048 * 8 * 128 * 2
        self.assertEqual(kv_size.kv_cache_bytes(arch, 2048), expected)

    def test_mla_cache_bytes(self):
        arch = make_arch(kv_lora_rank=512, qk_rope_head_dim=64)
        expected = 48 * 2048 * (512 + 64) * 2
        self.assertEqual(kv_size.kv_cache_bytes(arch, 2048), expected)

    def test_dsa_cache_bytes(self):
        arch = make_arch(
            kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128
        )
        expected = 48 * 2048 * (512 + 64 + 128) * 2
        self.assertEqual(kv_size.kv_cache_bytes(arch, 2048), expected)

    def test_detect_architecture(self):
        self.assertEqual(kv_size.detect_architecture(make_arch()), "gqa")
        self.assertEqual(
            kv_size.detect_architecture(
                make_arch(kv_lora_rank=512, qk_rope_head_dim=64)
            ),
            "mla",
        )
        self.assertEqual(
            kv_size.detect_architecture(
                make_arch(
                    kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128
                )
            ),
            "dsa",
        )

    def test_per_card_gqa_shards_by_tp(self):
        arch = make_arch()
        total = kv_size.kv_cache_bytes(arch, 2048)
        self.assertAlmostEqual(
            kv_size.per_card_cache_bytes(arch, 2048, 8), total / 8
        )

    def test_per_card_mla_not_sharded(self):
        arch = make_arch(kv_lora_rank=512, qk_rope_head_dim=64)
        total = kv_size.kv_cache_bytes(arch, 2048)
        self.assertAlmostEqual(
            kv_size.per_card_cache_bytes(arch, 2048, 8), float(total)
        )


class LoadModelArchitectureTest(unittest.TestCase):
    def _write_config(self, tmpdir: str, config: dict) -> str:
        Path(tmpdir, "config.json").write_text(json.dumps(config), encoding="utf-8")
        return tmpdir

    def test_load_gqa_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(
                tmp,
                {
                    "hidden_size": 5120,
                    "num_hidden_layers": 48,
                    "num_attention_heads": 40,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "torch_dtype": "bfloat16",
                },
            )
            arch = kv_size.load_model_architecture(tmp)
            self.assertEqual(arch.num_hidden_layers, 48)
            self.assertEqual(arch.num_key_value_heads, 8)
            self.assertEqual(arch.head_dim, 128)
            self.assertEqual(arch.dtype, "bfloat16")
            self.assertEqual(kv_size.detect_architecture(arch), "gqa")

    def test_load_mla_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(
                tmp,
                {
                    "hidden_size": 7168,
                    "num_hidden_layers": 61,
                    "num_attention_heads": 128,
                    "num_key_value_heads": 128,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                },
            )
            arch = kv_size.load_model_architecture(tmp)
            self.assertEqual(kv_size.detect_architecture(arch), "mla")
            self.assertEqual(arch.dtype, "bfloat16")

    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(kv_size.ModelConfigError):
                kv_size.load_model_architecture(tmp)

    def test_missing_fields_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(tmp, {"hidden_size": 5120})
            with self.assertRaises(kv_size.ModelConfigError):
                kv_size.load_model_architecture(tmp)

    def test_head_dim_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(
                tmp,
                {
                    "hidden_size": 5120,
                    "num_hidden_layers": 48,
                    "num_attention_heads": 40,
                },
            )
            arch = kv_size.load_model_architecture(tmp)
            self.assertEqual(arch.head_dim, 128)


class ModelTest(unittest.TestCase):
    def _inputs(self, **overrides) -> model.TtftInputs:
        base = dict(
            cache_size_bytes=402653184.0,
            tp=1,
            posix_bw_gbps=12.0,
            h2d_bw_gbps=60.0,
            ttft_prefill_ms=260.0,
            ttft_hbm_ms=3.2,
        )
        base.update(overrides)
        return model.TtftInputs(**base)

    def test_layered_uses_max(self):
        result = model.analyze(self._inputs(), "layered")
        self.assertAlmostEqual(
            result.t_cache_load_ms, result.storage_read_ms + result.h2d_ms
        )
        self.assertEqual(result.ttft_ucm_ms, max(3.2, result.t_cache_load_ms))

    def test_full_sums(self):
        result = model.analyze(self._inputs(), "full")
        self.assertAlmostEqual(result.ttft_ucm_ms, 3.2 + result.t_cache_load_ms)

    def test_gain_and_loss(self):
        result = model.analyze(self._inputs(), "full")
        self.assertAlmostEqual(result.gain_vs_prefill, 260.0 / result.ttft_ucm_ms)
        self.assertAlmostEqual(result.loss_vs_hbm, result.ttft_ucm_ms / 3.2)

    def test_zero_bandwidth_raises(self):
        with self.assertRaises(model.TtftAnalysisError):
            model.analyze(self._inputs(posix_bw_gbps=0), "layered")

    def test_zero_cache_size_raises(self):
        with self.assertRaises(model.TtftAnalysisError):
            model.analyze(self._inputs(cache_size_bytes=0), "layered")

    def test_storage_read_scales_with_tp_for_fixed_per_card(self):
        single = model.analyze(self._inputs(), "layered")
        multi = model.analyze(self._inputs(tp=4), "layered")
        self.assertAlmostEqual(multi.storage_read_ms, single.storage_read_ms * 4)
        self.assertAlmostEqual(multi.h2d_ms, single.h2d_ms)

    def test_invalid_tp_raises(self):
        with self.assertRaises(model.TtftAnalysisError):
            model.analyze(self._inputs(tp=0), "layered")


class ToolkitTest(unittest.TestCase):
    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_registered_top_level_tool(self):
        registry.init_builtin_tools()
        tool = registry.get("ttft-analyze")
        self.assertEqual(tool.name, "ttft-analyze")
        self.assertIn("ttft_analyze", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_run(self):
        registry.init_builtin_tools()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 5120,
                        "num_hidden_layers": 48,
                        "num_attention_heads": 40,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "run",
                        "ttft-analyze",
                        "--model-dir",
                        tmp,
                        "--posix-bw",
                        "12",
                        "--h2d-bw",
                        "60",
                        "--input-len",
                        "2048",
                        "--ttft-prefill",
                        "260",
                        "--ttft-hbm",
                        "3.2",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn("layered", output.getvalue())
            self.assertIn("full", output.getvalue())
            self.assertIn("bottleneck", output.getvalue())

    def test_cli_run_with_tp(self):
        registry.init_builtin_tools()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 5120,
                        "num_hidden_layers": 48,
                        "num_attention_heads": 40,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "run",
                        "ttft-analyze",
                        "--model-dir",
                        tmp,
                        "--posix-bw",
                        "12",
                        "--h2d-bw",
                        "60",
                        "--input-len",
                        "2048",
                        "--ttft-prefill",
                        "260",
                        "--ttft-hbm",
                        "3.2",
                        "--tp",
                        "8",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn("per-card", output.getvalue())

    def test_cli_run_invalid_bandwidth_returns_error(self):
        registry.init_builtin_tools()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 5120,
                        "num_hidden_layers": 48,
                        "num_attention_heads": 40,
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stderr(output):
                result = main(
                    [
                        "run",
                        "ttft-analyze",
                        "--model-dir",
                        tmp,
                        "--posix-bw",
                        "0",
                        "--h2d-bw",
                        "60",
                        "--input-len",
                        "2048",
                        "--ttft-prefill",
                        "260",
                        "--ttft-hbm",
                        "3.2",
                    ]
                )
            self.assertEqual(result, 1)
            self.assertIn("bandwidth must be positive", output.getvalue())


if __name__ == "__main__":
    unittest.main()
