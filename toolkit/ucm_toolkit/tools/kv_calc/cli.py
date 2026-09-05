"""kv-calc command-line interface.

Examples::

    kv-calc --list
    kv-calc --model qwen3-32b --input-len 4096 --num-requests 1000 --tp 1
    kv-calc --model deepseek-ai/DeepSeek-V3 --input-len 8192 --num-requests 256 --tp 8 --dp 2
    kv-calc --model hf://Qwen/Qwen3-32B --input-len 4096
    kv-calc --model ms://ZhipuAI/GLM-4.7-Flash --input-len 4096 --tp 4
    kv-calc --model ./local/model_dir --input-len 4096 --tp 2 --gqa-copy
"""

import argparse
import json
import sys

from . import fmt, presets
from .detect import CLASS_LABELS, classify
from .formulas import (
    DTYPE_BYTES,
    compute_seq_cache,
    default_precision,
    dtype_bytes,
)
from .loader import LoadError, load_config


def build_parser():
    p = argparse.ArgumentParser(
        prog="ucm-toolkit run kv-calc",
        description="KV cache size estimation for LLM serving planning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="list preset models and exit")
    p.add_argument(
        "--presets",
        metavar="FILE",
        help="merge extra preset models from a JSON file (list of entries "
        "in the same flat form as the built-ins; attention_class is "
        "auto-derived from architectures when omitted). Overrides "
        "built-ins with the same id.",
    )
    p.add_argument(
        "--model",
        help="preset name/alias, local path, hf://ID, ms://ID, " "or bare org/model id",
    )
    p.add_argument(
        "--model-dir",
        dest="model_dir",
        metavar="DIR",
        help="local model directory containing config.json " "(alternative to --model)",
    )
    p.add_argument(
        "--source",
        choices=["preset", "local", "hf", "ms"],
        help="force how --model is interpreted",
    )
    p.add_argument(
        "--input-len", type=int, default=1024, help="tokens per request (default 1024)"
    )
    p.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="concurrent request count (default 1)",
    )
    p.add_argument("--tp", type=int, default=1, help="tensor parallelism (default 1)")
    p.add_argument("--dp", type=int, default=1, help="data parallelism (default 1)")
    p.add_argument(
        "--gqa-copy",
        action="store_true",
        help="account for vLLM head-group replication when num_kv_heads "
        "is not divisible by TP (GQA/MHA only; MLA/DSA/V4 ignore)",
    )
    p.add_argument(
        "--kv-dtype",
        choices=sorted(DTYPE_BYTES.keys()),
        help="KV precision (default: bf16, or fp8 for DeepSeek V4 nope)",
    )
    p.add_argument(
        "--indexer-dtype",
        choices=sorted(DTYPE_BYTES.keys()),
        help="indexer precision for DSA/V4/MiniMax M3 (default fp4)",
    )
    p.add_argument(
        "--deployment",
        choices=["vllm", "vllm-ascend"],
        default="vllm",
        help="which DeepSeek V4 measured deployment to highlight (default vllm)",
    )
    p.add_argument(
        "--include-linear-state",
        action="store_true",
        help="include Qwen linear/Gated DeltaNet recurrent+conv state "
        "(off by default; conservative, matches kvcache.ai)",
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument(
        "--verbose", action="store_true", help="show config fields and extra detail"
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.presets:
        try:
            loaded = presets.load_user_presets(args.presets)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"error: failed to load --presets {args.presets!r}: {exc}",
                file=sys.stderr,
            )
            return 2
        if not loaded:
            print(
                f"error: --presets {args.presets!r} contained no preset entries",
                file=sys.stderr,
            )
            return 2

    if args.list:
        print(fmt.render_preset_table(presets.list_presets()))
        return 0

    if not args.model and not args.model_dir:
        print(
            "error: --model or --model-dir is required (or use --list)", file=sys.stderr
        )
        return 2

    # Validate numeric params (RFC acceptance: clear errors for dp/tp=0 etc.).
    for name, val in (
        ("input-len", args.input_len),
        ("num-requests", args.num_requests),
        ("tp", args.tp),
        ("dp", args.dp),
    ):
        if val < 1:
            print(f"error: --{name} must be >= 1 (got {val})", file=sys.stderr)
            return 2

    try:
        model = load_config(args.model, source=args.source, model_dir=args.model_dir)
    except LoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Classify. Presets carry a curated attention_class; externally loaded
    # configs go through the architecture-string registry + field inference.
    if model.is_preset:
        attention_class = model.preset_entry["attention_class"]
        classification = _CuratedClassification(attention_class, model.architectures)
    else:
        classification = classify(model.architectures, model.fields)
        attention_class = classification.attention_class

    precision = default_precision(attention_class, args.kv_dtype, args.indexer_dtype)

    try:
        seq = compute_seq_cache(
            model.fields,
            attention_class,
            args.input_len,
            precision,
            include_linear_state=args.include_linear_state,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        per_seq_per_gpu = seq.per_rank_bytes(args.tp, args.gqa_copy)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Aggregate accounting.
    #   total (cluster) = N * Σ(part.per_rank * mult(part))
    #     where mult = tp  for tp/heads-sharded parts (each request occupies
    #                       tp GPUs), 1 for fixed parts (state lives once).
    #   per_instance = total / dp   (one DP rank)
    #   per_gpu      = total / (dp * tp)   (uniform load)
    #   per_request_per_gpu = per_seq_per_gpu   (DP-free; the Q#3 answer)
    total = 0.0
    for part in seq.parts:
        per_rank = part.per_rank(args.tp, args.gqa_copy)
        mult = args.tp if part.shard in ("tp", "heads") else 1
        total += per_rank * mult
    total *= args.num_requests
    per_instance = total / args.dp
    per_gpu = total / (args.dp * args.tp)

    # V4 measured side-by-side.
    v4_measured = []
    if (
        attention_class == "deepseek_v4"
        and model.preset_entry
        and model.preset_entry.get("deployment_measured")
    ):
        dm = model.preset_entry["deployment_measured"]
        for dep in ("vllm", "vllm-ascend"):
            if dep not in dm:
                continue
            bpt = dm[dep]["bytes_per_token"]
            per_seq = bpt * args.input_len
            m_per_seq_pg = per_seq / args.tp
            m_total = args.num_requests * m_per_seq_pg * args.tp
            v4_measured.append(
                {
                    "deployment": dep,
                    "bytes_per_token": bpt,
                    "per_seq_bytes": per_seq,
                    "per_seq_per_gpu": m_per_seq_pg,
                    "total_bytes": m_total,
                    "selected": dep == args.deployment,
                }
            )

    # Effective dtype strings/bytes for display.
    if attention_class == "deepseek_v4":
        eff_kv_str = args.kv_dtype or "fp8"
        eff_kv_bytes = precision.v4_nope
    else:
        eff_kv_str = args.kv_dtype or "bf16"
        eff_kv_bytes = precision.kv
    eff_indexer_str = args.indexer_dtype or "fp4"
    eff_indexer_bytes = precision.indexer

    notes = []
    if model.is_preset and model.preset_entry.get("note"):
        notes.append(model.preset_entry["note"])
    if seq.note:
        notes.append(seq.note)
    if classification.method == "inferred" and getattr(classification, "note", ""):
        notes.append(classification.note)
    if attention_class == "deepseek_v4":
        notes.append(
            f"V4 RoPE dims kept at BF16 (2 B); nope at {eff_kv_str} ({eff_kv_bytes:.0f} B); "
            f"indexer at {eff_indexer_str} ({eff_indexer_bytes:.1f} B)."
        )
    if args.include_linear_state and attention_class == "qwen_linear_full":
        notes.append(
            "Linear-attention state is counted once per sequence (not sharded by "
            "TP in this model); the uniform per-GPU split is approximate for it."
        )

    verbose_fields = None
    if args.verbose:
        # Compact subset of fields that drove the formula.
        keys = (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_size",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "index_head_dim",
            "sliding_window",
            "global_head_dim",
            "num_global_key_value_heads",
            "swa_head_dim",
            "swa_v_head_dim",
            "v_head_dim",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "compress_ratios",
            "layer_types",
            "hybrid_layer_pattern",
        )
        verbose_fields = {k: model.fields[k] for k in keys if k in model.fields}

    result = {
        "model": model,
        "classification": classification,
        "params": {
            "tokens": args.input_len,
            "num_requests": args.num_requests,
            "tp": args.tp,
            "dp": args.dp,
            "kv_dtype": eff_kv_str,
            "kv_bytes": eff_kv_bytes,
            "indexer_dtype": eff_indexer_str,
            "indexer_bytes": eff_indexer_bytes,
            "gqa_copy": args.gqa_copy,
            "include_linear_state": args.include_linear_state,
        },
        "precision": precision,
        "seq": seq,
        "per_seq_per_gpu": per_seq_per_gpu,
        "total_bytes": total,
        "per_instance_bytes": per_instance,
        "per_gpu_bytes": per_gpu,
        "v4_measured": v4_measured,
        "notes": notes,
        "verbose_fields": verbose_fields,
    }

    if args.json:
        print(fmt.render_json(result))
    else:
        print(fmt.render_text(result))
    return 0


class _CuratedClassification:
    """Lightweight stand-in for presets (no inference, no note)."""

    def __init__(self, attention_class, architectures=None):
        self.attention_class = attention_class
        self.label = CLASS_LABELS.get(attention_class, attention_class)
        self.method = "curated"
        self.arch_string = (architectures or [None])[0]
        self.note = ""
