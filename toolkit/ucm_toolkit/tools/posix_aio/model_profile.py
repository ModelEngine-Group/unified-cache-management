"""Model-driven KV cache I/O sizing for the posix-aio tool (standard library only).

Reads a HuggingFace ``config.json`` and computes the POSIX store I/O granularity
(``shard_size`` / ``shard_number``) the UCM store would use for the model, in
both layerwise and non-layerwise modes.

Only GQA and the MLA family (plain MLA, and DSA = MLA + Lightning Indexer) are
sized here. Hybrid (e.g. DeepSeek V4), Mamba / linear-attention, and unknown
architectures are reported as unsupported so the adapter can warn and exit.

The detection rules mirror ``docs/source/_static/calculator.js``
(``detectArchitectureType``). The per-token formulas mirror the same source,
with the TP handling adjusted to match the real UCM store:

- MLA / DSA: the compressed latent is replicated across TP ranks and only
  rank 0 dumps, so the stored block is the *full* latent (not divided by TP).
- GQA: each TP rank stores its own KV, so ``num_kv_heads`` is divided by TP.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ALIGNMENT = 4096

SUPPORTED_ARCHITECTURES = ("mla", "dsa", "gqa")

_DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "bf16": 2,
    "half": 2,
    "float16": 2,
    "fp16": 2,
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8": 1,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
}


class ModelProfileError(ValueError):
    """Raised when a model profile cannot be computed."""


def load_config(model_dir: str | Path) -> dict[str, Any]:
    """Load a model's ``config.json`` from a directory or file path."""
    path = Path(model_dir)
    config_path = path if path.is_file() else path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with config_path.open(encoding="utf-8") as handler:
        return json.load(handler)


def dtype_to_elem_size(dtype: str | None) -> int:
    """Return the byte size of one element for a dtype string."""
    if dtype is None or dtype == "":
        return 2
    key = str(dtype).strip().lower()
    if key in _DTYPE_BYTES:
        return _DTYPE_BYTES[key]
    supported = ", ".join(sorted(set(_DTYPE_BYTES)))
    raise ModelProfileError(
        f"unsupported dtype '{dtype}'; supported: {supported}"
    )


def detect_architecture(cfg: dict[str, Any]) -> str:
    """Classify a model config as mla/dsa/gqa/hybrid/mamba/unknown.

    Detection mirrors ``calculator.js:detectArchitectureType``. The text_config
    sub-dict (present in multimodal configs) is consulted for layer-level
    indicators.
    """
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def has(key: str, *, sources: tuple[dict, ...] = (cfg, text)) -> bool:
        for src in sources:
            value = src.get(key)
            if value is not None and value is not False and value != "" and value != 0 and value != []:
                return True
        return False

    if _is_hybrid(cfg, text, has):
        return "hybrid"

    model_type = str(cfg.get("model_type") or text.get("model_type") or "").lower()
    if "mamba" in model_type or "jamba" in model_type:
        return "mamba"

    kv_lora_rank = cfg.get("kv_lora_rank")
    qk_rope_head_dim = cfg.get("qk_rope_head_dim")
    index_head_dim = cfg.get("index_head_dim")
    if kv_lora_rank and qk_rope_head_dim and index_head_dim:
        return "dsa"
    if kv_lora_rank and qk_rope_head_dim and not index_head_dim:
        return "mla"

    if has("num_key_value_heads") or has("num_kv_heads") or has("num_attention_heads"):
        return "gqa"

    return "unknown"


def _is_hybrid(cfg: dict, text: dict, has) -> bool:
    """Return True if config exhibits hybrid / non-standard attention signals."""
    layer_types = text.get("layer_types") or cfg.get("layer_types")
    if isinstance(layer_types, list) and any(t != "full_attention" for t in layer_types):
        return True

    attention_layers = cfg.get("attention_layers") or text.get("attention_layers")
    if isinstance(attention_layers, dict):
        return True
    if isinstance(cfg.get("attention_type") or text.get("attention_type"), list):
        return True
    if isinstance(
        cfg.get("layer_attention_type") or text.get("layer_attention_type"), list
    ):
        return True

    attention_mode = cfg.get("attention_mode") or text.get("attention_mode")
    if isinstance(attention_mode, str) and attention_mode.lower() in (
        "sliding",
        "linear",
        "mixed",
        "sparse",
    ):
        return True

    if has("compress_ratios"):
        return True
    if has("hybrid_layer_pattern"):
        return True
    if has("mixed_attention") or has("sparse_attention"):
        return True
    # A bare sliding_window / window_attention field marks a standard
    # sliding-window attention model (e.g. Qwen2 / Mistral), whose KV cache is
    # still computable with the GQA formula. calculator.js treats it as a hybrid
    # indicator; we deliberately do not, so plain sliding-window GQA models are
    # sized correctly. Genuine hybrids still trip on compress_ratios,
    # linear_attention, layer_types, swa_*, or paired full/sliding layer lists.
    if has("swa_num_key_value_heads") or has("swa_num_attention_heads"):
        return True
    if has("swa_head_dim") or has("add_swa_attention_sink_bias"):
        return True
    if has("full_attention_layers") and (
        has("sliding_attention_layers") or has("linear_attention_layers")
    ):
        return True
    if has("linear_attention", sources=(text, cfg)) or has(
        "linear_num_key_heads", sources=(text, cfg)
    ):
        return True
    if has("linear_key_head_dim", sources=(text, cfg)):
        return True
    if has("global_head_dim", sources=(text, cfg)) or has(
        "num_global_key_value_heads", sources=(text, cfg)
    ):
        return True
    return False


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    """Round ``value`` up to the nearest multiple of ``alignment``."""
    if value <= 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


def _resolve(cfg: dict[str, Any], key: str) -> Any:
    """Resolve a config field from the top level or text_config sub-dict."""
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    text = cfg.get("text_config")
    if isinstance(text, dict):
        value = text.get(key)
        if value is not None:
            return value
    return None


def compute_io_profile(
    cfg: dict[str, Any],
    *,
    tp: int,
    block_size: int,
    layerwise: bool,
    kv_dtype: str | None = None,
) -> dict[str, Any]:
    """Compute the POSIX store shard/block sizing for a supported model.

    ``tp`` only affects GQA (``num_kv_heads`` is divided by TP per rank); MLA
    and DSA store the full replicated latent and are TP-independent.
    """
    arch = detect_architecture(cfg)
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ModelProfileError(
            f"architecture '{arch}' is not supported; only GQA and MLA family "
            f"(MLA/DSA) are supported"
        )

    num_layers = _resolve(cfg, "num_hidden_layers")
    if not num_layers:
        raise ModelProfileError("num_hidden_layers is missing or zero")

    effective_dtype = kv_dtype or _resolve(cfg, "torch_dtype") or _resolve(cfg, "dtype") or "bfloat16"
    elem_size = dtype_to_elem_size(effective_dtype)
    tokens_per_block = block_size

    num_kv_heads_full = (
        _resolve(cfg, "num_key_value_heads")
        or _resolve(cfg, "num_kv_heads")
        or _resolve(cfg, "num_attention_heads")
    )
    num_attention_heads = _resolve(cfg, "num_attention_heads")

    if arch in ("mla", "dsa"):
        kv_lora_rank = _resolve(cfg, "kv_lora_rank")
        qk_rope_head_dim = _resolve(cfg, "qk_rope_head_dim")
        if kv_lora_rank is None or qk_rope_head_dim is None:
            raise ModelProfileError(
                "kv_lora_rank and qk_rope_head_dim are required for MLA/DSA"
            )
        latent = int(kv_lora_rank) + int(qk_rope_head_dim)
        if arch == "dsa":
            index_head_dim = _resolve(cfg, "index_head_dim")
            if index_head_dim is None:
                raise ModelProfileError("index_head_dim is required for DSA")
            latent += int(index_head_dim)
        head_dim = latent
        per_layer_block_bytes = latent * tokens_per_block * elem_size
        num_kv_heads_per_rank: int | None = None
    else:  # gqa
        if not num_kv_heads_full:
            raise ModelProfileError("num_key_value_heads is required for GQA")
        heads_per_rank = (
            max(1, int(num_kv_heads_full) // tp) if tp and tp > 0 else int(num_kv_heads_full)
        )
        num_kv_heads_per_rank = heads_per_rank
        head_dim = _resolve(cfg, "head_dim")
        if head_dim is None:
            if not num_attention_heads:
                raise ModelProfileError("cannot derive head_dim (missing head_dim/num_attention_heads)")
            hidden_size = _resolve(cfg, "hidden_size")
            if hidden_size is None:
                raise ModelProfileError("cannot derive head_dim (missing head_dim/hidden_size)")
            head_dim = int(hidden_size) // int(num_attention_heads)
        per_layer_block_bytes = 2 * heads_per_rank * int(head_dim) * tokens_per_block * elem_size

    if layerwise:
        shard_size = align_up(per_layer_block_bytes, ALIGNMENT)
        shard_number = int(num_layers)
    else:
        shard_size = align_up(int(num_layers) * per_layer_block_bytes, ALIGNMENT)
        shard_number = 1
    store_block_size = shard_size * shard_number

    return {
        "architecture": arch,
        "num_hidden_layers": int(num_layers),
        "head_dim": int(head_dim),
        "num_kv_heads_per_rank": num_kv_heads_per_rank,
        "elem_size": elem_size,
        "dtype": effective_dtype,
        "per_layer_block_bytes": per_layer_block_bytes,
        "shard_size": shard_size,
        "shard_number": shard_number,
        "store_block_size": store_block_size,
        "block_size_tokens": tokens_per_block,
        "tensor_parallel": int(tp) if tp and tp > 0 else 1,
        "layerwise": bool(layerwise),
    }
