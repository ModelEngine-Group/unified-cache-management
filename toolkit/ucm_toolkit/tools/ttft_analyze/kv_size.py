"""Model -> KV cache byte-size derivation for ttft-analyze.

Reads the Hugging Face ``config.json`` under ``--model-dir`` and derives the
per-request KV cache byte size for a prefix-hit of ``input_len`` tokens. The
derivation mirrors the KV Cache Size Calculator in
``docs/source/getting-started/kv_cache_calculator.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...errors import ToolkitError

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}

DEFAULT_DTYPE = "bfloat16"


class ModelConfigError(ToolkitError):
    """Raised when a model config is missing or lacks required fields."""


@dataclass
class ModelArchitecture:
    """KV cache architecture parameters derived from a model config."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    kv_lora_rank: int | None
    qk_rope_head_dim: int | None
    index_head_dim: int | None
    dtype: str


def _optional_int(config: dict, key: str) -> int | None:
    value = config.get(key)
    return int(value) if value is not None else None


def _dtype(config: dict) -> str:
    dtype = config.get("torch_dtype", DEFAULT_DTYPE)
    if isinstance(dtype, str):
        dtype = dtype.lower().split(".")[-1]
    if dtype in _DTYPE_BYTES:
        return dtype
    return DEFAULT_DTYPE


def load_model_architecture(model_dir: str | Path) -> ModelArchitecture:
    """Load and parse a model directory's config.json."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ModelConfigError(f"model config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigError(f"failed to parse {config_path}: {exc}") from exc

    num_hidden_layers = config.get("num_hidden_layers")
    num_attention_heads = config.get("num_attention_heads")
    if num_hidden_layers is None or num_attention_heads is None:
        raise ModelConfigError(
            f"{config_path} missing num_hidden_layers / num_attention_heads"
        )

    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden_size = config.get("hidden_size")
        if hidden_size is None:
            raise ModelConfigError(f"{config_path} missing head_dim / hidden_size")
        head_dim = hidden_size // num_attention_heads

    return ModelArchitecture(
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        kv_lora_rank=_optional_int(config, "kv_lora_rank"),
        qk_rope_head_dim=_optional_int(config, "qk_rope_head_dim"),
        index_head_dim=_optional_int(config, "index_head_dim"),
        dtype=_dtype(config),
    )


def detect_architecture(arch: ModelArchitecture) -> str:
    """Return ``dsa`` / ``mla`` / ``gqa`` following the calculator's rules."""
    if arch.kv_lora_rank and arch.qk_rope_head_dim and arch.index_head_dim:
        return "dsa"
    if arch.kv_lora_rank and arch.qk_rope_head_dim and not arch.index_head_dim:
        return "mla"
    return "gqa"


def dtype_bytes(dtype: str) -> int:
    """Return bytes-per-element for a dtype name."""
    return _DTYPE_BYTES.get(dtype.lower().split(".")[-1], 2)


def kv_cache_bytes(arch: ModelArchitecture, input_len: int) -> int:
    """Return per-request KV cache bytes for a prefix-hit of ``input_len`` tokens."""
    layers = arch.num_hidden_layers
    tokens = input_len
    dtype_size = dtype_bytes(arch.dtype)
    arch_type = detect_architecture(arch)

    if arch_type == "dsa":
        elements = layers * tokens * (
            arch.kv_lora_rank + arch.qk_rope_head_dim + arch.index_head_dim
        )
    elif arch_type == "mla":
        elements = layers * tokens * (arch.kv_lora_rank + arch.qk_rope_head_dim)
    else:
        elements = 2 * layers * tokens * arch.num_key_value_heads * arch.head_dim
    return elements * dtype_size


def per_card_cache_bytes(arch: ModelArchitecture, input_len: int, tp: int) -> float:
    """Return the KV cache bytes a single card must load into its HBM.

    GQA/MHA shard KV heads across ``tp`` cards, so each card loads ``total/tp``.
    MLA stores a shared latent that is not sharded, so each card loads the full
    latent. DSA follows the MLA path for its latent portion.
    """
    total = kv_cache_bytes(arch, input_len)
    if detect_architecture(arch) == "gqa":
        return total / tp
    return float(total)

