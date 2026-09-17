"""Per-sequence KV cache formulas for each attention class.

Formulas are aligned with the LMCache / kvcache.ai "paper"口径 (the web tool's
DeepSeek V4 measured constants live in :mod:`kv_calc.presets` and are applied
side-by-side by the CLI, not here).

The model is *parts-based*: each attention class produces a list of additive
parts, and every part carries a *shard mode* describing how it splits across
tensor parallelism:

* ``"tp"``   — the part's bytes are evenly divisible by TP (latent dims,
              head_dim, indexer dims, V4 compressed entries).
* ``"heads"``— sharded by KV-head groups; needs ``heads_total``. With
              ``gqa_copy`` the per-rank group count is ``ceil(K/tp)`` (vLLM
              head-group replication when ``K % tp != 0``); without the flag a
              non-divisible TP raises an error.
* ``"fixed"``— per-sequence state not sharded by TP (e.g. Gated DeltaNet
              recurrent state).

This lets one :func:`split_across_tp` implement every class's TP accounting.
"""

import math

DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
}


def dtype_bytes(name):
    if name is None:
        return None
    key = str(name).strip().lower()
    if key not in DTYPE_BYTES:
        raise ValueError(
            f"unknown dtype '{name}'; expected one of {sorted(DTYPE_BYTES)}"
        )
    return DTYPE_BYTES[key]


class Precision:
    """Bytes-per-element precision, with V4-specific nope/rope/indexer split."""

    __slots__ = ("kv", "indexer", "v4_nope", "v4_rope")

    def __init__(self, kv=2.0, indexer=0.5, v4_nope=1.0, v4_rope=2.0):
        self.kv = kv
        self.indexer = indexer
        self.v4_nope = v4_nope
        self.v4_rope = v4_rope


def default_precision(attention_class, kv_dtype=None, indexer_dtype=None):
    """Build a Precision. ``kv_dtype``/``indexer_dtype`` are raw CLI strings
    (None = class default)."""
    if kv_dtype is not None:
        kv = dtype_bytes(kv_dtype)
        # When the operator sets --kv-dtype explicitly, treat it as the overall
        # KV precision: V4 nope and rope both follow it. Otherwise keep the
        # paper defaults (nope=FP8, rope=BF16, indexer=FP4).
        v4_nope = v4_rope = kv
    else:
        kv = 2.0
        v4_nope = 1.0
        v4_rope = 2.0
    indexer = dtype_bytes(indexer_dtype) if indexer_dtype is not None else 0.5
    return Precision(kv=kv, indexer=indexer, v4_nope=v4_nope, v4_rope=v4_rope)


class Part:
    __slots__ = ("name", "bytes_per_seq", "shard", "heads_total", "note")

    def __init__(self, name, bytes_per_seq, shard, heads_total=None, note=""):
        if shard not in ("tp", "heads", "fixed"):
            raise ValueError(f"bad shard mode {shard!r}")
        self.name = name
        self.bytes_per_seq = float(bytes_per_seq)
        self.shard = shard
        self.heads_total = heads_total
        self.note = note

    def per_rank(self, tp, gqa_copy):
        if self.shard == "tp":
            return self.bytes_per_seq / tp
        if self.shard == "fixed":
            return self.bytes_per_seq
        # heads
        k = self.heads_total
        if not k:
            raise ValueError(f"part {self.name!r}: heads-shard has no heads_total")
        if k % tp == 0:
            eff = k // tp
        elif gqa_copy:
            eff = math.ceil(k / tp)
        else:
            raise ValueError(
                f"part {self.name!r}: num_kv_heads={k} not divisible by TP={tp}. "
                f"Pass --gqa-copy to account for vLLM head-group replication "
                f"(per-rank groups = ceil({k}/{tp}) = {math.ceil(k / tp)})."
            )
        return self.bytes_per_seq * (eff / k)


class SeqCache:
    __slots__ = ("attention_class", "parts", "note")

    def __init__(self, attention_class, parts, note=""):
        self.attention_class = attention_class
        self.parts = parts
        self.note = note

    @property
    def bytes_per_seq(self):
        return sum(p.bytes_per_seq for p in self.parts)

    @property
    def bytes_per_token(self):
        # Amortized; for V4 includes the constant sliding-window floor.
        return self.bytes_per_seq  # caller divides by tokens

    def per_rank_bytes(self, tp, gqa_copy):
        return sum(p.per_rank(tp, gqa_copy) for p in self.parts)


def compute_seq_cache(
    cfg, attention_class, tokens, precision, include_linear_state=False
):
    """Compute the whole-model per-sequence KV cache for one attention class.

    ``cfg`` is a dict of flattened config fields (see loader.py). Returns a
    :class:`SeqCache`.
    """
    tokens = int(tokens)
    if tokens <= 0:
        raise ValueError("tokens must be > 0")

    if attention_class == "standard":
        return _standard(cfg, tokens, precision)
    if attention_class == "mla":
        return _mla(cfg, tokens, precision)
    if attention_class == "dsa":
        return _dsa(cfg, tokens, precision)
    if attention_class == "deepseek_v4":
        return _deepseek_v4(cfg, tokens, precision)
    if attention_class == "mixed_full_sliding":
        return _mixed_full_sliding(cfg, tokens, precision)
    if attention_class == "qwen_linear_full":
        return _qwen_linear_full(cfg, tokens, precision, include_linear_state)
    if attention_class == "minimax_msa":
        return _minimax_msa(cfg, tokens, precision)
    raise ValueError(f"unknown attention class {attention_class!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num(v):
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return None
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def _int(v):
    n = _num(v)
    return int(n) if n is not None else None


def _resolve_head_dim(cfg):
    hd = _num(cfg.get("head_dim"))
    if hd:
        return hd
    hidden = _num(cfg.get("hidden_size"))
    attn = _num(cfg.get("num_attention_heads"))
    if hidden and attn:
        return hidden / attn
    raise ValueError(
        "cannot resolve head_dim: provide head_dim or hidden_size/num_attention_heads"
    )


def _layers(cfg):
    n = _int(cfg.get("num_hidden_layers")) or _int(cfg.get("num_layers"))
    if not n:
        raise ValueError("num_hidden_layers (or num_layers) missing from config")
    return n


def _has_layer_split(cfg):
    """True if the config distinguishes full/sliding/linear layer types."""
    return any(
        cfg.get(k)
        for k in (
            "full_attention_layers",
            "sliding_attention_layers",
            "linear_attention_layers",
            "layer_types",
            "hybrid_layer_pattern",
        )
    )


def _layer_count(cfg, kind):
    from .detect import _layer_count as _lc

    return _lc(cfg, kind)


# ---------------------------------------------------------------------------
# Class formulas
# ---------------------------------------------------------------------------


def _standard(cfg, tokens, prec):
    layers = _layers(cfg)
    attn = _int(cfg.get("num_attention_heads")) or 0
    kv = _int(cfg.get("num_key_value_heads")) or attn
    if not kv:
        raise ValueError("num_key_value_heads (or num_attention_heads) missing")
    head_dim = _resolve_head_dim(cfg)
    # Pure sliding-window attention (Mistral/Muse-Glimmer style: a global
    # sliding_window with no full/sliding layer split): vLLM caps the per-seq
    # cache at the window. Models with a layer split are handled by the
    # mixed class, not here.
    window = _num(cfg.get("sliding_window"))
    eff_tokens = tokens
    note = ""
    if window and 0 < window < tokens and not _has_layer_split(cfg):
        eff_tokens = int(window)
        note = (
            f"pure sliding-window attention (sliding_window={int(window)}, no "
            f"full/sliding layer split); per-seq tokens capped at the window "
            f"(vLLM SWA behavior): {tokens} -> {eff_tokens}"
        )
    bytes_per_seq = layers * 2 * kv * head_dim * eff_tokens * prec.kv
    from .detect import standard_variant

    sub = standard_variant(cfg)
    return SeqCache(
        "standard",
        [
            Part(f"KV ({sub})", bytes_per_seq, "heads", heads_total=kv),
        ],
        note=note,
    )


def _mla(cfg, tokens, prec):
    layers = _layers(cfg)
    kv_lora = _num(cfg.get("kv_lora_rank"))
    qk_rope = _num(cfg.get("qk_rope_head_dim"))
    if not (kv_lora and qk_rope):
        raise ValueError("MLA requires kv_lora_rank and qk_rope_head_dim")
    latent = kv_lora + qk_rope
    bytes_per_seq = layers * latent * tokens * prec.kv
    return SeqCache(
        "mla",
        [
            Part(
                "MLA latent KV (kv_lora_rank + qk_rope_head_dim)", bytes_per_seq, "tp"
            ),
        ],
    )


def _dsa(cfg, tokens, prec):
    layers = _layers(cfg)
    kv_lora = _num(cfg.get("kv_lora_rank"))
    qk_rope = _num(cfg.get("qk_rope_head_dim"))
    index_head_dim = _num(cfg.get("index_head_dim"))
    if not (kv_lora and qk_rope):
        raise ValueError("DSA requires kv_lora_rank and qk_rope_head_dim")
    if not index_head_dim:
        raise ValueError("DSA requires index_head_dim")
    ml_bytes = layers * (kv_lora + qk_rope) * tokens * prec.kv
    # GLM-5.2: indexer_full_layers / indexer_shared_layers -> shared layers
    # reuse the previous full indexer's top-k selection (no independent cache).
    indexer_layers = _int(cfg.get("indexer_full_layers")) or layers
    indexer_bytes = indexer_layers * index_head_dim * tokens * prec.indexer
    return SeqCache(
        "dsa",
        [
            Part("MLA latent KV (kv_lora_rank + qk_rope_head_dim)", ml_bytes, "tp"),
            Part("Lightning Indexer", indexer_bytes, "tp"),
        ],
    )


def _deepseek_v4(cfg, tokens, prec):
    layers = _layers(cfg)
    ratios = cfg.get("compress_ratios") or []
    if not isinstance(ratios, list) or not ratios:
        raise ValueError("DeepSeek V4 requires compress_ratios array")
    head_dim = _num(cfg.get("head_dim"))
    rope = _num(cfg.get("qk_rope_head_dim")) or 0.0
    window = _num(cfg.get("sliding_window"))
    index_head_dim = _num(cfg.get("index_head_dim")) or 0.0
    if not head_dim:
        raise ValueError("DeepSeek V4 requires head_dim")
    if not window:
        raise ValueError("DeepSeek V4 requires sliding_window")

    nope = head_dim - rope
    # Per-entry (per compressed token) bytes: nope dims at FP8, rope dims at BF16.
    # Follows the LMCache paper formula; kvcache.ai uses a single KV precision
    # (slightly less precise). Rope dims in the sliding-window reserve are also
    # kept at BF16 here.
    entry = nope * prec.v4_nope + rope * prec.v4_rope

    compressed = sum((tokens // r) * entry for r in ratios if r and r > 0)
    sliding = layers * window * entry
    indexer_layers = sum(1 for r in ratios if r == 4)
    indexer = indexer_layers * (tokens // 4) * index_head_dim * prec.indexer

    note = (
        "V4 paper formula (config-derived): nope*FP8 + rope*BF16 + indexer*FP4, "
        "sliding-window reserve per layer, compressed = sum(floor(T/ratio)). "
        "Differs from the measured vLLM/vLLM-Ascend bytes/token; the CLI prints "
        "both side by side."
    )
    return SeqCache(
        "deepseek_v4",
        [
            Part("V4 sliding-window reserve", sliding, "tp"),
            Part("V4 compressed KV (sum floor(T/ratio))", compressed, "tp"),
            Part("V4 Lightning Indexer (ratio==4)", indexer, "tp"),
        ],
        note=note,
    )


def _mixed_full_sliding(cfg, tokens, prec):
    layers = _layers(cfg)
    full_layers = _layer_count(cfg, "full")
    sliding_layers = _layer_count(cfg, "sliding")
    window = _num(cfg.get("sliding_window"))
    if not (full_layers and sliding_layers and window):
        raise ValueError(
            "mixed_full_sliding requires full + sliding layer counts and sliding_window"
        )

    # Full-attention layers may use separate "global" heads/dims (Gemma 4).
    full_kv = _int(cfg.get("num_global_key_value_heads")) or _int(
        cfg.get("num_key_value_heads")
    )
    full_hd = _num(cfg.get("global_head_dim")) or _num(cfg.get("head_dim"))
    full_vd = _num(cfg.get("v_head_dim")) or full_hd
    # Sliding-window layers use swa_* fields when present (MiMo), else the
    # standard kv/heads (Gemma 4 sliding layers reuse num_key_value_heads).
    swa_kv = _int(cfg.get("swa_num_key_value_heads")) or _int(
        cfg.get("num_key_value_heads")
    )
    swa_hd = _num(cfg.get("swa_head_dim")) or _num(cfg.get("head_dim"))
    swa_vd = _num(cfg.get("swa_v_head_dim")) or _num(cfg.get("v_head_dim")) or swa_hd

    if not (full_kv and full_hd and swa_kv and swa_hd):
        raise ValueError(
            "mixed_full_sliding: cannot resolve full/swa head counts and dims"
        )

    # Cross-layer KV sharing (Gemma 4 E2B/E4B num_kv_shared_layers): adjacent
    # layers reuse a neighbor's KV, so the allocated (stored) layer count is
    # L - shared. The full/sliding counts from layer_types are scaled by
    # (L - shared)/L, which reproduces the kvcache.ai stored-layer counts
    # exactly (E2B: 7/28 -> 3/12; E4B: 7/35 -> 4/20). When shared=0 (31B,
    # 26B-A4B) the scale is 1 and this is a no-op.
    shared = _int(cfg.get("num_kv_shared_layers")) or 0
    scale = 1.0
    if shared > 0:
        total = _layers(cfg)
        scale = (total - shared) / total
    full_layers_eff = max(1, round(full_layers * scale))
    sliding_layers_eff = max(1, round(sliding_layers * scale))

    # K + V dims are *added* (not 2x head_dim) because head_dim and v_head_dim
    # may differ (MiMo: head 192 / v 128).
    full_bytes = full_layers_eff * full_kv * (full_hd + full_vd) * tokens * prec.kv
    swa_bytes = (
        min(tokens, int(window))
        * sliding_layers_eff
        * swa_kv
        * (swa_hd + swa_vd)
        * prec.kv
    )

    note = ""
    if shared > 0:
        note = (
            f"Cross-layer KV sharing: num_kv_shared_layers={shared} -> effective "
            f"layers scaled by {scale:.3f} (full {full_layers}->{full_layers_eff}, "
            f"sliding {sliding_layers}->{sliding_layers_eff}); matches kvcache.ai "
            f"stored-layer counts."
        )
    else:
        note = (
            "Cross-layer KV sharing (Gemma 4 E2B/E4B num_kv_shared_layers) modeled as "
            "shared layers contributing 0 additional KV; verify for your config."
        )
    return SeqCache(
        "mixed_full_sliding",
        [
            Part("Full-attention KV", full_bytes, "heads", heads_total=full_kv),
            Part(
                "Sliding-window KV (capped at window)",
                swa_bytes,
                "heads",
                heads_total=swa_kv,
            ),
        ],
        note=note,
    )


def _qwen_linear_full(cfg, tokens, prec, include_linear_state):
    full_layers = _layer_count(cfg, "full")
    if not full_layers:
        raise ValueError("qwen_linear_full requires full-attention layer count")
    kv = _int(cfg.get("num_key_value_heads"))
    head_dim = _num(cfg.get("head_dim"))
    if not (kv and head_dim):
        raise ValueError("qwen_linear_full requires num_key_value_heads and head_dim")
    # Only full-attention layers hold token-linear KV; the linear (Gated
    # DeltaNet) layers hold a fixed recurrent/conv state instead.
    full_bytes = full_layers * 2 * kv * head_dim * tokens * prec.kv
    parts = [Part("Full-attention KV", full_bytes, "heads", heads_total=kv)]

    if include_linear_state:
        linear_layers = _layer_count(cfg, "linear")
        if not linear_layers:
            raise ValueError(
                "--include-linear-state set but no linear_attention layers found"
            )
        lkh = _int(cfg.get("linear_num_key_heads")) or 0
        lkhdim = _num(cfg.get("linear_key_head_dim")) or 0
        lvh = _int(cfg.get("linear_num_value_heads")) or 0
        lvhdim = _num(cfg.get("linear_value_head_dim")) or 0
        conv_k = _int(cfg.get("linear_conv_kernel_dim")) or 0
        if not (lkh and lkhdim and lvh and lvhdim and conv_k):
            raise ValueError("linear state requested but linear_* fields incomplete")
        # Conv state (BF16) + recurrent state (FP32), fixed per sequence.
        conv_state = linear_layers * conv_k * (2 * lkh * lkhdim + lvh * lvhdim) * 2.0
        recurrent_state = linear_layers * lvh * lkhdim * lvhdim * 4.0
        parts.append(
            Part(
                "Linear-attention state (conv+recurrent, fixed per seq)",
                conv_state + recurrent_state,
                "fixed",
                note="state is per-sequence; not sharded by TP in this model",
            )
        )
    return SeqCache("qwen_linear_full", parts)


def _minimax_msa(cfg, tokens, prec):
    layers = _layers(cfg)
    kv = _int(cfg.get("num_key_value_heads"))
    head_dim = _resolve_head_dim(cfg)
    if not kv:
        raise ValueError("minimax_msa requires num_key_value_heads")
    # MSA sparse layers still hold full KV (sparse attention reduces compute,
    # not memory) — so all layers count as standard GQA.
    kv_bytes = layers * 2 * kv * head_dim * tokens * prec.kv
    parts = [Part("KV (all layers, standard GQA)", kv_bytes, "heads", heads_total=kv)]
    index_head_dim = _num(cfg.get("index_head_dim"))
    note = ""
    if index_head_dim:
        sparse_layers = (
            _int(cfg.get("sparse_attention_layers"))
            or (layers - _layer_count(cfg, "full"))
            or layers
        )
        indexer_bytes = sparse_layers * index_head_dim * tokens * prec.indexer
        parts.append(Part("MSA Lightning Indexer (sparse layers)", indexer_bytes, "tp"))
    else:
        # MiniMax-M3: real config exposes only standard GQA fields; the MSA
        # indexer side-cache dims aren't in config -> not counted.
        note = (
            "MSA Lightning Indexer side-cache not computed: index_head_dim is "
            "not exposed in the HF config (e.g. MiniMax-M3). The number here is "
            "the standard-GQA KV on all layers only."
        )
    return SeqCache("minimax_msa", parts, note=note)
