"""Attention-architecture classification.

Two-tier strategy, as agreed in the design:

1. **Primary** — `architectures[0]` string looked up in :data:`ARCHITECTURE_REGISTRY`
   (exact match), then family-prefix fallback. This is the deterministic path for
   known model families (DeepSeek V3 / V4, Qwen2/3/3.5, Gemma 4, MiMo V2,
   MiniMax M3, Llama, ...).

2. **Fallback** — field-based inference for anything not in the registry. The
   decision order follows the LMCache calculator (with the bugs of the web tool
   removed: a bare ``sliding_window`` no longer flags hybrid; the SWA/linear
   checks require *both* full and sliding/linear layer counts together so a
   plain GQA model with a window is not misclassified).

The fallback prints an explicit "inferred" note telling the operator to verify.
"""

# Canonical attention class IDs.
STANDARD = "standard"
MLA = "mla"
DSA = "dsa"
DEEPSEEK_V4 = "deepseek_v4"
MIXED_FULL_SLIDING = "mixed_full_sliding"
QWEN_LINEAR_FULL = "qwen_linear_full"
MINIMAX_MSA = "minimax_msa"

ALL_CLASSES = (
    STANDARD,
    MLA,
    DSA,
    DEEPSEEK_V4,
    MIXED_FULL_SLIDING,
    QWEN_LINEAR_FULL,
    MINIMAX_MSA,
)

CLASS_LABELS = {
    STANDARD: "Standard (MHA/MQA/GQA)",
    MLA: "MLA (latent KV)",
    DSA: "DSA (MLA + Lightning Indexer)",
    DEEPSEEK_V4: "DeepSeek V4 (compressed sparse attention)",
    MIXED_FULL_SLIDING: "Mixed full/sliding attention",
    QWEN_LINEAR_FULL: "Qwen linear/full hybrid",
    MINIMAX_MSA: "MiniMax MSA (sparse attention)",
}

# architectures[0] -> attention class. All entries below were VERIFIED
# against the real HuggingFace config.json on 2026-08-18 via `hf download
# <repo> config.json` (see kv_calc/presets.py docstring for the exceptions:
# Llama is gated, Kimi-K2 base repo not separately published).
ARCHITECTURE_REGISTRY = {
    # DeepSeek (verified)
    "DeepseekV3ForCausalLM": MLA,  # V3, R1, V3.1-Terminus, Kimi K2.x
    "DeepseekV32ForCausalLM": DSA,  # V3.2 (note "V32", no underscore)
    "DeepseekV4ForCausalLM": DEEPSEEK_V4,
    # Qwen (verified)
    "Qwen3ForCausalLM": STANDARD,  # dense Qwen3 (32B)
    "Qwen3MoeForCausalLM": STANDARD,  # MoE Qwen3 (235B-A22B, Coder-480B)
    "Qwen3_5ForConditionalGeneration": QWEN_LINEAR_FULL,  # Qwen3.5/3.6
    # GLM (verified, zai-org)
    "Glm4MoeForCausalLM": STANDARD,  # GLM-4.5, 4.5-Air, 4.7 (GQA)
    "Glm4MoeLiteForCausalLM": MLA,  # GLM-4.7-Flash (MLA latent)
    "GlmMoeDsaForCausalLM": DSA,  # GLM-5, 5.1 (MLA + Lightning Indexer)
    # MiniMax (verified, MiniMaxAI)
    "MiniMaxM2ForCausalLM": STANDARD,  # M2, M2.1, M2.5 (GQA)
    # Llama (canonical HF class; gated repo, not freshly re-fetched)
    "LlamaForCausalLM": STANDARD,
    # Mistral (verified; Ministral-8B, Mistral-Large-2411)
    "MistralForCausalLM": STANDARD,
    # Verified via the raw repo JSON files (cross-checked against HF)
    "Gemma4ForConditionalGeneration": MIXED_FULL_SLIDING,  # google/gemma-4-31b-it
    "MiMoV2ForCausalLM": MIXED_FULL_SLIDING,  # XiaomiMiMo/MiMo-V2.5
    # --- newly verified 2026-08-18 (hf download) ---
    "MuseGlimmerForConditionalGeneration": STANDARD,  # meta-models/Muse-Glimmer-30B
    "KimiK3ForConditionalGeneration": MLA,  # moonshotai/Kimi-K3 (config exposes MLA fields; KDA state not in config)
    "KimiK25ForConditionalGeneration": MLA,  # moonshotai/Kimi-K2.7-Code
    "Dots3NoteForCausalLM": DSA,  # dots-studio/dots3-note-prev
    "BailingMoeV3ForCausalLM": MLA,  # inclusionAI/Ling-3.0-tiny/flash (config exposes MLA; "bailing_hybrid" KDA state not in config)
    "OpenPanguV2ForCausalLM": DSA,  # openpangu/openPangu-2.0-Pro/Flash
    "LongcatCausalLM": DSA,  # meituan-longcat/LongCat-2.0 (num_layers field)
    "Step3p5ForCausalLM": MIXED_FULL_SLIDING,  # stepfun-ai/Step-3.5-Flash (grouped head-gated; approx)
    "Step3p7ForConditionalGeneration": MIXED_FULL_SLIDING,  # stepfun-ai/Step-3.7-Flash (grouped head-gated; approx)
    "Qwen3_5MoeForCausalLM": QWEN_LINEAR_FULL,  # Qwen3.5/3.6/3.8 MoE (Gated DeltaNet)
    "Qwen3_5MoeForConditionalGeneration": QWEN_LINEAR_FULL,
    # MiniMax M3 — verified: real arch is MiniMaxM3SparseForConditionalGeneration
    # (model_type minimax_m3_vl). Config does NOT expose index_head_dim/sparse
    # layers, so the MSA indexer side-cache can't be computed from config; the
    # formula falls back to standard GQA on all layers + a note.
    "MiniMaxM3SparseForConditionalGeneration": MINIMAX_MSA,
}

# Family prefixes, ordered; first match wins. Only used when the exact
# string is not in ARCHITECTURE_REGISTRY (e.g. an unlisted variant like
# "Qwen3_6ForConditionalGeneration"). NOTE: do NOT add a bare "DeepseekV3"
# prefix — it would also match "DeepseekV32ForCausalLM" and mis-classify
# V3.2 as MLA; V3.2 is covered by the exact entry above.
ARCHITECTURE_PREFIXES = (
    ("Qwen3_6", QWEN_LINEAR_FULL),
    ("Qwen3_5", QWEN_LINEAR_FULL),
    ("DeepseekV4", DEEPSEEK_V4),
    ("MiMoV2", MIXED_FULL_SLIDING),
    ("Gemma4", MIXED_FULL_SLIDING),
    ("MiniMaxM3", MINIMAX_MSA),
)


class Classification:
    """Outcome of classifying a model config."""

    __slots__ = ("attention_class", "label", "method", "arch_string", "note")

    def __init__(self, attention_class, label, method, arch_string, note):
        self.attention_class = attention_class
        self.label = label
        self.method = method  # "registry" | "prefix" | "inferred"
        self.arch_string = arch_string
        self.note = note

    def __repr__(self):
        return (
            f"Classification(class={self.attention_class!r}, method={self.method!r}, "
            f"arch={self.arch_string!r})"
        )


def standard_variant(cfg):
    """Sub-label for the standard class: MHA / MQA / GQA.

    Falls back to the generic ratio formula when fields are incomplete.
    """
    kv = _num(cfg.get("num_key_value_heads"))
    attn = _num(cfg.get("num_attention_heads"))
    if kv is None or attn is None or attn == 0:
        return "GQA"
    if kv == attn:
        return "MHA"
    if kv == 1:
        return "MQA"
    return "GQA"


def classify(architectures, cfg):
    """Classify a loaded model config (dict of flattened fields).

    ``architectures`` is the list from the HF config (may be None/empty for
    externally-loaded configs whose source did not include it).
    """
    arch = (architectures or [None])[0]

    if arch:
        if arch in ARCHITECTURE_REGISTRY:
            cls = ARCHITECTURE_REGISTRY[arch]
            return Classification(cls, CLASS_LABELS[cls], "registry", arch, "")
        for prefix, cls in ARCHITECTURE_PREFIXES:
            if arch.startswith(prefix):
                return Classification(cls, CLASS_LABELS[cls], "prefix", arch, "")

    cls, detail = infer_attention_class(cfg)
    note = (
        f"Architecture '{arch or '?'}' not in the registry; inferred as "
        f"{CLASS_LABELS[cls]} ({detail}). Please verify against the model "
        f"config before trusting the number."
    )
    return Classification(cls, CLASS_LABELS[cls], "inferred", arch, note)


def infer_attention_class(cfg):
    """Field-based fallback. Returns (class, short_detail_string)."""
    head_dim = _num(cfg.get("head_dim"))
    has_compress_ratios = bool(cfg.get("compress_ratios"))
    kv_lora = _num(cfg.get("kv_lora_rank"))
    qk_rope = _num(cfg.get("qk_rope_head_dim"))
    index_head_dim = _num(cfg.get("index_head_dim"))

    full_layers = _layer_count(cfg, "full")
    sliding_layers = _layer_count(cfg, "sliding")
    linear_layers = _layer_count(cfg, "linear")
    window = _num(cfg.get("sliding_window"))

    # 1. DeepSeek V4-style compressed sparse attention: per-layer compress
    #    ratios + head_dim.
    if has_compress_ratios and head_dim:
        return DEEPSEEK_V4, "compress_ratios + head_dim"

    # 2. Full + sliding window mix (Gemma 4 / MiMo / Cohere R7B). Requires
    #    BOTH layer counts and a window so a bare sliding_window on a plain
    #    GQA model does not trip this.
    if (
        full_layers
        and sliding_layers
        and window
        and _num(cfg.get("num_key_value_heads"))
        and head_dim
    ):
        return MIXED_FULL_SLIDING, (
            f"full_attention={full_layers}, sliding_attention={sliding_layers}, "
            f"window={window}"
        )

    # 3. Full + linear mix (Qwen 3.5/3.6 Gated DeltaNet).
    if full_layers and linear_layers:
        return QWEN_LINEAR_FULL, (
            f"full_attention={full_layers}, linear_attention={linear_layers}"
        )

    # 4-5. MLA / DSA via latent KV dims.
    if kv_lora and qk_rope:
        if index_head_dim:
            return DSA, f"kv_lora_rank={kv_lora}, index_head_dim={index_head_dim}"
        return MLA, f"kv_lora_rank={kv_lora}, qk_rope_head_dim={qk_rope}"

    # 6. Gemma4-flavor fallback: global head fields without explicit layer
    #    split (older multi-modal configs).
    if (
        cfg.get("global_head_dim")
        or cfg.get("num_global_key_value_heads")
        or cfg.get("num_kv_shared_layers")
    ):
        return MIXED_FULL_SLIDING, "global_head_dim / num_kv_shared_layers present"

    # 7. Standard MHA/MQA/GQA (with explicit head_dim).
    if head_dim:
        return STANDARD, f"{standard_variant(cfg)} (kv_heads vs attn_heads ratio)"

    # 8. Generic fallback (Llama-style without head_dim).
    return STANDARD, "generic ratio fallback (no head_dim)"


def _num(v):
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return None
        f = float(v)
        return f if f == int(f) else f
    except (TypeError, ValueError):
        return None


def _layer_count(cfg, kind):
    """Count layers of a given attention kind from explicit counts or arrays.

    ``kind`` is one of "full" / "sliding" / "linear". Handles both the
    explicit ``<kind>_attention_layers`` numeric form and the
    ``layer_types`` / ``hybrid_layer_pattern`` array forms.
    """
    explicit = {
        "full": "full_attention_layers",
        "sliding": "sliding_attention_layers",
        "linear": "linear_attention_layers",
    }[kind]
    n = _num(cfg.get(explicit))
    if n is not None:
        return int(n)

    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list):
        target = {
            "full": "full_attention",
            "sliding": "sliding_attention",
            "linear": "linear_attention",
        }[kind]
        return sum(1 for t in layer_types if t == target)

    pattern = cfg.get("hybrid_layer_pattern")
    if isinstance(pattern, list):
        # MiMo convention: 0 = full, 1 = sliding (see presets.py note).
        target = {"full": 0, "sliding": 1, "linear": None}[kind]
        if target is None:
            return 0
        return sum(1 for v in pattern if v == target)

    return 0
