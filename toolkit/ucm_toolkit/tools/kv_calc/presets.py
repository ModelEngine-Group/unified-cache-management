"""Built-in preset model database.

All preset entries below were VERIFIED against the real HuggingFace
``config.json`` on 2026-08-18 via the ``hf`` CLI (``hf download <repo>
config.json``). The ``architectures`` strings, ``model_type`` values, and
attention-relevant fields are real — not guessed. Two exceptions:

* ``meta-llama/Llama-3.1-*`` repos are gated and could not be re-fetched;
  ``LlamaForCausalLM`` is the canonical HF class (certain, not a guess).
* ``moonshotai/Kimi-K2`` base repo is not separately published; its fields
  follow K2.5/K2.6 (same ``model_type kimi_k2``, same arch).

DeepSeek V4 deployment constants (``V4_MEASURED``) come from the web tool's
``calculator.js`` (DEEPSEEK_V4_CONFIGS — empirically validated on real
vLLM / vLLM-Ascend deployments; authoritative, do not recompute).
"""

from .detect import CLASS_LABELS


def _v4_compress_ratios_pro():
    # Verified from HF deepseek-ai/DeepSeek-V4-Pro config.json: 62 entries
    # (31 x 128, 30 x 4, 1 x 0 = trailing draft). ratio 0 = sliding-window-only
    # layer, 4 = CSA (top-k sparse), 128 = HCA (dense global summary).
    return [
        128,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ]


def _v4_compress_ratios_flash():
    # Verified from HF deepseek-ai/DeepSeek-V4-Flash config.json: 44 entries
    # (3 x 0, 21 x 4, 20 x 128).
    return [
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ]


# Measured DeepSeek V4 per-token bytes, from calculator.js DEEPSEEK_V4_CONFIGS.
# Absolute B/token (precision already baked in), whole-model figure;
# per-GPU = bpt * tokens / tp.
V4_MEASURED = {
    "deepseek-ai/DeepSeek-V4-Pro": {
        "vllm": {"bytes_per_token": 28415.4375, "block_tokens": 256},
        "vllm-ascend": {"bytes_per_token": 27175.0, "block_tokens": 512},
    },
    "deepseek-ai/DeepSeek-V4-Flash": {
        "vllm": {"bytes_per_token": 20058.25, "block_tokens": 256},
        "vllm-ascend": {"bytes_per_token": 19162.5, "block_tokens": 512},
    },
}


def _gemma4_layer_types():
    # Verified from HF google/gemma-4-31b-it text_config.layer_types: 60
    # entries, one full_attention every 6th layer -> 10 full + 50 sliding.
    return ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]


def _mimo_v2_5_pattern():
    # Verified from HF XiaomiMiMo/MiMo-V2.5 hybrid_layer_pattern: 48 entries,
    # 39 x 1 + 9 x 0 (zeros at 1-based positions 1, 6, 12, 18, 24, 30, 36, 42,
    # 48). Cross-checked with the kvcache.ai dataset (9 full / 39 sliding):
    # here 1 = sliding_attention layer, 0 = full_attention layer.
    zeros = {0, 5, 11, 17, 23, 29, 35, 41, 47}
    return [0 if i in zeros else 1 for i in range(48)]


def _qwen36_layer_types():
    # Verified from HF Qwen/Qwen3.6-27B text_config.layer_types: 64 entries,
    # full every 4th -> 16 full + 48 linear_attention.
    return ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]


_PRESET_MODELS = [
    # ------------------------------------------------------------------
    # MLA — DeepSeek V3 / R1 / V3.1-Terminus
    # arch: DeepseekV3ForCausalLM (verified)
    # ------------------------------------------------------------------
    dict(
        id="deepseek-ai/DeepSeek-V3",
        architectures=["DeepseekV3ForCausalLM"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=128,
        num_hidden_layers=61,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    dict(
        id="deepseek-ai/DeepSeek-R1",
        architectures=["DeepseekV3ForCausalLM"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=128,
        num_hidden_layers=61,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    dict(
        id="deepseek-ai/DeepSeek-V3.1-Terminus",
        architectures=["DeepseekV3ForCausalLM"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=128,
        num_hidden_layers=61,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    # ------------------------------------------------------------------
    # DSA — DeepSeek V3.2
    # arch: DeepseekV32ForCausalLM (verified; note "V32", no underscore)
    # ------------------------------------------------------------------
    dict(
        id="deepseek-ai/DeepSeek-V3.2",
        architectures=["DeepseekV32ForCausalLM"],
        attention_class="dsa",
        hidden_size=7168,
        num_attention_heads=128,
        num_hidden_layers=61,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=2048,
        v_head_dim=128,
    ),
    # ------------------------------------------------------------------
    # DeepSeek V4 — compressed sparse attention + measured deployments
    # arch: DeepseekV4ForCausalLM (verified); compress_ratios verified
    # ------------------------------------------------------------------
    dict(
        id="deepseek-ai/DeepSeek-V4-Pro",
        architectures=["DeepseekV4ForCausalLM"],
        attention_class="deepseek_v4",
        hidden_size=7168,
        num_attention_heads=128,
        num_hidden_layers=61,
        num_key_value_heads=1,
        head_dim=512,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=1024,
        qk_rope_head_dim=64,
        sliding_window=128,
        compress_ratios=_v4_compress_ratios_pro(),
        deployment_measured=V4_MEASURED["deepseek-ai/DeepSeek-V4-Pro"],
    ),
    dict(
        id="deepseek-ai/DeepSeek-V4-Flash",
        architectures=["DeepseekV4ForCausalLM"],
        attention_class="deepseek_v4",
        hidden_size=4096,
        num_attention_heads=64,
        num_hidden_layers=43,
        num_key_value_heads=1,
        head_dim=512,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        qk_rope_head_dim=64,
        sliding_window=128,
        compress_ratios=_v4_compress_ratios_flash(),
        deployment_measured=V4_MEASURED["deepseek-ai/DeepSeek-V4-Flash"],
    ),
    # ------------------------------------------------------------------
    # MLA — GLM-4.7-Flash (arch: Glm4MoeLiteForCausalLM, verified)
    # ------------------------------------------------------------------
    dict(
        id="zai-org/GLM-4.7-Flash",
        architectures=["Glm4MoeLiteForCausalLM"],
        attention_class="mla",
        hidden_size=2048,
        num_attention_heads=20,
        num_hidden_layers=47,
        num_key_value_heads=20,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=256,
    ),
    # ------------------------------------------------------------------
    # DSA — GLM-5 / 5.1 (arch: GlmMoeDsaForCausalLM, verified)
    # ------------------------------------------------------------------
    dict(
        id="zai-org/GLM-5",
        architectures=["GlmMoeDsaForCausalLM"],
        attention_class="dsa",
        hidden_size=6144,
        num_attention_heads=64,
        num_hidden_layers=78,
        num_key_value_heads=64,
        head_dim=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        v_head_dim=256,
    ),
    dict(
        id="zai-org/GLM-5.1",
        architectures=["GlmMoeDsaForCausalLM"],
        attention_class="dsa",
        hidden_size=6144,
        num_attention_heads=64,
        num_hidden_layers=78,
        num_key_value_heads=64,
        head_dim=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        v_head_dim=256,
    ),
    # ------------------------------------------------------------------
    # MLA — Kimi K2.x (reuse DeepseekV3ForCausalLM arch; model_type kimi_k2)
    # K2.5/K2.6 verified via HF (moonshotai/); K2 base repo not separately
    # published — fields follow K2.5/K2.6 (same model_type).
    # ------------------------------------------------------------------
    dict(
        id="moonshotai/Kimi-K2",
        architectures=["DeepseekV3ForCausalLM"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=64,
        num_hidden_layers=61,
        num_key_value_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    dict(
        id="moonshotai/Kimi-K2.5",
        architectures=["DeepseekV3ForCausalLM"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=64,
        num_hidden_layers=61,
        num_key_value_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    dict(
        id="moonshotai/Kimi-K2.6",
        architectures=["DeepseekV3ForCausalLM"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=64,
        num_hidden_layers=61,
        num_key_value_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    # ------------------------------------------------------------------
    # Standard (GQA) — Qwen3 dense (arch: Qwen3ForCausalLM, verified)
    # ------------------------------------------------------------------
    dict(
        id="Qwen/Qwen3-32B",
        architectures=["Qwen3ForCausalLM"],
        attention_class="standard",
        hidden_size=5120,
        num_attention_heads=64,
        num_hidden_layers=64,
        num_key_value_heads=8,
        head_dim=128,
    ),
    # ------------------------------------------------------------------
    # Standard (GQA) — Qwen3 MoE (arch: Qwen3MoeForCausalLM, verified;
    # MoE does not change the per-token KV cache)
    # ------------------------------------------------------------------
    dict(
        id="Qwen/Qwen3-235B-A22B",
        architectures=["Qwen3MoeForCausalLM"],
        attention_class="standard",
        hidden_size=4096,
        num_attention_heads=64,
        num_hidden_layers=94,
        num_key_value_heads=4,
        head_dim=128,
    ),
    dict(
        id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        architectures=["Qwen3MoeForCausalLM"],
        attention_class="standard",
        hidden_size=6144,
        num_attention_heads=96,
        num_hidden_layers=62,
        num_key_value_heads=8,
        head_dim=128,
    ),
    # ------------------------------------------------------------------
    # Standard (GQA) — Llama 3.1 (arch: LlamaForCausalLM, canonical HF class;
    # repo gated, not freshly re-fetched)
    # ------------------------------------------------------------------
    dict(
        id="meta-llama/Llama-3.1-70B-Instruct",
        architectures=["LlamaForCausalLM"],
        attention_class="standard",
        hidden_size=8192,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8,
    ),
    dict(
        id="meta-llama/Llama-3.1-405B",
        architectures=["LlamaForCausalLM"],
        attention_class="standard",
        hidden_size=16384,
        num_attention_heads=128,
        num_hidden_layers=126,
        num_key_value_heads=8,
    ),
    # ------------------------------------------------------------------
    # Standard (GQA) — GLM-4.5 family (arch: Glm4MoeForCausalLM, verified)
    # ------------------------------------------------------------------
    dict(
        id="zai-org/GLM-4.5",
        architectures=["Glm4MoeForCausalLM"],
        attention_class="standard",
        hidden_size=5120,
        num_attention_heads=96,
        num_hidden_layers=92,
        num_key_value_heads=8,
        head_dim=128,
    ),
    dict(
        id="zai-org/GLM-4.5-Air",
        architectures=["Glm4MoeForCausalLM"],
        attention_class="standard",
        hidden_size=4096,
        num_attention_heads=96,
        num_hidden_layers=46,
        num_key_value_heads=8,
        head_dim=128,
    ),
    dict(
        id="zai-org/GLM-4.7",
        architectures=["Glm4MoeForCausalLM"],
        attention_class="standard",
        hidden_size=5120,
        num_attention_heads=96,
        num_hidden_layers=92,
        num_key_value_heads=8,
        head_dim=128,
    ),
    # ------------------------------------------------------------------
    # Standard (GQA) — MiniMax M2 family (arch: MiniMaxM2ForCausalLM;
    # org MiniMaxAI, verified)
    # ------------------------------------------------------------------
    dict(
        id="MiniMaxAI/MiniMax-M2",
        architectures=["MiniMaxM2ForCausalLM"],
        attention_class="standard",
        hidden_size=3072,
        num_attention_heads=48,
        num_hidden_layers=62,
        num_key_value_heads=8,
        head_dim=128,
    ),
    dict(
        id="MiniMaxAI/MiniMax-M2.1",
        architectures=["MiniMaxM2ForCausalLM"],
        attention_class="standard",
        hidden_size=3072,
        num_attention_heads=48,
        num_hidden_layers=62,
        num_key_value_heads=8,
        head_dim=128,
    ),
    dict(
        id="MiniMaxAI/MiniMax-M2.5",
        architectures=["MiniMaxM2ForCausalLM"],
        attention_class="standard",
        hidden_size=3072,
        num_attention_heads=48,
        num_hidden_layers=62,
        num_key_value_heads=8,
        head_dim=128,
    ),
    # ------------------------------------------------------------------
    # Mixed full/sliding — Gemma 4 31B (arch: Gemma4ForConditionalGeneration;
    # verified at google/gemma-4-31b-it)
    # ------------------------------------------------------------------
    dict(
        id="google/gemma-4-31b-it",
        architectures=["Gemma4ForConditionalGeneration"],
        attention_class="mixed_full_sliding",
        hidden_size=5376,
        num_attention_heads=32,
        num_hidden_layers=60,
        num_key_value_heads=16,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=4,
        sliding_window=1024,
        num_kv_shared_layers=0,
        layer_types=_gemma4_layer_types(),
    ),
    # ------------------------------------------------------------------
    # Mixed full/sliding — MiMo V2.5 (arch: MiMoV2ForCausalLM; org XiaomiMiMo,
    # verified)
    # ------------------------------------------------------------------
    dict(
        id="XiaomiMiMo/MiMo-V2.5",
        architectures=["MiMoV2ForCausalLM"],
        attention_class="mixed_full_sliding",
        hidden_size=4096,
        num_attention_heads=64,
        num_hidden_layers=48,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_key_value_heads=8,
        swa_num_attention_heads=64,
        swa_head_dim=192,
        swa_v_head_dim=128,
        sliding_window=128,
        hybrid_layer_pattern=_mimo_v2_5_pattern(),
    ),
    # ------------------------------------------------------------------
    # Qwen linear/full hybrid — Qwen3.6-27B (arch: Qwen3_5ForConditionalGeneration,
    # verified at Qwen/Qwen3.6-27B)
    # ------------------------------------------------------------------
    dict(
        id="Qwen/Qwen3.6-27B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=5120,
        num_attention_heads=24,
        num_hidden_layers=64,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=_qwen36_layer_types(),
    ),
    # ==================================================================
    # Additional hot models — verified via `hf download` 2026-08-18.
    # Entries with a `note` flag an approximation that the operator should
    # sanity-check against the real serving engine.
    # ==================================================================
    # --- Qwen3.5 / 3.6 / 3.8 linear-full family (Gated DeltaNet) ---
    dict(
        id="Qwen/Qwen3.5-27B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=5120,
        num_attention_heads=24,
        num_hidden_layers=64,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-9B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=4096,
        num_attention_heads=16,
        num_hidden_layers=32,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-4B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=2560,
        num_attention_heads=16,
        num_hidden_layers=32,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-2B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=2048,
        num_attention_heads=8,
        num_hidden_layers=24,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=16,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-0.8B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=1024,
        num_attention_heads=8,
        num_hidden_layers=24,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=16,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-122B-A10B",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=3072,
        num_attention_heads=32,
        num_hidden_layers=48,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-397B-A17B",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=60,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=64,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.5-35B-A3B",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.6-35B-A3B",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.8-27B",
        architectures=["Qwen3_5ForConditionalGeneration"],
        attention_class="qwen_linear_full",
        hidden_size=5120,
        num_attention_heads=24,
        num_hidden_layers=64,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    dict(
        id="Qwen/Qwen3.8-2.4T-A95B",
        architectures=["Qwen3_5MoeForCausalLM"],
        attention_class="qwen_linear_full",
        hidden_size=8192,
        num_attention_heads=64,
        num_hidden_layers=92,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    ),
    # --- Kimi K3 / K2.7-Code (MLA; KDA recurrent state NOT in config) ---
    dict(
        id="moonshotai/Kimi-K3",
        architectures=["KimiK3ForConditionalGeneration"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=96,
        num_hidden_layers=93,
        num_key_value_heads=96,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        note="Kimi K3 is a KDA/MLA hybrid; the HF config exposes only the MLA "
        "latent fields, so the KDA recurrent/conv state is NOT counted "
        "(needs engine-specific info). The number here is the MLA-latent "
        "lower bound; add --include-linear-state once KDA dims are known.",
    ),
    dict(
        id="moonshotai/Kimi-K2.7-Code",
        architectures=["KimiK25ForConditionalGeneration"],
        attention_class="mla",
        hidden_size=7168,
        num_attention_heads=64,
        num_hidden_layers=61,
        num_key_value_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    # --- DSA family (MLA + Lightning Indexer), newly verified ---
    dict(
        id="dots-studio/dots3-note-prev",
        architectures=["Dots3NoteForCausalLM"],
        attention_class="dsa",
        hidden_size=5120,
        num_attention_heads=128,
        num_hidden_layers=46,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=2048,
        v_head_dim=128,
        swa_v_head_dim=128,
        swa_num_key_value_heads=64,
        note="Config carries swa_* fields but no full/sliding layer split; "
        "all layers counted as MLA+indexer (may overcount if some are "
        "pure sliding-window).",
    ),
    dict(
        id="openpangu/openPangu-2.0-Pro",
        architectures=["OpenPanguV2ForCausalLM"],
        attention_class="dsa",
        hidden_size=5120,
        num_attention_heads=64,
        num_hidden_layers=50,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        sliding_window=512,
        v_head_dim=128,
    ),
    dict(
        id="openpangu/openPangu-2.0-Flash",
        architectures=["OpenPanguV2ForCausalLM"],
        attention_class="dsa",
        hidden_size=2560,
        num_attention_heads=48,
        num_hidden_layers=46,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=24,
        index_topk=2048,
        sliding_window=512,
        v_head_dim=128,
    ),
    dict(
        id="meituan-longcat/LongCat-2.0",
        architectures=["LongcatCausalLM"],
        attention_class="dsa",
        num_layers=38,
        hidden_size=8192,
        num_attention_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        v_head_dim=128,
        note="Config uses 'num_layers' (not num_hidden_layers); both are "
        "read. attention_method=MLA + Lightning Indexer -> DSA.",
    ),
    dict(
        id="zai-org/GLM-5.2",
        architectures=["GlmMoeDsaForCausalLM"],
        attention_class="dsa",
        hidden_size=6144,
        num_attention_heads=64,
        num_hidden_layers=78,
        num_key_value_heads=64,
        head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        v_head_dim=256,
        note="GLM-5.2 shares the Lightning Indexer across layer groups in "
        "the real engine; the HF config does not expose "
        "indexer_full_layers/indexer_shared_layers, so the indexer is "
        "counted on all 78 layers (overcount vs the 21-full shared "
        "layout).",
    ),
    # --- MLA family, newly verified ---
    dict(
        id="inclusionAI/Ling-3.0-tiny",
        architectures=["BailingMoeV3ForCausalLM"],
        attention_class="mla",
        hidden_size=1536,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_key_value_heads=16,
        head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        note="model_type bailing_hybrid suggests KDA/linear layers, but the "
        "config exposes only MLA latent fields; KDA state not counted.",
    ),
    dict(
        id="inclusionAI/Ling-3.0-flash",
        architectures=["BailingMoeV3ForCausalLM"],
        attention_class="mla",
        hidden_size=2560,
        num_attention_heads=32,
        num_hidden_layers=42,
        num_key_value_heads=32,
        head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        note="model_type bailing_hybrid suggests KDA/linear layers, but the "
        "config exposes only MLA latent fields; KDA state not counted.",
    ),
    # --- Standard (GQA), newly verified ---
    dict(
        id="meta-models/Muse-Glimmer-30B",
        architectures=["MuseGlimmerForConditionalGeneration"],
        attention_class="standard",
        hidden_size=6656,
        num_attention_heads=32,
        num_hidden_layers=52,
        num_key_value_heads=2,
        head_dim=128,
        sliding_window=2048,
        note="Pure sliding-window attention (sliding_window=2048, no full/"
        "sliding layer split); per-seq cache is capped at the window "
        "(vLLM SWA behavior).",
    ),
    dict(
        id="mistralai/Mistral-Large-Instruct-2411",
        architectures=["MistralForCausalLM"],
        attention_class="standard",
        hidden_size=12288,
        num_attention_heads=96,
        num_hidden_layers=88,
        num_key_value_heads=8,
        head_dim=128,
    ),
    dict(
        id="mistralai/Ministral-8B-Instruct-2410",
        architectures=["MistralForCausalLM"],
        attention_class="standard",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=36,
        num_key_value_heads=8,
        head_dim=128,
        sliding_window=32768,
        note="Pure sliding-window attention (sliding_window=32768); per-seq "
        "cache capped at the window for T > 32768.",
    ),
    # --- Gemma 4 family (mixed full/sliding + cross-layer KV sharing) ---
    dict(
        id="google/gemma-4-e2b-it",
        architectures=["Gemma4ForConditionalGeneration"],
        attention_class="mixed_full_sliding",
        hidden_size=1536,
        num_attention_heads=8,
        num_hidden_layers=35,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        sliding_window=512,
        num_kv_shared_layers=20,
        full_attention_layers=7,
        sliding_attention_layers=28,
    ),
    dict(
        id="google/gemma-4-e4b-it",
        architectures=["Gemma4ForConditionalGeneration"],
        attention_class="mixed_full_sliding",
        hidden_size=2560,
        num_attention_heads=8,
        num_hidden_layers=42,
        num_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        sliding_window=512,
        num_kv_shared_layers=18,
        full_attention_layers=7,
        sliding_attention_layers=35,
    ),
    dict(
        id="google/gemma-4-26b-a4b-it",
        architectures=["Gemma4ForConditionalGeneration"],
        attention_class="mixed_full_sliding",
        hidden_size=2816,
        num_attention_heads=16,
        num_hidden_layers=30,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=2,
        sliding_window=1024,
        num_kv_shared_layers=0,
        full_attention_layers=5,
        sliding_attention_layers=25,
    ),
    # --- Step 3.5 / 3.7 Flash (mixed full/sliding; grouped head-gated) ---
    dict(
        id="stepfun-ai/Step-3.5-Flash",
        architectures=["Step3p5ForCausalLM"],
        attention_class="mixed_full_sliding",
        hidden_size=4096,
        num_attention_heads=64,
        num_hidden_layers=45,
        num_key_value_heads=8,
        head_dim=128,
        sliding_window=512,
        full_attention_layers=12,
        sliding_attention_layers=36,
        note="Step uses grouped head-gated attention; num_key_value_heads=8 "
        "is interpreted from num_attention_groups=8. The number is an "
        "approximation (upper bound on KV groups).",
    ),
    dict(
        id="stepfun-ai/Step-3.7-Flash",
        architectures=["Step3p7ForConditionalGeneration"],
        attention_class="mixed_full_sliding",
        hidden_size=4096,
        num_attention_heads=64,
        num_hidden_layers=45,
        num_key_value_heads=8,
        head_dim=128,
        sliding_window=512,
        full_attention_layers=12,
        sliding_attention_layers=36,
        note="Step uses grouped head-gated attention; num_key_value_heads=8 "
        "is interpreted from num_attention_groups=8. The number is an "
        "approximation (upper bound on KV groups).",
    ),
    # --- MiniMax M3 (MSA sparse attention; config exposes only GQA fields) ---
    dict(
        id="MiniMaxAI/MiniMax-M3",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        attention_class="minimax_msa",
        hidden_size=6144,
        num_attention_heads=64,
        num_hidden_layers=60,
        num_key_value_heads=4,
        head_dim=128,
        num_mtp_modules=7,
        num_nextn_predict_layers=1,
        note="MiniMax-M3 is MSA (sparse attention), but the HF config does "
        "not expose index_head_dim / sparse layer counts, so the MSA "
        "Lightning Indexer side-cache is NOT counted. The number here is "
        "the standard-GQA KV on all 60 layers only.",
    ),
]


_META_KEYS = (
    "id",
    "architectures",
    "attention_class",
    "deployment_measured",
    "fields",
    "note",
)


def _register(entry, presets, aliases, source="builtin"):
    """Register one preset dict (flat form: id + architectures + fields at
    top level, optional ``attention_class`` and ``deployment_measured``).

    If ``attention_class`` is omitted it is auto-derived from
    ``architectures`` via the detect registry/inference — so a new preset
    can be just an HF config dump with an id.
    """
    entry = dict(entry)
    mid = entry.get("id")
    if not mid:
        raise ValueError("preset entry missing 'id'")
    fields = {k: v for k, v in entry.items() if k not in _META_KEYS}
    archs = list(entry.get("architectures") or [])
    attention_class = entry.get("attention_class")
    if not attention_class:
        from .detect import classify

        attention_class = classify(archs, fields).attention_class
    record = {
        "id": mid,
        "architectures": archs,
        "attention_class": attention_class,
        "fields": fields,
        "source": source,
    }
    if "deployment_measured" in entry:
        record["deployment_measured"] = entry["deployment_measured"]
    if entry.get("note"):
        record["note"] = entry["note"]
    presets[mid] = record
    short = mid.split("/")[-1].lower()
    aliases[mid.lower()] = mid
    aliases[short] = mid
    aliases["/".join(p.lower() for p in mid.split("/"))] = mid
    return record


def _build_index():
    presets = {}
    aliases = {}
    for entry in _PRESET_MODELS:
        _register(entry, presets, aliases, source="builtin")
    return presets, aliases


PRESETS, ALIASES = _build_index()


def load_user_presets(path):
    """Merge extra presets from a JSON file into the global registries.

    The JSON is a list (or ``{"models": [...]}``) of preset entries in the
    same flat form as ``_PRESET_MODELS``. ``attention_class`` is optional
    (auto-derived from ``architectures``). Entries override built-ins with
    the same id. Returns the list of ids loaded.
    """
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data if isinstance(data, list) else data.get("models", [])
    loaded = []
    for entry in entries:
        rec = _register(entry, PRESETS, ALIASES, source=f"user:{path}")
        loaded.append(rec["id"])
    return loaded


def get_preset(name):
    """Return the preset entry for ``name`` (exact id or alias) or None."""
    preset_id = (
        name if name in PRESETS else ALIASES.get(name.lower().strip().lstrip("/"))
    )
    if preset_id is None:
        return None
    return PRESETS[preset_id]


def list_presets():
    """Return preset entries sorted by id."""
    return [PRESETS[i] for i in sorted(PRESETS)]


def preset_short_name(preset):
    return preset["id"].split("/")[-1]


def preset_arch_label(preset):
    arch = (preset.get("architectures") or ["(none)"])[0]
    cls = preset["attention_class"]
    return f"{arch} -> {CLASS_LABELS.get(cls, cls)} (preset)"


def key_params_str(entry):
    """Compact one-line summary of the fields that drive the formula, for
    the ``--list`` table."""
    f = entry["fields"]
    parts = []
    if "num_hidden_layers" in f or "num_layers" in f:
        parts.append(f"L={f.get('num_hidden_layers') or f.get('num_layers')}")
    if "kv_lora_rank" in f:
        parts.append(f"lora={f['kv_lora_rank']}")
    if "num_key_value_heads" in f:
        parts.append(f"kv={f['num_key_value_heads']}")
    if "head_dim" in f:
        parts.append(f"hd={f['head_dim']}")
    elif (
        "kv_lora_rank" not in f
        and "hidden_size" in f
        and "num_attention_heads" in f
        and f["num_attention_heads"]
    ):
        # Only derive head_dim for plain GQA/MHA models (Llama-style) where
        # it is a real driver; for MLA/DSA the latent dims matter instead.
        parts.append(f"hd={f['hidden_size'] // f['num_attention_heads']}")
    if "index_head_dim" in f:
        parts.append(f"idx={f['index_head_dim']}")
    if "sliding_window" in f:
        parts.append(f"sw={f['sliding_window']}")
    if "global_head_dim" in f:
        parts.append(f"ghd={f['global_head_dim']}")
    return " ".join(parts) if parts else "-"
