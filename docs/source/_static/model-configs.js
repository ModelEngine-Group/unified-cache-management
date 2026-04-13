/**
 * KV Cache Calculator - Model Configurations & Translations
 * 
 * This file contains:
 * 1. Translation strings (English only)
 * 2. Embedded preset model configurations
 */

// English-only translations
const translations = {
    en: {
        'title': 'KV Cache Size Calculator',
        'subtitle': 'Calculate KV cache size for large language models',
        'input-panel': 'Configuration',
        'model-source': 'Model Source',
        'preset-models': 'Preset Models',
        'custom-model': 'Custom Model',
        'select-model': 'Select Model',
        'loading': 'Loading models...',
        'model-url': 'Model URL',
        'data-type': 'Data Type',
        'token-count': 'Number of Tokens',
        'batch-size': 'Batch Size',
        'tp': 'Tensor Parallelism (TP)',
        'dp': 'Data Parallelism (DP)',
        'gpu-memory': 'Single-GPU Memory for KV Cache (GB)',
        'gpu-memory-hint': 'Memory available for KV cache (excluding model weights)',
        'calculate': 'Calculate KV Cache',
        'max-tokens-calculator': 'Maximum Tokens Calculator',
        'calculate-max-tokens': 'Calculate Max Tokens',
        'results': 'Results',
        'no-results': 'Configure your model and click calculate to see results.',
        'calculation-details': 'Calculation Details',
        'footer': 'KV Cache Calculator',
        'close': 'Close',
        'error': 'Error',
        'success': 'Success',
        'warning': 'Warning',
        'invalid-tokens': 'Please enter a valid number of tokens.',
        'model-not-found': 'Model configuration not found.',
        'calculation-success': 'KV cache size calculated successfully!',
        'model-url-invalid': 'Please enter a valid model URL.',
        'fetch-error': 'Failed to fetch model configuration. Please check the URL and try again.',
        'calculating': 'Calculating...'
    }
};

/**
 * Get embedded model configurations
 * Updated with 2025 mainstream models
 */
function getEmbeddedModelConfigs() {
    return {
        // DeepSeek Models
        "deepseek-ai/DeepSeek-V3": {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "deepseek-ai/DeepSeek-R1": {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "deepseek-ai/DeepSeek-V3.1-Terminus": {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "deepseek-ai/DeepSeek-V3.2": {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        // Qwen3 Models
        "Qwen/Qwen3-32B": {
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "num_hidden_layers": 64,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "Qwen/Qwen3-235B-A22B": {
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_hidden_layers": 94,
            "num_key_value_heads": 4,
            "head_dim": 128
        },
        "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
            "hidden_size": 6144,
            "num_attention_heads": 96,
            "num_hidden_layers": 62,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "Qwen/Qwen3-14B": {
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4
        },
        "Qwen/Qwen-7B": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32
        },
        // Qwen3.5 Series (GQA with explicit head_dim, some Hybrid)
        "Qwen/Qwen3.5-397B-A17B": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 60,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "is_hybrid": true
        },
        "Qwen/Qwen3.5-122B-A10B": {
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "is_hybrid": true
        },
        "Qwen/Qwen3.5-35B-A3B": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "is_hybrid": true
        },
        "Qwen/Qwen3.5-27B": {
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "is_hybrid": true
        },
        // Llama Models
        "meta-llama/Llama-3.1-70B-Instruct": {
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "num_hidden_layers": 80,
            "num_key_value_heads": 8
        },
        "meta-llama/Llama-3.1-405B": {
            "hidden_size": 16384,
            "num_attention_heads": 128,
            "num_hidden_layers": 126,
            "num_key_value_heads": 8
        },
        // GLM Series
        // GQA
        "zai-org/GLM-4.5":{
            "hidden_size": 5120,
            "num_attention_heads": 96,
            "num_hidden_layers": 92,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "zai-org/GLM-4.5-Air": {
            "hidden_size": 4096,
            "num_attention_heads": 96,
            "num_hidden_layers": 46,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "zai-org/GLM-4.7": {
            "hidden_size": 5120,
            "num_attention_heads": 96,
            "num_hidden_layers": 92,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        // MLA
        "zai-org/GLM-4.7-Flash": {
            "hidden_size": 2048,
            "num_attention_heads": 20,
            "num_hidden_layers": 47,
            "num_key_value_heads": 20,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "zai-org/GLM-5": {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_hidden_layers": 78,
            "num_key_value_heads": 64,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "zai-org/GLM-5.1": {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_hidden_layers": 78,
            "num_key_value_heads": 64,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },

        // MiniMax Series (GQA + MoE)
        "minimax/MiniMax-M2.5": {
            "hidden_size": 3072,
            "num_attention_heads": 48,
            "num_hidden_layers": 62,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "minimax/MiniMax-M2.1": {
            "hidden_size": 3072,
            "num_attention_heads": 48,
            "num_hidden_layers": 62,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        "minimax/MiniMax-M2": {
            "hidden_size": 3072,
            "num_attention_heads": 48,
            "num_hidden_layers": 62,
            "num_key_value_heads": 8,
            "head_dim": 128
        },
        // Kimi Series (MLA + Multimodal)
        "moonshot/Kimi-K2.5": {
            "hidden_size": 7168,
            "num_attention_heads": 64,
            "num_hidden_layers": 61,
            "num_key_value_heads": 64,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "moonshot/Kimi-K2": {
            "hidden_size": 7168,
            "num_attention_heads": 64,
            "num_hidden_layers": 61,
            "num_key_value_heads": 64,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        }
    };
}
