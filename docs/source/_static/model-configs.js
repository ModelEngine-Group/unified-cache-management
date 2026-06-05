/**
 * KV Cache Calculator - Model Configurations
 *
 * Standard Models: MHA, MQA, GQA, MLA, DSA architectures
 * Hybrid Models (DeepSeek V4) are handled separately in the Hybrid tab
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
 * Standard Models only (MHA, MQA, GQA, MLA, DSA)
 * DeepSeek V4 models are in Hybrid tab
 */
function getEmbeddedModelConfigs() {
    return {
        // DeepSeek V3 Series (MLA)
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
            "qk_rope_head_dim": 64,
            "index_head_dim": 128
        },

        // DeepSeek V4 Series (Hybrid - for Hybrid tab only)
        "deepseek-ai/DeepSeek-V4-Pro": {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "index_head_dim": 128,
            "sliding_window": 128,
            "qk_rope_head_dim": 64,
            "compress_ratios": [128,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0]
        },
        "deepseek-ai/DeepSeek-V4-Flash": {
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_hidden_layers": 43,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "index_head_dim": 128,
            "sliding_window": 128,
            "qk_rope_head_dim": 64,
            "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
        },

        // Qwen3 Series (GQA)
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

        // Llama Series (GQA)
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
        // GQA Models
        "zai-org/GLM-4.5": {
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
        // MLA Models
        "zai-org/GLM-4.7-Flash": {
            "hidden_size": 2048,
            "num_attention_heads": 20,
            "num_hidden_layers": 47,
            "num_key_value_heads": 20,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        // DSA Models
        "zai-org/GLM-5": {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_hidden_layers": 78,
            "num_key_value_heads": 64,
            "index_head_dim": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "zai-org/GLM-5.1": {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_hidden_layers": 78,
            "num_key_value_heads": 64,
            "index_head_dim": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },

        // MiniMax Series (GQA)
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

        // Kimi Series (MLA)
        "moonshot/Kimi-K2.5": {
            "hidden_size": 7168,
            "num_attention_heads": 64,
            "num_hidden_layers": 61,
            "num_key_value_heads": 64,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        },
        "moonshot/Kimi-K2.6": {
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