/**
 * KV Cache Calculator - Core Calculation Logic & UI
 *
 * This file contains:
 * 1. Global state management
 * 2. Formula data for different architectures
 * 3. Model source & config loading
 * 4. KV cache calculation (Standard Models only)
 * 5. Hybrid Models placeholder (calculation formula TBD)
 * 6. Display functions
 * 7. Toast notification system
 * 8. Event listeners
 */

// Global state
let currentLanguage = 'en';
let modelConfigs = {};
let currentModelSource = 'preset';
let currentConfigTab = 'standard';

// ============================================================
// Formula Data for Standard Architectures (MHA, MQA, GQA, MLA, DSA)
// ============================================================

const formulaData = {
    'MHA': {
        title: 'MHA (Multi-Head Attention)',
        icon: '🔹',
        color: '#3b82f6',
        formula: 'Single-GPU KV Cache = 2 × num_hidden_layers × num_tokens × batch_size × hidden_size / tensor_parallelism × dtype_bytes',
        params: [
            { name: '2', desc: 'Key and Value matrices' },
            { name: 'num_hidden_layers', desc: 'Number of layers in the model' },
            { name: 'num_tokens', desc: 'Number of tokens in sequence' },
            { name: 'batch_size', desc: 'Batch size' },
            { name: 'hidden_size', desc: 'Hidden layer dimension' },
            { name: 'tensor_parallelism', desc: 'Tensor parallelism degree' },
            { name: 'dtype_bytes', desc: 'Data type bytes (float16=2, float32=4)' }
        ],
        models: 'GPT-2, BERT',
        note: 'Each attention head has independent Key and Value. kv_heads = attn_heads.',
        why: 'Each head stores KV independently',
        keyPoint: 'Factor 2: K and V stored separately'
    },
    'MQA': {
        title: 'MQA (Multi-Query Attention)',
        icon: '🔹',
        color: '#10b981',
        formula: 'Single-GPU KV Cache = 2 × num_hidden_layers × num_tokens × batch_size × head_dim / tensor_parallelism × dtype_bytes',
        params: [
            { name: '2', desc: 'Key and Value matrices' },
            { name: 'num_hidden_layers', desc: 'Number of layers' },
            { name: 'head_dim', desc: 'Head dimension (hidden_size / attn_heads)' },
            { name: 'kv_heads', desc: 'KV head count = 1' }
        ],
        models: 'PaLM',
        note: 'All Query heads share single KV head. kv_heads = 1.',
        why: 'All heads share single KV, highest efficiency',
        keyPoint: 'Minimum KV Cache, but may affect quality'
    },
    'GQA': {
        title: 'GQA (Grouped-Query Attention)',
        icon: '🔹',
        color: '#8b5cf6',
        formula: 'Single-GPU KV Cache = 2 × num_hidden_layers × num_tokens × batch_size × num_kv_heads × head_dim / tensor_parallelism × dtype_bytes',
        params: [
            { name: '2', desc: 'Key and Value matrices' },
            { name: 'num_kv_heads', desc: 'KV head count (less than attn_heads)' },
            { name: 'head_dim', desc: 'Dimension per head' }
        ],
        models: 'LLaMA-3.1-70B, Qwen3-32B, Mistral-7B, GLM-4.5',
        note: 'Multiple Query heads share a group of KV heads. kv_heads < attn_heads.',
        why: 'Grouped sharing, balances efficiency and quality',
        keyPoint: 'Current mainstream architecture'
    },
    'MLA': {
        title: 'MLA (Multi-head Latent Attention)',
        icon: '🔸',
        color: '#f59e0b',
        formula: 'Single-GPU KV Cache = num_hidden_layers × num_tokens × batch_size × (kv_lora_rank + qk_rope_head_dim) / tensor_parallelism × dtype_bytes',
        params: [
            { name: 'No factor 2', desc: 'K and V compressed together' },
            { name: 'kv_lora_rank', desc: 'KV compressed latent dimension (e.g., 512)' },
            { name: 'qk_rope_head_dim', desc: 'RoPE positional encoding dimension (e.g., 64)' }
        ],
        models: 'DeepSeek V3, DeepSeek R1, Kimi K2, GLM-4.7-Flash',
        note: 'KV compressed to low-rank latent space, no factor 2.',
        why: 'KV compressed to latent space, saving memory',
        keyPoint: 'No factor 2, latent compression'
    },
    'DSA': {
        title: 'DSA (DeepSeek Sparse Attention)',
        icon: '🔮',
        color: '#9333ea',
        formula: 'Single-GPU KV Cache = num_hidden_layers × num_tokens × batch_size × (kv_lora_rank + qk_rope_head_dim + index_head_dim) / tensor_parallelism × dtype_bytes',
        params: [
            { name: 'No factor 2', desc: 'K and V compressed together (MLA)' },
            { name: 'kv_lora_rank', desc: 'KV compressed dimension (512)' },
            { name: 'qk_rope_head_dim', desc: 'RoPE dimension (64)' },
            { name: 'index_head_dim', desc: 'Lightning Indexer head dimension (128)' },
            { name: 'tensor_parallelism', desc: 'Tensor parallelism degree' }
        ],
        models: 'DeepSeek V3.2, GLM-5, GLM-5.1',
        note: 'MLA + Lightning Indexer, for sparse retrieval.',
        why: 'MLA with sparse retrieval + independent indexer precision',
        keyPoint: 'Additional index_head_dim'
    },
    'Standard': {
        title: 'Standard Transformer (MHA/MQA/GQA)',
        icon: '🔹',
        color: '#3b82f6',
        formula: 'Single-GPU KV Cache = 2 × num_hidden_layers × num_tokens × batch_size × hidden_size × (num_kv_heads / num_attn_heads) / tensor_parallelism × dtype_bytes',
        params: [
            { name: '2', desc: 'Key and Value matrices' },
            { name: 'num_kv_heads', desc: 'KV head count' },
            { name: 'num_attn_heads', desc: 'Attention head count' }
        ],
        models: 'Determined by kv_heads/attn_heads ratio',
        note: 'Auto-detect: kv_heads = attn_heads → MHA, kv_heads = 1 → MQA, otherwise → GQA.',
        why: 'Auto-detect architecture type',
        keyPoint: 'Generic formula, auto-adapt'
    }
};

// ============================================================
// Formula Display Functions
// ============================================================

function getFormulaInfo(modelArch) {
    let archKey = 'Standard';

    if (modelArch.isDSA) {
        archKey = 'DSA';
    } else if (modelArch.isMLA) {
        archKey = 'MLA';
    } else if (modelArch.isGQA) {
        archKey = 'GQA';
    } else {
        const kvHeads = modelArch.kv_heads || modelArch.num_key_value_heads;
        const attnHeads = modelArch.num_attention_heads;
        if (kvHeads === attnHeads) {
            archKey = 'MHA';
        } else if (kvHeads === 1) {
            archKey = 'MQA';
        } else {
            archKey = 'GQA';
        }
    }

    return formulaData[archKey] || formulaData['Standard'];
}

function generateFormulaCard(formulaInfo) {
    return `
        <div class="formula-card" style="border-left-color: ${formulaInfo.color}; margin-bottom: 1.5rem;">
            <div class="formula-header">
                <span>${formulaInfo.icon}</span>
                <span>${formulaInfo.title}</span>
            </div>
            <div class="formula-content">
                <div class="formula-main" style="font-size: 0.85rem; margin-bottom: 0.75rem;">
                    ${formulaInfo.formula}
                </div>
                <div style="background: rgba(${hexToRgb(formulaInfo.color)}, 0.1); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.75rem;">
                    <strong style="color: ${formulaInfo.color};">Key Point:</strong>
                    <span style="color: var(--text-primary); margin-left: 0.25rem;">${formulaInfo.keyPoint}</span>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
                    <strong>Why:</strong> ${formulaInfo.why}
                </div>
                <div class="formula-breakdown">
                    ${formulaInfo.params.map(param => `
                        <div class="formula-step">
                            <span class="formula-step-label">${param.name}:</span>
                            <span class="formula-step-value">${param.desc}</span>
                        </div>
                    `).join('')}
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">
                    <strong>Note:</strong> ${formulaInfo.note}
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-secondary);">
                    <strong>Models:</strong> ${formulaInfo.models}
                </div>
            </div>
        </div>
    `;
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : '59, 130, 246';
}

function updateFormulaReference(config) {
    const container = document.getElementById('dynamic-formula-container');
    if (!container) return;

    if (!config) {
        container.innerHTML = `
            <div class="text-center" style="padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <div class="subtitle" style="color: var(--text-secondary);">Select a model to see its KV Cache formula.</div>
            </div>
        `;
        return;
    }

    const modelArch = detectArchitectureType(config);
    modelArch.kv_heads = config.num_key_value_heads;
    modelArch.num_attention_heads = config.num_attention_heads;

    const formulaInfo = getFormulaInfo(modelArch);
    container.innerHTML = generateFormulaCard(formulaInfo);
}

// ============================================================
// Configuration Tab Switching
// ============================================================

function switchConfigTab(tab) {
    currentConfigTab = tab;

    // Update tab buttons
    document.querySelectorAll('.model-type-option').forEach(item => {
        item.classList.remove('active');
    });
    document.getElementById('config-tab-' + tab).classList.add('active');

    // Update tab content
    document.querySelectorAll('.config-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById('config-content-' + tab).classList.add('active');

    clearResults();

    if (tab === 'standard') {
        const presetSelect = document.getElementById('preset-model-select');
        if (presetSelect && presetSelect.value && modelConfigs[presetSelect.value]) {
            updateFormulaReference(modelConfigs[presetSelect.value]);
        } else {
            updateFormulaReference(null);
        }
    } else if (tab === 'hybrid') {
        const container = document.getElementById('dynamic-formula-container');
        container.innerHTML = `
            <div class="formula-card" style="margin-bottom: 1.5rem;">
                <div class="formula-header">
                    <span>🌟</span>
                    <span>DeepSeek V4 Hybrid (Sparse Attention)</span>
                </div>
                <div class="formula-content">
                    <div class="formula-main" style="font-size: 0.85rem; margin-bottom: 0.75rem;">
                        KV Cache = Bytes-per-Token × Tokens × Batch ÷ TP
                    </div>
                    <div style="background: rgba(81, 145, 238, 0.1); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.75rem;">
                        <strong style="color: var(--accent-primary);">Key Point:</strong>
                        <span style="color: var(--text-primary); margin-left: 0.25rem;">Block-based Cache (FA + WA), significant difference between vllm-ascend and vllm</span>
                    </div>
                    <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                        <strong>FA Cache (Fixed Attention) - Compressor:</strong>
                        <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                            <li>C4A Compressor Cache: block-based compression</li>
                            <li>C4A Indexer Cache: Lightning Indexer</li>
                            <li>C128A Compressor Cache: high compression ratio layers</li>
                        </ul>
                    </div>
                    <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                        <strong>WA Cache (Window Attention) - KV ×2:</strong>
                        <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                            <li>SWA Cache: sliding window layers</li>
                            <li>C4A/C128A KV Cache: compressed layer KV</li>
                        </ul>
                    </div>
                    <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                        <strong>Deployment Differences:</strong>
                        <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                            <li>vllm-ascend: 512 tokens/block</li>
                            <li>vllm: 256 tokens/block, FP8/FP4 quantization (smaller KV cache)</li>
                        </ul>
                    </div>
                    <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-secondary);">
                        <strong>Models:</strong> DeepSeek V4 Pro (30 C4A + 31 C128A), DeepSeek V4 Flash (21 C4A + 20 C128A)
                    </div>
                </div>
            </div>
        `;
    }
}

// ============================================================
// Helper Functions
// ============================================================

function getModelDisplayName(modelName) {
    if (modelName.startsWith('http://') || modelName.startsWith('https://')) {
        try {
            const urlObj = new URL(modelName);
            const pathParts = urlObj.pathname.split('/').filter(part => part);

            if (urlObj.hostname.includes('modelscope.cn') && pathParts[0] === 'models') {
                if (pathParts.length >= 3) return pathParts.slice(1, 3).join('/');
            } else if (urlObj.hostname.includes('huggingface.co')) {
                const modelPathParts = pathParts.filter(part =>
                    !['tree', 'blob', 'raw', 'commit', 'discussions', 'issues', 'pull', 'models'].includes(part)
                );
                if (modelPathParts.length >= 2) return modelPathParts.slice(0, 2).join('/');
            }
        } catch (e) {
            console.warn('Failed to parse model URL:', e);
        }
    }

    if (modelName.includes('/')) {
        const parts = modelName.split('/');
        if (parts.length >= 2) return parts.slice(0, 2).join('/');
    }
    return modelName;
}

function clearResults() {
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="text-center" style="padding: 3rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <div class="subtitle">Configure your model and click calculate to see results.</div>
            </div>
        `;
    }
    const detailsContainer = document.getElementById('calculation-details');
    if (detailsContainer) detailsContainer.classList.add('hidden');
    const stepsContainer = document.getElementById('calculation-steps');
    if (stepsContainer) stepsContainer.innerHTML = '';
}

// ============================================================
// Initialization
// ============================================================

window.onload = function() {
    loadModelConfigs();
    initializeEventListeners();
};

// ============================================================
// Model Source Management
// ============================================================

function setModelSource(source) {
    currentModelSource = source;

    const presetOption = document.getElementById('preset-option');
    const customOption = document.getElementById('custom-option');

    presetOption.classList.remove('active');
    customOption.classList.remove('active');

    document.getElementById('preset-model-section').classList.add('hidden');
    document.getElementById('custom-model-section').classList.add('hidden');

    if (source === 'custom') {
        customOption.classList.add('active');
        document.getElementById('custom-model-section').classList.remove('hidden');
        updateFormulaReference(null);
    } else {
        presetOption.classList.add('active');
        document.getElementById('preset-model-section').classList.remove('hidden');
        populateModelDropdown();
        const presetSelect = document.getElementById('preset-model-select');
        if (presetSelect && presetSelect.value && modelConfigs[presetSelect.value]) {
            updateFormulaReference(modelConfigs[presetSelect.value]);
        }
    }
}

// ============================================================
// Model Configuration Loading
// ============================================================

function loadModelConfigs() {
    modelConfigs = getEmbeddedModelConfigs();
    console.log('Model configurations loaded:', Object.keys(modelConfigs).length, 'models');
    populateModelDropdown();
}

function populateModelDropdown() {
    const presetModelSelect = document.getElementById('preset-model-select');
    presetModelSelect.innerHTML = '';

    // Filter out DeepSeek V4 models (they go in Hybrid tab)
    const standardModels = Object.keys(modelConfigs).filter(name =>
        !name.includes('DeepSeek-V4')
    );

    const sortedModelNames = standardModels.sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

    sortedModelNames.forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        presetModelSelect.appendChild(option);
    });

    if (sortedModelNames.length > 0) {
        presetModelSelect.value = sortedModelNames[0];
        updateFormulaReference(modelConfigs[sortedModelNames[0]]);
    }
}

function onModelSelect() {
    const modelName = document.getElementById('preset-model-select').value;
    if (modelName && modelConfigs[modelName]) {
        updateFormulaReference(modelConfigs[modelName]);
    }
}

function onHybridModelSelect() {
    // Placeholder - can add specific logic later
}

// ============================================================
// Detect Model Architecture Type (Standard Models Only)
// ============================================================

function detectArchitectureType(config) {
    // Check if it's a hybrid model (various indicators for mixed/hybrid/sparse attention)
    // Support nested config (text_config) for multimodal models

    const textConfig = config.text_config || config;

    const hybridIndicators = [
        // DeepSeek V4 style
        config.compress_ratios,
        // MiMo style: hybrid_layer_pattern or swa_* fields
        config.hybrid_layer_pattern,
        config.swa_num_key_value_heads,
        config.swa_num_attention_heads,
        config.swa_head_dim,
        config.add_swa_attention_sink_bias,
        // General sliding window indicators
        config.sliding_window,
        config.window_attention,
        config.attention_window,
        // Qwen/Gemma style: layer_types array in text_config
        (textConfig.layer_types && Array.isArray(textConfig.layer_types) &&
         textConfig.layer_types.some(t => t !== 'full_attention')),
        // Linear attention indicators (Qwen3.6)
        textConfig.linear_attention,
        textConfig.linear_num_key_heads,
        textConfig.linear_key_head_dim,
        // Gemma-4 global attention
        textConfig.global_head_dim,
        textConfig.num_global_key_value_heads,
        // Other indicators
        config.mixed_attention,
        config.sparse_attention,
        (config.full_attention_layers && config.sliding_attention_layers),
        (config.full_attention_layers && config.linear_attention_layers),
        (config.attention_layers && typeof config.attention_layers === 'object'),
        (Array.isArray(config.attention_type) || Array.isArray(config.layer_attention_type)),
        (config.attention_mode && ['sliding', 'linear', 'mixed', 'sparse'].includes(config.attention_mode.toLowerCase()))
    ];

    const isHybridModel = hybridIndicators.some(indicator => indicator);

    // DSA: MLA + Lightning Indexer
    const isDSA = config.kv_lora_rank && config.qk_rope_head_dim && config.index_head_dim && !config.compress_ratios;

    // MLA: has kv_lora_rank and qk_rope_head_dim (no index_head_dim)
    const isMLA = config.kv_lora_rank && config.qk_rope_head_dim && !config.index_head_dim && !config.compress_ratios;

    // GQA with explicit head_dim
    const isGQA = config.head_dim && !isMLA && !isDSA && !isHybridModel;

    return {
        isDSA,
        isMLA,
        isGQA,
        isHybridModel,
        kv_heads: config.num_key_value_heads,
        num_attention_heads: config.num_attention_heads
    };
}

// ============================================================
// Calculate KV Cache Size (Standard Models)
// ============================================================

async function calculateKVCache() {
    clearResults();

    const tokenInput = document.getElementById('token-input').value.trim();
    const tokens = parseInt(tokenInput);
    const dtype = document.getElementById('dtype-select').value;

    if (!tokenInput || isNaN(tokens) || tokens <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for tokens.');
        return;
    }

    let config;
    let modelName;
    let hasError = false;

    const calculateBtn = document.querySelector('button[onclick="calculateKVCache()"]');
    const originalText = calculateBtn.innerHTML;
    calculateBtn.innerHTML = '<span>⏳</span> <span>Calculating...</span>';
    calculateBtn.disabled = true;

    try {
        if (currentModelSource === 'preset') {
            const presetSelect = document.getElementById('preset-model-select');
            modelName = presetSelect.value;
            if (!modelName || !modelConfigs[modelName]) {
                displayError('Model Not Found', 'The selected preset model configuration is not available.');
                hasError = true;
                throw new Error('Model not found');
            }
            config = modelConfigs[modelName];
        } else {
            const modelUrlInput = document.getElementById('model-url');
            const modelUrl = modelUrlInput.value.trim();
            if (!modelUrl) {
                displayError('Invalid URL', 'Please enter a model URL.');
                modelUrlInput.focus();
                hasError = true;
                throw new Error('Invalid model URL');
            }

            try {
                new URL(modelUrl);
            } catch (urlError) {
                displayError('Invalid URL', 'The URL format is invalid.');
                modelUrlInput.focus();
                hasError = true;
                throw new Error('Invalid URL');
            }

            try {
                config = await fetchModelConfigFromUrl(modelUrl);
                modelName = config._modelName || modelUrl;
            } catch (fetchError) {
                displayError('Fetch Failed', fetchError.message || 'Failed to fetch model configuration.');
                hasError = true;
                throw fetchError;
            }
        }

        if (!config || !config.hidden_size || !config.num_attention_heads || !config.num_hidden_layers) {
            displayError('Invalid Configuration', 'The model configuration is incomplete.');
            hasError = true;
            throw new Error('Incomplete model configuration');
        }

        // Check if it's a hybrid model from Custom Model input
        const modelArch = detectArchitectureType(config);
        if (modelArch.isHybridModel && currentModelSource === 'custom') {
            showToast('warning', 'Hybrid Model Warning',
                'This appears to be a Hybrid model (e.g., DeepSeek V4, Qwen Hybrid). The calculation result may not be accurate. For Hybrid models, please use the Hybrid Models tab.');
        }

        const result = performCalculation(config, tokens, dtype, modelName);
        displayResults(result);

    } catch (error) {
        if (!hasError) console.error('Calculation error:', error);
    } finally {
        calculateBtn.innerHTML = originalText;
        calculateBtn.disabled = false;
    }
}

// ============================================================
// Core Calculation: KV Cache Size (Standard Models)
// ============================================================

function performCalculation(config, tokens, dtype, modelName) {
    const { hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads,
            kv_lora_rank, qk_rope_head_dim, index_head_dim, head_dim } = config;

    const batchSize = parseInt(document.getElementById('batch-size').value) || 1;
    const tp = parseInt(document.getElementById('tp').value) || 1;

    const dtypeSizes = { 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1 };
    const dtypeSize = dtypeSizes[dtype] || 2;

    const modelArch = detectArchitectureType(config);
    const kvHeads = num_key_value_heads || num_attention_heads;
    const hdim = head_dim || (hidden_size / num_attention_heads);

    let totalElements;
    let formula;

    if (modelArch.isDSA) {
        // DSA: MLA + Lightning Indexer
        const elementsPerToken = num_hidden_layers * (kv_lora_rank + qk_rope_head_dim + index_head_dim) / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = num_hidden_layers + ' × ' + tokens + ' × ' + batchSize + ' × (' + kv_lora_rank + ' + ' + qk_rope_head_dim + ' + ' + index_head_dim + ') ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    } else if (modelArch.isMLA) {
        // MLA: no factor 2
        const elementsPerToken = num_hidden_layers * (kv_lora_rank + qk_rope_head_dim) / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = num_hidden_layers + ' × ' + tokens + ' × ' + batchSize + ' × (' + kv_lora_rank + ' + ' + qk_rope_head_dim + ') ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    } else if (modelArch.isHybridModel) {
        // Hybrid Model: use GQA-like calculation but show warning
        // For hybrid models, use available head_dim or fallback to hidden_size calculation
        const effectiveHdim = hdim || (hidden_size / num_attention_heads);
        const elementsPerToken = 2 * num_hidden_layers * kvHeads * effectiveHdim / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = '2 × ' + num_hidden_layers + ' × ' + tokens + ' × ' + batchSize + ' × ' + kvHeads + ' × ' + effectiveHdim + ' ÷ ' + tp + ' × ' + dtypeSize + ' bytes (Hybrid - may not be accurate)';
    } else if (modelArch.isGQA) {
        // GQA with explicit head_dim
        const elementsPerToken = 2 * num_hidden_layers * kvHeads * hdim / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = '2 × ' + num_hidden_layers + ' × ' + tokens + ' × ' + batchSize + ' × ' + kvHeads + ' × ' + hdim + ' ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    } else {
        // Standard: MHA/MQA/GQA auto-detect
        const elementsPerToken = 2 * num_hidden_layers * hidden_size * (kvHeads / num_attention_heads) / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = '2 × ' + num_hidden_layers + ' × ' + tokens + ' × ' + batchSize + ' × ' + hidden_size + ' × (' + kvHeads + '/' + num_attention_heads + ') ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    }

    const totalBytes = totalElements * dtypeSize;
    const kvCacheSizeGiB = totalBytes / Math.pow(1024, 3);
    const kvCacheSizeGB = totalBytes / Math.pow(1000, 3);

    const dp = parseInt(document.getElementById('dp').value) || 1;
    const totalGPUs = tp * dp;
    const clusterKVCacheSizeGiB = kvCacheSizeGiB * totalGPUs;
    const clusterKVCacheSizeGB = kvCacheSizeGB * totalGPUs;

    // Determine architecture type for display
    let architectureType;
    if (modelArch.isDSA) {
        architectureType = 'DSA (DeepSeek Sparse Attention)';
    } else if (modelArch.isMLA) {
        architectureType = 'MLA (Multi-head Latent Attention)';
    } else if (modelArch.isHybridModel) {
        architectureType = 'Hybrid Model (Warning: result may not be accurate)';
    } else if (kvHeads === num_attention_heads) {
        architectureType = 'MHA (Multi-Head Attention)';
    } else if (kvHeads === 1) {
        architectureType = 'MQA (Multi-Query Attention)';
    } else {
        architectureType = 'GQA (Grouped-Query Attention)';
    }

    return {
        modelName,
        tokens,
        batchSize,
        tp,
        dp,
        totalGPUs,
        dtype,
        dtypeSize,
        kvCacheSizeGiB,
        kvCacheSizeGB,
        clusterKVCacheSizeGiB,
        clusterKVCacheSizeGB,
        totalElements,
        totalBytes,
        config,
        formula,
        architectureType,
        showHybridWarning: modelArch.isHybridModel
    };
}

// ============================================================
// Calculate Maximum Tokens (Standard Models)
// ============================================================

async function calculateMaxTokens() {
    clearResults();

    const gpuMemoryInput = document.getElementById('gpu-memory-input').value.trim();
    const gpuMemoryGiB = parseFloat(gpuMemoryInput);
    const dtype = document.getElementById('dtype-select').value;

    if (!gpuMemoryInput || isNaN(gpuMemoryGiB) || gpuMemoryGiB <= 0) {
        displayError('Invalid Input', 'Please enter a valid GPU memory size.');
        return;
    }

    let config;
    let modelName;

    const calculateBtn = document.querySelector('button[onclick="calculateMaxTokens()"]');
    const originalText = calculateBtn.innerHTML;
    calculateBtn.innerHTML = '<span>⏳</span> <span>Calculating...</span>';
    calculateBtn.disabled = true;

    try {
        if (currentModelSource === 'preset') {
            const presetSelect = document.getElementById('preset-model-select');
            modelName = presetSelect.value;
            config = modelConfigs[modelName];
        } else {
            const modelUrl = document.getElementById('model-url').value.trim();
            config = await fetchModelConfigFromUrl(modelUrl);
            modelName = config._modelName || modelUrl;
        }

        const result = calculateMaxTokensForMemory(config, gpuMemoryGiB, dtype, modelName);
        displayMaxTokensResults(result);

    } catch (error) {
        console.error('Max tokens calculation error:', error);
    } finally {
        calculateBtn.innerHTML = originalText;
        calculateBtn.disabled = false;
    }
}

function calculateMaxTokensForMemory(config, gpuMemoryGiB, dtype, modelName) {
    const { hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads,
            kv_lora_rank, qk_rope_head_dim, index_head_dim, head_dim } = config;

    const batchSize = parseInt(document.getElementById('batch-size').value) || 1;
    const tp = parseInt(document.getElementById('tp').value) || 1;

    const dtypeSizes = { 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1 };
    const dtypeSize = dtypeSizes[dtype] || 2;

    const modelArch = detectArchitectureType(config);
    const kvHeads = num_key_value_heads || num_attention_heads;
    const hdim = head_dim || (hidden_size / num_attention_heads);

    let elementsPerToken;
    let formula;

    if (modelArch.isDSA) {
        elementsPerToken = num_hidden_layers * (kv_lora_rank + qk_rope_head_dim + index_head_dim) / tp;
        formula = num_hidden_layers + ' × (' + kv_lora_rank + ' + ' + qk_rope_head_dim + ' + ' + index_head_dim + ') ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    } else if (modelArch.isMLA) {
        elementsPerToken = num_hidden_layers * (kv_lora_rank + qk_rope_head_dim) / tp;
        formula = num_hidden_layers + ' × (' + kv_lora_rank + ' + ' + qk_rope_head_dim + ') ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    } else if (modelArch.isGQA) {
        elementsPerToken = 2 * num_hidden_layers * kvHeads * hdim / tp;
        formula = '2 × ' + num_hidden_layers + ' × ' + kvHeads + ' × ' + hdim + ' ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    } else {
        elementsPerToken = 2 * hidden_size * (kvHeads / num_attention_heads) * num_hidden_layers / tp;
        formula = '2 × ' + hidden_size + ' × (' + kvHeads + '/' + num_attention_heads + ') × ' + num_hidden_layers + ' ÷ ' + tp + ' × ' + dtypeSize + ' bytes';
    }

    const totalMemoryBytes = gpuMemoryGiB * Math.pow(1024, 3);
    const maxTokens = Math.floor(totalMemoryBytes / (elementsPerToken * dtypeSize));

    let architectureType;
    if (modelArch.isDSA) architectureType = 'DSA';
    else if (modelArch.isMLA) architectureType = 'MLA';
    else if (modelArch.isHybridModel) architectureType = 'Hybrid Model';
    else if (kvHeads === num_attention_heads) architectureType = 'MHA';
    else if (kvHeads === 1) architectureType = 'MQA';
    else architectureType = 'GQA';

    return {
        modelName,
        batchSize,
        tp,
        gpuMemoryGiB,
        dtype,
        dtypeSize,
        maxTokens,
        elementsPerToken,
        formula,
        architectureType,
        isHybridModel: modelArch.isHybridModel,
        perTokenMemoryMiB: (elementsPerToken * dtypeSize) / Math.pow(1024, 2),
        config
    };
}

// ============================================================
// Hybrid Models Calculation (DeepSeek V4)
// ============================================================

// DeepSeek V4 预置参数 (基于用户验证的数据)
const DEEPSEEK_V4_CONFIGS = {
    'deepseek-ai/DeepSeek-V4-Pro': {
        c4aLayers: 30,
        c128aLayers: 31,
        // vllm-ascend (512 tokens/block)
        // Block size values derived from:
        // - c4aCompressor: 512 tokens × 256 KV heads × 1 byte (FP8) = 131072 B
        // - c4aIndexer: 512 tokens × 32 indexer heads × 4 bytes (FP32) × 1 layer = 16640 B (approx)
        // - c128aCompressor: 512 tokens × 8 heads × 1 byte = 4096 B (approx)
        // - swaCache: Same as c4aCompressor (sliding window attention uses same compression)
        // - c4aKVCache: 512 tokens × 32 heads × 1 byte = 16384 B (approx)
        // - c128aKVCache: 512 tokens × 8 heads × 1 byte = 4096 B (approx)
        // Values validated by user testing on actual vLLM-Ascend deployment
        vllmAscend: {
            blockTokens: 512,
            bytesPerToken: 27175,
            c4aCompressor: 131072,    // 512 × 256 × 1 = 131072
            c4aIndexer: 16640,        // Lightning indexer overhead
            c128aCompressor: 4096,    // 512 × 8 × 1 = 4096
            swaCache: 131072,         // Same compression as c4a, layers = 31, ×2
            c4aKVCache: 16384,       // 512 × 32 × 1 = 16384
            c128aKVCache: 4096,      // 512 × 8 × 1 = 4096
            swaLayers: 31,
            kvLayers: 30  // C4A和C128A用于KV cache的层数
        },
        // vllm (256 tokens/block, FP8/FP4量化)
        // Block values derived similarly but with 256 tokens/block and quantization
        vllm: {
            blockTokens: 256,
            bytesPerToken: 28415.4375,
            c4aCompressor: 37376,    // 256 × 146 × 1 ≈ 37376 (quantized)
            c4aIndexer: 8448,
            c128aCompressor: 1168,
            swaCache: 37376,         // layers = 62, ×2
            c4aKVCache: 8192,
            c128aKVCache: 32768,
            swaLayers: 62,
            kvLayers: 30
        }
    },
    'deepseek-ai/DeepSeek-V4-Flash': {
        c4aLayers: 21,
        c128aLayers: 20,
        // vllm-ascend (512 tokens/block)
        // Same calculation method as V4-Pro, but with fewer layers
        vllmAscend: {
            blockTokens: 512,
            bytesPerToken: 19162.5,
            c4aCompressor: 131072,   // 512 × 256 × 1 = 131072
            c4aIndexer: 16640,
            c128aCompressor: 4096,
            swaCache: 131072,        // layers = 22, ×2
            c4aKVCache: 16384,
            c128aKVCache: 4096,
            swaLayers: 22,
            kvLayers: 21
        },
        // vllm (256 tokens/block)
        vllm: {
            blockTokens: 256,
            bytesPerToken: 20058.25,
            c4aCompressor: 37376,
            c4aIndexer: 8448,
            c128aCompressor: 1168,
            swaCache: 37376,         // layers = 44, ×2
            c4aKVCache: 8192,
            c128aKVCache: 32768,
            swaLayers: 44,
            kvLayers: 21
        }
    }
};

function calculateHybrid() {
    clearResults();

    const modelName = document.getElementById('hybrid-model-select').value;
    const deployment = document.getElementById('hybrid-deployment').value;
    const tokensInput = document.getElementById('hybrid-token-input').value.trim();
    const tokens = parseInt(tokensInput) || 4096;
    const batchSizeInput = document.getElementById('hybrid-batch-size').value.trim();
    const batchSize = parseInt(batchSizeInput) || 1;
    const tpInput = document.getElementById('hybrid-tp').value.trim();
    const tp = parseInt(tpInput) || 1;
    const dp = parseInt(document.getElementById('hybrid-dp').value) || 1;

    // Input validation
    if (!tokensInput || isNaN(tokens) || tokens <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for tokens.');
        return;
    }
    if (!batchSizeInput || isNaN(batchSize) || batchSize <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for batch size.');
        return;
    }
    if (!tpInput || isNaN(tp) || tp <= 0) {
        displayError('Invalid Input', 'Tensor Parallelism must be at least 1.');
        return;
    }
    if (dp <= 0) {
        displayError('Invalid Input', 'Data Parallelism must be at least 1.');
        return;
    }

    // 获取模型配置
    const v4Config = DEEPSEEK_V4_CONFIGS[modelName];
    if (!v4Config) {
        displayError('Model Not Found', 'DeepSeek V4 configuration not found.');
        return;
    }

    // 根据部署方式选择参数
    const deployKey = deployment === 'vllm-ascend' ? 'vllmAscend' : 'vllm';
    const deployConfig = v4Config[deployKey];

    // 计算 KV Cache (使用用户验证的 bytesPerToken)
    const totalBytes = deployConfig.bytesPerToken * tokens * batchSize / tp;
    const kvCacheSizeGiB = totalBytes / Math.pow(1024, 3);
    const kvCacheSizeMiB = totalBytes / Math.pow(1024, 2);
    const kvCacheSizeMB = totalBytes / Math.pow(1000, 2);

    // 计算集群总大小
    const totalGPUs = tp * dp;
    const clusterKVCacheGiB = kvCacheSizeGiB * totalGPUs;

    // 计算block数量
    const blockCount = Math.ceil(tokens / deployConfig.blockTokens);

    // Block breakdown 计算 (按用户给的公式)
    const c4aLayers = v4Config.c4aLayers;
    const c128aLayers = v4Config.c128aLayers;
    const cfg = deployConfig;

    // FA Cache (Compressor)
    const c4aCompressorTotal = cfg.c4aCompressor * c4aLayers;
    const c4aIndexerTotal = cfg.c4aIndexer * c4aLayers;
    const c128aCompressorTotal = cfg.c128aCompressor * c128aLayers;

    // WA Cache (SWA ×2, KV)
    const swaTotal = cfg.swaCache * cfg.swaLayers * 2;
    const c4aKVTotal = cfg.c4aKVCache * cfg.kvLayers;
    const c128aKVTotal = cfg.c128aKVCache * cfg.kvLayers;

    // 每block总字节
    const blockBytes = c4aCompressorTotal + c4aIndexerTotal + c128aCompressorTotal +
                       swaTotal + c4aKVTotal + c128aKVTotal;

    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-primary);">${kvCacheSizeGiB.toFixed(4)} GiB</div>
            <div class="result-label" style="font-size: 0.75rem; color: var(--text-secondary);">= ${kvCacheSizeMiB.toFixed(2)} MiB (= ${kvCacheSizeMB.toFixed(2)} MB)</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Single-GPU KV Cache Size</div>
            ${totalGPUs > 1 ? `
            <div class="result-value" style="font-size: 1.2rem; font-weight: 600; color: var(--accent-primary); margin-top: 0.5rem;">${clusterKVCacheGiB.toFixed(4)} GiB</div>
            <div class="result-label" style="font-size: 0.75rem; color: var(--text-secondary);">Cluster-wide (TP=${tp} × DP=${dp} = ${totalGPUs} GPUs)</div>
            ` : ''}
        </div>

        <div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Model:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${modelName.split('/')[1]}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Deployment:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${deployment}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Tokens:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${tokens.toLocaleString()}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Blocks:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${blockCount}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Batch:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${batchSize}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">TP:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${tp}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">B/Token:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${deployConfig.bytesPerToken.toLocaleString()}</strong>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📐</span>
                <span>Calculation Formula</span>
            </div>
            <div class="formula-content">
                <div class="formula-main" style="font-size: 0.75rem;">
                    KV Cache = ${deployConfig.bytesPerToken.toLocaleString()} B/token × ${tokens.toLocaleString()} tokens × ${batchSize} batch ÷ ${tp} TP
                </div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📊</span>
                <span>Block Breakdown (${deployConfig.blockTokens} tokens/block)</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="margin-bottom: 0.5rem; font-weight: 600; color: var(--accent-primary);">FA Cache (Compressor):</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-left: 0.5rem;">
                    <div style="color: var(--text-secondary);">C4A Compressor:</div>
                    <div>${cfg.c4aCompressor.toLocaleString()} B × ${c4aLayers} = ${c4aCompressorTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C4A Indexer:</div>
                    <div>${cfg.c4aIndexer.toLocaleString()} B × ${c4aLayers} = ${c4aIndexerTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C128A Compressor:</div>
                    <div>${cfg.c128aCompressor.toLocaleString()} B × ${c128aLayers} = ${c128aCompressorTotal.toLocaleString()} B</div>
                </div>

                <div style="margin-top: 0.75rem; font-weight: 600; color: var(--accent-primary);">WA Cache:</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-left: 0.5rem;">
                    <div style="color: var(--text-secondary);">SWA Cache (×2):</div>
                    <div>${cfg.swaCache.toLocaleString()} B × ${cfg.swaLayers} × 2 = ${swaTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C4A KV Cache:</div>
                    <div>${cfg.c4aKVCache.toLocaleString()} B × ${cfg.kvLayers} = ${c4aKVTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C128A KV Cache:</div>
                    <div>${cfg.c128aKVCache.toLocaleString()} B × ${cfg.kvLayers} = ${c128aKVTotal.toLocaleString()} B</div>
                </div>

                <div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px dashed var(--border-color);">
                    <strong>Total per Block:</strong> ${blockBytes.toLocaleString()} B = ${(blockBytes / 1024).toFixed(2)} KiB = ${(blockBytes / 1024 / 1024).toFixed(4)} MiB
                </div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Layer Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <div style="color: var(--text-secondary);">C4A Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${c4aLayers}</div>
                    <div style="color: var(--text-secondary);">C128A Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${c128aLayers}</div>
                    <div style="color: var(--text-secondary);">Block Size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${deployConfig.blockTokens} tokens</div>
                </div>
            </div>
        </div>
    `;
}

function calculateHybridMaxTokens() {
    clearResults();

    const modelName = document.getElementById('hybrid-model-select').value;
    const deployment = document.getElementById('hybrid-deployment').value;
    const gpuMemoryInput = document.getElementById('hybrid-gpu-memory-input').value.trim();
    const gpuMemoryGiB = parseFloat(gpuMemoryInput) || 42;
    const batchSizeInput = document.getElementById('hybrid-batch-size').value.trim();
    const batchSize = parseInt(batchSizeInput) || 1;
    const tpInput = document.getElementById('hybrid-tp').value.trim();
    const tp = parseInt(tpInput) || 1;

    // Input validation
    if (!gpuMemoryInput || isNaN(gpuMemoryGiB) || gpuMemoryGiB <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for GPU memory.');
        return;
    }
    if (!batchSizeInput || isNaN(batchSize) || batchSize <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for batch size.');
        return;
    }
    if (!tpInput || isNaN(tp) || tp <= 0) {
        displayError('Invalid Input', 'Tensor Parallelism must be at least 1.');
        return;
    }

    // 获取模型配置
    const v4Config = DEEPSEEK_V4_CONFIGS[modelName];
    if (!v4Config) {
        displayError('Model Not Found', 'DeepSeek V4 configuration not found.');
        return;
    }

    // 根据部署方式选择参数
    const deployKey = deployment === 'vllm-ascend' ? 'vllmAscend' : 'vllm';
    const deployConfig = v4Config[deployKey];

    // 计算最大tokens
    const totalMemoryBytes = gpuMemoryGiB * Math.pow(1024, 3);
    const perTokenBytes = deployConfig.bytesPerToken * batchSize / tp;
    const maxTokens = Math.floor(totalMemoryBytes / perTokenBytes);

    // 计算需要的block数量
    const blockCount = Math.ceil(maxTokens / deployConfig.blockTokens);

    // Block breakdown 计算
    const c4aLayers = v4Config.c4aLayers;
    const c128aLayers = v4Config.c128aLayers;
    const cfg = deployConfig;

    // FA Cache (Compressor)
    const c4aCompressorTotal = cfg.c4aCompressor * c4aLayers;
    const c4aIndexerTotal = cfg.c4aIndexer * c4aLayers;
    const c128aCompressorTotal = cfg.c128aCompressor * c128aLayers;

    // WA Cache (SWA ×2, KV)
    const swaTotal = cfg.swaCache * cfg.swaLayers * 2;
    const c4aKVTotal = cfg.c4aKVCache * cfg.kvLayers;
    const c128aKVTotal = cfg.c128aKVCache * cfg.kvLayers;

    // 每block总字节
    const blockBytes = c4aCompressorTotal + c4aIndexerTotal + c128aCompressorTotal +
                       swaTotal + c4aKVTotal + c128aKVTotal;

    // 计算各部分占总内存的比例
    const totalBlockMemory = blockBytes * blockCount;

    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${maxTokens.toLocaleString()}</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Max Tokens ${tp > 1 ? '(TP=' + tp + ')' : ''}</div>
        </div>

        <div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Model:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${modelName.split('/')[1]}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Deployment:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${deployment}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">GPU Memory:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${gpuMemoryGiB} GiB</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Max Blocks:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">~${blockCount}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">B/Token:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${deployConfig.bytesPerToken.toLocaleString()}</strong>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📐</span>
                <span>Calculation Formula</span>
            </div>
            <div class="formula-content">
                <div class="formula-main" style="font-size: 0.75rem;">
                    Max Tokens = GPU Memory ÷ Bytes-per-Token = ${(gpuMemoryGiB * 1024).toFixed(0)} MiB ÷ ${(perTokenBytes / 1024).toFixed(2)} KiB
                </div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📊</span>
                <span>Block Breakdown (${deployConfig.blockTokens} tokens/block)</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="margin-bottom: 0.5rem; font-weight: 600; color: var(--accent-primary);">FA Cache (Compressor):</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-left: 0.5rem;">
                    <div style="color: var(--text-secondary);">C4A Compressor:</div>
                    <div>${cfg.c4aCompressor.toLocaleString()} B × ${c4aLayers} = ${c4aCompressorTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C4A Indexer:</div>
                    <div>${cfg.c4aIndexer.toLocaleString()} B × ${c4aLayers} = ${c4aIndexerTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C128A Compressor:</div>
                    <div>${cfg.c128aCompressor.toLocaleString()} B × ${c128aLayers} = ${c128aCompressorTotal.toLocaleString()} B</div>
                </div>

                <div style="margin-top: 0.75rem; font-weight: 600; color: var(--accent-primary);">WA Cache:</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-left: 0.5rem;">
                    <div style="color: var(--text-secondary);">SWA Cache (×2):</div>
                    <div>${cfg.swaCache.toLocaleString()} B × ${cfg.swaLayers} × 2 = ${swaTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C4A KV Cache:</div>
                    <div>${cfg.c4aKVCache.toLocaleString()} B × ${cfg.kvLayers} = ${c4aKVTotal.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">C128A KV Cache:</div>
                    <div>${cfg.c128aKVCache.toLocaleString()} B × ${cfg.kvLayers} = ${c128aKVTotal.toLocaleString()} B</div>
                </div>

                <div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px dashed var(--border-color);">
                    <strong>Total per Block:</strong> ${blockBytes.toLocaleString()} B = ${(blockBytes / 1024).toFixed(2)} KiB = ${(blockBytes / 1024 / 1024).toFixed(4)} MiB
                </div>
                <div style="margin-top: 0.25rem;">
                    <strong>Total Memory (${blockCount} blocks):</strong> ${(blockBytes * blockCount).toLocaleString()} B = ${((blockBytes * blockCount) / 1024 / 1024).toFixed(2)} MiB
                </div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Layer Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <div style="color: var(--text-secondary);">C4A Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${c4aLayers}</div>
                    <div style="color: var(--text-secondary);">C128A Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${c128aLayers}</div>
                    <div style="color: var(--text-secondary);">Block Size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${deployConfig.blockTokens} tokens</div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================
// Fetch Model Configuration from URL
// ============================================================

async function fetchModelConfigFromUrl(url) {
    try {
        let normalizedUrl = url.trim().replace(/\/+$/, '');
        normalizedUrl = normalizedUrl.replace(/\/(files|tree\/main|blob\/main|raw\/main|commits|issues|discussions).*$/, '');

        const urlObj = new URL(normalizedUrl);
        let modelIdentifier;
        let platform = '';

        if (urlObj.hostname.includes('huggingface.co')) {
            platform = 'huggingface';
            const pathParts = urlObj.pathname.split('/').filter(part => part && part !== 'models');
            const modelPathParts = pathParts.filter(part =>
                !['tree', 'blob', 'raw', 'commit', 'discussions', 'issues', 'pull'].includes(part)
            );
            if (modelPathParts.length >= 2) {
                modelIdentifier = modelPathParts.slice(0, 2).join('/');
            }
        } else if (urlObj.hostname.includes('modelscope.cn')) {
            platform = 'modelscope';
            const pathParts = urlObj.pathname.split('/').filter(part => part);
            if (pathParts.length >= 3 && pathParts[0] === 'models') {
                modelIdentifier = pathParts.slice(1, 3).join('/');
            }
        }

        if (!modelIdentifier) {
            throw new Error('Could not extract model identifier from URL.');
        }

        console.log('Fetching config for ' + platform + ' model: ' + modelIdentifier);

        let configData = null;

        // Try direct fetch
        try {
            if (platform === 'huggingface') {
                const apiUrl = 'https://huggingface.co/' + modelIdentifier + '/raw/main/config.json';
                const response = await fetch(apiUrl);
                if (response.ok) {
                    configData = await response.json();
                }
            } else if (platform === 'modelscope') {
                const endpoints = [
                    'https://modelscope.cn/api/v1/models/' + modelIdentifier + '/repo?Revision=master&FilePath=config.json',
                    'https://modelscope.cn/' + modelIdentifier + '/raw/master/config.json'
                ];
                for (const apiUrl of endpoints) {
                    try {
                        const response = await fetch(apiUrl);
                        if (response.ok) {
                            const contentType = response.headers.get('content-type');
                            if (contentType && contentType.includes('application/json')) {
                                const data = await response.json();
                                let rawContent = data.Data || data.data || data;
                                if (rawContent && rawContent.Content) {
                                    try {
                                        const decodedContent = atob(rawContent.Content);
                                        configData = JSON.parse(decodedContent);
                                    } catch (e) {
                                        configData = JSON.parse(rawContent.Content);
                                    }
                                } else if (typeof rawContent === 'object') {
                                    configData = rawContent;
                                }
                            } else {
                                const textData = await response.text();
                                configData = JSON.parse(textData);
                            }
                            if (configData && configData.hidden_size) break;
                        }
                    } catch (e) {
                        continue;
                    }
                }
            }
        } catch (e) {
            console.log('Direct fetch failed:', e);
        }

        // Check local configs
        if (!configData && modelConfigs[modelIdentifier]) {
            return modelConfigs[modelIdentifier];
        }

        if (!configData) {
            throw new Error('Unable to fetch model configuration. Please check the URL.');
        }

        const sourceConfig = configData.text_config || configData;

        // Preserve all fields including hybrid model indicators
        const transformedConfig = {
            hidden_size: sourceConfig.hidden_size,
            num_attention_heads: sourceConfig.num_attention_heads,
            num_hidden_layers: sourceConfig.num_hidden_layers,
            num_key_value_heads: sourceConfig.num_key_value_heads,
            kv_lora_rank: sourceConfig.kv_lora_rank,
            qk_rope_head_dim: sourceConfig.qk_rope_head_dim,
            head_dim: sourceConfig.head_dim,
            index_head_dim: sourceConfig.index_head_dim,
            compress_ratios: sourceConfig.compress_ratios || configData.compress_ratios,
            // Hybrid model indicators
            hybrid_layer_pattern: sourceConfig.hybrid_layer_pattern || configData.hybrid_layer_pattern,
            sliding_window: sourceConfig.sliding_window || configData.sliding_window,
            sliding_window_size: sourceConfig.sliding_window_size || configData.sliding_window_size,
            swa_num_key_value_heads: sourceConfig.swa_num_key_value_heads || configData.swa_num_key_value_heads,
            swa_num_attention_heads: sourceConfig.swa_num_attention_heads || configData.swa_num_attention_heads,
            swa_head_dim: sourceConfig.swa_head_dim || configData.swa_head_dim,
            add_swa_attention_sink_bias: sourceConfig.add_swa_attention_sink_bias || configData.add_swa_attention_sink_bias,
            layer_types: sourceConfig.layer_types,
            linear_attention: sourceConfig.linear_attention,
            linear_num_key_heads: sourceConfig.linear_num_key_heads,
            linear_key_head_dim: sourceConfig.linear_key_head_dim,
            global_head_dim: sourceConfig.global_head_dim,
            num_global_key_value_heads: sourceConfig.num_global_key_value_heads,
            window_attention: sourceConfig.window_attention || configData.window_attention,
            attention_window: sourceConfig.attention_window || configData.attention_window,
            mixed_attention: sourceConfig.mixed_attention || configData.mixed_attention,
            sparse_attention: sourceConfig.sparse_attention || configData.sparse_attention,
            full_attention_layers: sourceConfig.full_attention_layers || configData.full_attention_layers,
            sliding_attention_layers: sourceConfig.sliding_attention_layers || configData.sliding_attention_layers,
            linear_attention_layers: sourceConfig.linear_attention_layers || configData.linear_attention_layers,
            _modelName: modelIdentifier
        };

        Object.keys(transformedConfig).forEach(key => {
            if (key !== '_modelName' && transformedConfig[key] === undefined) {
                delete transformedConfig[key];
            }
        });

        return transformedConfig;

    } catch (error) {
        console.error('Error fetching model config:', error);
        throw error;
    }
}

// ============================================================
// Display Functions
// ============================================================

function displayError(title, message) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    const detailsContainer = document.getElementById('calculation-details');
    if (detailsContainer) detailsContainer.classList.add('hidden');

    resultsContainer.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
            <h3 style="color: var(--accent-error); margin-bottom: 0.5rem; font-size: 1.2rem;">${title}</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">${message}</p>
        </div>
    `;
}

function displayResults(result) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    const config = result.config;
    const kvHeads = config.num_key_value_heads || config.num_attention_heads;

    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-primary);">${result.kvCacheSizeGiB.toFixed(4)} GiB</div>
            <div class="result-label" style="font-size: 0.75rem; color: var(--text-secondary);">= ${result.kvCacheSizeGB.toFixed(5)} GB</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Single-GPU KV Cache Size</div>
            ${result.totalGPUs > 1 ? `
            <div class="result-value" style="font-size: 1.2rem; font-weight: 600; color: var(--accent-primary); margin-top: 0.5rem;">${result.clusterKVCacheSizeGiB.toFixed(4)} GiB</div>
            <div class="result-label" style="font-size: 0.75rem; color: var(--text-secondary);">Cluster-wide (TP=${result.tp} × DP=${result.dp} = ${result.totalGPUs} GPUs)</div>
            ` : ''}
        </div>

        ${result.showHybridWarning ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-warning); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: var(--accent-warning); font-size: 0.85rem;">Hybrid Model Warning</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                This appears to be a Hybrid model. The calculation result may not be accurate. Please use the Hybrid Models tab for accurate results.
            </div>
        </div>
        ` : ''}

        <div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Model:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${getModelDisplayName(result.modelName)}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Type:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.architectureType}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Tokens:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.tokens.toLocaleString()}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Batch:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.batchSize}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">DType:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.dtype}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">TP:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.tp}</strong>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📐</span>
                <span>Calculation Formula</span>
            </div>
            <div class="formula-content">
                <div class="formula-main" style="font-size: 0.75rem;">${result.formula}</div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Model Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_hidden_layers}</div>
                    <div style="color: var(--text-secondary);">Hidden Size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.hidden_size}</div>
                    <div style="color: var(--text-secondary);">Attn Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_attention_heads}</div>
                    <div style="color: var(--text-secondary);">KV Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${kvHeads}</div>
                    ${config.kv_lora_rank ? '<div style="color: var(--text-secondary);">KV LoRA Rank:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.kv_lora_rank + '</div>' : ''}
                    ${config.qk_rope_head_dim ? '<div style="color: var(--text-secondary);">QK RoPE Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.qk_rope_head_dim + '</div>' : ''}
                    ${config.index_head_dim ? '<div style="color: var(--text-secondary);">Index Head Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.index_head_dim + '</div>' : ''}
                </div>
            </div>
        </div>
    `;
}

function displayMaxTokensResults(result) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    const config = result.config;
    const kvHeads = config.num_key_value_heads || config.num_attention_heads;

    // Show toast warning for hybrid models (same as KV Cache calculation)
    if (result.isHybridModel) {
        showToast('warning', 'Hybrid Model Warning',
            'This appears to be a Hybrid model. The max tokens calculation may not be accurate. For Hybrid models, please use the Hybrid Models tab.');
    }

    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${result.maxTokens.toLocaleString()}</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Max Tokens ${result.tp > 1 ? '(TP=' + result.tp + ')' : ''}</div>
        </div>

        ${result.isHybridModel ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-warning); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: var(--accent-warning); font-size: 0.85rem;">Hybrid Model Warning</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                This appears to be a Hybrid model. The max tokens calculation may not be accurate. Please use the Hybrid Models tab for accurate results.
            </div>
        </div>
        ` : ''}

        <div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Model:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${getModelDisplayName(result.modelName)}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">Type:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.isHybridModel ? 'Hybrid Model (Warning: result may not be accurate)' : result.architectureType}</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">GPU Memory:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.gpuMemoryGiB} GiB</strong>
            </div>
            <div class="metric-item">
                <span style="color: var(--text-secondary);">DType:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.dtype}</strong>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Model Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_hidden_layers}</div>
                    <div style="color: var(--text-secondary);">Hidden Size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.hidden_size}</div>
                    <div style="color: var(--text-secondary);">Attn Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_attention_heads}</div>
                    <div style="color: var(--text-secondary);">KV Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${kvHeads}</div>
                    ${config.kv_lora_rank ? '<div style="color: var(--text-secondary);">KV LoRA Rank:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.kv_lora_rank + '</div>' : ''}
                    ${config.qk_rope_head_dim ? '<div style="color: var(--text-secondary);">QK RoPE Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.qk_rope_head_dim + '</div>' : ''}
                    ${config.index_head_dim ? '<div style="color: var(--text-secondary);">Index Head Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.index_head_dim + '</div>' : ''}
                </div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📐</span>
                <span>Per-Token Formula</span>
            </div>
            <div class="formula-content">
                <div class="formula-main" style="font-size: 0.7rem;">${result.formula}</div>
            </div>
        </div>

        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>🔢</span>
                <span>Max Tokens Calculation</span>
            </div>
            <div class="formula-content">
                <div class="formula-breakdown">
                    <div class="formula-step">
                        <span class="formula-step-label">Memory:</span>
                        <span class="formula-step-value">${(result.gpuMemoryGiB * 1024).toFixed(0)} MiB</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">Per Token:</span>
                        <span class="formula-step-value">${result.perTokenMemoryMiB.toFixed(3)} MiB</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">Max Tokens:</span>
                        <span class="formula-step-value" style="color: var(--accent-success); font-weight: 600;">${result.maxTokens.toLocaleString()}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================
// Toast Notification System
// ============================================================

function showToast(type, title, message) {
    const container = document.getElementById('toast-container');

    const icons = { 'error': '❌', 'success': '✅', 'warning': '⚠️' };

    const toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-icon">${icons[type] || '❌'}</div>
            <div class="toast-info">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
        </div>
        <button class="toast-close" onclick="closeToast(this.parentElement)">×</button>
    `;

    container.appendChild(toast);
    setTimeout(function() { toast.classList.add('show'); }, 10);

    const timeout = type === 'error' ? 8000 : 5000;
    setTimeout(function() { closeToast(toast); }, timeout);
}

function closeToast(toast) {
    if (toast) {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(function() { toast.remove(); }, 300);
    }
}

// ============================================================
// Event Listeners
// ============================================================

function initializeEventListeners() {
    // Enter key support
    const tokenInput = document.getElementById('token-input');
    if (tokenInput) {
        tokenInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') calculateKVCache();
        });
    }

    const modelUrlInput = document.getElementById('model-url');
    if (modelUrlInput) {
        modelUrlInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') calculateKVCache();
        });
    }

    // Preset model selection change
    const presetSelect = document.getElementById('preset-model-select');
    if (presetSelect) {
        presetSelect.addEventListener('change', function() {
            const modelName = this.value;
            if (modelName && modelConfigs[modelName]) {
                updateFormulaReference(modelConfigs[modelName]);
            } else {
                updateFormulaReference(null);
            }
        });

        // Initial formula display
        setTimeout(function() {
            if (presetSelect.value && modelConfigs[presetSelect.value]) {
                updateFormulaReference(modelConfigs[presetSelect.value]);
            }
        }, 100);
    }
}