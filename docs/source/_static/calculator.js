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
        formula: 'block_data_size = 2 × num_hidden_layers × block_size × (num_kv_heads / TP) × head_dim × dtype_bytes   (kv_heads = attn_heads)',
        params: [
            { name: '2', desc: 'Key and Value stored separately' },
            { name: 'num_hidden_layers', desc: 'Number of layers in the model' },
            { name: 'block_size', desc: 'vLLM block size (hash / allocation granularity, default 128)' },
            { name: 'num_kv_heads', desc: 'KV head count (= num_attention_heads for MHA)' },
            { name: 'TP', desc: 'Tensor parallelism — KV is sharded by head, so divided by TP' },
            { name: 'head_dim', desc: 'Head dimension' },
            { name: 'dtype_bytes', desc: 'Data type bytes (float16/bfloat16=2, float32=4, int8=1)' }
        ],
        models: 'GPT-2, BERT',
        note: 'Each attention head has independent Key and Value. kv_heads = attn_heads. KV is sharded by head across TP ranks, so UCM dumps every TP rank.',
        why: 'Each head stores KV independently',
        keyPoint: 'Factor 2 (K+V); KV divided by TP'
    },
    'MQA': {
        title: 'MQA (Multi-Query Attention)',
        icon: '🔹',
        color: '#10b981',
        formula: 'block_data_size = 2 × num_hidden_layers × block_size × (1 / TP) × head_dim × dtype_bytes   (kv_heads = 1)',
        params: [
            { name: '2', desc: 'Key and Value matrices' },
            { name: 'block_size', desc: 'vLLM block size (default 128)' },
            { name: '1', desc: 'KV head count = 1 (shared by all query heads)' },
            { name: 'TP', desc: 'Tensor parallelism — divided by TP' },
            { name: 'head_dim', desc: 'Head dimension' }
        ],
        models: 'PaLM',
        note: 'All Query heads share a single KV head. kv_heads = 1. Still factor 2 and divided by TP.',
        why: 'All heads share single KV, highest efficiency',
        keyPoint: 'Minimum KV Cache; still ×2 and /TP'
    },
    'GQA': {
        title: 'GQA (Grouped-Query Attention)',
        icon: '🔹',
        color: '#8b5cf6',
        formula: 'block_data_size = 2 × num_hidden_layers × block_size × (num_kv_heads / TP) × head_dim × dtype_bytes',
        params: [
            { name: '2', desc: 'Key and Value stored separately' },
            { name: 'num_kv_heads', desc: 'KV head count (less than attn_heads)' },
            { name: 'block_size', desc: 'vLLM block size (default 128)' },
            { name: 'TP', desc: 'Tensor parallelism — KV sharded by head, divided by TP' },
            { name: 'head_dim', desc: 'Dimension per head (= hidden_size / attn_heads when not given)' }
        ],
        models: 'LLaMA-3.1-70B, Qwen3-32B, Mistral-7B, GLM-4.5',
        note: 'Multiple Query heads share a group of KV heads. kv_heads < attn_heads. KV sharded by head across TP ranks, so UCM dumps every TP rank.',
        why: 'Grouped sharing, balances efficiency and quality',
        keyPoint: 'Factor 2 (K+V); KV divided by TP'
    },
    'MLA': {
        title: 'MLA (Multi-head Latent Attention)',
        icon: '🔸',
        color: '#f59e0b',
        formula: 'block_data_size = num_hidden_layers × block_size × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes   (rank-shared, NOT divided by TP)',
        params: [
            { name: 'No factor 2', desc: 'K and V compressed into one latent' },
            { name: 'kv_lora_rank', desc: 'KV compressed latent dimension (e.g., 512)' },
            { name: 'qk_rope_head_dim', desc: 'RoPE positional encoding dimension (e.g., 64)' },
            { name: 'block_size', desc: 'vLLM block size (default 128)' },
            { name: 'No /TP', desc: 'Latent is rank-shared — one physical page holds the full latent; UCM dumps only TP0' }
        ],
        models: 'DeepSeek V3, DeepSeek R1, Kimi K2, GLM-4.7-Flash',
        note: 'KV compressed to low-rank latent space. The latent is replicated across TP ranks, so it is NOT divided by TP and UCM dumps only TP0.',
        why: 'KV compressed to latent space, saving memory',
        keyPoint: 'No ×2, no /TP; rank-shared latent'
    },
    'DSA': {
        title: 'DSA (DeepSeek Sparse Attention)',
        icon: '🔮',
        color: '#9333ea',
        formula: 'block_data_size = num_hidden_layers × block_size × (kv_lora_rank + qk_rope_head_dim + index_head_dim) × dtype_bytes   (rank-shared, NOT divided by TP)',
        params: [
            { name: 'No factor 2', desc: 'K and V compressed into one latent (MLA)' },
            { name: 'kv_lora_rank', desc: 'KV compressed dimension (512)' },
            { name: 'qk_rope_head_dim', desc: 'RoPE dimension (64)' },
            { name: 'index_head_dim', desc: 'Lightning Indexer head dimension (128)' },
            { name: 'block_size', desc: 'vLLM block size (default 128)' },
            { name: 'No /TP', desc: 'Latent is rank-shared — NOT divided by TP; UCM dumps only TP0' }
        ],
        models: 'DeepSeek V3.2, GLM-5, GLM-5.1',
        note: 'MLA + Lightning Indexer, for sparse retrieval. Latent is rank-shared → NOT divided by TP; UCM dumps only TP0.',
        why: 'MLA with sparse retrieval + independent indexer precision',
        keyPoint: 'Additional index_head_dim; no ×2, no /TP'
    },
    'Standard': {
        title: 'Standard Transformer (MHA/MQA/GQA)',
        icon: '🔹',
        color: '#3b82f6',
        formula: 'block_data_size = 2 × num_hidden_layers × block_size × (num_kv_heads / TP) × head_dim × dtype_bytes',
        params: [
            { name: '2', desc: 'Key and Value matrices' },
            { name: 'num_kv_heads', desc: 'KV head count' },
            { name: 'block_size', desc: 'vLLM block size (default 128)' },
            { name: 'TP', desc: 'Tensor parallelism — divided by TP' },
            { name: 'head_dim', desc: 'Head dimension' }
        ],
        models: 'Auto-detected by kv_heads / attn_heads ratio',
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
        const hybridSelect = document.getElementById('hybrid-model-select');
        updateHybridInputsVisibility();
        updateHybridFormulaReference(hybridSelect ? hybridSelect.value : 'deepseek-ai/DeepSeek-V4-Pro');
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

    // Filter out DeepSeek V4 and linear-attention hybrid models (they go in Hybrid tab)
    const standardModels = Object.keys(modelConfigs).filter(name =>
        !name.includes('DeepSeek-V4') && !modelConfigs[name].is_linear_hybrid
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
    const modelName = document.getElementById('hybrid-model-select').value;
    updateHybridInputsVisibility();
    updateHybridFormulaReference(modelName);
}

// Show/hide DS-V4-specific inputs based on the selected model's config flags:
//   vLLM Block Size → only DeepSeek V4 (linear models derive block_size)
//   MTP              → only models with mtp_supported (DS V4, Qwen3.6; Kimi-K3 off)
function updateHybridInputsVisibility() {
    const modelName = document.getElementById('hybrid-model-select').value;
    const cfg = modelConfigs[modelName] || {};
    const bsGroup = document.getElementById('hybrid-vllm-block-size-group');
    const mtpGroup = document.getElementById('hybrid-mtp-group');
    if (bsGroup) bsGroup.style.display = cfg.is_dsv4 ? '' : 'none';
    if (mtpGroup) mtpGroup.style.display = cfg.mtp_supported ? '' : 'none';
    updateHybridBlockSizeOptions();
}

// Rebuild the vLLM Block Size dropdown per Deployment Method:
//   vllm → 256 only (DSV4 MLA kernel supports [256], multiples of 256); vllm-ascend → 32/64/128
function updateHybridBlockSizeOptions() {
    const select = document.getElementById('hybrid-base-block-size');
    if (!select) return;
    const isAscend = document.getElementById('hybrid-deployment').value === 'vllm-ascend';
    const opts = isAscend ? [128, 64, 32] : [256];
    const cur = parseInt(select.value) || (isAscend ? 128 : 256);
    select.innerHTML = '';
    for (const v of opts) {
        const o = document.createElement('option');
        o.value = String(v);
        o.textContent = String(v);
        select.appendChild(o);
    }
    select.value = String(opts.includes(cur) ? cur : opts[0]);
}

// Bottom "Current Model Formula Reference" card for the Hybrid tab:
// linear-attention hybrid vs DeepSeek V4 (sparse attention) have different formulas.
function updateHybridFormulaReference(modelName) {
    const container = document.getElementById('dynamic-formula-container');
    if (!container) return;

    if (modelConfigs[modelName] && modelConfigs[modelName].is_linear_hybrid) {
        const isMLA = !!modelConfigs[modelName].is_mla;
        container.innerHTML = `
            <div class="formula-card" style="margin-bottom: 1.5rem;">
                <div class="formula-header">
                    <span>🌟</span>
                    <span>${isMLA ? 'MLA + Linear (Kimi-K3)' : 'GQA + Linear (Qwen3.6)'} Hybrid Attention</span>
                </div>
                <div class="formula-content">
                    <div class="formula-main" style="font-size: 0.85rem; margin-bottom: 0.75rem;">
                        block_data_size = num_tensors × page_size<br>
                        UCM dump = (fa_blocks × fa_tp_factor + mamba_blocks × TP) × block_data_size
                    </div>
                    <div style="background: rgba(81, 145, 238, 0.1); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.75rem;">
                        <strong style="color: var(--accent-primary);">Key Point:</strong>
                        <span style="color: var(--text-primary); margin-left: 0.25rem;">vLLM block_size is engine-derived (page must hold one FA block AND the mamba state); chunk prefill decides how many mamba state blocks get dumped.</span>
                    </div>
                    <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                        <strong>Block counts (0-hit dump):</strong>
                        <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                            <li>fa_blocks = batch × ⌈seq_len / block_size⌉</li>
                            <li>mamba_blocks = batch × num_chunks × num_mamba_groups</li>
                            <li>num_chunks = ⌈seq_len / actual_chunk⌉, actual_chunk = ⌊chunk_prefill / block_size⌋ × block_size</li>
                        </ul>
                    </div>
                    <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                        <strong>TP dump factors:</strong>
                        <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                            <li>FA: ${isMLA ? 'MLA latent rank-shared → UCM dumps TP0 only (×1)' : 'GQA KV head-sharded → UCM dumps every TP rank (×TP)'}</li>
                            <li>Mamba state: head-sharded (both) → UCM dumps every TP rank (×TP)</li>
                        </ul>
                    </div>
                    <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                        <strong>block_size alignment (not user-set):</strong>
                        <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                            <li>Ascend: 128 × ⌈ssm_size / (128 × attn_single_token_k)⌉</li>
                            <li>CUDA: 16 × ⌈mamba_total / (16 × attn_token)⌉</li>
                        </ul>
                    </div>
                    <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-secondary);">
                        <strong>Models:</strong> Qwen3.6-35B-A3B, Qwen3.6-27B, Kimi-K3
                    </div>
                </div>
            </div>
        `;
        return;
    }

    // DeepSeek V4 (default) reference card
    container.innerHTML = `
        <div class="formula-card" style="margin-bottom: 1.5rem;">
            <div class="formula-header">
                <span>🌟</span>
                <span>DeepSeek V4 (MLA + SWA + Compressor)</span>
            </div>
            <div class="formula-content">
                <div style="background: rgba(81, 145, 238, 0.1); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.75rem;">
                    <strong style="color: var(--accent-primary);">Key Point:</strong>
                    <span style="color: var(--text-primary); margin-left: 0.25rem;">HBM block-data-size (per block_id) ≠ UCM dump block-data-size (per hash block) — computed separately.</span>
                </div>
                <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                    <strong>hash_block_size:</strong>
                    <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                        <li>Ascend: base_block_size × 4 (32/64/128 → 128/256/512)</li>
                        <li>CUDA: 256 (fixed)</li>
                    </ul>
                </div>
                <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                    <strong>HBM block-data-size (per block_id):</strong>
                    <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                        <li>Ascend: num_layer_tuples × bs × 1154 (+ bs×1024 if MTP)</li>
                        <li>CUDA: block_stride = max(group page_size_sum) — MLA group (bs=256, fp8_ds_mla, 584B/token, alignment=576) typically determines block_stride; SWA/compressor fixed</li>
                    </ul>
                </div>
                <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                    <strong>UCM dump (per hash block):</strong>
                    <ul style="margin-left: 1rem; margin-top: 0.25rem;">
                        <li>Ascend FA = bs×(1154×num_c4a + 32×num_c128a); WA = 131072×num_total + 40960×num_c4a</li>
                        <li>CUDA FA = 45824×num_c4a + 1168×num_c128a; WA = 74752×num_total + 40960×num_c4a (hash_bs=256 fixed)</li>
                        <li>round_up to 4096</li>
                    </ul>
                </div>
                <div style="font-size: 0.75rem; margin-bottom: 0.5rem;">
                    <strong>TP dump:</strong> FA &amp; WA both TP-partitioned → total ×1 (each hash block dumped once network-wide).
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-secondary);">
                    <strong>Models:</strong> DeepSeek V4 Pro (30 C4A + 31 C128A), DeepSeek V4 Flash (21 C4A + 20 C128A + 2 SWA-only)
                </div>
            </div>
        </div>
    `;
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
// Core Calculation: KV Cache Size (Standard Models, block granularity)
// ============================================================

// Format a byte count into a compact "GiB / MiB / KiB" string.
function formatBytes(bytes) {
    if (bytes >= Math.pow(1024, 3)) return (bytes / Math.pow(1024, 3)).toFixed(4) + ' GiB';
    if (bytes >= Math.pow(1024, 2)) return (bytes / Math.pow(1024, 2)).toFixed(2) + ' MiB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' KiB';
    return bytes + ' B';
}

function performCalculation(config, tokens, dtype, modelName) {
    const { hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads,
            kv_lora_rank, qk_rope_head_dim, index_head_dim, head_dim } = config;

    const batchSize = parseInt(document.getElementById('batch-size').value) || 1;
    const tp = parseInt(document.getElementById('tp').value) || 1;
    const blockSize = parseInt(document.getElementById('block-size').value) || 128;

    const dtypeSizes = { 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1 };
    const dtypeSize = dtypeSizes[dtype] || 2;

    const modelArch = detectArchitectureType(config);
    const kvHeads = num_key_value_heads || num_attention_heads;
    const hdim = head_dim || (hidden_size / num_attention_heads);

    // block_data_size: single-TP-rank, one block, physical bytes in HBM
    let rankShared = false;   // MLA/DSA: latent replicated across TP, UCM dumps TP0 only
    let blockDataSize = 0;
    let blockFormula = '';

    if (modelArch.isDSA) {
        // DSA: MLA + Lightning Indexer (rank-shared, NOT divided by TP)
        rankShared = true;
        blockDataSize = num_hidden_layers * blockSize * (kv_lora_rank + qk_rope_head_dim + index_head_dim) * dtypeSize;
        blockFormula = `${num_hidden_layers} × ${blockSize} × (${kv_lora_rank} + ${qk_rope_head_dim} + ${index_head_dim}) × ${dtypeSize}`;
    } else if (modelArch.isMLA) {
        // MLA: no factor 2, rank-shared, NOT divided by TP
        rankShared = true;
        blockDataSize = num_hidden_layers * blockSize * (kv_lora_rank + qk_rope_head_dim) * dtypeSize;
        blockFormula = `${num_hidden_layers} × ${blockSize} × (${kv_lora_rank} + ${qk_rope_head_dim}) × ${dtypeSize}`;
    } else {
        // GQA-family (MHA / MQA / GQA, and the hybrid-model fallback)
        rankShared = false;
        const kvHeadsPerRank = kvHeads / tp;
        blockDataSize = 2 * num_hidden_layers * blockSize * kvHeadsPerRank * hdim * dtypeSize;
        blockFormula = `2 × ${num_hidden_layers} × ${blockSize} × (${kvHeads} / ${tp}) × ${hdim} × ${dtypeSize}`;
    }

    // Architecture label for display
    let archLabel;
    if (modelArch.isDSA) {
        archLabel = 'DSA (DeepSeek Sparse Attention)';
    } else if (modelArch.isMLA) {
        archLabel = 'MLA (Multi-head Latent Attention)';
    } else if (modelArch.isHybridModel) {
        archLabel = 'Hybrid Model (GQA-family fallback — may not be accurate)';
    } else if (kvHeads === num_attention_heads) {
        archLabel = 'MHA (Multi-Head Attention)';
    } else if (kvHeads === 1) {
        archLabel = 'MQA (Multi-Query Attention)';
    } else {
        archLabel = 'GQA (Grouped-Query Attention)';
    }

    // Number of blocks for one request, then for the whole batch.
    // Only full blocks (with hash) are dumped to UCM and persist in HBM.
    // Partial block has no hash → not dumped, freed to free_deque head (immediate reuse).
    const blocksPerRequest = Math.floor(tokens / blockSize);
    const totalBlocks = batchSize * blocksPerRequest;

    // HBM occupancy: only full blocks (with hash) persist in HBM after prefill.
    // Partial block has no hash → freed to free_deque head (immediate reuse).
    const hashBlocksPerRequest = Math.floor(tokens / blockSize);
    const totalHashBlocks = batchSize * hashBlocksPerRequest;
    const hbmOccupancyBytes = totalHashBlocks * blockDataSize;

    // UCM dump volume for one request across all TP ranks:
    //   GQA → dumps every TP rank (×TP); MLA/DSA → rank-shared, dumps TP0 only (×1)
    const tpFactor = rankShared ? 1 : tp;
    const totalBytes = totalBlocks * blockDataSize * tpFactor;
    const totalGiB = totalBytes / Math.pow(1024, 3);
    const totalMiB = totalBytes / Math.pow(1024, 2);

    return {
        modelName,
        tokens,
        batchSize,
        tp,
        blockSize,
        dtype,
        dtypeSize,
        config,
        archLabel,
        rankShared,
        kvHeads,
        hdim,
        blockDataSize,
        blockFormula,
        blocksPerRequest,
        totalBlocks,
        tpFactor,
        totalBytes,
        totalGiB,
        totalMiB,
        hashBlocksPerRequest,
        totalHashBlocks,
        hbmOccupancyBytes,
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

    const tp = parseInt(document.getElementById('tp').value) || 1;
    const blockSize = parseInt(document.getElementById('block-size').value) || 128;

    const dtypeSizes = { 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1 };
    const dtypeSize = dtypeSizes[dtype] || 2;

    const modelArch = detectArchitectureType(config);
    const kvHeads = num_key_value_heads || num_attention_heads;
    const hdim = head_dim || (hidden_size / num_attention_heads);

    // block_data_size: single-TP-rank, one block, physical bytes in HBM
    let rankShared = false;
    let blockDataSize = 0;
    let blockFormula = '';

    if (modelArch.isDSA) {
        rankShared = true;
        blockDataSize = num_hidden_layers * blockSize * (kv_lora_rank + qk_rope_head_dim + index_head_dim) * dtypeSize;
        blockFormula = `${num_hidden_layers} × ${blockSize} × (${kv_lora_rank} + ${qk_rope_head_dim} + ${index_head_dim}) × ${dtypeSize}`;
    } else if (modelArch.isMLA) {
        rankShared = true;
        blockDataSize = num_hidden_layers * blockSize * (kv_lora_rank + qk_rope_head_dim) * dtypeSize;
        blockFormula = `${num_hidden_layers} × ${blockSize} × (${kv_lora_rank} + ${qk_rope_head_dim}) × ${dtypeSize}`;
    } else {
        rankShared = false;
        blockDataSize = 2 * num_hidden_layers * blockSize * (kvHeads / tp) * hdim * dtypeSize;
        blockFormula = `2 × ${num_hidden_layers} × ${blockSize} × (${kvHeads} / ${tp}) × ${hdim} × ${dtypeSize}`;
    }

    // A single GPU holds floor(gpu_mem / block_data_size) blocks → that many tokens.
    const totalMemoryBytes = gpuMemoryGiB * Math.pow(1024, 3);
    const maxBlocks = Math.floor(totalMemoryBytes / blockDataSize);
    const maxTokens = maxBlocks * blockSize;

    let archLabel;
    if (modelArch.isDSA) archLabel = 'DSA (DeepSeek Sparse Attention)';
    else if (modelArch.isMLA) archLabel = 'MLA (Multi-head Latent Attention)';
    else if (modelArch.isHybridModel) archLabel = 'Hybrid Model (GQA-family fallback — may not be accurate)';
    else if (kvHeads === num_attention_heads) archLabel = 'MHA (Multi-Head Attention)';
    else if (kvHeads === 1) archLabel = 'MQA (Multi-Query Attention)';
    else archLabel = 'GQA (Grouped-Query Attention)';

    return {
        modelName,
        tp,
        blockSize,
        gpuMemoryGiB,
        dtype,
        dtypeSize,
        maxBlocks,
        maxTokens,
        blockDataSize,
        blockFormula,
        rankShared,
        archLabel,
        isHybridModel: modelArch.isHybridModel,
        config
    };
}

// ============================================================
// Hybrid Models Calculation
// ============================================================
// DeepSeek V4 (sparse-attention hybrid) — calculation logic removed; to be
// reimplemented from scratch (pure model config lives in model-configs.js).

function calculateHybrid() {
    clearResults();

    const modelName = document.getElementById('hybrid-model-select').value;

    // Dispatch: linear-attention hybrid models have their own derivation path;
    // DeepSeek V4 (sparse-attention hybrid) has its own derivation too.
    if (modelConfigs[modelName] && modelConfigs[modelName].is_linear_hybrid) {
        calculateLinearHybrid();
        return;
    }
    if (modelConfigs[modelName] && modelConfigs[modelName].is_dsv4) {
        calculateDSv4();
        return;
    }

    displayError('Model Not Found', 'Unknown hybrid model.');
}

// ============================================================
// Linear-Attention Hybrid Models (HLA: Qwen3.6 / Kimi-K3)
// ============================================================

// Ceiling-division helper.
function cdiv(a, b) { return Math.ceil(a / b); }

// Round up x to the next multiple of alignment.
function roundUp(x, alignment) { return Math.ceil(x / alignment) * alignment; }

// GCD and LCM helpers (for alignment_tokens computation).
function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }
function lcm(a, b) { return (a * b) / gcd(a, b); }

// Count full blocks that have hash (are cached) for a SlidingWindow group.
//   need = cdiv(sw - 1, lbs)  — consecutive blocks needed for a hit
//   per_segment = alignment // lbs  — blocks per alignment segment
//   need >= per_segment → all full blocks reachable
//   need < per_segment → only the last `need` blocks of each segment are reachable
function reachableHashCount(tokens, lbs, sw, alignment) {
    const need = cdiv(sw - 1, lbs);
    const perSegment = Math.floor(alignment / lbs);
    const numFull = Math.floor(tokens / lbs);
    if (need >= perSegment) return numFull;
    return Math.floor(numFull / perSegment) * need
         + Math.max(0, numFull % perSegment - (perSegment - need));
}

// Derive vLLM block_size, page_size, block_data_size and GDN state sizes for a
// linear-attention hybrid model straight from its config. Shared by the
// dump-volume and max-tokens calculations.
// `mtp`: when true, MTP adds one extra full-attn tuple (num_tensors += 1);
//        page_size is unchanged, so block_data_size grows by one page_size.
//        fa_blocks / mamba_blocks / num_mamba_groups are unaffected (MTP reuses
//        the FA group's block ids).
function deriveLinearHybridParams(config, tp, isAscend, mtp) {
    const dtypeSizes = { 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1 };
    const modelDtype = config.dtype || 'bfloat16';
    const modelDtypeSize = dtypeSizes[modelDtype] || 2;
    const ssmDtype = config.mamba_ssm_dtype || modelDtype;
    const ssmDtypeSize = dtypeSizes[ssmDtype] || 4;
    const isMLA = !!config.is_mla;

    // Layer partitioning (1 full-attn group + N mamba-align groups).
    // MTP adds one full-attn layer → group_size grows by 1 (for Qwen3.6 where
    // num_linear = 3×num_full, group_size = num_full + (mtp?1:0)).
    const numFull = config.num_full_attn_layers;
    const numLinear = config.num_linear_layers;
    const numFullEff = numFull + (mtp ? 1 : 0);
    const groupSize = Math.min(numFullEff, numLinear);
    const numTensors = groupSize;
    const numMambaGroups = cdiv(numLinear, groupSize);

    // GDN state sizes (per rank; conv/ssm are head-sharded → divided by TP)
    const linKeyHeads = config.linear_num_key_heads;
    const linValueHeads = config.linear_num_value_heads;
    const linKeyDim = config.linear_key_head_dim;
    const linValueDim = config.linear_value_head_dim;
    const convKernel = config.linear_conv_kernel_dim;

    const valueHeadsPerRank = linValueHeads / tp;
    const ssmSize = valueHeadsPerRank * linValueDim * linKeyDim * ssmDtypeSize;
    const convDim = linKeyDim * linKeyHeads * 2 + linValueDim * linValueHeads; // ×2 = GDN double key
    const convDimPerRank = convDim / tp;
    const convSize = (convKernel - 1) * convDimPerRank * modelDtypeSize;
    const mambaTotal = convSize + ssmSize;

    // Full-attention per-token size:
    //   GQA: K+V = 2 × head_dim × (kv_heads/TP) × dtype (head-sharded)
    //   MLA: latent = (kv_lora_rank + qk_rope_head_dim) × 1 × dtype (num_kv_heads=1, rank-shared, NOT /TP)
    let attnTokenPerPageToken, attnSingleTokenK, faArchLabel, faPerTokenFormula;
    if (isMLA) {
        const kvLora = config.kv_lora_rank;
        const qkRope = config.qk_rope_head_dim;
        attnTokenPerPageToken = (kvLora + qkRope) * 1 * modelDtypeSize;
        attnSingleTokenK = kvLora * 1 * modelDtypeSize; // nope = kv_lora_rank
        faArchLabel = 'MLA (latent, rank-shared)';
        faPerTokenFormula = `(${kvLora} + ${qkRope}) × 1 × ${modelDtypeSize} = ${attnTokenPerPageToken} B/token`;
    } else {
        const headDim = config.head_dim;
        const kvHeadsPerRank = config.num_key_value_heads / tp;
        attnTokenPerPageToken = 2 * headDim * kvHeadsPerRank * modelDtypeSize;
        attnSingleTokenK = headDim * kvHeadsPerRank * modelDtypeSize;
        faArchLabel = 'GQA (K+V, head-sharded)';
        faPerTokenFormula = `2 × ${headDim} × (${config.num_key_value_heads}/${tp}) × ${modelDtypeSize} = ${attnTokenPerPageToken} B/token`;
    }

    // vLLM block_size (engine-aligned, NOT user-set) + page_size + block_data_size
    let blockSize, pageSize, blockSizeFormula, pageSizeFormula;
    if (isAscend) {
        // block_size chosen so block_size × attn_single_token_k = ssm_size (K shares middle with ssm)
        const KERNEL = 128; // NPU alignment granularity
        const ratio = cdiv(ssmSize, KERNEL * attnSingleTokenK);
        blockSize = KERNEL * ratio;
        pageSize = blockSize * attnTokenPerPageToken + convSize; // ssm overlays K; conv added
        blockSizeFormula = `128 × ⌈${ssmSize.toLocaleString()} / (128 × ${attnSingleTokenK})⌉ = 128 × ${ratio} = ${blockSize}`;
        pageSizeFormula = `${blockSize} × ${attnTokenPerPageToken} (K+V) + ${convSize.toLocaleString()} (conv) = ${pageSize.toLocaleString()} B`;
    } else {
        // CUDA: block_size × attn_token ≥ mamba_total (mamba padded into the K+V page)
        const KERNEL = 16; // GPU FlashAttention alignment
        const ratio = cdiv(mambaTotal, KERNEL * attnTokenPerPageToken);
        blockSize = KERNEL * ratio;
        pageSize = blockSize * attnTokenPerPageToken;
        blockSizeFormula = `16 × ⌈${mambaTotal.toLocaleString()} / (16 × ${attnTokenPerPageToken})⌉ = 16 × ${ratio} = ${blockSize}`;
        pageSizeFormula = `${blockSize} × ${attnTokenPerPageToken} = ${pageSize.toLocaleString()} B`;
    }
    const blockDataSize = numTensors * pageSize;

    const archLabel = isMLA
        ? 'Linear-Attention Hybrid — MLA-based (Kimi-K3)'
        : 'Linear-Attention Hybrid — GQA-based (Qwen3.6)';

    return {
        isMLA, mtp, modelDtype, modelDtypeSize, ssmDtype, ssmDtypeSize,
        numFull, numLinear, numTensors, numMambaGroups,
        linKeyHeads, linValueHeads, linKeyDim, linValueDim,
        ssmSize, convSize, mambaTotal,
        attnTokenPerPageToken, faArchLabel, faPerTokenFormula,
        blockSize, pageSize, blockDataSize, blockSizeFormula, pageSizeFormula,
        archLabel
    };
}

function calculateLinearHybrid() {
    const modelName = document.getElementById('hybrid-model-select').value;
    const config = modelConfigs[modelName];
    if (!config || !config.is_linear_hybrid) {
        displayError('Model Not Found', 'Linear-attention hybrid model configuration not found.');
        return;
    }

    const deployment = document.getElementById('hybrid-deployment').value;
    const isAscend = deployment === 'vllm-ascend';
    const tokens = parseInt(document.getElementById('hybrid-token-input').value) || 4096;
    const batchSize = parseInt(document.getElementById('hybrid-batch-size').value) || 1;
    const tp = parseInt(document.getElementById('hybrid-tp').value) || 1;
    const chunkPrefill = parseInt(document.getElementById('hybrid-chunk-prefill').value) || 2048;
    const mtp = config.mtp_supported ? document.getElementById('hybrid-mtp').checked : false;

    if (tokens <= 0 || batchSize <= 0 || tp <= 0) {
        displayError('Invalid Input', 'Tokens, batch size, and TP must be positive.');
        return;
    }

    const p = deriveLinearHybridParams(config, tp, isAscend, mtp);

    // Block counts (0-hit dump scenario)
    // Only full blocks (with hash) are dumped to UCM and persist in HBM.
    // Partial block has no hash → not dumped, freed to free_deque head (immediate reuse).
    const faBlocksPerReq = Math.floor(tokens / p.blockSize);
    const faBlocks = batchSize * faBlocksPerReq;
    const actualChunk = Math.floor(chunkPrefill / p.blockSize) * p.blockSize; // align down to block_size
    const numChunks = actualChunk > 0 ? cdiv(tokens, actualChunk) : 1;
    const mambaBlocks = batchSize * numChunks * p.numMambaGroups;

    // TP factors for UCM dump (the key MLA vs GQA difference)
    //   MLA full-attn latent is rank-shared → UCM dumps TP0 only (×1)
    //   GQA full-attn KV is head-sharded → UCM dumps every TP rank (×TP)
    //   Linear/mamba state is head-sharded for both → UCM dumps every TP rank (×TP)
    const faTpFactor = p.isMLA ? 1 : tp;
    const mambaTpFactor = tp;

    const totalBytes = faBlocks * p.blockDataSize * faTpFactor + mambaBlocks * p.blockDataSize * mambaTpFactor;
    const totalGiB = totalBytes / Math.pow(1024, 3);

    // HBM occupancy: full FA blocks + mamba state blocks (all have hash, no reachable mask)
    const faHashPerReq = Math.floor(tokens / p.blockSize);
    const mambaHashPerReq = numChunks * p.numMambaGroups;
    const hashBlocksPerReq = faHashPerReq + mambaHashPerReq;
    const totalHashBlocks = batchSize * hashBlocksPerReq;
    const hbmOccupancyBytes = totalHashBlocks * p.blockDataSize;

    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="result-display" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${formatBytes(hbmOccupancyBytes)}</div>
                <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">KV Cache HBM Occupancy ${batchSize > 1 ? `(${batchSize} reqs)` : '(1 req)'}</div>
                <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">${totalHashBlocks.toLocaleString()} hash blocks × ${p.blockDataSize.toLocaleString()} B = ${formatBytes(hbmOccupancyBytes)}</div>
            </div>
            <div style="text-align: center;">
                <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-primary);">${formatBytes(totalBytes)}</div>
                <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">KV Cache UCM Dump ${batchSize > 1 ? `(${batchSize} reqs)` : '(1 req)'}</div>
                <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">${totalBytes.toLocaleString()} B = ${totalGiB.toFixed(4)} GiB</div>
            </div>
        </div>

        <!-- Sub-panel 1: Model Configuration -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Model Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Model:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${modelName.split('/')[1] || modelName}</div>
                    <div style="color: var(--text-secondary);">Architecture:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.archLabel}</div>
                    <div style="color: var(--text-secondary);">Total layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_hidden_layers} (${p.numFull} full + ${p.numLinear} linear)${p.mtp ? ' + 1 MTP' : ''}</div>
                    <div style="color: var(--text-secondary);">KVCacheTensors:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.numTensors}${p.mtp ? ' (incl. MTP tuple)' : ''}</div>
                    <div style="color: var(--text-secondary);">Mamba groups:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.numMambaGroups}</div>
                    <div style="color: var(--text-secondary);">FA type:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.faArchLabel}</div>
                    <div style="color: var(--text-secondary);">Model dtype:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.modelDtype} (${p.modelDtypeSize} B) — FA / conv</div>
                    <div style="color: var(--text-secondary);">ssm dtype:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.ssmDtype} (${p.ssmDtypeSize} B) — mamba state</div>
                    <div style="color: var(--text-secondary);">Linear heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.linKeyHeads} key / ${p.linValueHeads} value × ${p.linKeyDim}/${p.linValueDim}</div>
                    <div style="color: var(--text-secondary);">Platform:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${isAscend ? 'vllm-ascend (Ascend)' : 'vllm (CUDA)'}</div>
                </div>
            </div>
        </div>

        <!-- Sub-panel 2: Block Data Size (derived vLLM block_size + page) -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📦</span>
                <span>Block Data Size</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">vLLM block_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.blockSize} tokens/block <span style="font-weight: 400; color: var(--text-secondary);">(derived, not user-set)</span></div>
                    <div style="color: var(--text-secondary);">page_size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.pageSize.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">block_data_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.blockDataSize.toLocaleString()} B = ${formatBytes(p.blockDataSize)}</div>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-primary); background: rgba(81, 145, 238, 0.08); padding: 0.4rem; border-radius: 4px; margin-bottom: 0.3rem; font-family: monospace;">
                    block_size = ${p.blockSizeFormula}<br>
                    page_size = ${p.pageSizeFormula}<br>
                    block_data_size = ${p.numTensors} × ${p.pageSize.toLocaleString()} = ${p.blockDataSize.toLocaleString()} B
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); line-height: 1.4;">
                    ssm_state (per rank) = ${p.ssmSize.toLocaleString()} B | conv_state (per rank) = ${p.convSize.toLocaleString()} B | mamba_total = ${p.mambaTotal.toLocaleString()} B<br>
                    FA per-token (per rank) = ${p.faPerTokenFormula}
                </div>
            </div>
        </div>

        <!-- Sub-panel 3: KV Cache HBM Occupancy -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>💾</span>
                <span>KV Cache HBM Occupancy (single card, after prefill)</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">FA hash blocks / req:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">⌊${tokens.toLocaleString()}/${p.blockSize}⌋ = ${faHashPerReq}</div>
                    <div style="color: var(--text-secondary);">Mamba hash blocks / req:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${numChunks} chunks × ${p.numMambaGroups} groups = ${mambaHashPerReq}</div>
                    <div style="color: var(--text-secondary);">Total hash blocks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize} × ${hashBlocksPerReq} = ${totalHashBlocks.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">× block_data_size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.blockDataSize.toLocaleString()} B</div>
                </div>
                <div style="margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--border-color);">
                    <strong style="color: var(--accent-success);">HBM occupancy:</strong> ${totalHashBlocks.toLocaleString()} × ${p.blockDataSize.toLocaleString()} = ${hbmOccupancyBytes.toLocaleString()} B = ${formatBytes(hbmOccupancyBytes)}
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.3rem; line-height: 1.4;">
                    FA + mamba state blocks all have hash (no reachable mask). Single-card: block_data_size already accounts for TP.
                </div>
            </div>
        </div>

        <!-- Sub-panel 4: KV Cache UCM Dump -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>🧮</span>
                <span>KV Cache UCM Dump (across all TP ranks)</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <div style="color: var(--text-secondary);">Tokens / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${tokens.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">Batch size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize}</div>
                    <div style="color: var(--text-secondary);">vLLM block_size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.blockSize}</div>
                    <div style="color: var(--text-secondary);">FA blocks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize} × ⌈${tokens.toLocaleString()}/${p.blockSize}⌉ = ${faBlocks.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">Chunk prefill:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${chunkPrefill} → aligned ${actualChunk} → ${numChunks} chunks</div>
                    <div style="color: var(--text-secondary);">Mamba blocks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize} × ${numChunks} × ${p.numMambaGroups} = ${mambaBlocks.toLocaleString()}</div>
                </div>
                <div style="margin-top: 0.5rem; padding: 0.4rem; background: rgba(81, 145, 238, 0.08); border-radius: 4px; font-family: monospace; font-size: 0.68rem;">
                    FA: ${faBlocks.toLocaleString()} × ${p.blockDataSize.toLocaleString()} × ${faTpFactor} ${p.isMLA ? '(MLA rank-shared → TP0)' : '(GQA → all TP)'} = ${(faBlocks * p.blockDataSize * faTpFactor).toLocaleString()} B<br>
                    Mamba: ${mambaBlocks.toLocaleString()} × ${p.blockDataSize.toLocaleString()} × ${mambaTpFactor} (state head-sharded → all TP) = ${(mambaBlocks * p.blockDataSize * mambaTpFactor).toLocaleString()} B
                </div>
                <div style="margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--border-color);">
                    <strong style="color: var(--accent-primary);">Total dump:</strong> ${formatBytes(faBlocks * p.blockDataSize * faTpFactor)} + ${formatBytes(mambaBlocks * p.blockDataSize * mambaTpFactor)} = ${formatBytes(totalBytes)}
                </div>
            </div>
        </div>
    `;
}

// Maximum tokens for a linear-attention hybrid model on a single GPU.
// HBM peak = null_block(1) + mamba rolling window (2 × num_mamba_groups) + full-attn blocks.
//   max_tokens = (total_blocks − 1 − 2×num_mamba_groups) × block_size
function calculateLinearMaxTokens() {
    const modelName = document.getElementById('hybrid-model-select').value;
    const config = modelConfigs[modelName];
    if (!config || !config.is_linear_hybrid) {
        displayError('Model Not Found', 'Linear-attention hybrid model configuration not found.');
        return;
    }

    const deployment = document.getElementById('hybrid-deployment').value;
    const isAscend = deployment === 'vllm-ascend';
    const gpuMemoryGiB = parseFloat(document.getElementById('hybrid-gpu-memory-input').value) || 42;
    const tp = parseInt(document.getElementById('hybrid-tp').value) || 1;
    const mtp = config.mtp_supported ? document.getElementById('hybrid-mtp').checked : false;

    if (gpuMemoryGiB <= 0 || tp <= 0) {
        displayError('Invalid Input', 'GPU memory and TP must be positive.');
        return;
    }

    const p = deriveLinearHybridParams(config, tp, isAscend, mtp);

    const totalMemoryBytes = gpuMemoryGiB * Math.pow(1024, 3);
    const totalBlocks = Math.floor(totalMemoryBytes / p.blockDataSize);
    const nullBlocks = 1;                       // block_id=0 reserved by BlockPool, never freed
    const mambaPeak = 2 * p.numMambaGroups;     // rolling window: pre-copy source + running state
    const reserved = nullBlocks + mambaPeak;
    const availableFaBlocks = Math.max(0, totalBlocks - reserved);
    const maxTokens = availableFaBlocks * p.blockSize;

    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${maxTokens.toLocaleString()}</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Max tokens for one request on a single GPU ${tp > 1 ? '(TP=' + tp + ')' : ''}</div>
            <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">= ${availableFaBlocks.toLocaleString()} full-attn blocks × ${p.blockSize} tokens/block</div>
        </div>

        <!-- Sub-panel 1: Model Configuration -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Model Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Model:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${modelName.split('/')[1] || modelName}</div>
                    <div style="color: var(--text-secondary);">Architecture:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.archLabel}</div>
                    <div style="color: var(--text-secondary);">Mamba groups:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.numMambaGroups}</div>
                    <div style="color: var(--text-secondary);">FA type:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.faArchLabel}</div>
                    <div style="color: var(--text-secondary);">Platform:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${isAscend ? 'vllm-ascend (Ascend)' : 'vllm (CUDA)'}</div>
                    <div style="color: var(--text-secondary);">GPU memory:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${gpuMemoryGiB} GiB</div>
                </div>
            </div>
        </div>

        <!-- Sub-panel 2: Block Data Size -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📦</span>
                <span>Block Data Size</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">vLLM block_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.blockSize} tokens/block</div>
                    <div style="color: var(--text-secondary);">block_data_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.blockDataSize.toLocaleString()} B = ${formatBytes(p.blockDataSize)}</div>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-primary); background: rgba(81, 145, 238, 0.08); padding: 0.4rem; border-radius: 4px; font-family: monospace;">
                    block_size = ${p.blockSizeFormula}<br>
                    page_size = ${p.pageSizeFormula}<br>
                    block_data_size = ${p.numTensors} × ${p.pageSize.toLocaleString()} = ${p.blockDataSize.toLocaleString()} B
                </div>
            </div>
        </div>

        <!-- Sub-panel 3: Max Tokens Calculation -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>🔢</span>
                <span>Max Tokens Calculation</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div class="formula-breakdown">
                    <div class="formula-step">
                        <span class="formula-step-label">GPU memory:</span>
                        <span class="formula-step-value">${(gpuMemoryGiB * 1024).toFixed(0)} MiB</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">÷ block_data_size:</span>
                        <span class="formula-step-value">${p.blockDataSize.toLocaleString()} B</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">= total blocks:</span>
                        <span class="formula-step-value">${totalBlocks.toLocaleString()}</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">− null block:</span>
                        <span class="formula-step-value">1 (BlockPool reserves block_id=0)</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">− mamba peak:</span>
                        <span class="formula-step-value">${p.numMambaGroups} × 2 = ${mambaPeak} (rolling window: pre-copy + running)</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">= full-attn blocks:</span>
                        <span class="formula-step-value">${availableFaBlocks.toLocaleString()}</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">× block_size:</span>
                        <span class="formula-step-value">${p.blockSize}</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">= max tokens:</span>
                        <span class="formula-step-value" style="color: var(--accent-success); font-weight: 600;">${maxTokens.toLocaleString()}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================
// DeepSeek V4 (sparse-attention hybrid: MLA indexer + SWA + Compressor)
// ============================================================
// All formulas derived from model config (compress_ratios, head_dim,
// index_head_dim, sliding_window, dtypes) and verified against
// design_hma_fawa_connector.md Appendix A + hma_connector.py:579-602.

// Mirror of vLLM's _approximate_gcd: pick d in [lb, max(sizes)] minimizing
// total round-up padding Σ(cdiv(s,d)*d - s).
function approximateGcd(sizes, lb) {
    let bestD = lb, bestPad = Infinity;
    const hi = Math.max(...sizes);
    for (let d = lb; d <= hi; d++) {
        let pad = 0;
        for (const s of sizes) pad += cdiv(s, d) * d - s;
        if (pad < bestPad) { bestPad = pad; bestD = d; }
    }
    return bestD;
}

// Derive the three block-data-sizes for DeepSeek V4 from config.
//   bs   : cache_config.block_size (Ascend 32/64/128, CUDA 32/64/128/256)
//   mtp  : whether speculative decode adds 1 MTP layer
function deriveDSv4Params(config, bs, isAscend, mtp) {
    const dtypeSizes = { 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1 };
    const modelDtype = config.dtype || 'bfloat16';
    const modelDt = dtypeSizes[modelDtype] || 2;          // MLA / SWA / conv (bf16=2)
    const ssmDt = 4;                                       // compressor state always fp32
    const indexerDt = 1;                                   // indexer K is int8/fp8 (1 byte)

    const headDim = config.head_dim;                        // MLA latent dim = 512 (rope folded in)
    const indexHeadDim = config.index_head_dim;             // 128
    const windowSize = config.sliding_window;               // 128

    // --- Layer counts from compress_ratios (trailing 0 = MTP slot, excluded) ---
    const ratios = config.compress_ratios;
    const decoder = ratios.slice(0, -1);                    // drop trailing MTP slot
    let numC4a = 0, numC128a = 0, numSwaOnly = 0;
    for (const r of decoder) {
        if (r === 4) numC4a++;
        else if (r === 128) numC128a++;
        else numSwaOnly++;                                  // r == 0/1 (SWA-only, Flash only)
    }
    const numTotalDecoder = decoder.length;                // Pro 61, Flash 43
    const numTotalLayers = numTotalDecoder + (mtp ? 1 : 0); // WA file_size uses this

    // --- per-token constants (derived) ---
    // Ascend MLA: kv_lora latent per token = head_dim × model_dtype (512×2 = 1024)
    const mlaPerToken = headDim * modelDt;                  // 1024 (bf16, Ascend)
    // fp8_ds_mla per-token: 448 NoPE + 128 RoPE + 8 fp8 scale = 584 (CUDA default)
    const mlaPerTokenFp8 = 584;
    // bucket0 (C4 indexer K): index_head_dim × indexer_dtype + scale (128×1 + 2 = 130)
    const indexerPerTokenAscend = indexHeadDim * indexerDt + 2;
    // CUDA indexer (fp8 layout): index_head_dim + scale (128 + 128/128*4 = 132)
    const indexerPerTokenCUDA = indexHeadDim + Math.floor(indexHeadDim / 128) * 4; // 132
    // Alignment for fp8_ds_mla layout (CUDA): page padded up to multiple of 576
    const fp8Alignment = 576;
    // C128 contributes 1 MLA page per (compress_ratio/c4_ratio) hash-blocks: 1024×4/128 = 32 (per bs-token)
    const c128PerBsToken = mlaPerToken * 4 / 128;           // 32
    // SWA per hash block = window_size × mlaPerToken (128×1024 = 131072)
    const swaPerHash = windowSize * mlaPerToken;             // 131072
    // C4 compressor state_dim = 2×coff×head_dim: C4 coff=2 → 4×512=2048; C128 coff=1 → 2×512=1024
    const c4StateDim = 4 * headDim;                          // 2048
    const c128StateDim = 2 * headDim;                        // 1024
    const c4AttnCompPerHash = 4 * c4StateDim * ssmDt;        // 4×2048×4 = 32768  (compressor block_size=4)
    const c4IdxCompPerHash = 4 * headDim * ssmDt;           // 4×512×4  = 8192
    const c4CompPerHash = c4AttnCompPerHash + c4IdxCompPerHash; // 40960
    const c128CompPerHash = 8 * c128StateDim * ssmDt;        // 8×1024×4 = 32768 (block_size=8)

    // --- num_layer_tuples (engine _approximate_gcd over the 5 buckets, NO MTP) ---
    // HBM uses this bucket-tensor count; MTP adds a separate tensor (+ bs×1024).
    const buckets = [numC4a, numC128a, numTotalDecoder, numC4a, numC128a];
    const lb = Math.min(numC4a, numC128a) || 1;
    const numLayerTuples = approximateGcd(buckets, lb);    // Pro 31, Flash 22 (constant, MTP-independent)

    // --- hash_block_size ---
    const hashBlockSize = isAscend ? bs * 4 : 256;           // Ascend bs×4, CUDA fixed 256

    // --- HBM block_data_size (per block_id) ---
    let hbmBlockDataSize, hbmFormula;
    if (isAscend) {
        // num_layer_tuples × bs × 1154 (+ bs×1024 if MTP, own MTP tensor)
        const tupleBytes = bs * (mlaPerToken + indexerPerTokenAscend); // bs × 1154
        hbmBlockDataSize = numLayerTuples * tupleBytes + (mtp ? bs * mlaPerToken : 0);
        hbmFormula = `${numLayerTuples} × ${bs} × ${mlaPerToken + indexerPerTokenAscend}` +
            (mtp ? ` + ${bs}×${mlaPerToken} (MTP)` : '');
    } else {
        // CUDA packed slab: block_stride = max group page_size_sum
        // fp8_ds_mla layout (default): MLA/SWA per-token = 584, alignment = 576
        //   G0 MLA (scales with bs): C4 MLA + C128 MLA + indexer
        //   G1/G2 SWA (fixed, block_size 64)
        //   G3 C4 comp / G4 C128 comp (fixed, block_size 4/8)
        const g0Mla = numC4a * roundUp(bs / 4 * mlaPerTokenFp8, fp8Alignment)
            + numC128a * roundUp(bs / 128 * mlaPerTokenFp8, fp8Alignment)
            + numC4a * roundUp(bs / 4 * indexerPerTokenCUDA, fp8Alignment);
        const swaSubgroups = cdiv(numTotalDecoder, numLayerTuples); // SWA split count (2)
        const swaSub1 = cdiv(numTotalDecoder, swaSubgroups);        // larger sub-group (Pro 31, Flash 22)
        const g1Swa = swaSub1 * roundUp(64 * mlaPerTokenFp8, fp8Alignment);
        const g3C4comp = numC4a * roundUp(c4AttnCompPerHash, fp8Alignment)
            + numC4a * roundUp(c4IdxCompPerHash, fp8Alignment);
        const g4C128comp = numC128a * roundUp(c128CompPerHash, fp8Alignment);
        hbmBlockDataSize = Math.max(g0Mla, g1Swa, g3C4comp, g4C128comp);
        hbmFormula = `max(G0_MLA=${g0Mla.toLocaleString()}, G1_SWA=${g1Swa.toLocaleString()}, ` +
            `G3_C4comp=${g3C4comp.toLocaleString()}, G4_C128comp=${g4C128comp.toLocaleString()})`;
    }

    // --- UCM dump file_size (per hash block), split FA / WA ---
    let faFile, waFile, faFormula, waFormula;
    if (isAscend) {
        faFile = bs * ((mlaPerToken + indexerPerTokenAscend) * numC4a + c128PerBsToken * numC128a);
        waFile = swaPerHash * numTotalLayers + c4CompPerHash * numC4a;
        faFormula = `${bs} × (${mlaPerToken + indexerPerTokenAscend}×${numC4a} + ${c128PerBsToken}×${numC128a})`;
        waFormula = `${swaPerHash}×${numTotalLayers} + ${c4CompPerHash}×${numC4a}`;
    } else {
        // CUDA constants from hma_connector.py:595-600 (fp8_ds_mla layout, hash_bs=256 fixed)
        // 37376 = C4A MLA page (64×584, storage_bs=256//4=64), coincidentally = SWA page (64×584)
        // 8448 = C4 Indexer page (64×132, per_token=128+4), 1168 = C128 MLA page (2×584)
        // 74752 = 37376×2 (SWA ×2 blocks per boundary: tail_blocks=window_size/64=2)
        // 40960 = C4 comp per hash block (32768 attn + 8192 idx, unpadded)
        const swaPageCUDA = 64 * 584;                        // 37376 (fp8 UE8M0: 448 nope + 128 rope + 8 scale)
        const c4IndexerPageCUDA = 64 * indexerPerTokenCUDA;  // 8448 = 64×132
        const c128MlaPerLayerCUDA = 2 * mlaPerTokenFp8;     // 1168 = 2×584
        faFile = (swaPageCUDA + c4IndexerPageCUDA) * numC4a + c128MlaPerLayerCUDA * numC128a;
        waFile = swaPageCUDA * 2 * numTotalLayers + c4CompPerHash * numC4a;
        faFormula = `(37376+8448)×${numC4a} + 1168×${numC128a}`;
        waFormula = `74752×${numTotalLayers} + 40960×${numC4a}`;
    }
    faFile = Math.ceil(faFile / 4096) * 4096;                // round_up to 4096
    waFile = Math.ceil(waFile / 4096) * 4096;

    // --- alignment_tokens and group info (for HBM hash block counting) ---
    // alignment_tokens = lcm(effective block_size) where effective = lbs for each group.
    // Ascend patch overrides SWA manager's scheduler_block_size to this lcm value,
    // making reachable_block_mask sparse (only segment-tail blocks cached).
    const groups = isAscend ? [
        {name: 'G0 C4 MLA',   lbs: bs * 4,     isFA: true,  sw: null},
        {name: 'G1 C128 MLA', lbs: bs * 128,   isFA: true,  sw: null},
        {name: 'G2 SWA',      lbs: bs,          isFA: false, sw: windowSize},
        {name: 'G3 SWA+MTP',  lbs: bs,          isFA: false, sw: windowSize},
        {name: 'G4 C4 comp',  lbs: bs / 16,     isFA: false, sw: 8},
        {name: 'G5 C128 comp', lbs: bs / 4,     isFA: false, sw: 128},
    ] : [
        {name: 'G0 MLA',       lbs: bs,  isFA: true,  sw: null},
        {name: 'G1 SWA',       lbs: 64,  isFA: false, sw: windowSize},
        {name: 'G2 SWA+MTP',   lbs: 64,  isFA: false, sw: windowSize},
        {name: 'G3 C4 comp',   lbs: 4,   isFA: false, sw: 8},
        {name: 'G4 C128 comp', lbs: 8,   isFA: false, sw: 128},
    ];
    const alignmentTokens = groups.map(g => g.lbs).reduce((a, b) => lcm(a, b));

    return {
        bs, isAscend, mtp, modelDtype, hashBlockSize,
        headDim, indexHeadDim, windowSize,
        numC4a, numC128a, numSwaOnly, numTotalDecoder, numTotalLayers, numLayerTuples,
        mlaPerToken, c128PerBsToken, swaPerHash, c4CompPerHash, c128CompPerHash,
        hbmBlockDataSize, hbmFormula,
        faFile, waFile, faFormula, waFormula,
        alignmentTokens, groups
    };
}

function calculateDSv4() {
    const modelName = document.getElementById('hybrid-model-select').value;
    const config = modelConfigs[modelName];
    if (!config || !config.is_dsv4) {
        displayError('Model Not Found', 'DeepSeek V4 configuration not found.');
        return;
    }
    const deployment = document.getElementById('hybrid-deployment').value;
    const isAscend = deployment === 'vllm-ascend';
    let bs = parseInt(document.getElementById('hybrid-base-block-size').value) || 256;
    const mtp = document.getElementById('hybrid-mtp').checked;
    const tokens = parseInt(document.getElementById('hybrid-token-input').value) || 4096;
    const batchSize = parseInt(document.getElementById('hybrid-batch-size').value) || 1;
    const chunkPrefill = parseInt(document.getElementById('hybrid-chunk-prefill').value) || 2048;

    if (isAscend && bs === 256) {
        displayError('Invalid Base Block Size', 'Ascend only supports base block size 32/64/128.');
        return;
    }
    if (tokens <= 0 || batchSize <= 0) {
        displayError('Invalid Input', 'Tokens and batch size must be positive.');
        return;
    }

    const p = deriveDSv4Params(config, bs, isAscend, mtp);

    // UCM dump (0-hit), TP-partitioned → ×1. FA is dumped every hash block in
    // both modes; WA differs:
    //   block-wise (default): WA dumped at every hash-block boundary
    //   chunk-wise:           WA dumped only at chunk-end boundaries (1 per chunk)
    const numHashBlocks = cdiv(tokens, p.hashBlockSize);
    const totalHashBlocks = batchSize * numHashBlocks;
    // chunk aligned to hash_block_size (WA is dumped at hash-block boundaries)
    const actualChunk = Math.max(p.hashBlockSize,
        Math.floor(chunkPrefill / p.hashBlockSize) * p.hashBlockSize);
    const numChunks = cdiv(tokens, actualChunk);
    const totalChunks = batchSize * numChunks;

    const blockwiseBytes = totalHashBlocks * (p.faFile + p.waFile);
    const chunkwiseBytes = totalHashBlocks * p.faFile + totalChunks * p.waFile;

    // HBM occupancy: FA full blocks (all have hash) + WA reachable blocks (reachable mask)
    let hashBlocksPerReq = 0;
    const groupHashDetails = [];
    for (const g of p.groups) {
        const numFull = Math.floor(tokens / g.lbs);
        let hashCount;
        if (g.isFA) {
            hashCount = numFull;
        } else {
            hashCount = reachableHashCount(tokens, g.lbs, g.sw, p.alignmentTokens);
        }
        hashBlocksPerReq += hashCount;
        groupHashDetails.push({name: g.name, lbs: g.lbs, numFull, hashCount, isFA: g.isFA, sw: g.sw});
    }
    const totalHBMHashBlocks = batchSize * hashBlocksPerReq;
    const hbmOccupancyBytes = totalHBMHashBlocks * p.hbmBlockDataSize;

    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="result-display" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${formatBytes(hbmOccupancyBytes)}</div>
                <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">KV Cache HBM Occupancy ${batchSize > 1 ? `(${batchSize} reqs)` : '(1 req)'}</div>
                <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">${totalHBMHashBlocks.toLocaleString()} hash blocks × ${p.hbmBlockDataSize.toLocaleString()} B = ${formatBytes(hbmOccupancyBytes)}</div>
            </div>
            <div style="text-align: center;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.3rem;">
                    <div>
                        <div class="result-value" style="font-size: 1.4rem; font-weight: 700; color: var(--accent-primary);">${formatBytes(blockwiseBytes)}</div>
                        <div class="result-label" style="font-size: 0.65rem; color: var(--accent-primary);">block-wise WA</div>
                        <div class="result-label" style="font-size: 0.6rem; color: var(--text-secondary);">WA per hash block</div>
                    </div>
                    <div>
                        <div class="result-value" style="font-size: 1.4rem; font-weight: 700; color: var(--accent-primary);">${formatBytes(chunkwiseBytes)}</div>
                        <div class="result-label" style="font-size: 0.65rem; color: var(--accent-primary);">chunk-wise WA</div>
                        <div class="result-label" style="font-size: 0.6rem; color: var(--text-secondary);">WA per chunk-end</div>
                    </div>
                </div>
                <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">KV Cache UCM Dump ${batchSize > 1 ? `(${batchSize} reqs)` : '(1 req)'} (TP ×1)</div>
            </div>
        </div>

        <!-- Sub-panel 1: Model Configuration -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>⚙️</span><span>Model Configuration</span></div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Model:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${modelName.split('/')[1] || modelName}</div>
                    <div style="color: var(--text-secondary);">Architecture:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">DS V4 (MLA indexer + SWA + Compressor)</div>
                    <div style="color: var(--text-secondary);">Layers (C4A/C128A/SWA-only):</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.numC4a} / ${p.numC128a} / ${p.numSwaOnly} (decoder ${p.numTotalDecoder})</div>
                    <div style="color: var(--text-secondary);">num_layer_tuples:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.numLayerTuples} (MTP ${p.mtp ? 'on' : 'off'})</div>
                    <div style="color: var(--text-secondary);">Base block size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.bs}</div>
                    <div style="color: var(--text-secondary);">hash_block_size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.hashBlockSize} tokens</div>
                    <div style="color: var(--text-secondary);">head_dim / index / window:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${p.headDim} / ${p.indexHeadDim} / ${p.windowSize}</div>
                    <div style="color: var(--text-secondary);">Platform:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${isAscend ? 'vllm-ascend (Ascend)' : 'vllm (CUDA)'}</div>
                </div>
            </div>
        </div>

        <!-- Sub-panel 2: HBM Block Data Size (per block_id) -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>💽</span><span>HBM Block Data Size (per block_id)</span></div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">Value:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.hbmBlockDataSize.toLocaleString()} B = ${formatBytes(p.hbmBlockDataSize)}</div>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-primary); background: rgba(81, 145, 238, 0.08); padding: 0.4rem; border-radius: 4px; margin-bottom: 0.3rem; font-family: monospace;">
                    ${isAscend ? `Ascend: ${p.hbmFormula} = ${p.hbmBlockDataSize.toLocaleString()}` : `CUDA block_stride = ${p.hbmFormula} = ${p.hbmBlockDataSize.toLocaleString()}`}
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); line-height: 1.4;">
                    One block_id's physical HBM footprint. HBM block count = single-card HBM ÷ this value.
                </div>
            </div>
        </div>

        <!-- Sub-panel 3: UCM Dump Block Data Size (per hash block) -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>📦</span><span>UCM Dump Block Data Size (per hash block)</span></div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">FA file_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.faFile.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">WA file_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.waFile.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">Total per hash block:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${(p.faFile + p.waFile).toLocaleString()} B = ${formatBytes(p.faFile + p.waFile)}</div>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-primary); background: rgba(81, 145, 238, 0.08); padding: 0.4rem; border-radius: 4px; margin-bottom: 0.3rem; font-family: monospace;">
                    FA = round_up(${p.faFormula}, 4096) = ${p.faFile.toLocaleString()}<br>
                    WA = round_up(${p.waFormula}, 4096) = ${p.waFile.toLocaleString()}
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); line-height: 1.4;">
                    hash_block_size = ${p.hashBlockSize} tokens (Ascend: bs×4; CUDA: 256 fixed). FA = MLA indexer (C4/C128), WA = SWA + Compressor tail.
                </div>
            </div>
        </div>

        <!-- Sub-panel 4: KV Cache HBM Occupancy -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>💾</span><span>KV Cache HBM Occupancy (single card, after prefill)</span></div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.25rem; margin-bottom: 0.4rem; font-weight: 600; font-size: 0.65rem; color: var(--text-secondary);">
                    <div>Group</div><div>Full blocks</div><div>Hash blocks (cached)</div>
                </div>
                ${groupHashDetails.map(g => `
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.25rem; font-size: 0.68rem;">
                    <div style="color: ${g.isFA ? 'var(--accent-success)' : 'var(--accent-warning)'};">${g.name}${g.isFA ? ' (FA)' : ' (WA)'}</div>
                    <div style="color: var(--text-primary);">⌊${tokens.toLocaleString()}/${g.lbs}⌋ = ${g.numFull}</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${g.hashCount}${!g.isFA && g.hashCount < g.numFull ? ` <span style="font-size: 0.6rem; color: var(--text-secondary);">(reachable: need=${cdiv(g.sw-1,g.lbs)}, per_seg=${Math.floor(p.alignmentTokens/g.lbs)})</span>` : ''}</div>
                </div>`).join('')}
                <div style="margin-top: 0.5rem; padding: 0.4rem; background: rgba(81, 145, 238, 0.08); border-radius: 4px; font-family: monospace; font-size: 0.68rem;">
                    Hash blocks / req: ${groupHashDetails.map(g => g.hashCount).join(' + ')} = ${hashBlocksPerReq}<br>
                    Total hash blocks: ${batchSize} × ${hashBlocksPerReq} = ${totalHBMHashBlocks.toLocaleString()}<br>
                    HBM occupancy: ${totalHBMHashBlocks.toLocaleString()} × ${p.hbmBlockDataSize.toLocaleString()} = ${hbmOccupancyBytes.toLocaleString()} B = ${formatBytes(hbmOccupancyBytes)}
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.3rem; line-height: 1.4;">
                    FA groups: all full blocks have hash. WA groups: only reachable blocks (segment tail <code>need</code> blocks) have hash; the rest are allocated but not cached (freed to free_deque head, immediate reuse). Single-card: hbm_block_data_size is per-block_id, shared across all groups.
                </div>
            </div>
        </div>

        <!-- Sub-panel 5: KV Cache UCM Dump -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>🧮</span><span>KV Cache UCM Dump (across all TP ranks)</span></div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <div style="color: var(--text-secondary);">Tokens / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${tokens.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">Batch size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize}</div>
                    <div style="color: var(--text-secondary);">hash blocks / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">⌈${tokens.toLocaleString()}/${p.hashBlockSize}⌉ = ${numHashBlocks}</div>
                    <div style="color: var(--text-secondary);">Total hash blocks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize} × ${numHashBlocks} = ${totalHashBlocks.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">chunks / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">⌈${tokens.toLocaleString()}/${actualChunk}⌉ = ${numChunks} (chunk prefill ${chunkPrefill} → aligned ${actualChunk})</div>
                    <div style="color: var(--text-secondary);">Total chunks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${batchSize} × ${numChunks} = ${totalChunks.toLocaleString()}</div>
                </div>
                <div style="margin-top: 0.5rem; padding: 0.4rem; background: rgba(81, 145, 238, 0.08); border-radius: 4px; font-family: monospace; font-size: 0.68rem;">
                    <strong style="color: var(--accent-primary);">block-wise (default, WA per hash block):</strong><br>
                    ${totalHashBlocks.toLocaleString()} × (${p.faFile.toLocaleString()} + ${p.waFile.toLocaleString()}) × 1 = ${blockwiseBytes.toLocaleString()} B = ${formatBytes(blockwiseBytes)}<br>
                    <strong style="color: var(--accent-primary);">chunk-wise (WA per chunk-end):</strong><br>
                    ${totalHashBlocks.toLocaleString()} × ${p.faFile.toLocaleString()} + ${totalChunks.toLocaleString()} × ${p.waFile.toLocaleString()} = ${chunkwiseBytes.toLocaleString()} B = ${formatBytes(chunkwiseBytes)}
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.3rem;">
                    FA dumped every hash block in both modes; WA differs (every hash block vs every chunk-end). TP-partitioned → ×1.
                </div>
            </div>
        </div>
    `;
}

// Maximum tokens for DeepSeek V4 on a single GPU.
// HBM holds: null(1) + FA resident blocks (whole request) + WA peak (one chunk).
//   Ascend: FA = G0(C4 MLA, tb=bs×4) + G1(C128 MLA, tb=bs×128); WA = G2/G3 SWA + G4/G5 comp
//   CUDA:   FA = G0(MLA, tb=bs);                       WA = G1/G2 SWA + G3/G4 comp
function calculateDSv4MaxTokens() {
    const modelName = document.getElementById('hybrid-model-select').value;
    const config = modelConfigs[modelName];
    if (!config || !config.is_dsv4) {
        displayError('Model Not Found', 'DeepSeek V4 configuration not found.');
        return;
    }
    const deployment = document.getElementById('hybrid-deployment').value;
    const isAscend = deployment === 'vllm-ascend';
    let bs = parseInt(document.getElementById('hybrid-base-block-size').value) || 256;
    const mtp = document.getElementById('hybrid-mtp').checked;
    const gpuMemoryGiB = parseFloat(document.getElementById('hybrid-gpu-memory-input').value) || 42;
    const chunkPrefill = parseInt(document.getElementById('hybrid-chunk-prefill').value) || 2048;

    if (isAscend && bs === 256) {
        displayError('Invalid vLLM Block Size', 'Ascend only supports 32/64/128.');
        return;
    }
    if (gpuMemoryGiB <= 0) {
        displayError('Invalid Input', 'GPU memory must be positive.');
        return;
    }

    const p = deriveDSv4Params(config, bs, isAscend, mtp);
    const totalMemoryBytes = gpuMemoryGiB * Math.pow(1024, 3);
    const hbmBlocks = Math.floor(totalMemoryBytes / p.hbmBlockDataSize);

    // chunk aligned to base block size (bs is a multiple of every group's block size)
    const actualChunk = Math.max(bs, Math.floor(chunkPrefill / bs) * bs);

    // WA peak = one chunk's worth of all window groups
    let waPeak, waBreakdown, residentTbs, faLabel;
    if (isAscend) {
        const g2g3 = 2 * cdiv(actualChunk, bs);                 // G2/G3 SWA, tb=bs
        const g4 = cdiv(actualChunk, bs / 16);                  // G4 C4 comp, tb=bs/16
        const g5 = cdiv(actualChunk, bs / 4);                   // G5 C128 comp, tb=bs/4
        waPeak = g2g3 + g4 + g5;
        waBreakdown = `G2/G3 SWA: 2×⌈${actualChunk}/${bs}⌉=${g2g3}, G4 C4comp: ⌈${actualChunk}/${bs / 16}⌉=${g4}, G5 C128comp: ⌈${actualChunk}/${bs / 4}⌉=${g5}`;
        residentTbs = [bs * 4, bs * 128];                      // G0, G1
        faLabel = 'G0 (C4 MLA, tb=' + (bs * 4) + ') + G1 (C128 MLA, tb=' + (bs * 128) + ')';
    } else {
        const g1g2 = 2 * cdiv(actualChunk, 64);                 // G1/G2 SWA, tb=64
        const g3 = cdiv(actualChunk, 4);                       // G3 C4 comp, tb=4
        const g4 = cdiv(actualChunk, 8);                       // G4 C128 comp, tb=8
        waPeak = g1g2 + g3 + g4;
        waBreakdown = `G1/G2 SWA: 2×⌈${actualChunk}/64⌉=${g1g2}, G3 C4comp: ⌈${actualChunk}/4⌉=${g3}, G4 C128comp: ⌈${actualChunk}/8⌉=${g4}`;
        residentTbs = [bs];                                      // G0 MLA
        faLabel = 'G0 (MLA, tb=' + bs + ')';
    }

    const nullBlock = 1;
    const R = hbmBlocks - nullBlock - waPeak;                  // blocks available for FA resident

    // Max seq with Σ ceil(seq/tb) ≤ R (binary search; bs is min token_block multiple)
    let maxTokens = 0;
    if (R >= 1) {
        const minTb = Math.min(...residentTbs);
        let lo = 0, hi = (R + 1) * minTb;                       // safe upper bound
        while (lo < hi) {
            const mid = Math.ceil((lo + hi + 1) / 2);
            const resident = residentTbs.reduce((s, tb) => s + cdiv(mid, tb), 0);
            if (resident <= R) lo = mid; else hi = mid - 1;
        }
        maxTokens = lo;
    }

    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${maxTokens.toLocaleString()}</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Max tokens for one request on a single GPU</div>
            <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">FA resident ≤ ${R < 0 ? 0 : R} blocks (HBM ${hbmBlocks.toLocaleString()} − null 1 − WA peak ${waPeak})</div>
        </div>

        <!-- Sub-panel 1: Model Configuration -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>⚙️</span><span>Model Configuration</span></div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Model:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${modelName.split('/')[1] || modelName}</div>
                    <div style="color: var(--text-secondary);">vLLM block size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${bs} (MTP ${mtp ? 'on' : 'off'})</div>
                    <div style="color: var(--text-secondary);">GPU memory:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${gpuMemoryGiB} GiB</div>
                    <div style="color: var(--text-secondary);">chunk prefill:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${chunkPrefill} → aligned ${actualChunk}</div>
                    <div style="color: var(--text-secondary);">Platform:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${isAscend ? 'vllm-ascend (Ascend)' : 'vllm (CUDA)'}</div>
                </div>
            </div>
        </div>

        <!-- Sub-panel 2: HBM Block Data Size -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>💽</span><span>HBM Block Data Size (per block_id)</span></div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <div style="color: var(--text-secondary);">block_data_size:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${p.hbmBlockDataSize.toLocaleString()} B = ${formatBytes(p.hbmBlockDataSize)}</div>
                    <div style="color: var(--text-secondary);">HBM blocks (pool):</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${hbmBlocks.toLocaleString()}</div>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.3rem;">${isAscend ? 'Ascend' : 'CUDA'}: ${p.hbmFormula}</div>
            </div>
        </div>

        <!-- Sub-panel 3: Max Tokens Calculation -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header"><span>🔢</span><span>Max Tokens Calculation</span></div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div class="formula-breakdown">
                    <div class="formula-step"><span class="formula-step-label">HBM blocks:</span><span class="formula-step-value">${hbmBlocks.toLocaleString()}</span></div>
                    <div class="formula-step"><span class="formula-step-label">− null block:</span><span class="formula-step-value">1</span></div>
                    <div class="formula-step"><span class="formula-step-label">− WA peak (one chunk ${actualChunk}):</span><span class="formula-step-value">${waPeak.toLocaleString()}</span></div>
                    <div class="formula-step"><span class="formula-step-label">= FA resident budget R:</span><span class="formula-step-value">${R < 0 ? 0 : R}</span></div>
                </div>
                <div style="margin-top: 0.5rem; padding: 0.4rem; background: rgba(81, 145, 238, 0.08); border-radius: 4px; font-family: monospace; font-size: 0.68rem;">
                    FA resident (${faLabel}): Σ⌈seq/tb⌉ ≤ ${R < 0 ? 0 : R}<br>
                    WA peak: ${waBreakdown}
                </div>
                ${R < 0 ? `<div style="margin-top: 0.4rem; color: var(--accent-error); font-size: 0.7rem;">⚠ HBM too small — cannot fit even one chunk's WA blocks.</div>` : ''}
                <div style="margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--border-color);">
                    <strong style="color: var(--accent-success);">Max tokens:</strong> ${maxTokens.toLocaleString()}
                </div>
            </div>
        </div>
    `;
}

function calculateHybridMaxTokens() {
    clearResults();

    const modelName = document.getElementById('hybrid-model-select').value;

    if (modelConfigs[modelName] && modelConfigs[modelName].is_linear_hybrid) {
        calculateLinearMaxTokens();
        return;
    }
    if (modelConfigs[modelName] && modelConfigs[modelName].is_dsv4) {
        calculateDSv4MaxTokens();
        return;
    }

    displayError('Model Not Found', 'Unknown hybrid model.');
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
        <div class="result-display" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${formatBytes(result.hbmOccupancyBytes)}</div>
                <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">KV Cache HBM Occupancy ${result.batchSize > 1 ? `(${result.batchSize} reqs)` : '(1 req)'}</div>
                <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">${result.totalHashBlocks.toLocaleString()} hash blocks × ${result.blockDataSize.toLocaleString()} B = ${formatBytes(result.hbmOccupancyBytes)}</div>
            </div>
            <div style="text-align: center;">
                <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-primary);">${formatBytes(result.totalBytes)}</div>
                <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">KV Cache UCM Dump ${result.batchSize > 1 ? `(${result.batchSize} reqs)` : '(1 req)'}</div>
                <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">${result.totalBlocks.toLocaleString()} blocks × ${result.blockDataSize.toLocaleString()} B × ${result.tpFactor} = ${formatBytes(result.totalBytes)}</div>
            </div>
        </div>

        ${result.showHybridWarning ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-warning); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: var(--accent-warning); font-size: 0.85rem;">Hybrid Model Warning</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                This appears to be a Hybrid model. The calculation uses a GQA-family fallback and may not be accurate. For accurate results, please use the Hybrid Models tab.
            </div>
        </div>
        ` : ''}

        <!-- Sub-panel 1: Model Configuration -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Model Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Model:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${getModelDisplayName(result.modelName)}</div>
                    <div style="color: var(--text-secondary);">Architecture:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.archLabel}</div>
                    <div style="color: var(--text-secondary);">Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_hidden_layers}</div>
                    <div style="color: var(--text-secondary);">Hidden Size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.hidden_size}</div>
                    <div style="color: var(--text-secondary);">Attn Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_attention_heads}</div>
                    <div style="color: var(--text-secondary);">KV Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${kvHeads}</div>
                    <div style="color: var(--text-secondary);">Head Dim:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.hdim}</div>
                    ${config.kv_lora_rank ? '<div style="color: var(--text-secondary);">KV LoRA Rank:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.kv_lora_rank + '</div>' : ''}
                    ${config.qk_rope_head_dim ? '<div style="color: var(--text-secondary);">QK RoPE Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.qk_rope_head_dim + '</div>' : ''}
                    ${config.index_head_dim ? '<div style="color: var(--text-secondary);">Index Head Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.index_head_dim + '</div>' : ''}
                    <div style="color: var(--text-secondary);">DType:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.dtype} (${result.dtypeSize} B)</div>
                </div>
            </div>
        </div>

        <!-- Sub-panel 2: Block Data Size -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📦</span>
                <span>Block Data Size</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">Value:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${result.blockDataSize.toLocaleString()} B (= ${(result.blockDataSize / 1024).toFixed(2)} KiB)</div>
                    <div style="color: var(--text-secondary);">Rank-shared:</div>
                    <div style="color: var(--text-primary);">${result.rankShared ? 'yes (latent replicated, UCM dumps TP0 only)' : 'no (KV sharded by head, divided by TP)'}</div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-primary); background: rgba(81, 145, 238, 0.08); padding: 0.4rem; border-radius: 4px; margin-bottom: 0.4rem; font-family: monospace;">
                    ${result.blockFormula}
                </div>
                <div style="font-size: 0.7rem; color: var(--text-secondary); line-height: 1.4;">
                    Single-TP-rank, one block's physical bytes in HBM. For standard models this equals the UCM dump file's block-data-size (vLLM allocates N blocks → UCM dumps N blocks).
                </div>
            </div>
        </div>

        <!-- Sub-panel 3: KV Cache HBM Occupancy -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>💾</span>
                <span>KV Cache HBM Occupancy (single card, after prefill)</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">Hash blocks / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">⌊${result.tokens.toLocaleString()} / ${result.blockSize}⌋ = ${result.hashBlocksPerRequest} <span style="font-size: 0.65rem; color: var(--text-secondary);">(full blocks only, partial excluded)</span></div>
                    <div style="color: var(--text-secondary);">Total hash blocks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.batchSize} × ${result.hashBlocksPerRequest} = ${result.totalHashBlocks.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">× block_data_size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.blockDataSize.toLocaleString()} B</div>
                </div>
                <div style="margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--border-color);">
                    <strong style="color: var(--accent-success);">HBM occupancy:</strong> ${result.totalHashBlocks.toLocaleString()} × ${result.blockDataSize.toLocaleString()} = ${result.hbmOccupancyBytes.toLocaleString()} B = ${formatBytes(result.hbmOccupancyBytes)}
                </div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.3rem; line-height: 1.4;">
                    Only blocks with hash (cached / prefix-cache-able) are counted. Blocks without hash (partial / unfilled) are freed to free_deque head and immediately reused — they don't persist. Single-card: block_data_size already accounts for TP (GQA ÷ TP, MLA rank-shared).
                </div>
            </div>
        </div>

        <!-- Sub-panel 4: KV Cache UCM Dump -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>🧮</span>
                <span>KV Cache UCM Dump (across all TP ranks)</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <div style="color: var(--text-secondary);">Tokens / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.tokens.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">Batch size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.batchSize}</div>
                    <div style="color: var(--text-secondary);">vLLM block size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.blockSize}</div>
                    <div style="color: var(--text-secondary);">Blocks / request:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">⌈${result.tokens.toLocaleString()} / ${result.blockSize}⌉ = ${result.blocksPerRequest}</div>
                    <div style="color: var(--text-secondary);">Total blocks:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.batchSize} × ${result.blocksPerRequest} = ${result.totalBlocks.toLocaleString()}</div>
                    <div style="color: var(--text-secondary);">× block_data_size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.blockDataSize.toLocaleString()} B</div>
                    <div style="color: var(--text-secondary);">× TP factor:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.tpFactor} ${result.rankShared ? '(rank-shared → TP0 only)' : '(GQA → all TP ranks)'}</div>
                </div>
                <div style="margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--border-color);">
                    <strong style="color: var(--accent-primary);">Total:</strong> ${result.totalBlocks.toLocaleString()} × ${result.blockDataSize.toLocaleString()} × ${result.tpFactor} = ${result.totalBytes.toLocaleString()} B = ${formatBytes(result.totalBytes)}
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
            'This appears to be a Hybrid model. The max tokens calculation uses a GQA-family fallback and may not be accurate. For Hybrid models, please use the Hybrid Models tab.');
    }

    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${result.maxTokens.toLocaleString()}</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Max tokens for one request on a single GPU ${result.tp > 1 ? '(TP=' + result.tp + ')' : ''}</div>
            <div class="result-label" style="font-size: 0.7rem; color: var(--text-secondary);">= ${result.maxBlocks.toLocaleString()} blocks × ${result.blockSize} tokens/block</div>
        </div>

        ${result.isHybridModel ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-warning); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: var(--accent-warning); font-size: 0.85rem;">Hybrid Model Warning</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                This appears to be a Hybrid model. The max tokens calculation uses a GQA-family fallback and may not be accurate. Please use the Hybrid Models tab for accurate results.
            </div>
        </div>
        ` : ''}

        <!-- Sub-panel 1: Model Configuration -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>⚙️</span>
                <span>Model Configuration</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem; font-family: inherit;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem;">
                    <div style="color: var(--text-secondary);">Model:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${getModelDisplayName(result.modelName)}</div>
                    <div style="color: var(--text-secondary);">Architecture:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.archLabel}</div>
                    <div style="color: var(--text-secondary);">Layers:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.num_hidden_layers}</div>
                    <div style="color: var(--text-secondary);">Hidden Size:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${config.hidden_size}</div>
                    <div style="color: var(--text-secondary);">KV Heads:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${kvHeads}</div>
                    ${config.kv_lora_rank ? '<div style="color: var(--text-secondary);">KV LoRA Rank:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.kv_lora_rank + '</div>' : ''}
                    ${config.qk_rope_head_dim ? '<div style="color: var(--text-secondary);">QK RoPE Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.qk_rope_head_dim + '</div>' : ''}
                    ${config.index_head_dim ? '<div style="color: var(--text-secondary);">Index Head Dim:</div><div style="color: var(--text-primary); font-weight: 500;">' + config.index_head_dim + '</div>' : ''}
                    <div style="color: var(--text-secondary);">DType:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.dtype} (${result.dtypeSize} B)</div>
                    <div style="color: var(--text-secondary);">GPU Memory:</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${result.gpuMemoryGiB} GiB</div>
                </div>
            </div>
        </div>

        <!-- Sub-panel 2: Block Data Size -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📦</span>
                <span>Block Data Size</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--text-secondary);">Value:</div>
                    <div style="color: var(--text-primary); font-weight: 600;">${result.blockDataSize.toLocaleString()} B (= ${(result.blockDataSize / 1024).toFixed(2)} KiB)</div>
                    <div style="color: var(--text-secondary);">Rank-shared:</div>
                    <div style="color: var(--text-primary);">${result.rankShared ? 'yes (latent replicated, UCM dumps TP0 only)' : 'no (KV sharded by head, divided by TP)'}</div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-primary); background: rgba(81, 145, 238, 0.08); padding: 0.4rem; border-radius: 4px; font-family: monospace;">
                    ${result.blockFormula}
                </div>
            </div>
        </div>

        <!-- Sub-panel 3: Max Tokens Calculation -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>🔢</span>
                <span>Max Tokens Calculation</span>
            </div>
            <div class="formula-content" style="font-size: 0.7rem;">
                <div class="formula-breakdown">
                    <div class="formula-step">
                        <span class="formula-step-label">GPU memory:</span>
                        <span class="formula-step-value">${(result.gpuMemoryGiB * 1024).toFixed(0)} MiB</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">÷ block_data_size:</span>
                        <span class="formula-step-value">${result.blockDataSize.toLocaleString()} B</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">= max blocks:</span>
                        <span class="formula-step-value">${result.maxBlocks.toLocaleString()}</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">× block_size:</span>
                        <span class="formula-step-value">${result.blockSize}</span>
                    </div>
                    <div class="formula-step">
                        <span class="formula-step-label">= max tokens:</span>
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