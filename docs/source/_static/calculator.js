/**
 * KV Cache Calculator - Core Calculation Logic & UI
 * 
 * This file contains:
 * 1. Global state management
 * 2. Model source & config loading
 * 3. KV cache calculation (performCalculation)
 * 4. Max tokens calculation
 * 5. Display functions
 * 6. Toast notification system
 * 7. Event listeners
 */

// Global state
let currentLanguage = 'en'; // Always English
let modelConfigs = {};
let currentModelSource = 'preset';

// ============================================================
// Helper Functions
// ============================================================

/**
 * Extract display name from model identifier or URL
 */
function getModelDisplayName(modelName) {
    // If it's a URL, extract the model identifier
    if (modelName.startsWith('http://') || modelName.startsWith('https://')) {
        try {
            const urlObj = new URL(modelName);
            const pathParts = urlObj.pathname.split('/').filter(part => part);

            // Handle ModelScope URLs: /models/organization/model
            if (urlObj.hostname.includes('modelscope.cn') && pathParts[0] === 'models') {
                if (pathParts.length >= 3) {
                    return pathParts.slice(1, 3).join('/');
                }
            }
            // Handle HuggingFace URLs: /organization/model
            else if (urlObj.hostname.includes('huggingface.co')) {
                // Filter out 'models' if present
                const modelPathParts = pathParts.filter(part =>
                    !['tree', 'blob', 'raw', 'commit', 'discussions', 'issues', 'pull', 'models'].includes(part)
                );
                if (modelPathParts.length >= 2) {
                    return modelPathParts.slice(0, 2).join('/');
                }
            }
        } catch (e) {
            console.warn('Failed to parse model URL:', e);
        }
    }

    // If it's already a simple identifier (org/model), return as-is
    // Otherwise, just return the last part
    if (modelName.includes('/')) {
        const parts = modelName.split('/');
        // If it looks like org/model format, return both parts
        if (parts.length >= 2) {
            return parts.slice(0, 2).join('/');
        }
    }

    return modelName;
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
    console.log('Setting model source to:', source);

    // Update selector state
    const presetOption = document.getElementById('preset-option');
    const customOption = document.getElementById('custom-option');

    // Reset all options
    presetOption.classList.remove('active');
    customOption.classList.remove('active');

    // Hide all sections
    document.getElementById('preset-model-section').classList.add('hidden');
    document.getElementById('custom-model-section').classList.add('hidden');

    // Activate selected option and show corresponding section
    if (source === 'custom') {
        customOption.classList.add('active');
        document.getElementById('custom-model-section').classList.remove('hidden');
    } else { // preset
        presetOption.classList.add('active');
        document.getElementById('preset-model-section').classList.remove('hidden');
        // Repopulate with preset models
        populateModelDropdown();
    }
}

// ============================================================
// Model Configuration Loading
// ============================================================

function loadModelConfigs() {
    // Use embedded model configurations (defined in model-configs.js)
    modelConfigs = getEmbeddedModelConfigs();
    console.log('Model configurations loaded:', Object.keys(modelConfigs).length, 'models');
    populateModelDropdown();
}

function populateModelDropdown() {
    const presetModelSelect = document.getElementById('preset-model-select');
    presetModelSelect.innerHTML = '';

    const sortedModelNames = Object.keys(modelConfigs).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

    console.log('Populating preset model dropdown:', sortedModelNames);

    sortedModelNames.forEach(modelName => {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        presetModelSelect.appendChild(option);
    });

    // Select the first model by default
    if (sortedModelNames.length > 0) {
        presetModelSelect.value = sortedModelNames[0];
    }
}

// ============================================================
// Fetch Model Configuration from URL
// ============================================================

async function fetchModelConfigFromUrl(url) {
    try {
        // Normalize URL: remove trailing slash, 'files', 'tree/main', etc.
        let normalizedUrl = url.trim();

        // Remove trailing slashes
        normalizedUrl = normalizedUrl.replace(/\/+$/, '');

        // Remove common suffixes that aren't part of model name
        normalizedUrl = normalizedUrl.replace(/\/(files|tree\/main|blob\/main|raw\/main|commits|issues|discussions).*$/, '');

        // Parse URL to determine platform and extract model identifier
        const urlObj = new URL(normalizedUrl);
        let modelIdentifier;
        let platform = '';

        if (urlObj.hostname.includes('huggingface.co')) {
            platform = 'huggingface';
            // Extract model path from Hugging Face URL
            // Expected format: https://huggingface.co/organization/model
            const pathParts = urlObj.pathname.split('/').filter(part => part && part !== 'models');

            // Filter out non-model paths
            const modelPathParts = pathParts.filter(part =>
                !['tree', 'blob', 'raw', 'commit', 'discussions', 'issues', 'pull', 'blob'].includes(part)
            );

            if (modelPathParts.length >= 2) {
                modelIdentifier = modelPathParts.slice(0, 2).join('/');
            }
        } else if (urlObj.hostname.includes('modelscope.cn')) {
            platform = 'modelscope';
            // Extract model path from ModelScope URL
            // Expected format: https://www.modelscope.cn/models/organization/model
            const pathParts = urlObj.pathname.split('/').filter(part => part);

            if (pathParts.length >= 3 && pathParts[0] === 'models') {
                // Extract organization/model from /models/organization/model
                modelIdentifier = pathParts.slice(1, 3).join('/');
            }
        }

        if (!modelIdentifier) {
            throw new Error('Could not extract model identifier from URL. Please check the URL format.');
        }

        console.log(`Fetching config for ${platform} model: ${modelIdentifier}`);

        // Store modelIdentifier for later use
        const fetchedModelIdentifier = modelIdentifier;

        // Try to fetch from online APIs with CORS proxy fallbacks
        let configData = null;

        // Try direct fetch first (might work in some environments)
        try {
            if (platform === 'huggingface') {
                const apiUrl = `https://huggingface.co/${modelIdentifier}/raw/main/config.json`;
                console.log('Trying HuggingFace API:', apiUrl);
                const response = await fetch(apiUrl);
                if (response.ok) {
                    configData = await response.json();
                    console.log('Successfully fetched from HuggingFace');
                }
            } else if (platform === 'modelscope') {
                // Try multiple ModelScope endpoints
                const modelScopeEndpoints = [
                    // Method 1: Direct raw file (most reliable)
                    `https://modelscope.cn/api/v1/models/${modelIdentifier}/repo?Revision=master&FilePath=config.json`,
                    // Method 2: Alternative raw endpoint
                    `https://modelscope.cn/${modelIdentifier}/raw/master/config.json`,
                    // Method 3: Using www subdomain
                    `https://www.modelscope.cn/api/v1/models/${modelIdentifier}/repo?Revision=master&FilePath=config.json`
                ];

                for (const apiUrl of modelScopeEndpoints) {
                    console.log('Trying ModelScope endpoint:', apiUrl);
                    try {
                        const response = await fetch(apiUrl);
                        console.log('ModelScope response status:', response.status, 'type:', response.headers.get('content-type'));
                        if (response.ok) {
                            // Try to parse as JSON first
                            const contentType = response.headers.get('content-type');
                            let data;

                            if (contentType && contentType.includes('application/json')) {
                                data = await response.json();
                                console.log('ModelScope API JSON response:', data);

                                // ModelScope API returns data in different formats:
                                // 1. API endpoint: { Data: { Content: "base64-encoded-json" } }
                                // 2. Alternative: { data: { Content: "base64-encoded-json" } }
                                // 3. Direct: { config fields directly }
                                let rawContent = data.Data || data.data || data;

                                // Check if Content field exists and is base64 encoded
                                if (rawContent && rawContent.Content) {
                                    try {
                                        // Decode base64 and parse JSON
                                        const decodedContent = atob(rawContent.Content);
                                        configData = JSON.parse(decodedContent);
                                        console.log('Successfully decoded base64 content from ModelScope');
                                    } catch (decodeError) {
                                        console.warn('Failed to decode base64 content:', decodeError.message);
                                        // Try using Content directly as JSON
                                        try {
                                            configData = JSON.parse(rawContent.Content);
                                        } catch (e) {
                                            // Use as-is
                                            configData = rawContent.Content;
                                        }
                                    }
                                } else if (typeof rawContent === 'object') {
                                    // Direct JSON config
                                    configData = rawContent;
                                }

                                if (configData && (configData.hidden_size || configData.num_attention_heads)) {
                                    console.log('Successfully fetched from ModelScope, config keys:', Object.keys(configData));
                                    break;
                                }
                            } else {
                                // Try to get text response
                                const textData = await response.text();
                                console.log('ModelScope text response (first 200 chars):', textData.substring(0, 200));
                                try {
                                    configData = JSON.parse(textData);
                                    if (configData && (configData.hidden_size || configData.num_attention_heads)) {
                                        console.log('Successfully parsed text response as JSON');
                                        break;
                                    }
                                } catch (parseError) {
                                    console.warn('Failed to parse text response as JSON:', parseError.message);
                                }
                            }
                        }
                    } catch (endpointError) {
                        console.warn('Endpoint failed:', endpointError.message);
                        continue;
                    }
                }
            }
        } catch (directError) {
            console.warn('Direct fetch failed, trying fallback methods:', directError.message);
        }

        // If direct fetch failed, try multiple CORS proxies
        if (!configData) {
            // List of CORS proxies to try
            const corsProxies = [
                { name: 'corsproxy.io', url: 'https://corsproxy.io/?' },
                { name: 'allorigins', url: 'https://api.allorigins.win/raw?url=' },
                { name: 'cors-anywhere-temp', url: 'https://cors-anywhere.herokuapp.com/' },
                { name: 'thingproxy', url: 'https://thingproxy.freeboard.io/fetch/' }
            ];

            for (const proxy of corsProxies) {
                try {
                    console.log(`Trying CORS proxy: ${proxy.name}`);
                    let targetUrl;
                    if (platform === 'huggingface') {
                        targetUrl = `https://huggingface.co/${modelIdentifier}/raw/main/config.json`;
                    } else if (platform === 'modelscope') {
                        // Try with www subdomain
                        targetUrl = `https://www.modelscope.cn/api/v1/models/${modelIdentifier}/repo?Revision=master&FilePath=config.json`;
                    }

                    if (targetUrl) {
                        let proxyUrl;
                        if (proxy.name === 'cors-anywhere-temp') {
                            // cors-anywhere requires temporary access request
                            proxyUrl = proxy.url + targetUrl;
                        } else {
                            proxyUrl = proxy.url + encodeURIComponent(targetUrl);
                        }

                        console.log(`Proxy URL: ${proxy.name}`, proxyUrl.substring(0, 100) + '...');
                        const response = await fetch(proxyUrl);

                        if (response.ok) {
                            const data = await response.json();
                            console.log(`CORS proxy ${proxy.name} response data:`, data);

                            // Handle different response formats
                            if (platform === 'modelscope') {
                                let rawContent = data.Data || data.data || data;

                                // Check if Content field exists and is base64 encoded
                                if (rawContent && rawContent.Content) {
                                    try {
                                        // Decode base64 and parse JSON
                                        const decodedContent = atob(rawContent.Content);
                                        configData = JSON.parse(decodedContent);
                                        console.log(`Successfully decoded base64 content via CORS proxy: ${proxy.name}`);
                                    } catch (decodeError) {
                                        console.warn(`Failed to decode base64 content via ${proxy.name}:`, decodeError.message);
                                        // Try using Content directly as JSON
                                        try {
                                            configData = JSON.parse(rawContent.Content);
                                        } catch (e) {
                                            // Use as-is
                                            configData = rawContent.Content;
                                        }
                                    }
                                } else if (typeof rawContent === 'object') {
                                    // Direct JSON config
                                    configData = rawContent;
                                }
                            } else {
                                // HuggingFace or other platforms
                                configData = data;
                            }

                            // Validate config data
                            if (configData && (configData.hidden_size || configData.num_attention_heads)) {
                                console.log(`Successfully fetched via CORS proxy: ${proxy.name}, config keys:`, Object.keys(configData));
                                break;
                            } else {
                                console.warn(`CORS proxy ${proxy.name} returned invalid config data`);
                                configData = null;
                            }
                        } else {
                            console.warn(`CORS proxy ${proxy.name} returned status:`, response.status);
                        }
                    }
                } catch (proxyError) {
                    console.warn(`CORS proxy ${proxy.name} failed:`, proxyError.message);
                    continue;
                }
            }
        }

        // If all online methods fail, check if we have this model in our local configs
        if (!configData && modelConfigs[modelIdentifier]) {
            console.log('Using local configuration for model:', modelIdentifier);
            const localConfig = modelConfigs[modelIdentifier];
            // Add _modelName if not already present
            if (!localConfig._modelName) {
                localConfig._modelName = modelIdentifier;
            }
            return localConfig;
        }

        // If still no config, throw error with helpful message
        if (!configData) {
            if (platform === 'modelscope') {
                throw new Error(`ModelScope API is blocked by CORS policy.\n\nAll public CORS proxies (corsproxy.io, allorigins, etc.) are blocked by ModelScope.\n\nTo use ModelScope models:\n1. Restart Chrome with: chrome.exe --disable-web-security --user-data-dir="C:/temp"\n2. Or use HuggingFace models (recommended)\n3. Or use preset models from the dropdown`);
            } else {
                throw new Error(`Unable to fetch configuration for model "${modelIdentifier}". Please:\n1. Check if the model exists on ${platform}\n2. Verify the URL is correct\n3. Try a different model or use manual configuration`);
            }
        }

        // Transform to our format
        // For multimodal models, check text_config first
        const sourceConfig = configData.text_config || configData;

        const transformedConfig = {
            hidden_size: sourceConfig.hidden_size,
            num_attention_heads: sourceConfig.num_attention_heads,
            num_hidden_layers: sourceConfig.num_hidden_layers,
            num_key_value_heads: sourceConfig.num_key_value_heads,
            kv_lora_rank: sourceConfig.kv_lora_rank,
            qk_rope_head_dim: sourceConfig.qk_rope_head_dim,
            head_dim: sourceConfig.head_dim,
            sliding_window: sourceConfig.sliding_window || sourceConfig.sliding_window_size,
            attention_layer_count: sourceConfig.attention_layer_count,
            layer_types: sourceConfig.layer_types,
            hybrid_layer_pattern: sourceConfig.hybrid_layer_pattern,
            _modelName: fetchedModelIdentifier  // Store model identifier
        };

        // Filter out undefined values (but keep _modelName)
        Object.keys(transformedConfig).forEach(key => {
            if (key !== '_modelName' && transformedConfig[key] === undefined) {
                delete transformedConfig[key];
            }
        });

        console.log('Transformed config:', transformedConfig);

        return transformedConfig;

    } catch (error) {
        console.error('Error fetching model config:', error);
        throw error;
    }
}

// ============================================================
// Detect Model Architecture Type
// ============================================================

/**
 * Detect architecture type from model config.
 * Returns: { isMLAModel, isHybrid, isGQAWithHeadDim, hybridSubType, attentionLayerCount, sliding_window }
 */
function detectArchitectureType(config) {
    // MLA (Multi-head Latent Attention): has kv_lora_rank and qk_rope_head_dim
    const isMLAModel = config.kv_lora_rank && config.qk_rope_head_dim;

    const layerTypes = config.layer_types;
    const hybridLayerPattern = config.hybrid_layer_pattern;
    const hasSlidingWindow = config.sliding_window || config.sliding_window_size;

    // Check if layer_types contains multiple distinct attention types
    let hasHybridLayerTypes = false;
    let hybridSubType = '';
    // BUG FIX: use config.num_hidden_layers instead of the yet-unassigned local variable
    let attentionLayerCount = config.num_hidden_layers; // default: all layers

    if (layerTypes && Array.isArray(layerTypes) && layerTypes.length > 0) {
        const uniqueTypes = [...new Set(layerTypes)];
        if (uniqueTypes.length > 1) {
            hasHybridLayerTypes = true;
            // Determine sub-type
            if (uniqueTypes.includes('linear_attention')) {
                hybridSubType = 'Linear + Full Attention';
                // Count full_attention layers for KV calculation
                attentionLayerCount = layerTypes.filter(t => t === 'full_attention').length;
            } else if (uniqueTypes.includes('sliding_attention')) {
                hybridSubType = 'Sliding + Full Attention';
                // All layers have KV cache, but sliding layers use window
                attentionLayerCount = config.num_hidden_layers;
            }
        }
    }

    // Check hybrid_layer_pattern (0=SSM/no-attn, 1=Attention)
    let hasHybridPattern = false;
    if (hybridLayerPattern && Array.isArray(hybridLayerPattern) && hybridLayerPattern.length > 0) {
        const uniqueVals = [...new Set(hybridLayerPattern)];
        if (uniqueVals.length > 1) {
            hasHybridPattern = true;
            if (!hybridSubType) hybridSubType = 'Attention + SSM';
            attentionLayerCount = hybridLayerPattern.filter(v => v === 1).length;
        }
    }

    // Combined hybrid flag
    const isHybrid = hasHybridLayerTypes || hasHybridPattern || (config.is_hybrid || false);

    // If is_hybrid flag but no detailed layer info, use all layers
    if (config.is_hybrid && !hasHybridLayerTypes && !hasHybridPattern) {
        hybridSubType = 'Linear + Full Attention';
        attentionLayerCount = config.num_hidden_layers;
    }

    // GQA with explicit head_dim (but NOT MLA, NOT Hybrid)
    const isGQAWithHeadDim = config.head_dim && !isMLAModel && !isHybrid;

    return {
        isMLAModel,
        isHybrid,
        isGQAWithHeadDim,
        hybridSubType,
        attentionLayerCount,
        sliding_window: hasSlidingWindow || null
    };
}

// ============================================================
// Calculate KV Cache Size
// ============================================================

async function calculateKVCache() {
    // Clear previous results before starting new calculation
    const resultsContainer = document.getElementById('results-container');
    const detailsContainer = document.getElementById('calculation-details');
    const stepsContainer = document.getElementById('calculation-steps');

    if (resultsContainer) resultsContainer.innerHTML = '';
    if (detailsContainer) detailsContainer.innerHTML = '';
    if (stepsContainer) stepsContainer.innerHTML = '';

    // Get and validate token input
    const tokenInput = document.getElementById('token-input').value.trim();
    const tokens = parseInt(tokenInput);
    const dtype = document.getElementById('dtype-select').value;

    // Validate input
    if (!tokenInput) {
        displayError('Invalid Input', 'Please enter the number of tokens.');
        return;
    }

    if (isNaN(tokens) || tokens <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for tokens.');
        return;
    }

    if (tokens > 1000000) {
        console.warn('Large token count detected, calculation may take some time');
    }

    let config;
    let modelName;
    let hasError = false;

    // Show loading state
    const calculateBtn = document.querySelector('button[onclick="calculateKVCache()"]');
    const originalText = calculateBtn.innerHTML;
    calculateBtn.innerHTML = '<span>⏳</span> <span>Calculating...</span>';
    calculateBtn.disabled = true;

    try {
        console.log('Current model source:', currentModelSource);

        if (currentModelSource === 'preset') {
            const presetSelect = document.getElementById('preset-model-select');
            modelName = presetSelect.value;
            console.log('Selected preset model:', modelName);
            if (!modelName || !modelConfigs[modelName]) {
                displayError('Model Not Found', 'The selected preset model configuration is not available. Please select another model.');
                hasError = true;
                throw new Error('Model not found');
            }
            config = modelConfigs[modelName];
            console.log('Using preset config for:', modelName);
        } else {
            // Custom model URL
            const modelUrlInput = document.getElementById('model-url');
            const modelUrl = modelUrlInput.value.trim();
            if (!modelUrl) {
                displayError('Invalid URL', 'Please enter a model URL.');
                modelUrlInput.focus();
                hasError = true;
                throw new Error('Invalid model URL');
            }

            // Basic URL validation
            try {
                new URL(modelUrl);
            } catch (urlError) {
                displayError('Invalid URL', 'The URL format is invalid. Please enter a valid URL (e.g., https://huggingface.co/org/model).');
                modelUrlInput.focus();
                hasError = true;
                throw new Error('Invalid URL');
            }

            try {
                config = await fetchModelConfigFromUrl(modelUrl);
                // Use the model name from config if available, otherwise use the identifier
                modelName = config._modelName || modelUrl;
            } catch (fetchError) {
                let errorMessage = 'Failed to fetch model configuration. ';
                if (fetchError.message) {
                    errorMessage += fetchError.message;
                } else {
                    errorMessage += 'Please check if the model exists and the URL is correct.';
                }
                displayError('Fetch Failed', errorMessage);
                hasError = true;
                throw fetchError;
            }
        }

        // Validate model config
        if (!config || !config.hidden_size || !config.num_attention_heads || !config.num_hidden_layers) {
            displayError('Invalid Configuration', 'The model configuration is incomplete or invalid. Required fields: hidden_size, num_attention_heads, num_hidden_layers.');
            hasError = true;
            throw new Error('Incomplete model configuration');
        }

        // Perform calculation
        const result = performCalculation(config, tokens, dtype, modelName);
        displayResults(result);

        console.log('Calculation completed successfully');

    } catch (error) {
        if (!hasError) {
            console.error('Calculation error:', error);
        }
    } finally {
        // Always restore button state
        calculateBtn.innerHTML = originalText;
        calculateBtn.disabled = false;

        // Update translations for the button text
        const calcText = document.querySelector('button[onclick="calculateKVCache()"] span:last-child');
        if (calcText) calcText.textContent = translations[currentLanguage]['calculate'] || 'Calculate KV Cache';
    }
}

// ============================================================
// Calculate Maximum Tokens
// ============================================================

async function calculateMaxTokens() {
    // Get and validate GPU memory input
    const gpuMemoryInput = document.getElementById('gpu-memory-input').value.trim();
    const gpuMemoryGB = parseFloat(gpuMemoryInput);
    const dtype = document.getElementById('dtype-select').value;

    // Validate input
    if (!gpuMemoryInput) {
        displayError('Invalid Input', 'Please enter the GPU memory size in GB.');
        return;
    }

    if (isNaN(gpuMemoryGB) || gpuMemoryGB <= 0) {
        displayError('Invalid Input', 'Please enter a valid positive number for GPU memory size (GB).');
        return;
    }

    let config;
    let modelName;

    // Show loading state
    const calculateBtn = document.querySelector('button[onclick="calculateMaxTokens()"]');
    const originalText = calculateBtn.innerHTML;
    calculateBtn.innerHTML = '<span>⏳</span> <span>Calculating...</span>';
    calculateBtn.disabled = true;

    try {
        // Get model configuration (same logic as calculateKVCache)
        if (currentModelSource === 'preset') {
            const presetSelect = document.getElementById('preset-model-select');
            modelName = presetSelect.value;
            if (!modelName || !modelConfigs[modelName]) {
                displayError('Model Not Found', 'The selected preset model configuration is not available. Please select another model.');
                return;
            }
            config = modelConfigs[modelName];
        } else {
            const modelUrlInput = document.getElementById('model-url');
            const modelUrl = modelUrlInput.value.trim();
            if (!modelUrl) {
                displayError('Invalid URL', 'Please enter a model URL.');
                modelUrlInput.focus();
                return;
            }

            // Basic URL validation
            try {
                new URL(modelUrl);
            } catch (urlError) {
                displayError('Invalid URL', 'The URL format is invalid. Please enter a valid URL (e.g., https://huggingface.co/org/model).');
                modelUrlInput.focus();
                return;
            }

            try {
                config = await fetchModelConfigFromUrl(modelUrl);
                // Use the model name from config if available, otherwise use the identifier
                modelName = config._modelName || modelUrl;
            } catch (fetchError) {
                let errorMessage = 'Failed to fetch model configuration. ';
                if (fetchError.message) {
                    errorMessage += fetchError.message;
                } else {
                    errorMessage += 'Please check if the model exists and the URL is correct.';
                }
                displayError('Fetch Failed', errorMessage);
                return;
            }
        }

        // Validate model config
        if (!config || !config.hidden_size || !config.num_attention_heads || !config.num_hidden_layers) {
            displayError('Invalid Configuration', 'The model configuration is incomplete or invalid. Required fields: hidden_size, num_attention_heads, num_hidden_layers.');
            return;
        }

        // Calculate maximum tokens
        const result = calculateMaxTokensForMemory(config, gpuMemoryGB, dtype, modelName);
        displayMaxTokensResults(result);

        console.log('Maximum tokens calculated successfully');

    } catch (error) {
        console.error('Max tokens calculation error:', error);
    } finally {
        // Restore button state
        calculateBtn.innerHTML = originalText;
        calculateBtn.disabled = false;

        // Update translations for the button text
        const calcText = document.querySelector('button[onclick="calculateMaxTokens()"] span:last-child');
        if (calcText) calcText.textContent = translations[currentLanguage]['calculate-max-tokens'] || 'Calculate Max Tokens';
    }
}

// ============================================================
// Core Calculation: KV Cache Size
// ============================================================

/**
 * Perform KV Cache calculation for given model config and parameters.
 * BUG FIX: Hybrid architecture detection now correctly uses config.num_hidden_layers
 *           instead of the unassigned local variable num_hidden_layers.
 */
function performCalculation(config, tokens, dtype, modelName) {
    let hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads;
    let kv_lora_rank, qk_rope_head_dim; // for MLA models
    let head_dim;
    let sliding_window;

    // Detect model type based on configuration parameters
    const {
        isMLAModel,
        isHybrid,
        isGQAWithHeadDim,
        hybridSubType,
        attentionLayerCount,
        sliding_window: detectedSlidingWindow
    } = detectArchitectureType(config);

    sliding_window = detectedSlidingWindow;

    // Extract config fields based on architecture type
    if (isMLAModel) {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, kv_lora_rank, qk_rope_head_dim } = config);
        console.log('Detected MLA architecture for:', modelName);
    } else if (isHybrid) {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, head_dim } = config);
        console.log('Detected Hybrid architecture (' + hybridSubType + ') for:', modelName, 'attention layers:', attentionLayerCount);
    } else if (isGQAWithHeadDim) {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, head_dim } = config);
        console.log('Detected GQA (with head_dim) architecture for:', modelName);
    } else {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads } = config);
        console.log('Detected Standard architecture for:', modelName);
    }

    // Validate required fields
    const requiredFields = ['hidden_size', 'num_attention_heads', 'num_hidden_layers'];
    for (const field of requiredFields) {
        if (!config[field]) {
            throw new Error(`Missing required field: ${field}`);
        }
    }

    // Get additional parameters
    const batchSize = parseInt(document.getElementById('batch-size').value) || 1;
    const tp = parseInt(document.getElementById('tp').value) || 1;
    const dp = parseInt(document.getElementById('dp').value) || 1;

    // Data type sizes in bytes
    const dtypeSizes = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1
    };

    if (!dtypeSizes[dtype]) {
        throw new Error(`Unsupported data type: ${dtype}`);
    }

    const dtypeSize = dtypeSizes[dtype];

    // Calculate KV cache size (Single GPU)
    let totalElements;
    let formula;
    let elementsPerToken;
    let effectiveTokens = tokens;
    let hasHybridWarning = false;

    const kvHeads = num_key_value_heads || num_attention_heads;
    const hdim = head_dim || (hidden_size / num_attention_heads);

    if (isMLAModel) {
        // MLA: layers × tokens × batch × (kv_lora_rank + qk_rope_head_dim) / tp × dtype
        // No factor of 2 (K and V compressed together)
        elementsPerToken = num_hidden_layers * (kv_lora_rank + qk_rope_head_dim) / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = `${num_hidden_layers} × ${tokens} × ${batchSize} × (${kv_lora_rank} + ${qk_rope_head_dim}) ÷ ${tp} × ${dtypeSize} bytes`;
    } else if (isHybrid) {
        // Hybrid architecture - unified handling
        // All hybrid models get a warning
        hasHybridWarning = true;

        if (sliding_window) {
            // Hybrid with sliding window (e.g., MiMo-V2-Flash, Gemma4, GPT-OSS)
            // KV = 2 × attn_layers × min(tokens, window) × batch × kv_heads × head_dim / tp × dtype
            effectiveTokens = Math.min(tokens, sliding_window);
            elementsPerToken = 2 * attentionLayerCount * kvHeads * hdim / tp;
            totalElements = elementsPerToken * effectiveTokens * batchSize;
            formula = `2 × ${attentionLayerCount} × ${sliding_window} × ${batchSize} × ${kvHeads} × ${hdim} ÷ ${tp} × ${dtypeSize} bytes`;
        } else {
            // Hybrid without sliding window (e.g., Qwen3.5 with Linear+Full)
            // KV = 2 × attn_layers × tokens × batch × kv_heads × head_dim / tp × dtype
            elementsPerToken = 2 * attentionLayerCount * kvHeads * hdim / tp;
            totalElements = elementsPerToken * tokens * batchSize;
            formula = `2 × ${attentionLayerCount} × ${tokens} × ${batchSize} × ${kvHeads} × ${hdim} ÷ ${tp} × ${dtypeSize} bytes`;
        }
    } else if (isGQAWithHeadDim) {
        // GQA with explicit head_dim
        elementsPerToken = 2 * num_hidden_layers * kvHeads * hdim / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = `2 × ${num_hidden_layers} × ${tokens} × ${batchSize} × ${kvHeads} × ${hdim} ÷ ${tp} × ${dtypeSize} bytes`;
    } else {
        // Standard Transformer with or without GQA
        elementsPerToken = 2 * num_hidden_layers * hidden_size * (kvHeads / num_attention_heads) / tp;
        totalElements = elementsPerToken * tokens * batchSize;
        formula = `2 × ${num_hidden_layers} × ${tokens} × ${batchSize} × ${hidden_size} × (${kvHeads}/${num_attention_heads}) ÷ ${tp} × ${dtypeSize} bytes`;
    }

    const totalBytes = totalElements * dtypeSize;
    const kvCacheSizeGB = totalBytes / (1024 ** 3);

    // Calculate cluster-wide KV cache (all GPUs)
    const totalGPUs = tp * dp;
    const clusterKVCacheSizeGB = kvCacheSizeGB * totalGPUs;

    // Calculate model parameters (approximate)
    const modelParams = num_hidden_layers * hidden_size * hidden_size * 3;
    const modelSizeGB = (modelParams * dtypeSize * 2 / tp) / (1024 ** 3); // 2 * n / tp

    // Calculate FLOPs
    const prefillFLOPs = 2 * modelParams * batchSize * tokens / tp;
    const decodeFLOPs = 2 * modelParams * batchSize * 1 / tp;

    // Create details object based on model type
    const details = {
        tokens,
        batch_size: batchSize,
        tp,
        dp,
        dtype,
        dtype_size: dtypeSize,
        model_params: modelParams,
        model_size_gb: modelSizeGB,
        prefill_flops: prefillFLOPs,
        decode_flops: decodeFLOPs,
        calculation_formula: formula,
        elements_per_token: elementsPerToken
    };

    // Determine architecture type for display
    let architectureType = 'Standard Transformer';
    let showHybridWarning = false;

    if (isMLAModel) {
        architectureType = 'MLA (Multi-head Latent Attention)';
        details.hidden_size = hidden_size;
        details.num_attention_heads = num_attention_heads;
        details.num_hidden_layers = num_hidden_layers;
        details.num_key_value_heads = num_key_value_heads;
        details.kv_lora_rank = kv_lora_rank;
        details.qk_rope_head_dim = qk_rope_head_dim;
    } else if (isHybrid) {
        architectureType = 'Hybrid (' + hybridSubType + ')';
        details.hidden_size = hidden_size;
        details.num_attention_heads = num_attention_heads;
        details.num_hidden_layers = num_hidden_layers;
        details.num_key_value_heads = num_key_value_heads;
        details.head_dim = hdim;
        details.attention_layer_count = attentionLayerCount;
        if (sliding_window) details.sliding_window = sliding_window;
        showHybridWarning = true;
    } else if (isGQAWithHeadDim) {
        architectureType = 'GQA (Grouped-Query Attention)';
        details.hidden_size = hidden_size;
        details.num_attention_heads = num_attention_heads;
        details.num_hidden_layers = num_hidden_layers;
        details.num_key_value_heads = num_key_value_heads;
        details.head_dim = head_dim;
    } else {
        // Standard: determine MHA/MQA/GQA
        if (kvHeads === num_attention_heads) {
            architectureType = 'MHA (Multi-Head Attention)';
        } else if (kvHeads === 1) {
            architectureType = 'MQA (Multi-Query Attention)';
        } else {
            architectureType = 'GQA (Grouped-Query Attention)';
        }
        details.hidden_size = hidden_size;
        details.num_attention_heads = num_attention_heads;
        details.num_hidden_layers = num_hidden_layers;
        details.num_key_value_heads = kvHeads;
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
        kvCacheSizeGB,
        clusterKVCacheSizeGB,
        modelSizeGB,
        prefillFLOPs,
        decodeFLOPs,
        totalElements,
        totalBytes,
        config,
        formula,
        details,
        architectureType,  // Add architecture type for display
        showHybridWarning  // Add warning flag
    };
}

// ============================================================
// Core Calculation: Max Tokens for Given Memory
// ============================================================

function calculateMaxTokensForMemory(config, gpuMemoryGB, dtype, modelName) {
    let hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads;
    let kv_lora_rank, qk_rope_head_dim; // for MLA models
    let head_dim;
    let sliding_window;

    // Detect model type - same logic as performCalculation
    const {
        isMLAModel,
        isHybrid,
        isGQAWithHeadDim,
        hybridSubType,
        attentionLayerCount,
        sliding_window: detectedSlidingWindow
    } = detectArchitectureType(config);

    sliding_window = detectedSlidingWindow;

    // Extract config fields
    if (isMLAModel) {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, kv_lora_rank, qk_rope_head_dim } = config);
    } else if (isHybrid) {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, head_dim } = config);
    } else if (isGQAWithHeadDim) {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, head_dim } = config);
    } else {
        ({ hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads } = config);
    }

    // Validate required fields
    const requiredFields = ['hidden_size', 'num_attention_heads', 'num_hidden_layers'];
    for (const field of requiredFields) {
        if (!config[field]) {
            throw new Error(`Missing required field: ${field}`);
        }
    }

    // Get additional parameters
    const batchSize = parseInt(document.getElementById('batch-size').value) || 1;
    const tp = parseInt(document.getElementById('tp').value) || 1;
    const dp = parseInt(document.getElementById('dp').value) || 1;

    // Data type sizes in bytes
    const dtypeSizes = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1
    };

    if (!dtypeSizes[dtype]) {
        throw new Error(`Unsupported data type: ${dtype}`);
    }

    const dtypeSize = dtypeSizes[dtype];

    // Calculate elements per token using model-specific formula
    let elementsPerToken;
    const kvHeads = num_key_value_heads || num_attention_heads;
    const hdim = head_dim || (hidden_size / num_attention_heads);

    if (isMLAModel) {
        elementsPerToken = num_hidden_layers * batchSize * (kv_lora_rank + qk_rope_head_dim) / tp;
    } else if (isHybrid) {
        if (sliding_window) {
            elementsPerToken = 2 * attentionLayerCount * batchSize * kvHeads * hdim / tp;
        } else {
            elementsPerToken = 2 * attentionLayerCount * batchSize * kvHeads * hdim / tp;
        }
    } else if (isGQAWithHeadDim) {
        elementsPerToken = 2 * num_hidden_layers * batchSize * kvHeads * hdim / tp;
    } else {
        elementsPerToken = 2 * batchSize * hidden_size * (kvHeads / num_attention_heads) * num_hidden_layers / tp;
    }

    // Calculate model parameters (approximate)
    const modelParams = num_hidden_layers * hidden_size * hidden_size * 3;
    const modelSizeGB = (modelParams * dtypeSize * 2 / tp) / (1024 ** 3);

    // Calculate maximum tokens per request
    // maxTokens = single GPU memory / per-token memory on that GPU
    const totalMemoryBytes = gpuMemoryGB * (1024 ** 3);
    let maxTokens;

    // For sliding window models, max tokens is limited by window size
    if (isHybrid && sliding_window) {
        maxTokens = sliding_window;
    } else {
        maxTokens = Math.floor(totalMemoryBytes / (elementsPerToken * dtypeSize));
    }

    // Create formula based on model type
    let formula;
    if (isMLAModel) {
        formula = `${num_hidden_layers} × ${batchSize} × (${kv_lora_rank} + ${qk_rope_head_dim}) ÷ ${tp} × ${dtypeSize} bytes`;
    } else if (isHybrid) {
        if (sliding_window) {
            formula = `2 × ${attentionLayerCount} × ${sliding_window} × ${batchSize} × ${kvHeads} × ${hdim} ÷ ${tp} × ${dtypeSize} bytes`;
        } else {
            formula = `2 × ${attentionLayerCount} × ${batchSize} × ${kvHeads} × ${hdim} ÷ ${tp} × ${dtypeSize} bytes`;
        }
    } else if (isGQAWithHeadDim) {
        formula = `2 × ${num_hidden_layers} × ${batchSize} × ${kvHeads} × ${hdim} ÷ ${tp} × ${dtypeSize} bytes`;
    } else {
        formula = `2 × ${batchSize} × ${hidden_size} × (${kvHeads}/${num_attention_heads}) × ${num_hidden_layers} ÷ ${tp} × ${dtypeSize} bytes`;
    }

    // Create config object for display
    const displayConfig = {
        num_hidden_layers: num_hidden_layers,
        hidden_size: hidden_size,
        num_attention_heads: num_attention_heads,
        num_key_value_heads: isMLAModel ? num_key_value_heads : (num_key_value_heads || num_attention_heads)
    };

    if (isMLAModel) {
        displayConfig.kv_lora_rank = kv_lora_rank;
        displayConfig.qk_rope_head_dim = qk_rope_head_dim;
    } else if (isHybrid) {
        if (head_dim) displayConfig.head_dim = head_dim;
        if (sliding_window) displayConfig.sliding_window = sliding_window;
        displayConfig.attention_layer_count = attentionLayerCount;
    } else if (isGQAWithHeadDim) {
        if (head_dim) displayConfig.head_dim = head_dim;
    }

    // Determine architecture type for display
    let architectureType = 'Standard Transformer';
    let showHybridWarning = false;

    if (isMLAModel) {
        architectureType = 'MLA (Multi-head Latent Attention)';
    } else if (isHybrid) {
        architectureType = 'Hybrid (' + hybridSubType + ')';
        showHybridWarning = true;
    } else if (isGQAWithHeadDim) {
        architectureType = 'GQA (Grouped-Query Attention)';
    } else {
        if (kvHeads === num_attention_heads) {
            architectureType = 'MHA (Multi-Head Attention)';
        } else if (kvHeads === 1) {
            architectureType = 'MQA (Multi-Query Attention)';
        } else {
            architectureType = 'GQA (Grouped-Query Attention)';
        }
    }

    return {
        modelName,
        batchSize,
        tp,
        dp,
        totalGPUs: tp * dp,
        gpuMemoryGB,
        dtype,
        dtypeSize,
        maxTokens,
        elementsPerToken,
        totalMemoryBytes,
        config: displayConfig,
        formula,
        modelSizeGB,
        modelParams,
        perTokenMemoryMB: (elementsPerToken * dtypeSize) / (1024 ** 2),
        architectureType,
        showHybridWarning
    };
}

// ============================================================
// Display Functions
// ============================================================

function displayError(title, message) {
    const resultsContainer = document.getElementById('results-container');
    const detailsContainer = document.getElementById('calculation-details');
    const stepsContainer = document.getElementById('calculation-steps');

    if (!resultsContainer) {
        console.error('Results container not found');
        return;
    }

    // Hide details section
    if (detailsContainer) {
        detailsContainer.classList.add('hidden');
    }
    if (stepsContainer) {
        stepsContainer.innerHTML = '';
    }

    // Display error in results panel
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
    const detailsContainer = document.getElementById('calculation-details');
    const stepsContainer = document.getElementById('calculation-steps');

    // Check if required elements exist
    if (!resultsContainer) {
        console.error('Results container not found');
        return;
    }

    // Main result display
    const config = result.config;
    const kvHeads = config.num_key_value_heads || config.num_attention_heads;

    // Use the architecture type from performCalculation
    const modelTypeText = result.architectureType || 'Standard Transformer';

    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-primary);">${result.kvCacheSizeGB.toFixed(4)} GB</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Single-GPU KV Cache Size</div>
            ${result.totalGPUs > 1 ? `
            <div class="result-value" style="font-size: 1.2rem; font-weight: 600; color: var(--accent-primary); margin-top: 0.5rem;">${result.clusterKVCacheSizeGB.toFixed(4)} GB</div>
            <div class="result-label" style="font-size: 0.75rem; color: var(--text-secondary);">Cluster-wide KV Cache (TP=${result.tp} × DP=${result.dp} = ${result.totalGPUs} GPUs)</div>
            ` : ''}
        </div>

        ${result.showHybridWarning ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-warning); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: var(--accent-warning); font-size: 0.85rem;">Hybrid Architecture Warning</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                This model contains special layers (e.g., Linear Attention, SSM). Calculation may not be accurate. Further adaptation needed.
            </div>
        </div>
        ` : ''}

        <!-- Single-line metrics for high density -->
        <div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Model:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${getModelDisplayName(result.modelName)}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Type:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${modelTypeText}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Tokens:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.tokens.toLocaleString()}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Batch:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.batchSize}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">DType:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.dtype}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">TP:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.tp}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">DP:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.dp}</strong>
            </div>
        </div>

        <!-- Calculation Formula Card -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📐</span>
                <span>Single-GPU Formula</span>
            </div>
            <div class="formula-content">
                <div class="formula-main" style="font-size: 0.75rem;">${result.formula}</div>
            </div>
        </div>

        <!-- Model Configuration -->
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
                    ${config.kv_lora_rank ? `<div style="color: var(--text-secondary);">KV LoRA Rank:</div><div style="color: var(--text-primary); font-weight: 500;">${config.kv_lora_rank}</div>` : ''}
                    ${config.qk_rope_head_dim ? `<div style="color: var(--text-secondary);">QK RoPE Dim:</div><div style="color: var(--text-primary); font-weight: 500;">${config.qk_rope_head_dim}</div>` : ''}
                    ${config.head_dim && !config.kv_lora_rank ? `<div style="color: var(--text-secondary);">Head Dim:</div><div style="color: var(--text-primary); font-weight: 500;">${config.head_dim}</div>` : ''}
                </div>
            </div>
        </div>
    `;

    // Hide the separate calculation details section since we now show everything in results
    if (detailsContainer) {
        detailsContainer.classList.add('hidden');
    }
    if (stepsContainer) {
        stepsContainer.innerHTML = '';
    }
}

function displayMaxTokensResults(result) {
    const resultsContainer = document.getElementById('results-container');
    const detailsContainer = document.getElementById('calculation-details');
    const stepsContainer = document.getElementById('calculation-steps');

    // Check if required elements exist
    if (!resultsContainer) {
        console.error('Results container not found');
        return;
    }

    // Main result display
    const config = result.config;
    const kvHeads = config.num_key_value_heads || config.num_attention_heads;

    // Use the architecture type from calculateMaxTokensForMemory
    const modelTypeText = result.architectureType || 'Standard Transformer';

    resultsContainer.innerHTML = `
        <div class="result-display" style="text-align: center; margin-bottom: 1rem;">
            <div class="result-value" style="font-size: 1.8rem; font-weight: 700; color: var(--accent-success);">${result.maxTokens.toLocaleString()}</div>
            <div class="result-label" style="font-size: 0.8rem; color: var(--text-secondary);">Max Tokens ${result.tp > 1 ? `(Per-Request, TP=${result.tp})` : '(Per Request)'}</div>
        </div>

        ${result.showHybridWarning ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid var(--accent-warning); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1rem;">⚠️</span>
                <strong style="color: var(--accent-warning); font-size: 0.85rem;">Hybrid Architecture Warning</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                This model contains special layers (e.g., Linear Attention, SSM). Calculation may not be accurate. Further adaptation needed.
            </div>
        </div>
        ` : ''}

        <!-- Single-line metrics for high density -->
        <div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Model:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${getModelDisplayName(result.modelName)}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Type:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${modelTypeText}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Single-GPU:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.gpuMemoryGB}GB</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">Batch:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.batchSize}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">DType:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.dtype}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">TP:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.tp}</strong>
            </div>
            <div class="metric-item" style="background: var(--bg-secondary); padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">
                <span style="color: var(--text-secondary);">DP:</span>
                <strong style="color: var(--text-primary); margin-left: 0.25rem;">${result.dp}</strong>
            </div>
        </div>

        <!-- Calculation Details Card -->
        <div class="formula-card" style="margin-bottom: 0.625rem;">
            <div class="formula-header">
                <span>📐</span>
                <span>Calculation Details</span>
            </div>
            <div class="formula-content">
                <!-- Per-Token Formula -->
                <div style="margin-bottom: 0.75rem;">
                    <div style="font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Per-Token Formula (Single-GPU):</div>
                    <div class="formula-main" style="font-size: 0.7rem;">${result.formula}</div>
                </div>

                <!-- Max Tokens Breakdown -->
                <div style="padding-top: 0.75rem; border-top: 1px dashed var(--border-color);">
                    <div style="font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 0.375rem;">Max Tokens Calculation:</div>
                    <div class="formula-breakdown">
                        <div class="formula-step">
                            <span class="formula-step-label">Memory for KV Cache:</span>
                            <span class="formula-step-value">${(result.gpuMemoryGB * 1024).toFixed(0)} MB</span>
                        </div>
                        <div class="formula-step">
                            <span class="formula-step-label">Per Token:</span>
                            <span class="formula-step-value">${result.perTokenMemoryMB.toFixed(3)} MB</span>
                        </div>
                        <div class="formula-step" style="margin-top: 0.25rem;">
                            <span class="formula-step-label">Per-Request Result:</span>
                            <span class="formula-step-value" style="color: var(--accent-success); font-weight: 600;">${result.maxTokens.toLocaleString()} tokens</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Configuration -->
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
                    ${config.kv_lora_rank ? `<div style="color: var(--text-secondary);">KV LoRA Rank:</div><div style="color: var(--text-primary); font-weight: 500;">${config.kv_lora_rank}</div>` : ''}
                    ${config.qk_rope_head_dim ? `<div style="color: var(--text-secondary);">QK RoPE Dim:</div><div style="color: var(--text-primary); font-weight: 500;">${config.qk_rope_head_dim}</div>` : ''}
                    ${config.head_dim && !config.kv_lora_rank ? `<div style="color: var(--text-secondary);">Head Dim:</div><div style="color: var(--text-primary); font-weight: 500;">${config.head_dim}</div>` : ''}
                </div>
            </div>
        </div>
    `;

    // Calculation details (only if elements exist)
    if (stepsContainer || detailsContainer) {
        const config = result.config;
        const kvHeads = config.num_key_value_heads || config.num_attention_heads;

        const stepsHTML = `
            <div class="formula-card">
                <div class="formula-header">
                    <span>📐</span>
                    <span>Per-Token Formula</span>
                </div>
                <div class="formula-content">
                    <div class="formula-main">${result.formula}</div>
                </div>
            </div>

            <div class="formula-card">
                <div class="formula-header">
                    <span>🔢</span>
                    <span>Max Tokens Calculation</span>
                </div>
                <div class="formula-content">
                    <div class="formula-main">
                        ${(result.gpuMemoryGB * 1024).toFixed(0)} MB ÷ ${(result.perTokenMemoryMB).toFixed(3)} MB = ${result.maxTokens.toLocaleString()} Tokens
                    </div>
                </div>
            </div>
        `;

        // Hide the separate calculation details section since we now show everything in results
        if (detailsContainer) {
            detailsContainer.classList.add('hidden');
        }
        if (stepsContainer) {
            stepsContainer.innerHTML = '';
        }

        // Apply translations to elements with data-i18n attributes
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            if (translations[currentLanguage][key]) {
                element.textContent = translations[currentLanguage][key];
            }
        });
    }
}

// ============================================================
// Toast Notification System
// ============================================================

function showToast(type, title, message) {
    const container = document.getElementById('toast-container');

    // Remove any existing toasts of the same type
    const existingToasts = container.querySelectorAll(`.toast.${type}`);
    existingToasts.forEach(toast => toast.remove());

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    // Set icon based on type
    const icons = {
        'error': '❌',
        'success': '✅',
        'warning': '⚠️'
    };

    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-icon">${icons[type] || icons['error']}</div>
            <div class="toast-info">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
        </div>
        <button class="toast-close" onclick="closeToast(this.parentElement)">×</button>
    `;

    // Add to container
    container.appendChild(toast);

    // Trigger animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    // Auto remove after 5 seconds for success/warning, 8 seconds for error
    const timeout = type === 'error' ? 8000 : 5000;
    setTimeout(() => {
        closeToast(toast);
    }, timeout);
}

function closeToast(toast) {
    if (toast) {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 300);
    }
}

// ============================================================
// Get Current Model Configuration (utility)
// ============================================================

async function getCurrentModelConfig() {
    let config;
    let modelName;

    try {
        console.log('Current model source:', currentModelSource);

        if (currentModelSource === 'preset') {
            const presetSelect = document.getElementById('preset-model-select');
            if (!presetSelect) {
                console.log('Preset model select element not found');
                return null;
            }
            modelName = presetSelect.value;
            console.log('Selected preset model:', modelName);
            if (!modelName || !modelConfigs[modelName]) {
                console.log('Preset model not found:', modelName);
                return null;
            }
            config = modelConfigs[modelName];
            console.log('Using preset config for:', modelName);
        } else {
            // Custom model URL
            const modelUrlElement = document.getElementById('model-url');
            if (!modelUrlElement) {
                console.log('Model URL element not found');
                return null;
            }
            const modelUrl = modelUrlElement.value.trim();
            if (!modelUrl) {
                console.log('No model URL provided');
                return null;
            }

            // Try to fetch config from URL (async)
            try {
                config = await fetchModelConfigFromUrl(modelUrl);
            } catch (fetchError) {
                console.log('Failed to fetch config from URL:', fetchError);
                return null;
            }

            if (!config) {
                console.log('Failed to fetch config from URL');
                return null;
            }
            modelName = modelUrl;
        }

        // Validate required fields
        const requiredFields = ['hidden_size', 'num_attention_heads', 'num_hidden_layers'];
        for (const field of requiredFields) {
            if (!config[field]) {
                console.log(`Missing required field: ${field}`);
                return null;
            }
        }

        // Add model name to config for display
        config._name = modelName;
        return config;

    } catch (error) {
        console.error('Error getting model config:', error);
        return null;
    }
}

// ============================================================
// Event Listeners
// ============================================================

function initializeEventListeners() {
    // Enter key support for token input
    document.getElementById('token-input').addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            calculateKVCache();
        }
    });

    // Enter key support for model URL
    document.getElementById('model-url').addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            calculateKVCache();
        }
    });

    // Toast notifications don't need escape key handling
}
