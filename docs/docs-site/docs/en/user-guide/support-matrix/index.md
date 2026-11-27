# Support Matrix

<div class="ucm-support-matrix" markdown="1">

Prefix Cache integration status by model. All three engine columns describe compatibility recorded for the development branch (main). Check [installation](../quick_start/index.md) for released combinations and verify the target runtime.

## Model compatibility

<div class="ucm-matrix-legend"><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span><span class="ucm-matrix-status ucm-matrix-status--unverified">Unverified</span></div>

<div class="ucm-matrix-scroll" tabindex="0" role="region" aria-label="Model compatibility table, horizontally scrollable">
<table class="ucm-support-table">
<colgroup><col style="width:40%"><col style="width:20%"><col style="width:20%"><col style="width:20%"></colgroup>
<thead><tr><th scope="col">Model</th><th scope="col">vLLM</th><th scope="col">vLLM-Ascend</th><th scope="col">SGLang</th></tr></thead>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">DeepSeek</th></tr>
<tr><th scope="row">DeepSeek V3/3.1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">DeepSeek R1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">DeepSeek V3.2</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">DeepSeek V4 Pro</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">DeepSeek V4 Flash</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">DeepSeek V4.1 Flash</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">Qwen</th></tr>
<tr><th scope="row">Qwen2.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">Qwen3</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">Qwen3-MoE</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">Qwen3-Next</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">Qwen3.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">Qwen3.6</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">Qwen3.8</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">Qwen3.8-Flash-Next</th><td><span class="ucm-matrix-status ucm-matrix-status--unverified">Unverified</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">GLM</th></tr>
<tr><th scope="row">GLM-4.x</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">GLM-5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">GLM-5.1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">GLM-5.2</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">GLM-5.3</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">Unverified</span></td></tr>
<tr><th scope="row">GLM-5.3-Flash</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">MiniMax</th></tr>
<tr><th scope="row">MiniMax-M2.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">MiniMax-M2.7</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td></tr>
<tr><th scope="row">MiniMax-M3</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">MiniMax-M3.1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">Unverified</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">Kimi</th></tr>
<tr><th scope="row">Kimi-K2.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">Kimi-K2.7</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">Supported</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
<tr><th scope="row">Kimi-K3</th><td><span class="ucm-matrix-status ucm-matrix-status--unverified">Unverified</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">Unverified</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">Unsupported</span></td></tr>
</tbody>
</table>
</div>

See [Model Tour](../model-tour/index.md) for model families and the [developer guide](../../developer-guide/cache-configuration/index.md) for storage configuration.

## Parallelism support

Per-model support for parallel strategies and their vLLM flags: DP (`--data-parallel-size`), TP (`--tensor-parallel-size`), PP (`--pipeline-parallel-size`), PCP (`--prefill-context-parallel-size`), and DCP (`--decode-context-parallel-size`). EP (`--enable-expert-parallel`) is not listed: it only shards MoE expert weights and leaves attention and the KV cache unaffected, so UCM has no model-specific restriction for it. Values apply to vLLM and vLLM-Ascend; SGLang is not covered here. The table uses ✅ (supported) and ❌ (unsupported).

<div class="ucm-matrix-scroll" tabindex="0" role="region" aria-label="Parallelism support table, horizontally scrollable">
<table class="ucm-support-table">
<colgroup><col style="width:40%"><col style="width:12%"><col style="width:12%"><col style="width:12%"><col style="width:12%"><col style="width:12%"></colgroup>
<thead><tr><th scope="col">Model</th><th scope="col">DP</th><th scope="col">TP</th><th scope="col">PP</th><th scope="col">PCP</th><th scope="col">DCP</th></tr></thead>
<tbody>
<tr class="ucm-matrix-family"><th colspan="6" scope="rowgroup">DeepSeek</th></tr>
<tr><th scope="row">DeepSeek V3/3.1</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">DeepSeek R1</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">DeepSeek V3.2</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">DeepSeek V4 Pro</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
<tr><th scope="row">DeepSeek V4 Flash</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
<tr><th scope="row">DeepSeek V4.1 Flash</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="6" scope="rowgroup">Qwen</th></tr>
<tr><th scope="row">Qwen2.5</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">Qwen3</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">Qwen3-MoE</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">Qwen3-Next</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
<tr><th scope="row">Qwen3.5</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
<tr><th scope="row">Qwen3.6</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
<tr><th scope="row">Qwen3.8</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
<tr><th scope="row">Qwen3.8-Flash-Next</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="6" scope="rowgroup">GLM</th></tr>
<tr><th scope="row">GLM-4.x</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">GLM-5</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">GLM-5.1</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">GLM-5.2</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">GLM-5.3</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">GLM-5.3-Flash</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="6" scope="rowgroup">MiniMax</th></tr>
<tr><th scope="row">MiniMax-M2.5</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">MiniMax-M2.7</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">MiniMax-M3</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">MiniMax-M3.1</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="6" scope="rowgroup">Kimi</th></tr>
<tr><th scope="row">Kimi-K2.5</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">Kimi-K2.7</th><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
<tr><th scope="row">Kimi-K3</th><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td></tr>
</tbody>
</table>
</div>

The DeepSeek V4 family and the hybrid linear-attention models support context parallelism through PCP only: PP and DCP are unsupported, and for hybrid linear-attention models PP > 1 is rejected at startup with a plain-language error (the KV hash namespace has no pipeline-stage dimension).

## Compute platforms and devices

<div class="ucm-matrix-scroll" tabindex="0" role="region" aria-label="Compute platforms and devices table">
<table class="ucm-support-table ucm-platform-table">
<colgroup><col style="width:25%"><col style="width:25%"><col style="width:50%"></colgroup>
<thead><tr><th scope="col">Platform</th><th scope="col">Vendor</th><th scope="col">Devices</th></tr></thead>
<tbody>
<tr><th scope="row">CANN</th><td>Ascend</td><td>910C, 910B, 950PR, 950DT</td></tr>
<tr><th scope="row">CUDA</th><td>NVIDIA</td><td>H100, H20, L40, L20</td></tr>
<tr><th scope="row">ROCm</th><td>AMD</td><td>MI250X (gfx90a), Radeon Pro W7800 (gfx1100), Radeon PRO V710 (gfx1101), Radeon RX 9070 XT (gfx1201), Radeon 8060S (gfx1151)</td></tr>
</tbody>
</table>
</div>

The tables list representative models and devices. Engine versions, model variants, runtime settings and backends affect behavior. Unverified means no corresponding validation conclusion is available.

**Note**: Ascend 950PR and 950DT have only been validated with DeepSeek-V4-Flash and GLM-5.2. Qwen3.8-Flash-Next runs on vLLM-Ascend with the dedicated image `qwen3.8-next-a3-openeuler`; MiniMax-M3.1 runs on vLLM with the dedicated image `minimax-m31-flash`.

</div>
