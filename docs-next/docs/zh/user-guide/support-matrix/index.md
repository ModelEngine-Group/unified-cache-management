# 支持矩阵

<div class="ucm-support-matrix" markdown="1">

本页展示各模型的 Prefix Cache 集成状态。三个引擎列均对应开发分支（main）的兼容性记录；具体发布组合见[安装](../quick_start/index.md)，目标环境仍需完成运行验证。

## 模型兼容性

<div class="ucm-matrix-legend"><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></div>

<div class="ucm-matrix-scroll" tabindex="0" role="region" aria-label="模型兼容性表，可横向滚动">
<table class="ucm-support-table">
<colgroup><col style="width:40%"><col style="width:20%"><col style="width:20%"><col style="width:20%"></colgroup>
<thead><tr><th scope="col">模型</th><th scope="col">vLLM</th><th scope="col">vLLM-Ascend</th><th scope="col">SGLang</th></tr></thead>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">DeepSeek</th></tr>
<tr><th scope="row">DeepSeek V3/3.1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">DeepSeek R1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">DeepSeek V3.2</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">DeepSeek V4 Pro</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">DeepSeek V4 Flash</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">Qwen</th></tr>
<tr><th scope="row">Qwen2.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">Qwen3</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">Qwen3-MoE</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">Qwen3-Next</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">Qwen3.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">Qwen3.6</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">Qwen3.8</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">Qwen3.8-Flash-Next</th><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">GLM</th></tr>
<tr><th scope="row">GLM-4.x</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">GLM-5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">GLM-5.1</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">GLM-5.2</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">GLM-5.3-Flash</th><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">MiniMax</th></tr>
<tr><th scope="row">MiniMax-M2.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">MiniMax-M2.7</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td></tr>
<tr><th scope="row">MiniMax-M3</th><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
</tbody>
<tbody>
<tr class="ucm-matrix-family"><th colspan="4" scope="rowgroup">Kimi</th></tr>
<tr><th scope="row">Kimi-K2.5</th><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--supported">支持</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
<tr><th scope="row">Kimi-K3</th><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unverified">待验证</span></td><td><span class="ucm-matrix-status ucm-matrix-status--unsupported">不支持</span></td></tr>
</tbody>
</table>
</div>

各家族教程见 [Model Tour](../model-tour/index.md)。存储配置见[开发者指南](../../developer-guide/cache-configuration/index.md)。

## 计算平台与设备

<div class="ucm-matrix-scroll" tabindex="0" role="region" aria-label="计算平台与设备表">
<table class="ucm-support-table ucm-platform-table">
<colgroup><col style="width:25%"><col style="width:25%"><col style="width:50%"></colgroup>
<thead><tr><th scope="col">计算平台</th><th scope="col">厂商</th><th scope="col">设备</th></tr></thead>
<tbody>
<tr><th scope="row">CANN</th><td>Ascend</td><td>910C, 910B</td></tr>
<tr><th scope="row">CUDA</th><td>NVIDIA</td><td>H100, H20, L40, L20</td></tr>
</tbody>
</table>
</div>

表中列出代表性模型和设备。实际行为受引擎版本、模型变体、运行参数及后端影响；“待验证”表示尚无对应验证结论。

</div>
