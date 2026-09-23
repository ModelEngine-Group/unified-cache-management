# GLM

Zhipu GLM model tutorials currently published by vLLM Ascend.

## Models

| Model | vLLM Ascend latest guide |
| --- | --- |
| GLM-4.x(4.5/4.6/4.7) | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM4.x.html) |
| GLM-5 & GLM-5.1 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html) |
| GLM-5.2 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html) |

## Enable UCM with this model

Use the official recipe above for the model's engine settings, then follow the
[vLLM](../../quick_start/index.md#vllm),
[vLLM-Ascend](../../quick_start/index.md#vllm-ascend), or
[SGLang](../../quick_start/index.md#sglang) integration guide. Confirm the
model and feature in the [support matrix](../../support-matrix/index.md).
An official engine tutorial establishes engine usage; it does not independently
verify UCM external-cache behavior for that model.
