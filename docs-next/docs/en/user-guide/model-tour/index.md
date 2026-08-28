# Model Tour

Use these recipes to start representative GLM, Qwen, DeepSeek, and Kimi models
with UCM-backed prefix caching. Each family page separates model-specific flags
from the common UCM configuration and provides commands for the engines that
the repository currently integrates.

The family tables mirror the current
[vLLM Ascend Model Tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/).
Only model tutorials published under `latest` are listed.

## vLLM Ascend runtime images

vLLM Ascend uses shared runtime images rather than a different image for every
model. Pull the image for the target hardware, then follow the model-specific
official guide for model weights, environment variables, and launch arguments.

### Official runtime images

| Hardware | Official image pull |
| --- | --- |
| Atlas A2 | `docker pull quay.io/ascend/vllm-ascend:v0.23.0` |
| Atlas A3 | `docker pull quay.io/ascend/vllm-ascend:v0.23.0-a3` |

## Model families

| Family | First recipe | UCM validation status |
| --- | --- | --- |
| [GLM](glm/index.md) | GLM-4.5-Air / GLM-4.7-W8A8 | GLM-4.x validated on vLLM, vLLM Ascend, and SGLang |
| [Qwen](qwen3/index.md) | Qwen3-8B | Qwen3 validated on vLLM, vLLM Ascend, and SGLang |
| [DeepSeek](deepseek/index.md) | DeepSeek-R1 | DeepSeek-R1 validated on vLLM, vLLM Ascend, and SGLang |
| [Kimi](kimi/index.md) | Kimi-K2-Thinking | Preview recipe; Kimi-K2 is not yet release-matrix validated |

The validation status describes UCM prefix-cache coverage in this repository,
not only whether the upstream engine can load the model. Model variants with a
different attention architecture, quantization, or modality require separate
validation even when they belong to the same family.

For openEuler images, append `-openeuler` to the A2 tag or use
`v0.23.0-a3-openeuler` for A3. A5 is not included in the runtime-image table
because the referenced model guides do not provide a common, verified A5
deployment contract. Each family page keeps the current model catalog and
official vLLM Ascend links directly above the **vLLM**, **vLLM Ascend**, and
**SGLang** tabs.
