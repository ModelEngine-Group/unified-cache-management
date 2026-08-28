# GLM

This tour uses `zai-org/GLM-4.5-Air` as the CUDA and SGLang example. It is the
smaller GLM-4.5 MoE checkpoint but still requires multiple accelerator cards.
For Ascend, the command follows the upstream-validated
`Eco-Tech/GLM-4.7-W8A8-floatmtp` recipe.

**UCM status:** GLM-4.x prefix caching is validated on vLLM, vLLM Ascend, and
SGLang. GLM-5/5.1/5.2 are currently validated only on vLLM and vLLM Ascend.

## Models

| Model | vLLM Ascend latest guide |
| --- | --- |
| GLM-4.x(4.5/4.6/4.7) | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM4.x.html) |
| GLM-5 & GLM-5.1 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html) |
| GLM-5.2 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html) |

## Before you start

--8<-- "docs/en/user-guide/model-tour/_shared/setup.md"

## Serving engine

=== "vLLM"

    --8<-- "docs/en/user-guide/model-tour/glm/vllm.md"

=== "vLLM Ascend"

    --8<-- "docs/en/user-guide/model-tour/glm/ascend.md"

=== "SGLang"

    --8<-- "docs/en/user-guide/model-tour/glm/sglang.md"

## Verify UCM reuse

```bash
export MODEL_ALIAS=glm
```

--8<-- "docs/en/user-guide/model-tour/_shared/verify.md"
