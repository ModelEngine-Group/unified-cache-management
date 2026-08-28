# DeepSeek

This tour uses `deepseek-ai/DeepSeek-R1`, the full 671B MoE reasoning model.
It is a production-scale recipe, not a single-GPU quickstart. Use the official
quantized checkpoint for the target device and reduce `--max-model-len` during
initial validation.

**UCM status:** DeepSeek-R1 and DeepSeek-V3/3.1/3.2 prefix caching is validated
on vLLM, vLLM Ascend, and SGLang. DeepSeek-V4 is not yet validated on SGLang.

## Models

| Model | vLLM Ascend latest guide |
| --- | --- |
| DeepSeek-V3 & 3.1 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.1.html) |
| DeepSeek-V3.2 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html) |
| DeepSeek-V4-Flash | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html) |
| DeepSeek-V4-Pro | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html) |
| DeepSeek-R1 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-R1.html) |
| DeepSeek-OCR-2 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeekOCR2.html) |

## Before you start

--8<-- "docs/en/user-guide/model-tour/_shared/setup.md"

## Serving engine

=== "vLLM"

    --8<-- "docs/en/user-guide/model-tour/deepseek/vllm.md"

=== "vLLM Ascend"

    --8<-- "docs/en/user-guide/model-tour/deepseek/ascend.md"

=== "SGLang"

    --8<-- "docs/en/user-guide/model-tour/deepseek/sglang.md"

## Verify UCM reuse

```bash
export MODEL_ALIAS=deepseek-r1
```

--8<-- "docs/en/user-guide/model-tour/_shared/verify.md"
