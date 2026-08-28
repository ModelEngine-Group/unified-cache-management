# Kimi

This tour uses `moonshotai/Kimi-K2-Thinking`, a 1T-parameter MoE reasoning
model. It requires a multi-accelerator server and model-specific reasoning/tool
parsers.

!!! warning "Preview UCM recipe"

    Upstream vLLM, vLLM Ascend, and SGLang can serve Kimi-K2 variants, but the
    current UCM release support matrix still marks Kimi-K2.5 as unsupported.
    Treat the commands below as an integration-validation recipe: qualify cache
    correctness and model accuracy before production use. Do not infer support
    for Kimi-K2.5 or Kimi-K2.6 from this Kimi-K2-Thinking example.

## Models

| Model | vLLM Ascend latest guide |
| --- | --- |
| Kimi-K2-Thinking | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2-Thinking.html) |
| Kimi-K2.5 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2.5.html) |
| Kimi-K2.6 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2.6.html) |

## Before you start

--8<-- "docs/en/user-guide/model-tour/_shared/setup.md"

## Serving engine

=== "vLLM"

    --8<-- "docs/en/user-guide/model-tour/kimi/vllm.md"

=== "vLLM Ascend"

    --8<-- "docs/en/user-guide/model-tour/kimi/ascend.md"

=== "SGLang"

    --8<-- "docs/en/user-guide/model-tour/kimi/sglang.md"

## Verify UCM reuse

```bash
export MODEL_ALIAS=kimi-k2-thinking
```

--8<-- "docs/en/user-guide/model-tour/_shared/verify.md"
