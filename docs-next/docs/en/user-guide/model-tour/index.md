# Model Tour

Explore UCM deployment recipes by model family. Each family keeps its model
catalog in one place and presents engine-specific guidance as tabs on the same
page.

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

- [GLM](glm/index.md)
- [Qwen](qwen3/index.md)
- [DeepSeek](deepseek/index.md)
- [MiniMax](minimax/index.md)
- [Kimi](kimi/index.md)

For openEuler images, append `-openeuler` to the A2 tag or use
`v0.23.0-a3-openeuler` for A3. A5 is not included in the runtime-image table
because the referenced model guides do not provide a common, verified A5
deployment contract. Each family page keeps the current model catalog and
official vLLM Ascend links directly above the **vLLM**, **vLLM Ascend**, and
**SGLang** tabs.
