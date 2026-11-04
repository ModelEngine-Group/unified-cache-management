# Features

Feature guides that take a supported capability from zero to verified: what the feature does, how to enable it, and how to measure the effect. Usage is kept free of internal implementation details — those live in [Design Documents](../design-docs/index.md).

| Feature | What it gives you |
| --- | --- |
| [Multimodal Encoder Cache reuse](ec-connector.md) | Store vLLM encoder cache in UCM's tiered storage so requests with identical media skip encoder recomputation |
