---
hide:
  - navigation
  - toc
---

<div align="center" markdown>

![UCM](../assets/images/UCM-light.png#only-light){ style="height:96px;width:auto" }
![UCM](../assets/images/UCM-dark.png#only-dark){ style="height:96px;width:auto" }

</div>

# Unified Cache Manager

**Unified Cache Manager (UCM) is a KV Cache management and reuse system for LLM inference. It persists reusable KV Cache to external storage and shares it across requests and compatible inference instances, reducing repeated Prefill computation.**

In multi-turn conversations, long-context reasoning, coding and multi-agent collaboration, project code, task instructions and conversation history often recur across successive requests. UCM reuses the KV Cache already generated for that context, reducing computation for repeated prefixes and lowering response latency.

Compared with KV Cache kept only within an inference process, UCM uses external storage to expand cache capacity. It supports **reuse across requests, persistence beyond process lifetimes and sharing between compatible instances**, so existing computation results remain reusable.

When integrated with vLLM, UCM achieves a **3–10x reduction in inference latency** in scenarios with substantial prefix repetition, including multi-turn conversations and long-context reasoning.

## Start with your task

<div class="ucm-home-cards" markdown>

[<span class="ucm-home-card__icon" aria-hidden="true">:material-play-circle-outline:</span>
<span class="ucm-home-card__title">Quickstart</span>
<span class="ucm-home-card__description">Choose artifacts, configure your engine and run your first service with UCM.</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](user-guide/quick_start/index.md){ .ucm-home-card }

[<span class="ucm-home-card__icon" aria-hidden="true">:material-view-grid-outline:</span>
<span class="ucm-home-card__title">Supported configurations</span>
<span class="ucm-home-card__description">Check engine, model, device and feature support before choosing a deployment.</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](user-guide/support-matrix/index.md){ .ucm-home-card }

[<span class="ucm-home-card__icon" aria-hidden="true">:material-layers-triple-outline:</span>
<span class="ucm-home-card__title">Start with a model</span>
<span class="ucm-home-card__description">Find official model guides and UCM examples for deployment and API calls.</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](user-guide/model-tour/index.md){ .ucm-home-card }

[<span class="ucm-home-card__icon" aria-hidden="true">:material-code-braces:</span>
<span class="ucm-home-card__title">Understand and extend UCM</span>
<span class="ucm-home-card__description">Follow a request through the architecture, then add a storage backend or metrics.</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](developer-guide/index.md){ .ucm-home-card }

</div>

## Operate and measure

[Operations](user-guide/observability/index.md) explains how to check cache reuse, backend health and time spent in each stage. The [toolkit](toolkit/index.md) provides deployment checks, storage tests and metric inspection; use the [KV Cache calculator](toolkit/kv-cache-calculator.md) for capacity estimates.

## Supported integrations

UCM provides integration paths for vLLM, vLLM-Ascend and SGLang. Check the [support matrix](user-guide/support-matrix/index.md) for model and platform coverage and [installation](user-guide/quick_start/index.md) for released engine/backend combinations.

[Source on GitHub](https://github.com/ModelEngine-Group/unified-cache-management) · [Contribute](developer-guide/contribute.md) · [About UCM](about.md)
