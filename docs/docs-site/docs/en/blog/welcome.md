---
title: "The UCM Blog Is Live: Making KV Cache Continuously Reusable"
date: 2026-11-09
authors: UCM Team
summary: The UCM blog is now live. This first post introduces what UCM is, why KV Cache reuse matters, and how UCM persists and shares KV Cache across requests, processes and instances.
cover: assets/images/blog/UCM-logo.png
---

# The UCM Blog Is Live: Making KV Cache Continuously Reusable

The UCM blog is now live.

This is our first post. Going forward, we will use this space to share design notes, deployment experience, benchmarks and field reports from the UCM team: why we designed things the way we did, what we ran into under real workloads, which optimizations helped, and which results only hold under specific conditions.

Before the deep dives, here is an introduction to UCM itself.

## What UCM is

Unified Cache Manager (UCM) is a KV Cache management and reuse system for LLM inference.

During Prefill, the model builds KV Cache for the input context. Multi-turn chat, long-context reasoning, coding and multi-agent workflows all have something in common: system prompts, tool definitions, repository context and conversation history repeat across successive requests. Recomputing those prefixes on every request means paying for the same Prefill again and again.

KV Cache kept inside a single inference process is also bounded by GPU memory and process lifetime. After a restart, a scale-out event or a move to another instance, the work is often lost.

UCM persists reusable KV Cache to external storage and shares it across requests and compatible inference instances:

- **Cross-request reuse**: repeated prefixes reuse existing KV Cache instead of running Prefill again;
- **Persistence beyond process lifetime**: cache survives restarts and scaling events;
- **Sharing between compatible instances**: instances with the same model and configuration can reuse cache produced by each other.

External storage turns cache capacity from a per-process resource into a larger shared pool, so results that were computed once remain reusable.

## What it changes

Integrated with vLLM, UCM delivers a **3–10x reduction in inference latency** in prefix-heavy workloads such as multi-turn conversation and long-context reasoning.

Beyond Prefix Cache, the team is also working on sparsification and PD disaggregation to improve throughput and reduce inference cost. The goal is not merely a larger cache, but cache as part of the inference infrastructure: persistent, shareable, observable and plannable.

## Where to start

- **Run it first**: use the [Quickstart](../user-guide/quick_start/index.md) to choose artifacts, configure your engine and start your first UCM-enabled service;
- **Check your stack**: see the [support matrix](../user-guide/support-matrix/index.md) for engine, model, device and feature coverage;
- **Understand the internals**: follow a request through the [developer guide](../developer-guide/index.md) to learn about cache keys, storage backends and metrics;
- **Debug in production**: read [Operations](../user-guide/observability/index.md) to check whether cache reuse is working and backends are healthy;
- **Plan capacity**: use the [KV Cache calculator](../toolkit/kv-cache-calculator.md) to estimate memory and storage requirements.

UCM provides integration paths for vLLM, vLLM-Ascend and SGLang. The exact engine and backend combinations available for installation are defined by the release artifacts on the Quickstart page.
