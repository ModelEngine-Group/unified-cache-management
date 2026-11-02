# AISBench Overview

Benchmarked against internationally advanced computing benchmarks such as MLPerf, the China Electronics Standardization Institute (CESI), supported by the Key Laboratory of New-Generation Artificial Intelligence Standards and Applications of the Ministry of Industry and Information Technology, has incubated AISBench, an AI system performance benchmark, and developed a companion standards-compliance testing tool. AISBench is a performance testing benchmark for AI computing products, comprising testing standards, testing outlines, and a companion testing tool. Its test targets cover multiple forms of AI computing products, including AI accelerator cards, servers, clusters, and AI computing centers.

The AISBench tool currently supports test categories covering training/inference performance for conventional small and medium models, as well as training/inference performance testing for large models (ranging from billions to hundreds of billions of parameters); it supports general-purpose, dedicated-scenario, open, closed, training, and inference testing scenarios, modes, and types; it supports multiple metrics including runtime, power consumption, throughput, resource utilization, and energy efficiency. It also supports semi-automated to fully automated evaluation, one-click execution, plugin-based development, upgrades without service interruption, multi-channel concurrency and real-time state management, multi-framework (PyTorch, MindSpore, TensorFlow) and multi-device testing. It adopts a CS architecture, with encrypted communication between Tester and Stubs using certificates plus WebSocket, enabling the full workflow of pre-run checks, test execution, process monitoring, result and performance data aggregation, and test report generation.

All of our large-model testing is based on AISBench, and we have adapted it for UCM to implement the corresponding performance and accuracy tests.

## Common Evaluation Parameters

| Parameter | Description |
|---|---|
| `temperature` | Sampling parameter controlling model generation randomness; lower values yield more deterministic and conservative outputs. A very small value of 0.01 is used to ensure reproducibility. |
| `ignore_eos` | Ignores early termination by EOS, forcing generation up to `max_out_len` to ensure controllable output length; set to `true` for performance testing and `false` for accuracy testing. |
| `CUSTOM_TEST_IMAGE` | For models such as GLM-5.3, use the image `<private-registry>/aisbench_benchmark:v3.1_x86_64_py_310-transformersv5`;<br/>otherwise use `<private-registry>/aisbench_benchmark:v3.1_x86_64_py_310`. |

## Performance Testing

### Dataset Construction

The dataset for this round of performance testing is built on AISBench's Prefix Cache test scenario. The core objective is to generate a request set with controllable total input length, controllable prefill ratio, and reproducible content, so as to observe the performance gains from prefix cache hits under PD-disaggregated + UCM deployment.

### Construction Method

The dataset adopts a three-segment structure of “common prefix + private prefix + private suffix” and outputs three JSONL files serving the prefill and formal testing stages respectively:

| File | Content Composition | Purpose |
|---|---|---|
| `common.jsonl` | Common prefix (shared by all requests) | Global shared prefix prefill |
| `prefill.jsonl` | Common prefix + private prefix | Prefill stage: write reusable prefixes into KV Cache |
| `full.jsonl` | Common prefix + private prefix + private suffix | Formal testing: prefix portion expected to hit, only suffix needs computation |

Here, **prefill rate** is defined as the ratio of the prefillable prefix length to the total input length. When the common prefix is 0, the prefill rate is realized by the split ratio between private prefix and private suffix, i.e., in the formal request, approximately the specified proportion of input is the pre-embedded, hittable prefix portion, and the rest is the suffix portion requiring real-time computation.

### Construction Parameters

A test case is described by a six-tuple:

**`[input_len, output_len, num_requests, hit_rate, concurrency, seed]`**

The meaning of each parameter is as follows:

| Parameter | Description |
|---|---|
| `input_len` | Total input length (number of tokens), determining the input scale of the formal request; during dataset generation, it is split into prefillable prefix and computable suffix according to the prefill rate |
| `output_len` | Output length (number of tokens), handled by the AISBench load-testing configuration side, controlling the generation length of each request |
| `num_requests` | Number of requests, i.e., the number of formal test requests under this case |
| `hit_rate` | Prefill rate, i.e., the proportion of prefillable prefix in the input, determining the split ratio between private prefix and private suffix |
| `concurrency` | Concurrency, handled by the AISBench load-testing configuration side, controlling concurrency scheduling during load testing |
| `seed` | Random seed, ensuring reproducibility of dataset content |

Among these, `input_len`, `num_requests`, `hit_rate`, and `seed` directly affect dataset construction, determining input scale, number of requests, prefix/suffix split, and content reproducibility; `output_len` and `concurrency` do not participate in dataset generation and are handled by the AISBench load-testing configuration, controlling generation length and concurrency scheduling respectively.

### Prefill and Formal Testing

Dataset usage follows a two-stage “prefill first, formal test later” workflow:

1. **Storage prefill**: Run AISBench performance mode with the prefill dataset to write the reusable prefixes of each request into KV Cache;
2. **Formal testing**: Run performance testing with the full dataset; the prefix portion is expected to hit the cache, and only the private suffix triggers actual computation;
3. **Hit observation**: Collect cache hit rates of each inference pod before and after prefill and before and after formal testing, and combine with latency and throughput metrics to verify the actual gains of UCM prefix caching.

This construction separates “pre-embedded prefix” and “newly added suffix” within the same request, thereby isolating the KV Cache hit gains from overall latency for separate observation, providing a controllable and reproducible input basis for UCM performance evaluation under PD-disaggregated architecture.

For each constructed dataset, testing consists of two rounds of AISBench runs with a hit-rate snapshot in between, forming a closed loop of “prefill → record baseline → formal test → record results → comparison”.

```text
① Storage prefill (prefill.jsonl)
        ↓
② Connectivity check + save hit_rate_before.json
        ↓
③ Formal testing (full.jsonl)
        ↓
④ Save hit_rate_after.json
        ↓
⑤ Before/after hit-rate comparison (hbm / ext)
```

Each round of AISBench outputs the following metrics.

| Metric (the table gives **Average / Min / Max / Median / P75 / P90 / P99 / N** statistics for each metric) | Meaning |
|---|---|
| `E2EL` | End-to-End Latency |
| `TTFT` | Time To First Token |
| `TPOT` | Time Per Output Token |
| `ITL` | Inter-Token Latency |
| `InputTokens` | Number of input tokens per request |
| `OutputTokens` | Number of output tokens per request |
| `OutputTokenThroughput` | Output token throughput |

| Metric | Meaning |
|---|---|
| `Benchmark Duration` | Total duration of one round, from start to completion of all requests |
| `Total Requests` | Total number of requests |
| `Failed Requests` | Number of failed requests |
| `Success Requests` | Number of successful requests |
| `Concurrency` | Actual average concurrency |
| `Max Concurrency` | Configured maximum concurrency |
| `Request Throughput` | Request throughput |
| `Total Input Tokens` | Total number of input tokens |
| `Prefill Token Throughput` | Prefill-stage token throughput |
| `Total Generated Tokens` | Total number of generated tokens |
| `Input Token Throughput` | Input token throughput |
| `Output Token Throughput` | Output token throughput |
| `Total Token Throughput` | Total token throughput (input + output) |

A snapshot is saved before and after formal testing (`hit_rate_before.json` / `hit_rate_after.json`), and a comparison table is printed to the terminal listing cache hit information.

| Metric | Meaning |
|---|---|
| `endpoint` | Tested service endpoint (IP:Port) |
| `hbm_hit_rate` | HBM cache hit rate |
| `hbm(hit/query)` | HBM cache hit count / query count |
| `ext_hit_rate` | External storage (e.g., UCM extended cache) hit rate |
| `ext(hit/query)` | External storage hit count / query count |
| `TOTAL` | Summary row across all endpoints |

## Accuracy Testing

### Datasets

Accuracy testing requires **explicit selection of a standard dataset**; custom JSONL cannot be used arbitrarily as in performance testing. AISBench provides configuration templates for common datasets, typically including:

- **The dataset itself**: such as ARC, GPQA, C-Eval, GSM8K, etc.;
- **Shot count**: number of few-shot examples, such as 0-shot, 4-shot, 25-shot;
- **Prompt format**: plain prompt or chain-of-thought (CoT) prompt;
- **Interface format**: chat format or plain completion format.

For example, the configuration name `gsm8k_gen_4_shot_cot_chat_prompt` means: GSM8K dataset, generative mode, 4-shot, CoT prompt, chat format.

### Evaluation Modes

The common mode is `gen` (generative): the model freely generates answers, and the evaluator scores them according to the metric. Different datasets may be suited to different modes; refer to the dataset configuration for details.

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | Accuracy, number of correct samples / total samples; the most commonly used single-dataset metric |
| `exact_match` | Exact match; correct only if the output exactly matches the reference answer |
| `naive_average` | Simple average; arithmetic mean of subset metrics within a group |
| `weighted_average` | Weighted average; weighted by subset sample size |

Single datasets typically output only `accuracy` or `exact_match`; multi-subject datasets (such as C-Eval) additionally output grouped and overall average metrics in the summary stage for cross-comparison.

### Execution Workflow

```text
① Load model config + dataset config + summarizer
        ↓
② Create experiment directory
        ↓
③ Inference stage: batch-request the tested service by dataset, collect answers
        ↓
④ Evaluation stage: score item by item according to dataset metrics
        ↓
⑤ Summary stage: generate summary (txt / csv / md)
```
