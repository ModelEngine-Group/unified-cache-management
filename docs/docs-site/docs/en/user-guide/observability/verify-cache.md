# Verify external cache

After starting the service, confirm that an initial request saves reusable blocks and a new process loads them with correct output. This separates external reuse from cache inside the engine process.

## Prepare a reproducible run

- Use a dedicated test namespace instead of clearing shared production cache.
- Fix model/tokenizer revisions, KV dtype, parallel settings, block size and UCM configuration.
- Use identical prompts spanning several complete blocks. Fix sampling settings and retain requests and outputs.
- Enable metrics and verify collection. [Metrics setup](metrics.md) covers vLLM; use the [SGLang quickstart](../quick_start/index.md#sglang) for SGLang observations.

## First request: compute and save

1. Check service health and model endpoints, then send the request.
2. Wait for cache saves to finish and inspect UCM backend selection, save activity and errors.
3. For filesystems, check committed KV files. Health probes also create small objects, so correlate them with logs.

Submission and timing counters establish that a path ran. Combine completion, errors and later replay to establish successful saving.

## Keep external data and replay after restart

Stop the test service while retaining external data, restart with the same settings and replay the prompt. Restart removes engine-process cache. If shared host cache survives its clients, check whether it remains populated and select observations for the storage layer being measured.

For vLLM with `Cache|Posix`, use these observations together:

| Observation | Purpose |
| --- | --- |
| `ucm:ucm_hit_tokens_total` increases within this run | Scheduler lookup found reusable external tokens |
| Load tasks complete without corresponding errors | Worker obtained the required data |
| Posix task activity and filesystem reads | Establish filesystem access when that is the target layer |
| Output or task score matches the baseline | Verify replay correctness |

`cache_posix_load_success_shards_total` counts shards delivered after waiting for host buffers; it does not uniquely identify Posix. Posix task counters can also include failed tasks. See [metric semantics](metrics-reference.md). Counters reset on restart, so compare deltas within one run.

## Cross-node and PD checks

For sharing, replay on another instance with the same store and a compatible layout. Shared-store PD also requires Prefill writes to be visible to Decode. Transport with UCM requires separate checks for P/D handoff and Prefill UCM reuse. See [PD Disaggregation](../capabilities/pd-disaggregation/index.md).

## Investigate unexpected results

For no external hits, check prompt tokens, model/layout compatibility, complete blocks and the actual namespace. For hits followed by failed loads, check health, visibility, transfer errors and resource budgets. Continue with [troubleshooting](../../reference/troubleshooting.md).

Once reuse is correct, compare cold runs, external replay and warm engine cache separately. Report actual hits, read paths, latency and throughput rather than treating one faster response as cache verification.
