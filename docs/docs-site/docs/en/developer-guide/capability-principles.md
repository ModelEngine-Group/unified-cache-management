# How caching works

A prompt prefix can reuse previously computed KV under compatible execution conditions. UCM separates block matching, loading and saving so the engine can reduce Prefill computation when cached data is available.

## From tokens to blocks

In the regular vLLM path, the Connector generates chained block identifiers from the token sequence at the cache granularity. Identifiers encode the preceding prefix: the same text fragment in a different context does not imply interchangeable KV. This persistence path handles complete blocks; the engine still handles the final partial block.

The Connector also incorporates model and execution settings into identifiers and layout handling, including model directory name, parallel configuration, dtype and rank. A matching directory name does not replace weight and tokenizer version management. Sharing instances require compatible weights, tokenizers, block sizes, KV layouts and execution settings; isolate incompatible configurations in separate storage namespaces.

## Lookup budgets compute; loading supplies data

The Scheduler already knows how many tokens are covered by engine memory. The Connector queries the following contiguous external blocks and converts the match into reusable tokens. `lookup_on_prefix()` returns the index of the last contiguous hit, or `-1` when the first block misses. Its result is not a block count.

A block found during lookup can be missing or fail to load later. The Worker therefore waits for transfer completion and reports errors after the Scheduler has observed a hit. The current implementation may also retain some token recomputation to satisfy execution and hook requirements. External matching and actual skipped computation are distinct quantities.

Hybrid layouts may use `lookup_on_reverse()` to find the rightmost existing state block. Follow the model-specific Connector instead of applying contiguous-prefix semantics to every layout.

## How layered storage moves KV

`Cache|Posix` loads from a ready host buffer or has Posix fill that buffer from files before transferring data to the device. Saving waits for the required device-compute synchronization, moves device data into host buffers and submits filesystem writes.

Cache hides buffer management, Posix hides filesystem I/O, and the Connector provides layouts and device addresses. Layerwise transfer can overlap the next layer's load with the current layer's compute, provided each layer waits before using its data. See the [request lifecycle](request-lifecycle.md).

Host buffers can be shared or allocated per worker. Backends own capacity reclamation; a configured storage budget does not reserve filesystem space. See the [filesystem pipeline](cache-configuration/pipeline.md) for deployment settings.

## When reading cache is worthwhile

External reuse replaces part of Prefill with lookup, reading, transfer and synchronization. End-to-end latency can improve only when the avoided compute outweighs those costs. Prefix repetition, block granularity, storage bandwidth, host-cache state and service queuing all affect the outcome.

[Verify external reuse](../user-guide/observability/verify-cache.md) first, then compare TTFT, throughput and correctness under the same input and concurrency. The [metric reference](../user-guide/observability/metrics-reference.md) distinguishes query tokens, transfer shards and byte counters.

## Source reading route

- `ucm/integration/vllm/ucm_connector.py`: block identifiers, scheduling matches and Worker transfer.
- `ucm/store/ucmstore_v1.py`: lookup and task contracts.
- `ucm/store/pipeline/connector.py`: stage composition and the Python/native boundary.
- `ucm/store/cache/cc/load_queue.cc` and `ucm/store/posix/`: host-buffer waits and backend I/O.
