# Decode KV writeback

See [the Chinese implementation and validation guide](decode_writeback_zh.md)
for detailed behavior, logs and the acceptance matrix.

This change targets vLLM/vLLM-Ascend 0.30, non-layerwise GLM5.2 replicated MLA,
DeepSeek V4 and V4.1 FAWA, with PP=PCP=DCP=1. Hardware/E2E validation is required;
CPU planner/layout tests are not a claim of validated NPU support.

`save_decode_cache` is the only new feature flag (boolean, default false).
Put it in the existing UCM YAML or override it through
`kv_connector_extra_config` alongside `UCM_CONFIG_FILE`. Both endpoints use
`use_layerwise: false` and compatible weights, KV formats, blocks and shared storage.

When enabled, all three roles retain saving of computed Prefill/re-prefill KV
and also save Decode KV. This includes consumer-side Prefill after failover
and recomputation of prompts plus generated history after preemption. Role alone
does not suppress writes; complete-unit, source-safety and deduplication checks
still apply. This feature does not implement failover routing. A reader need
not enable the flag. Disabled behavior retains the original prompt-only saving.
There is no `consumer_is_to_put` option.

FAWA's existing `load_tokens_threshold` defaults to 2048. For short functional
tests, temporarily set `load_tokens_threshold: 0` and `persist_token_threshold: 0`;
otherwise a successful Decode write may intentionally not be queried/loaded.
Restore or tune the threshold separately for performance tests.

Request progress follows authoritative computed/committed tokens, not new-block
allocation. Incremental hashes use the same parent and extra-key chain as lookup.
MRV1/MRV2 recovery captures the complete re-prefill range and resets the job
generation. All TP participants must confirm a job before its saved range moves;
failed/partial jobs retry with existence deduplication.

Replicated target records contain every persistent MLA/indexer component and
exclude draft/private/transient state. Their versioned namespace omits execution
TP and MTP options but retains compatible target record semantics. This cold-starts
old cache objects. Dense-GQA heterogeneous TP and arbitrary format conversion are
out of scope. V4.1 retains whole-page records and requires compatible page payloads.

FAWA writes FA data plus one final WA snapshot. For MTP steps crossing a hash
boundary, it selects retained complete boundary pages, never the currently
written partial page. A remainder reaching any WA page span, or an expired
source block, skips that opportunity. Exact-boundary scheduling is not required. The
guard includes scheduled draft writes, not only committed tokens; a large MTP
span may prevent safe checkpoints and should be reduced for validation. The
worker waits for both stores before the next forward can overwrite source pages.
FA/WA backend BlockIDs are family-namespaced for lookup/load/save/prefetch, so
the same logical hash cannot collide in one global Mooncake pool. Debug records
include `backend_key_fa` and `backend_key_wa`; native Mooncake may add `_0`.
This correctness fence can affect TPOT; measure it as well as TTFT. No historical
16384-token transfer constant is imposed.

Set `UCM_LOG_LEVEL=debug` for exact keys, token ranges, rank, generation/job ID,
save skips/plans/completion and next-turn lookup/load results. Plans are not PUT
success. Disable verbose logs for final performance measurements.

```bash
python benchmarks/verify_decode_writeback.py \
  --url http://ROUTER:8000/v1/chat/completions --model MODEL \
  --min-tokens 512 --max-tokens 2048 --report /tmp/reuse.json
```

The verifier carries actual output and reasoning fields into turn 2 and never
uses `ignore_eos=true`. Verify rendered token-prefix identity for the model's chat
template. Require successful Decode keys **beyond the original prompt**, matching
Prefill lookup and load, and exclude local HBM hits. Compare with an uncached
reference on the same reader TP. Test both TP1↔TP2 and TP2↔TP4 directions,
MTP/no-MTP, partial/zero/full acceptance, preemption and Store/rank failures.
Client TTFT or usage alone is not proof of correct KV reuse.
