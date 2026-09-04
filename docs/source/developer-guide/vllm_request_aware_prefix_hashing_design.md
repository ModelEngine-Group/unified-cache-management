# vLLM Integration Request-Aware Prefix Hashing Design

## Document status

- Status: Implemented
- Scope: UCM vLLM integration
- Primary objective: make external KV prefix-cache keys request-semantic-aware and reproducible across processes without relying on `PYTHONHASHSEED`

## 1. Background

UCM currently computes external KV-cache block IDs independently from vLLM. The
current algorithm chains the parent block hash with the token IDs of the current
block:

```text
block_hash[i] = H(block_hash[i - 1], block_token_ids[i])
```

This is sufficient for text-only requests whose KV state is fully determined by
token IDs. It is not sufficient for requests where the same token sequence can
produce different KV states.

Multimodal models commonly expand an image into repeated media placeholder token
IDs. Two different images can therefore have identical placeholder-token prefixes.
When UCM hashes only token IDs, an old small-image KV prefix can be reported as a
hit for a later large-image request. The remaining large-image blocks are then
computed from the new image embeddings, causing one inference to contain KV
features from both images.

vLLM avoids this problem by adding request-semantic data to its block hash. The
current extra keys include:

- multimodal feature identifier and its position relative to the block;
- LoRA identity;
- request cache salt;
- prompt-embedding content hash.

UCM must include the same request semantics while preserving its own deterministic,
external-store-oriented key chain.

## 2. Goals

This design has the following goals:

1. Different multimodal inputs must not share KV blocks solely because their
   placeholder token IDs are equal.
2. The same request and deployment configuration must produce the same UCM block
   IDs across processes and service restarts.
3. UCM must not depend on vLLM's process-local `NONE_HASH` or require users to set
   `PYTHONHASHSEED`.
4. UCM must reuse vLLM's definition of request hash extra keys instead of
   maintaining a multimodal-only copy.
5. Direct, LayerWise, CP, Mock, Lite, HLA/Mamba, and FAWA must use one common
   request-aware hashing implementation.
6. Existing block sizing, cache lookup, load, dump, rank scoping, and physical
   layout behavior must remain unchanged.
7. Unsupported request semantics must fail closed: UCM may miss, but it must not
   load or persist a token-only key for a request whose KV state depends on
   additional data.

## 3. Non-goals

The following work is intentionally excluded:

- reusing `request.block_hashes` as UCM store keys;
- changing the UCM Store `BlockId` width;
- replacing the current MD5/pickle encoding in `RequestHasher`;
- changing the model/config/rank namespace currently constructed by
  `RequestHasher`;
- optimizing HLA by computing one shared fine-grained semantic chain for all
  Full Attention groups;
- adding request-aware hashing to CacheBlend chunk-local keys;
- implementing the unfinished PD Connector;
- changing Inference Monitor behavior;
- moving the general `Config` class from `ucm.utils`.

FAWA participates in the common API refactor, but no FAWA-specific multimodal
behavior is introduced. DeepSeek V4 does not currently expose a multimodal request
path in the supported configuration.

## 4. Design principles

### 4.1 Separate request semantics from process state

UCM continues to calculate its own deterministic hash chain. It does not reuse
vLLM's final block hashes because those hashes are rooted in a process-specific
random value when `PYTHONHASHSEED` is unset.

### 4.2 Let vLLM define KV-affecting request semantics

UCM calls vLLM's `generate_block_hash_extra_keys()` for each UCM hash block. This
keeps multimodal, LoRA, cache-salt, and prompt-embedding semantics aligned with
vLLM.

### 4.3 Keep physical cache identity in UCM

`RequestHasher` continues to include the existing UCM model, dtype, parallelism,
rank, speculative-decoding, and sparse-attention metadata. Worker-side rank key
derivation remains unchanged.

### 4.4 One implementation, multiple block sizes

The request hashing loop belongs to `RequestHasher`. Connectors bind a reusable
closure with their required block size and initial parent hash.

## 5. Component placement

Move `RequestHasher` out of `ucm_connector.py` into:

```text
ucm/integration/vllm/request_hasher.py
```

The class is vLLM-specific because it consumes vLLM `Request` objects and calls a
vLLM block-extra-key helper. It should therefore remain under the vLLM integration,
not under a framework-neutral `ucm.integration.utils` package.

The general `Config` class remains in:

```text
ucm/utils.py
```

Moving `Config` into an integration package would create an invalid dependency
from core/sparse modules back into the framework-adapter layer.

## 6. RequestHasher API

`RequestHasher` retains its current generic object-hashing interface and gains a
request block-hasher factory.

```python
class RequestHasher:
    def __init__(self, vllm_config, rank_id):
        # Existing model/config/rank namespace construction.
        ...
        self.seed = self("UCM_HASH_SEED")

    def __call__(self, input_data) -> bytes:
        # Existing meta + pickle + MD5 implementation.
        ...

    def make_request_block_hasher(
        self,
        block_size: int,
        initial_hash: bytes | None = None,
    ) -> Callable[["Request"], list[bytes]]:
        ...
```

The returned closure computes:

```text
parent[0] = initial_hash or RequestHasher.seed

block_hash[i] = RequestHasher(
    parent[i],
    block_token_ids[i],
    block_extra_keys[i],
)

parent[i + 1] = block_hash[i]
```

Reference behavior:

```python
def make_request_block_hasher(self, block_size, initial_hash=None):
    root = initial_hash if initial_hash is not None else self.seed

    def hash_request(request):
        token_ids = request.all_token_ids
        parent = root
        curr_mm_idx = 0
        hashes = []

        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            if end > len(token_ids):
                break

            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request,
                start,
                end,
                curr_mm_idx,
            )
            parent = self(
                (
                    parent,
                    tuple(token_ids[start:end]),
                    extra_keys,
                )
            )
            hashes.append(parent)

        return hashes

    return hash_request
```

`parent` and `curr_mm_idx` must be initialized inside `hash_request()`. Placing
either variable in the outer factory scope would leak state between requests and
make one reusable closure unsafe.

Partial blocks are not hashed, matching current UCM behavior.

## 7. Extra-key compatibility layer

`request_hasher.py` owns a small compatibility wrapper around vLLM's
`generate_block_hash_extra_keys()`.

The wrapper must follow these rules:

1. Delegate to vLLM when the helper is available.
2. Preserve and return the multimodal cursor so each request/group scan remains
   linear.
3. If the helper is unavailable, return `None` only when the request has no
   multimodal features, LoRA, cache salt, or prompt embeddings.
4. If the helper is unavailable for a request with additional semantics, raise a
   dedicated request-hash exception.
5. Identifier validation is delegated to vLLM. If the helper reaches a
   multimodal feature with a missing identifier, its failure is converted to a
   request-hash error; token-only fallback is forbidden.

The connector catches this dedicated exception and returns an external miss without
creating dump metadata. The request is recomputed normally and no invalid external
KV entry is written.

## 8. Hash seed and cache compatibility

The seed remains unchanged:

```python
self.seed = self("UCM_HASH_SEED")
```

No seed version is required. The block input changes from:

```text
(parent_hash, block_token_ids)
```

to:

```text
(parent_hash, block_token_ids, extra_keys)
```

Even when `extra_keys` is `None`, the serialized tuple is different, so new block
IDs cannot match old token-only block IDs. Existing cache data naturally becomes a
cold miss.

This design preserves reproducibility under the current UCM operational contract:
the same UCM/Python software, model configuration, and rank configuration produce
the same keys. Stable hashes across Python or pickle protocol changes are not part
of this change.

## 9. Connector changes

### 9.1 Direct Connector

Remove the existing token-list-based `generate_hash()` implementation. Bind one
request closure after `hash_block_size` and `RequestHasher` are initialized:

```python
self.request_block_hasher = self.request_hasher.make_request_block_hasher(
    self.hash_block_size
)
```

Lookup becomes:

```python
ucm_block_ids = self.request_block_hasher(request)
```

Store lookup, request metadata, load/dump planning, and rank-specific key derivation
are unchanged.

### 9.2 LayerWise and Mock Connectors

Both inherit Direct's lookup and request hasher. No independent hash implementation
is required.

### 9.3 CP Connector

CP keeps its current relationship:

```text
hash_block_size = configured base block size
physical/scheduler block size = base block size * cp_world_size
```

After CP reconstructs `RequestHasher` with its normalized TP/rank configuration, it
must rebind `request_block_hasher`. Existing `[current_rank::cp_world_size]` key
slicing is unchanged.

### 9.4 Lite Connector

Remove Lite's duplicated token-only `generate_hash()` implementation. Lite creates
and invokes the same RequestHasher closure as Direct.

### 9.5 HLA Full Attention groups

Change the group API from token-list input to Request input:

```python
compute_block_hashes(group, request)
compute_all_group_block_ids(request)
```

Each Full Attention group binds a closure using its own block size and group seed:

```python
group.block_hasher = request_hasher.make_request_block_hasher(
    block_size=group.block_size,
    initial_hash=group.seed,
)
```

Each group scans the Request independently. This preserves the existing group hash
structure and block mapping. Token serialization was already repeated per Full
Attention group; this change adds only extra-key generation to those existing
passes.

### 9.6 HLA Mamba align groups

Mamba align groups continue to produce empty per-block placeholders. Their persisted
state key remains derived from:

```text
Mamba group seed
+ state tag
+ sequence length
+ primary Full Attention prefix hash
```

No extra-key scan is needed for a Mamba group. Once the primary Full Attention
prefix hash includes request extra keys, a different image, LoRA, salt, or prompt
embedding also produces a different Mamba state key.

LCM alignment, two-stage lookup, reverse state lookup, and HLA load/dump dispatch
remain unchanged.

### 9.7 FAWA

FAWA binds the common closure after selecting its canonical hash block size:

```text
GPU default: 256
Ascend: configured base block size * C4 compression ratio
```

FAWA then replaces its inherited call with:

```python
canonical_hashes = self.request_block_hasher(request)
```

For current DeepSeek V4 text requests, `extra_keys` is normally `None`. Canonical
block boundaries, FA/WA store separation, prefix lookup, reverse WA lookup, and
load/dump mapping are unchanged. Concrete hash values change because the hash tuple
now contains a third element, causing the expected cold-cache transition.

### 9.8 CacheBlend compatibility

CacheBlend is explicitly limited to plain-text request semantics in this change.
Its ordinary prefix path uses the common request-aware closure, while chunk-local
keys reset the parent chain but use the same three-field input with
`extra_keys=None`. This keeps plain-text chunk build and lookup keys identical.
A multimodal-capable model may still start and serve text requests; CacheBlend
logs a startup warning instead of rejecting the model. Requests containing
multimodal features, LoRA, cache salt, or prompt embeddings bypass CacheBlend
external caching because reusable chunk-relative semantics for those inputs
require a separate design.

## 10. Error handling

Request hashing errors are handled at scheduler-side external lookup entry points.

On an unsupported or invalid semantic request:

1. log the request ID and reason;
2. return zero external hit tokens;
3. do not create connector request metadata;
4. do not dump external KV for that request.

HLA treats a hash failure in any group as a failure for the whole external lookup.
It must never continue with a partial set of group keys.

## 11. Performance considerations

The multimodal path does not reload an image, run its processor, or hash raw pixels.
It consumes the identifier already created during vLLM input processing.

For text-only requests, the additional helper checks and hashing of a `None` tuple
element are negligible relative to existing tuple serialization and block hashing.

For HLA with multiple Full Attention groups, request tokens and extra keys are
processed once per group. This preserves the current architecture and is accepted
for the first correctness-focused implementation. A shared fine-grained semantic
chain is a future optimization.

Prompt embeddings may require additional hashing when UCM and vLLM use different
block boundaries. When boundaries match, vLLM's request-level prompt-embedding hash
cache can be reused.

The computed UCM block IDs remain stored in connector request metadata so lookup,
load, and dump do not recompute the request hash.

## 12. Testing strategy

### 12.1 Common RequestHasher tests

- identical tokens and identical extra keys produce identical block IDs;
- identical placeholder tokens with different multimodal identifiers diverge at
  the first image-overlapping block;
- text blocks before the first multimodal feature remain reusable;
- identical multimodal content at different positions produces different keys;
- divergence propagates through every subsequent chained block;
- different LoRA, cache salt, and prompt embeddings produce different keys;
- missing multimodal identifiers fail closed when the helper reaches their
  overlapping hash block; un-hashed tails are not pre-scanned by UCM;
- incomplete final blocks are not hashed;
- repeated calls to one closure do not share parent or multimodal cursor state.

### 12.2 Connector tests

- Direct, LayerWise, Mock, Lite, CP, and FAWA call the common request closure;
- CP preserves fine-grained hash count and rank slicing;
- FAWA preserves canonical block count and FA/WA dispatch mapping when extra keys
  are absent;
- one-FA-group HLA requests produce different Mamba state keys for different
  multimodal identifiers;
- multi-FA-group HLA requests generate semantic keys at each group's own block
  size;
- an HLA hash error in any group disables the entire external hit.

### 12.3 Reproducibility and migration tests

- processes with different or unset `PYTHONHASHSEED` produce identical UCM keys
  under the same deployment configuration;
- old two-element token-only hash input and new three-element request-aware input
  produce different block IDs;
- a new process can read persisted keys written by another process running the
  same UCM/Python version and configuration.

## 13. Rollout

The new hash tuple invalidates all existing token-only keys. Deployment must treat
the change as a cold-cache rollout.

Recommended sequence:

1. disable multimodal external-cache traffic to old instances;
2. upgrade all cache producers and consumers to the request-aware implementation;
3. re-enable multimodal external caching;
4. monitor semantic hash failures, hash latency, and external hit rate;
5. allow normal Store GC to reclaim unreachable token-only entries.

Mixed-version deployment is not safe for multimodal external caching because old
instances can still query token-only keys.

## 14. Implementation files

Implemented changes:

```text
ucm/integration/vllm/request_hasher.py      new RequestHasher module
ucm/integration/vllm/ucm_connector.py      Direct/CP/Lite integration
ucm/integration/vllm/hla_connector.py      group Request input and closures
ucm/integration/vllm/hma_connector.py      FAWA common closure binding
ucm/integration/vllm/blend_connector.py    chunk-local compatibility
ucm/default_metrics_config.py              request-hash failure counter
docs/source/user-guide/metrics/metrics_list.md  counter documentation
test/test_ucm_request_hasher.py             semantic hash tests
test/test_ucm_hla_hash.py                   HLA/Mamba tests
test/test_ucm_blend_hash.py                 text-only chunk hash compatibility
test/test_ucm_connector_metrics.py          counter registration test
```

Existing imports of `RequestHasher` from `ucm_connector.py` must move to the new
module. `Config` imports remain unchanged.

## 15. Future extensions

### 15.1 Stable serialization and stronger digest

Replace pickle/MD5 with a canonical encoding and SHA-256 or BLAKE3, truncating the
result to the Store's 16-byte `BlockId`. This would make the persisted key protocol
less dependent on Python implementation details.

### 15.2 Stronger model and layout fingerprint

Replace the current model-basename namespace with a stable model revision or
explicit cache namespace and include every KV-layout-affecting configuration.

### 15.3 Official vLLM connector API

Promote request extra-key generation or a stable semantic block descriptor to a
public vLLM KV Connector API. This removes reliance on an internal helper location.

### 15.4 Shared HLA semantic chain

Generate request semantics once at the greatest common divisor of Full Attention
group block sizes, then derive group keys at each group boundary. This avoids
repeated token serialization and extra-key scanning for multi-group HLA models.

### 15.5 FAWA multimodal support

If DeepSeek V4 gains multimodal support, validate that the FAWA canonical hash
block size and multimodal feature ranges preserve the intended FA/WA boundary
semantics and add dedicated regression tests.

### 15.6 CacheBlend request semantics

Design chunk-local semantic hashing with a reset parent chain, original request
feature ranges, chunk-relative positions, and cache-salt isolation.

### 15.7 Store-side semantic validation

Persist a full semantic digest and schema metadata alongside each 16-byte Store key.
Validate the digest before loading KV so future key-construction bugs degrade to a
miss rather than loading semantically incompatible KV.

### 15.8 Safe text-prefix truncation without the vLLM helper

When `generate_block_hash_extra_keys()` is unavailable, derive the earliest
multimodal insertion offset from `request.mm_features`, align it down to the UCM
hash block boundary, and hash only the complete text blocks before that point.
The block overlapping the multimodal insertion and every chained successor must
be excluded from lookup and dump. HLA must further align the common cutoff to its
group LCM and restrict Mamba state reuse to boundaries before the cutoff.

This partial fallback applies only when the unsafe semantics have a reliable
position, such as multimodal features. Request-wide semantics such as LoRA or
cache salt must continue to fail closed from block zero; prompt embeddings also
fail closed unless their affected token range can be established safely.

## 16. Acceptance criteria

The change is complete when:

1. no included connector contains an independent token-only request hash loop;
2. all included connector block IDs are produced by `RequestHasher` closures;
3. different multimodal inputs cannot share image-overlapping or subsequent prefix
   block IDs;
4. HLA Mamba state keys inherit the corrected primary prefix semantics;
5. UCM external keys remain reproducible without `PYTHONHASHSEED`;
6. unsupported semantic requests fail closed;
7. existing block sizing and physical load/dump mappings remain unchanged;
8. the specified unit and regression tests pass.
