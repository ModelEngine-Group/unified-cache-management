# vLLM Integration Request-Aware Prefix Hashing Design

## 1. Background

UCM computes external KV-cache block IDs independently from vLLM. The original
algorithm chains the parent block hash with the token IDs of the current block:

```text
block_hash[i] = H(block_hash[i - 1], block_token_ids[i])
```

This is sufficient for text-only requests whose KV state is fully determined by
token IDs. It breaks down when the same token sequence can produce different KV
states.

Multimodal models commonly expand an image into repeated media placeholder token
IDs. Two different images can therefore have identical placeholder-token prefixes.
When UCM hashes only token IDs, a previously written small-image KV prefix can be
reported as a hit for a later large-image request, causing one inference to
contain KV features from both images.

vLLM avoids this by adding request-semantic data (multimodal identifier +
position, LoRA identity, cache salt, prompt-embedding content hash) to its block
hash. UCM now folds the same semantics into its own hash chain while preserving
its deterministic, external-store-oriented key derivation.

## 2. Design

### 2.1 RequestHasher API

`RequestHasher` is extracted from `ucm_connector.py` into a dedicated module:

```text
ucm/integration/vllm/request_hasher.py
```

It retains its existing generic object-hashing interface (model + dtype + TP +
rank + speculative + sparse metadata → MD5) and gains a request block-hasher
factory:

```python
class RequestHasher:
    def __init__(self, vllm_config, rank_id):
        # ... existing meta namespace construction ...
        self.seed = self("UCM_HASH_SEED")

    def __call__(self, input_data) -> bytes:
        # meta_bytes + pickle.dumps(input_data) → MD5 digest
        ...

    def make_request_block_hasher(
        self,
        block_size: int,
        initial_hash: bytes | None = None,
    ) -> Callable[["Request"], list[bytes]]:
        ...
```

`make_request_block_hasher` returns a reusable closure bound to a specific
block size and chain root. Each call to the closure scans one `Request` and
returns a list of 16-byte block hashes.

### 2.2 Hash chain

The block hash input changes from a 2-tuple to a 3-tuple:

```text
parent[0] = initial_hash or RequestHasher.seed

block_hash[i] = RequestHasher(
    parent[i],
    block_token_ids[i],
    extra_keys[i],
)

parent[i + 1] = block_hash[i]
```

`parent` and the multimodal cursor `curr_mm_idx` are initialized inside the
closure body so each call is independent — sharing them in the outer factory
scope would leak state between requests. Partial blocks are not hashed, matching
existing UCM behavior.

### 2.3 Extra-key compatibility layer

`request_hasher.py` wraps vLLM's `generate_block_hash_extra_keys()` with the
following rules:

1. Delegate to vLLM when the helper is available, preserving and returning the
   multimodal cursor so each scan remains linear.
2. If the helper is unavailable, return `None` only when the request has no
   multimodal features, LoRA, cache salt, or prompt embeddings.
3. If the helper is unavailable for a request with additional semantics, raise
   `RequestHashError` — token-only fallback is forbidden.
4. Any exception from the vLLM helper (e.g. missing multimodal identifier) is
   converted to `RequestHashError`.

### 2.4 Seed and cache compatibility

The seed is unchanged:

```python
self.seed = self("UCM_HASH_SEED")
```

Because the block input now contains a third element (`extra_keys`), new block IDs
cannot match old token-only block IDs — even when `extra_keys` is `None`, the
serialized tuple differs. Existing cache data naturally becomes a cold miss.
Reproducibility is preserved: the same UCM/Python software, model configuration,
and rank configuration produce the same keys across processes and restarts,
without depending on `PYTHONHASHSEED`.

## 3. Connector changes

### 3.1 Direct / LayerWise / Mock

Remove the token-list-based `generate_hash()` method. After `hash_block_size`
and `RequestHasher` are initialized, bind one request closure:

```python
self.request_block_hasher = self.request_hasher.make_request_block_hasher(
    self.hash_block_size, self._seed
)
```

Lookup becomes `ucm_block_ids = self.request_block_hasher(request)`. Store
lookup, request metadata, load/dump planning, and rank-specific key derivation
are unchanged. LayerWise and Mock inherit Direct's implementation.

### 3.2 CP

CP keeps `hash_block_size = base block size` and
`physical block size = base block size * cp_world_size`. After reconstructing
`RequestHasher` with its normalized TP/rank configuration, it rebinds
`request_block_hasher`. Existing `[current_rank::cp_world_size]` key slicing is
unchanged.

### 3.3 Lite

Remove Lite's duplicated `generate_hash()`. Lite creates and invokes the same
`RequestHasher` closure as Direct.

### 3.4 HLA

Each Full Attention group binds a closure using its own block size and group
seed:

```python
group.block_hasher = request_hasher.make_request_block_hasher(
    block_size=group.block_size,
    initial_hash=group.seed,
)
```

The group API changes from token-list input to `Request` input. Each group scans
the request independently, preserving the existing group hash structure and block
mapping.

Mamba align groups continue to produce empty per-block placeholders. Their
persisted state key is derived from the primary Full Attention prefix hash, so
once that prefix includes request extra keys, a different image/LoRA/salt/prompt
embedding also produces a different Mamba state key.

### 3.5 FAWA

FAWA binds the common closure after selecting its canonical hash block size
(GPU default: 256; Ascend: base block size × C4 compression ratio). For current
DeepSeek V4 text requests, `extra_keys` is normally `None`. Canonical block
boundaries, FA/WA store separation, and load/dump mapping are unchanged.

### 3.6 CacheBlend

CacheBlend is limited to plain-text semantics. Its prefix path uses the common
closure; chunk-local keys reset the parent chain but use the same three-field
input with `extra_keys=None`, keeping text chunk keys identical. Requests with
multimodal features, LoRA, cache salt, or prompt embeddings bypass CacheBlend
external caching.

## 4. Error handling

Request hashing errors are handled at scheduler-side external lookup entry
points. Connectors catch `Exception` broadly (not just `RequestHashError`) so
that any failure during hashing — whether from vLLM's extra-key helper,
`pickle.dumps`, or `request.all_token_ids` access — degrades to a miss rather
than crashing the vLLM scheduler:

```python
try:
    ucm_block_ids = self.request_block_hasher(request)
except Exception as e:
    logger.error(f"request {request.request_id} hash error. {type(e).__name__}: {e}")
    return 0, False
```

On failure the connector returns zero external hit tokens, does not create
connector request metadata, and does not dump external KV for that request. HLA
treats a hash failure in any group as a failure for the whole external lookup.

## 5. Implementation files

```text
ucm/integration/vllm/request_hasher.py      new RequestHasher module
ucm/integration/vllm/ucm_connector.py      Direct/CP/Lite integration
ucm/integration/vllm/hla_connector.py      group Request input and closures
ucm/integration/vllm/hma_connector.py      FAWA common closure binding
ucm/integration/vllm/blend_connector.py    chunk-local compatibility
test/test_ucm_request_hasher.py             semantic hash tests
test/test_ucm_hla_hash.py                   HLA/Mamba tests
test/test_ucm_blend_hash.py                 text-only chunk hash compatibility
```
