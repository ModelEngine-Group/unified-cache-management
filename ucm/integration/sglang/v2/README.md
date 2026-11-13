# SGLang Store v2 scaffold

The first target is GLM-5.2 DSA. This directory is not a runnable linker backend.

- `linker.py`: `UcmDirectLinker` initializes Runtime from `extra_config`.
  The first lookup registers the request and selects a branch. Layerwise Get
  reuses that registration. The load thread submits ahead across layers
  and publishes readiness in layer
  order. It closes the Get windows after the load. The request route remains
  registered until retirement and completion of its queued Puts.
  Dump groups known pages by their associated request IDs and puts each group
  through the original request route. Unassociated pages use the default
  offload request. A node completes after all its Put windows close.
  Store offload runs independently: scheduler completion checks and Put
  completion wake an instance-level drain on the I/O executor. It admits one
  batch of plans at a time and returns each receipt after execution. Slow or
  failed lower-tier writes do not delay or change node results. Multiple Dump
  jobs can be in flight. Load and Dump wait on device events and prepare
  payloads on separate threads. A shared executor submits and polls Store
  jobs, delivering results by job ID. Device prefetch remains pseudocode.
- `request_hooks.py`: associates the page-aligned radix insert with its request
  before SGLang dispatches write-through. Request retirement and abandonment
  release those associations.
- `runtime.py`: initializes one StoreAssembler and obtains both Scheduler and
  Worker interfaces. Methods directly forward native arguments and results.
  There are no locks, request/window mirrors, job allocation, completion caches,
  polling loops or shutdown state machines.
- `device_layout.py`: retained DSA layout prototype (KV + indexer in one FA
  object per logical page). The linker uses its object IDs for lookup.

Runtime reads the YAML path from `extra_config["UCM_CONFIG_FILE"]`, as in V1.
The YAML must contain exactly one `ucm_connectors` entry; its
`ucm_connector_config` dictionary is passed to StoreAssembler, with
`llm_role: both` defaulted for each tier that does not specify a role.
The V1 connector factory/name is not used to create a V2 assembler.

```yaml
ucm_connectors:
  - ucm_connector_config:
      # Native StoreAssembler configuration goes here.
```

Query keys use the binding's packed bytes. Load submits native payloads with
per-fragment offsets and waits with one job ID per call. The linker serializes
Scheduler calls between lookup and the load worker.

Load waits for the index event before preparing device addresses. Set
`forward_event_to_store: true` in the linker extra config only when the selected
Store backend waits on CUDA prerequisite events. Load then passes the forward
event handle to Store and retains the event until the jobs finish. By default,
or if there is no usable handle, the worker waits on the forward event before
submitting Get jobs.

Rank-replicated layouts partition each node by complete logical pages across
the attention TP cache group (or the TP cache group when no attention group
is present). Partitioned Dump requires one Store branch shared by the group.
The linker rejects a multi-branch topology when partitioning is active because
each rank registers an independent Put request. Ranks with no pages still
acknowledge the node in order.

Store offload plans own cache-tier pins, not node device pages. Rejected plans
are receipted without waiting for a nonexistent job; accepted plans are receipted
only after completion, including failure. Failures are logged without replaying
tokens. The protocol does not currently define automatic failed-offload retry.
One batch bounds concurrent plan issuance, but the Store controls each batch's
size; a byte-level pin budget is not exposed by this interface.

`reset` and `close` drain accepted offloads and remaining committed pending plans
while the shared I/O executor is still alive. Callers must quiesce producers
before reset/close; close rejects new lookup/load/offload calls. Receipt or
preparation errors propagate from the drain rather than reporting a successful
shutdown with unreleased pins or pending work. Direct Posix Put still waits for
its own write; cache-tier Put success is not a durability/fsync guarantee.

The local built-in Cache and Posix still return Unsupported for Setup, Load
and Dump.
Real device I/O requires a working V2 Store plugin.

`test/test_sglang_v2_protocol.py` checks assembler ownership and thin interface
forwarding. `test/test_sglang_v2_offload.py` exercises the real linker executors
with a controllable Store: independent node/disk completion, FIFO node results,
shared Get completion routing, rejected/failed plans and reset/close draining.
These tests do not validate native plugins or device I/O.
