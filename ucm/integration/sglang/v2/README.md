# SGLang Store v2 scaffold

The first target is GLM-5.2 DSA. This directory is not a runnable linker backend.

- `linker.py`: `UcmDirectLinker` initializes Runtime from `extra_config`.
  The first lookup registers the request and selects a branch. Layerwise Get
  reuses that registration. The load thread submits ahead across layers
  and publishes readiness in layer
  order. It closes the Get windows before
  releasing the registrations. Dump queues one full-object Put job per node
  and publishes results in submission order after CompletePut. Multiple Dump
  jobs can be in flight. Load and Dump wait on device events and prepare
  payloads on separate threads. A shared executor submits and polls Store
  jobs, delivering results by job ID. Device prefetch remains pseudocode.
- `runtime.py`: initializes one StoreAssembler and obtains both Scheduler and
  Worker interfaces. Methods directly forward native arguments and results.
  There are no locks, request/window mirrors, job allocation, completion caches,
  polling loops or shutdown state machines.
- `device_layout.py`: retained DSA layout prototype (KV + indexer in one FA
  object per logical page). The linker uses its object IDs for lookup.

Runtime reads the YAML path from `extra_config["UCM_CONFIG_FILE"]`, as in V1.
The YAML must contain exactly one `ucm_connectors` entry; its
`ucm_connector_config` dictionary is passed directly to StoreAssembler.
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
is present). This assumes the single-host group shares a Store view and can
read its peers' writes. Ranks with no pages still acknowledge the node in order.

The local built-in Cache and Posix still return Unsupported for Setup, Load
and Dump.
Real device I/O requires a working V2 Store plugin.

`test/test_sglang_v2_protocol.py` now checks only assembler ownership and thin
interface forwarding. The previous executable linker lifecycle/round-trip tests
were removed with that implementation; these tests do not validate device IO.
