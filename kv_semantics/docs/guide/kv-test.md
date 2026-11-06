<!-- Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. -->
# kv-test command line tool

`kv-test` is a command line validation tool for the ASU KV client path. The
current implementation supports local smoke testing, basic request validation,
consistency checks, and simple benchmark metrics.

The tool reads one key-value config file. It loads the ASU client runtime
library through a small proxy, calls the ASU client config parser exported from
that library, then reads kv-test-specific options from the same file.

## Build and environment

`kv-test` is built from `ucm/transport/kv/kv-test/CMakeLists.txt`. The
`kv_client` and `kv_transport` shared libraries are built as separate
artifacts and loaded by `kv-test` at runtime with `dlopen`.

`kv-test` is included only when ASU support is enabled:

```bash
cmake -S . -B build-kv-test -DBUILD_UCM_ASU=ON -DBUILD_UCM_STORE=OFF -DBUILD_UNIT_TESTS=OFF -DRUNTIME_ENVIRONMENT=ascend -DBUILD_KV_CLIENT_PROVIDER_FAKE=ON
cmake --build build-kv-test --target kv_transport kv_client
cmake --build build-kv-test --target kv-test
```

Provider implementations are selected at build time:

| CMake option | Default | Extra dependency |
| --- | --- | --- |
| `BUILD_KV_CLIENT_PROVIDER_AICPU` | `OFF` | Reserved for the AICPU provider library. |
| `BUILD_KV_CLIENT_PROVIDER_FAKE` | `ON` | None. |
| `BUILD_KV_CLIENT_PROVIDER_AIV` | `OFF` | `libumc.a`, found through `KV_CLIENT_AIV_PROVIDER_ROOT`. |

The configured `transport.provider_type` must be built into `kv_transport`.
For example, real AIV testing needs:

```bash
cmake -S . -B build-kv-test -DBUILD_UCM_ASU=ON -DBUILD_UCM_STORE=OFF -DBUILD_UNIT_TESTS=OFF -DRUNTIME_ENVIRONMENT=ascend -DBUILD_KV_CLIENT_PROVIDER_FAKE=ON -DBUILD_KV_CLIENT_PROVIDER_AIV=ON -DKV_CLIENT_AIV_PROVIDER_ROOT=/path/to/aiv/provider
```

The same build can be started from any working directory with:

```bash
bash ucm/transport/kv/kv-test/build.sh
```

If the shared libraries are not discoverable from the dynamic linker search
path or from the default build-tree sibling directory, set
`asu.client_library_path` and `asu.transport_library_path` in the config file,
or export `KV_TEST_ASU_CLIENT_LIB` and `KV_TEST_ASU_TRANSPORT_LIB`.

The example environment script is:

```bash
source ucm/transport/kv/kv-test/set_kvtest_env.sh
```

Use `source` if the exported `KV_TEST_CONFIG` and `PATH` must remain visible in
the current shell. Running the script as `./set_kvtest_env.sh` only updates the
script process and does not update the caller's shell.

The bundled example config is:

```text
ucm/transport/kv/kv-test/kv_test.conf
```

The bundled example view file is:

```text
kv_semantics/kv_test/kv_client_view.conf
```

The sample config uses relative paths such as `view.config_path`,
`fake_backend.path`, and `output.path`. They are resolved against the process
working directory, not against the config file directory. The bundled examples
assume commands are run from the repository root.

## Command format

```bash
kv-test <command> [options]
kv-test <command> --help
kv-test --help
kv-test --version
```

`--configpath <path>` selects the config file for one run. If it is omitted,
`KV_TEST_CONFIG` is used. Commands other than `--help` and `--version` fail when
neither is set.

Naming rules:

- CLI commands and long options use kebab-case, for example `batch-store`,
  `power-cycle`, `--batch-size`, and `--read-ratio`.
- Config keys use dot-separated snake_case, for example `bench.read_ratio`,
  `limits.memory_max_bytes`, and `transport.asu_ids`.
- Compatibility spellings may exist in lower-level parsers, but kv-test help,
  docs, examples, and generated configs should only use the canonical forms
  above.

Supported commands:

| Command | Current behavior |
| --- | --- |
| `connect` | Initializes the ASU client and exits. |
| `config check` | Loads config, validates fixed kv-test behavior constraints, prints selected config values, and exits. |
| `version` | Prints `kv-test version <value>`, where the value is read from `version.ini`. |
| `store` | Stores all selected entries in one ASU client call in the current implementation. |
| `retrieve` | Retrieves all selected entries in one ASU client call in the current implementation. |
| `delete` | Deletes all selected keys in one ASU client call. |
| `exist` | Runs a per-key ASU query and prints existence summary. |
| `batch-store` | Stores all selected entries in one ASU client call. |
| `batch-retrieve` | Retrieves all selected entries in one ASU client call. |
| `power-cycle prepare` | Same execution path as `batch-store`/store-like commands. |
| `power-cycle verify` | Same execution path as retrieve-like commands and always performs value consistency checking. |
| `bench` | Runs an asynchronous, interval-driven benchmark for `store`, `retrieve`, `batch-store`, `batch-retrieve`, or `mix`. |

## Common options

Options can use either `--option value` or `--option=value`.

| Option | Description |
| --- | --- |
| `--configpath <path>` | Config path for this run. Overrides `KV_TEST_CONFIG`. |
| `--help`, `-h` | Prints general or command-specific help. |
| `--version` | Prints tool version. Does not accept positional arguments. |
| `--check` | Enables consistency checking where supported. |
| `--timeout <ms>` | Overrides `wait_timeout_ms` for this run. |
| `--output <path>` | Overrides `output.path`. |
| `--progress` | For `bench`, prints one progress line per measured second. |

## Key selection

At most one key selector can be used in a command:

| Selector | Description |
| --- | --- |
| `--key <key>` | Selects one key. |
| `--keys <k1,k2,...>` | Selects comma-separated keys. Empty items are rejected. |
| `--keys-file <path>` | Reads comma-separated and/or newline-separated keys from a file. Empty items are rejected. |
| `--count <n>` | Generates `n` keys as `<kv.key_prefix><index>`, starting at index `0`. |
| `--prefix <p> --key-start <n> --key-end <n>` | Generates keys in the closed interval `[key-start, key-end]` as `<prefix><index>`. |

`--prefix`, `--key-start`, and `--key-end` must be provided together.
`--key-start` must be less than or equal to `--key-end`.

When count-based generation is used, `kv.key_prefix` is required. For commands
that need values, `kv.value_size` is also required.

## Data generation

For normal commands, values are generated deterministically from:

- key
- seed
- value size

`delete` and `exist` are key-only commands and do not generate value buffers.

The generator uses a FNV-1a-style seed mix and SplitMix64 byte generation.
Consistency digests are CRC64-ECMA hex strings.

For `bench`, data generation is separate:

- key prefix defaults to `bench-key-` when `kv.key_prefix` is empty
- value bytes are filled from `(index + byteIndex + seed) & 0xFF`
- key count is at least `concurrency * entries_per_operation * 16`

## Config file

The config file format is `key=value`. Empty lines and lines beginning with
`#` are ignored. The current implementation does not use YAML.

Example:

```ini
client.client_id=kv-test-client-0
wait_timeout_ms=5000
# Optional when libasu_client.so/libasu_transport.so are not discoverable.
# asu.client_library_path=/path/to/libasu_client.so
# asu.transport_library_path=/path/to/libasu_transport.so

fake_backend.path=./kv-test-fake-backend-store
fake_backend.latency_us=1000
fake_backend.worker_threads=4
fake_backend.complete_immediately=true

view.config_path=./kv_semantics/kv_test/kv_client_view.conf
client.use_batch_io_for_single_key=true
client.hash_table.type=RING_HASH
client.ring_hash.virtual_node_count=128
client.asu_shared_provider=0

transport.asu_ids=1,2,3
transport.provider_type=FAKE
transport.device_id=0
asu_info.1=protocol=TCP,local.comm_id=127.0.0.1,port=19001
asu_info.2=protocol=TCP,local.comm_id=127.0.0.1,port=19002
asu_info.3=protocol=TCP,local.comm_id=127.0.0.1,port=19003

kv.key_prefix=kv-test-key-
kv.seed=20260530
kv.value_size=4096
kv.count=16

limits.memory_max_bytes=4294967296

bench.io_size=4096
bench.io_interval_us=1000
bench.concurrency=1
bench.duration_sec=10
bench.warmup_sec=1
bench.read_ratio=50
bench.write_ratio=50
bench.batch_size=16

output.path=./kv-test-output
output.realtime_file_max_bytes=104857600
```

### ASU client fields

Use the complete canonical file keys below, including the `client.` and
`transport.` prefixes. Internal Router attrs such as `hash_table.type` are not
standalone file keys. Compatibility aliases are omitted.

| Field | Description |
| --- | --- |
| `client.client_id` | Client identifier. |
| `client.max_inflight_tasks` | Client submission queue capacity; not the total number of unfinished tasks. |
| `client.use_batch_io_for_single_key` | Use batch Load/Store protocol for one entry; defaults to true. |
| `client.asu_shared_provider` | 0: independent provider per transport; 1: shared provider. |
| `view_service_addrs` | Comma-separated view service addresses. |
| `view.config_path` | File used by the default ConfigFileViewServer. |
| `wait_timeout_ms` | Default Client Wait/drain timeout and initial transport timeout, in milliseconds. |
| `client.hash_table.type` | Router type; documented supported types are RING_HASH and MAGLEV. |
| `client.ring_hash.virtual_node_count` | Number of Ring Hash virtual nodes per ASU. |
| `client.maglev.table_size` | Number of slots in the Maglev lookup table. |
| `client.contiguous_block_affinity.block_count` | Affinity block count; parser option outside the documented supported Router scope. |
| `client.contiguous_block_affinity.full_spread_type` | Underlying full-spread Router for contiguous-block affinity. |
| `client.contiguous_block_affinity.dynamic_adjust_enabled` | Parsed dynamic-adjust flag; does not imply a complete dynamic control plane. |
| `client.batch_topk_affinity.top_k` | Number of candidate ASUs for batch top-k affinity; outside the documented supported Router scope. |
| `client.batch_topk_affinity.dynamic_adjust_enabled` | Parsed dynamic-adjust flag for batch top-k affinity. |
| `transport.asu_ids` | Comma-separated ASU IDs for which transport configurations are created. |
| `transport.provider_type` | AICPU, FAKE, or AIV; the provider must be enabled at build time. |
| `transport.device_id` | Local logical device ID; AICPU first uses the caller's current ACL device. |
| `transport.max_inflight_tasks` | Per-transport task capacity. |
| `transport.max_error_count` | Consecutive channel failure threshold before draining. |
| `transport.completion_poll_spin_limit` | Empty completion polling rounds before yielding the CPU. |
| `transport.batch_load_io_num` | Configured maximum entries per batch-load sub-batch; server capabilities may lower it. |
| `transport.batch_store_io_num` | Configured maximum entries per batch-store sub-batch; server capabilities may lower it. |
| `transport.delete_io_num` | Configured maximum keys per delete sub-batch. |
| `transport.query_io_num` | Configured maximum keys per query sub-batch. |
| `transport.send_buffer_slot_size` | Bytes per send-buffer slot. |
| `transport.send_buffer_slot_num` | Number of send-buffer slots. |
| `transport.flag_buffer_slot_size` | Bytes per response/flag-buffer slot. |
| `transport.flag_buffer_slot_num` | Number of response/flag-buffer slots. |
| `asu_info.<id>` | Endpoint configuration for one ASU; multiple endpoints are separated by semicolons. |

When `view.config_path` is set, the loaded view is the ASU membership used by
`KvClient`. Every ASU id in that view must have a matching `transport.asu_ids`
entry so Client can build a transport for it. Non-mocked transports also need
endpoint information such as `asu_info.<id>`. If `view.config_path` is omitted
and `view_service_addrs` is empty, the default view is derived from
`transport.asu_ids`.

### SQE and provider send fields

Unknown `transport.` keys are retained as provider attrs; being retained does not
mean a consumer exists. The following fields have explicit consumers.

| Field | Actual meaning |
| --- | --- |
| `transport.kv_ns_id` | KV namespace encoded in SQEs; does not select a separate Client Router. |
| `transport.dtype` | Store SQE data-type field (3 bits); must match the server protocol. |
| `transport.dspec` | Store SQE data-specification field (5 bits); must match the server protocol. |
| `transport.lr` | SQE LR flag for Load/Store, passed through without a Client-level policy. |
| `transport.sc` | Query seek-control flag; true requests per-key results. |
| `transport.kernel_count` | Send kernel-count argument passed to the provider; kv-test supplies 1 if absent. AICPU currently ignores this argument. |
| `transport.quiet_count` | Send quiet-count argument passed to the provider; kv-test supplies 1 if absent. AICPU currently ignores this argument. |
| `transport.timeout` | Legacy AICPU send-timeout fallback in milliseconds; not TransportConfig.timeoutMs. Use wait_timeout_ms for Client/transport task timeout. |

### AICPU fields not all shown in the sample

| Field | Actual meaning |
| --- | --- |
| `transport.aicpu_hcomm_protocol` | Required data protocol selector: ubg or ubc_ctp (ub / ub_ctp are CTP aliases). Endpoint protocol does not select this path. |
| `transport.aicpu_local_eid.<logical_device_id>` | Local UBG EID for an ACL logical device. |
| `transport.aicpu_local_eid` | Common UBG EID fallback when the device-specific key is absent. |
| `transport.aicpu_local_ip.<logical_device_id>` | Local UBC_CTP address for an ACL logical device. |
| `transport.aicpu_local_ip` | Common UBC_CTP address fallback. |
| `transport.local_ip` | Final local endpoint address fallback after protocol-specific settings and the CreateConnection local-address argument. |
| `transport.logical_device_id` | Compatibility logical-device selector read by AICPU; prefer transport.device_id. |
| `transport.aicpu_notify_num` | HCOMM channel notification-resource count, not an IO concurrency setting. |
| `transport.aicpu_ub_sq_depth` | Requested UB send-queue depth; the default sentinel delegates selection to HCOMM. |
| `transport.aicpu_qos` | HCOMM channel QoS value; not a bandwidth rate. |
| `transport.aicpu_send_timeout_ms` | HIXL send timeout in milliseconds; separate from Client Wait and transport task timeout. |
| `transport.aicpu_channel_name` | Name used when creating HCOMM channel resources. |
| `transport.aicpu_hixl_kernel_json` | HIXL kernel JSON path; otherwise HIXL_KERNEL_JSON then ASCEND_HOME_PATH-based resolution is used. |
| `transport.aicpu_staged_oob_host` | Staged control-plane address override; defaults to the remote endpoint address. |
| `transport.aicpu_staged_oob_port` | Staged control-plane port override; defaults to the remote endpoint port. |
| `transport.aicpu_staged_client_id` | Requested staged client ID; defaults to the ASU node ID. Negotiation may return a different effective client ID. |
| `transport.aicpu_staged_kato` | Staged channel keep-alive field passed to HCOMM, default 0; not the Client Wait timeout. |
| `transport.aicpu_staged_rm_uasid` | Staged channel RM UASID field, default 0. |
| `transport.aicpu_staged_mami_tag` | Staged channel MAMI tag field, default 0. |
| `transport.endpoint_loc` | HCOMM endpoint location: host selects HOST; other values select DEVICE. |

Current local and remote EIDs must be written explicitly into config/view files.
Future work must discover EIDs automatically, without writing EIDs into config;
that capability is not implemented yet. AICPU accepts compatibility aliases
without the `aicpu_` prefix for notify_num, ub_sq_depth, qos, send_timeout_ms,
channel_name, staged_oob_host, staged_oob_port, staged_client_id, staged_kato,
staged_rm_uasid, and staged_mami_tag (each still prefixed with `transport.`).
`transport.hixl_kernel_json` aliases `transport.aicpu_hixl_kernel_json`;
`transport.localIp` aliases `transport.local_ip`. The shorter `transport.oob_host`,
`transport.oob_port`, and `transport.client_id` are further staged-setting aliases.

### Fields inside asu_info.<id>

These are comma-separated subfields, not top-level file keys. Separate endpoint
records with semicolons. The file parser only retains its recognized subfields.
Fields accepted by the C++ NodeEndpoint attrs API are not necessarily accepted
inside an asu_info file record.

| Field | Actual meaning |
| --- | --- |
| `local.comm_id` | Remote communication address (EID/IP according to provider and protocol). |
| `port` | Remote port. |
| `protocol` | Parsed endpoint protocol metadata; does not replace transport.aicpu_hcomm_protocol. |
| `placement` | Parsed endpoint location metadata. |
| `tc` | Parsed endpoint traffic-class attr. |
| `sl` | Parsed endpoint service-level attr. |
| `send_size` | Parsed endpoint send-size attr. |
| `flag_size` | Parsed endpoint flag-size attr. |
| `remote_send_addr` | Parsed remote send address attr. |
| `remote_flag_addr` | Parsed remote flag address attr. |

Endpoint attrs are parsed metadata, not a guarantee that every provider uses
each attr. AICPU staged overrides should use the top-level transport keys above.
AIV has no additional provider-specific .conf keys in its adapter; its device,
endpoint and send-count inputs come from the common fields.

### kv-test-only fields

These fields are parsed by `kv-test` itself:

| Field | Description |
| --- | --- |
| `asu.client_library_path` | Optional `libasu_client.so` path used by kv-test's runtime proxy. The environment variable `KV_TEST_ASU_CLIENT_LIB` is also accepted. |
| `asu.transport_library_path` | Optional `libasu_transport.so` path used by kv-test's runtime proxy. The environment variable `KV_TEST_ASU_TRANSPORT_LIB` is also accepted. |
| `fake_backend.path` | FAKE provider storage root. Defaults to `./kv-test-fake-backend-store`. |
| `fake_backend.latency_us` | Mock backend completion delay in microseconds. Default is `1000`. |
| `fake_backend.worker_threads` | Number of FAKE provider IO workers. Default is `4`. |
| `fake_backend.complete_immediately` | Complete FAKE provider requests without backend IO. Default is `false`. |
| `kv.key_prefix` | Prefix for count-based key generation. |
| `kv.seed` | Seed for deterministic value generation. |
| `kv.value_size` | Value size for normal commands. |
| `kv.count` | Default count for count-based generation. |
| `limits.memory_max_bytes` | Maximum value payload bytes held by kv-test. For normal commands this limits generated value bytes. For `bench`, this limits the reusable buffer pool. Default is 4 GiB. |
| `bench.io_size` | Bench value size for one key. Must not exceed the protocol 24-bit length limit `0xFFFFFF`. |
| `bench.io_interval_us` | Interval between asynchronous I/O submissions in microseconds. Must be greater than zero. |
| `bench.concurrency` | Number of completion waiter threads and reusable benchmark buffer slots. |
| `bench.duration_sec` | Measured benchmark duration. Must be greater than zero. |
| `bench.warmup_sec` | Warmup duration. |
| `bench.read_ratio` | Read ratio used by `bench mix`. |
| `bench.write_ratio` | Write ratio used by `bench mix`. |
| `bench.batch_size` | Entries per batch operation. |
| `output.path` | Base output directory. Empty value uses `.`. |
| `output.realtime_file_max_bytes` | Maximum realtime CSV size before rolling. Default is 100 MiB. |

Additional kv-test keys supported even when absent from the sample:

| Field | Actual meaning |
| --- | --- |
| `bench.io_count` | Measured entry count (not batch count); a positive value selects count-based measurement instead of duration. |
| `metrics.enabled` | Enable the standalone kv-test Prometheus endpoint. |
| `metrics.config_path` | Metric-definition YAML path; metrics.definition_path is a compatibility alias. |
| `metrics.listen_address` | Prometheus listener bind address; must not be empty when enabled. |
| `metrics.port` | Prometheus listener TCP port, range 1–65535. |
| `metrics.path` | HTTP metrics path; must start with / when enabled. |
| `metrics.aggregation_interval_ms` | Metric aggregation interval in milliseconds; must be positive when enabled. |
| `metrics.shutdown_grace_ms` | Standalone metrics HTTP-server shutdown grace period in milliseconds. |

For FAKE, kv-test fills provider attrs from the top-level fake_backend keys above.
`transport.fake_backend.path`, `transport.fake_backend.latency_us`,
`transport.fake_backend.worker_threads`, `transport.fake_backend.complete_immediately`,
and `transport.fake_backend.device_id` are direct-provider attrs, overwritten by
kv-test runtime preparation; configure the top-level fields instead. The provider
device ID is derived at runtime, not a separate fake_backend.device_id tool key.

Batch and sub-batch limits are left to Client and Transport. New kv-test configs
should not include older `limits.batch_store_max`, `limits.batch_retrieve_max`,
`limits.delete_max`, or `limits.exist_max` fields.

## FAKE provider

`transport.provider_type=FAKE` selects the local mock provider while preserving
the normal `KvClient` and `AsuTransportImpl` path. ASU transport sends are
completed by `FakeTransProvider`.

Use this mode to validate:

- `KvClient` routing and per-ASU subtask splitting
- `KvClient` task aggregation, `Wait`, and result merging
- `AsuTransportImpl` async submit paths for store, retrieve, delete, and query
- SQE packing into send buffers
- flag buffer/CQE polling
- CQE status handling and result-buffer parsing
- current register/bind bookkeeping paths, including placeholder memory handles
  and token ids returned through the provider interface

Mocked or not covered in this mode:

- real device or network `Send`
- real ASU backend execution
- real CQE writeback by device
- real RDMA memory registration, `rkey`, and `lkey` semantics
- real connection failure, drain, and recovery behavior

For every transport configured with the FAKE provider, kv-test fills required
SQE/send attrs and passes `fake_backend.path`, `fake_backend.latency_us`,
`fake_backend.worker_threads`, `fake_backend.complete_immediately`, and `fake_backend.device_id` through
`TransportConfig.attrs`. Other provider entries are left unchanged.
`FakeTransProvider::CreateConnection` returns placeholder connection handles so
`ConnectionManager` can create channels during this software-only integration
phase.

The fake backend stores each key under:

```text
<fake_backend.path>/asu-<kv_ns_id>/<fnv64-key-hash>.bin
```

If `fake_backend.path` is empty, fake backend uses:

```text
./kv-test-fake-backend-store
```

During this temporary integration stage, fake backend maps each
`TransportConfig.nodeId` into `TransportConfig.attrs["kv_ns_id"]` before
`Transport::Init`. The packed SQE carries `kv_ns_id`, so the mock backend can
recover a per-ASU namespace from the send buffer without changing the Transport
`Send` interface. This is a kv-test-only temporary semantic mapping, not a
long-term protocol statement that KVNS_ID is ASU ID.

The fake backend copies and queues each SQE before `Send` returns. Dedicated IO
workers process queued requests in parallel and publish the CQE/flag buffer when
the mock file operation completes.
Query returns CQE status `0` when every key exists and `0x732` with a
result-buffer payload when only some keys exist. Delete treats missing keys as
successful entries; a result-buffer byte value of `0` means success and `1`
means delete failed.

FakeBackend smoke scripts live under:

```text
ucm/transport/kv/kv-test/scripts/
```

They can be run after `kv-test` is built and discoverable in `PATH`, or by
setting `KV_TEST_BIN` to the executable path:

```bash
export KV_TEST_BIN=./build-kv-test/ucm/transport/kv/kv-test/kv-test
bash ucm/transport/kv/kv-test/scripts/test_fake_backend_single_asu.sh
bash ucm/transport/kv/kv-test/scripts/test_fake_backend_multi_asu.sh
bash ucm/transport/kv/kv-test/scripts/test_fake_backend_protocol_results.sh
```

Each script keeps its generated config, command logs, fake store, trace files,
and kv-test reports under:

```text
./kv-test-output/fake-backend-scripts/run-<timestamp>-<pid>/
```

Set `KV_TEST_SCRIPT_LOG_DIR` to place these artifacts elsewhere.

## Command behavior

### connect

```bash
kv-test connect --configpath ./ucm/transport/kv/kv-test/kv_test.conf
```

The tool loads config, creates the ASU client, initializes it, opens the output
directory, writes a summary, shuts the client down, and exits.

### config check

```bash
kv-test config check --configpath ./ucm/transport/kv/kv-test/kv_test.conf
```

The command validates config loading and fixed kv-test behavior constraints. It
does not initialize the ASU client and does not open the result writer.

### store and batch-store

```bash
kv-test store --key key1 --check
kv-test store --keys key1,key2,key3
kv-test store --count 16
kv-test store --prefix user- --key-start 100 --key-end 199
kv-test batch-store --count 16 --batch-size 16 --check
```

In the current implementation, both `store` and `batch-store` submit all
selected entries in one ASU `StoreAsync` call. The command names are kept for
CLI compatibility while the ASU stack currently exposes the batch path.

With `--check`, the tool retrieves the same entries and compares the returned
bytes with generated expected values.

### retrieve and batch-retrieve

```bash
kv-test retrieve --key key1 --check
kv-test retrieve --keys key1,key2,key3 --check
kv-test batch-retrieve --count 16 --batch-size 16 --check
```

In the current implementation, both `retrieve` and `batch-retrieve` submit all
selected entries in one ASU `LoadAsync` call. The command names are kept for
CLI compatibility while the ASU stack currently exposes the batch path.

With `--check`, retrieved bytes are compared with generated expected values.

### delete

```bash
kv-test delete --key key1 --check
kv-test delete --keys key1,key2,key3
```

The command submits one ASU `DeleteAsync` call containing all selected keys.
Any DHT routing or per-transport sub-batch splitting is handled inside Client.
With `--check`, the tool runs an exist query and expects all selected keys to be
missing. Delete treats a missing key as a successful delete unless the backend
reports an actual IO/delete failure.

### exist

```bash
kv-test exist --key key1
kv-test exist --keys key1,key2,key3
```

The command runs ASU `Query` in `PER_KEY` mode with all selected keys. Any DHT
routing or per-transport sub-batch splitting is handled inside Client.
Single-key output includes the key and either `exists` or `missing`; multi-key
output prints total, existing, and missing counts.

### power-cycle prepare and verify

```bash
kv-test power-cycle prepare --count 16 --value-size 4096
kv-test power-cycle verify --count 16 --value-size 4096
```

`power-cycle prepare` uses the same store-like path as `batch-store` and writes
`power-cycle-metadata.conf` under `output.path` or `.`. The metadata records
key prefix, seed, value size, and count.

`power-cycle verify` uses the retrieve-like path, reads the same metadata file,
checks that the current generation settings match it, and always performs
consistency checking even when `--check` is omitted.

### bench

```bash
kv-test bench store --duration 10 --concurrency 1 --io-size 4096
kv-test bench retrieve --duration 10 --concurrency 1 --io-size 4096
kv-test bench --op batch-store --batch-size 16 --duration 10 --concurrency 1
kv-test bench --op batch-retrieve --batch-size 16 --duration 10 --concurrency 1
kv-test bench mix --read-ratio 70 --write-ratio 30 --duration 10 --concurrency 1
kv-test bench batch-store --duration 10 --progress
```

Bench operation can be provided as `kv-test bench <op>` or through `--op` /
`--bench-op`.

Requirements:

- `bench.op` must be set by config, positional op, `--op`, or `--bench-op`.
- `bench.concurrency` must be greater than zero.
- `bench.io_interval_us` must be greater than zero.
- `bench.duration_sec` must be greater than zero.
- `bench.io_size` must be greater than zero.
- `bench.io_size` must be less than or equal to `0xFFFFFF`.
- `read_ratio` and `write_ratio` must each be in `0..100`.
- For `mix`, at least one of `read_ratio` or `write_ratio` must be greater than
  zero.
- For `mix`, `read_ratio + write_ratio` must equal `100`.

The benchmark runner submits one asynchronous task every `bench.io_interval_us`.
Submission does not wait for the preceding task to complete. Completion waiters
collect results in the background, and the runner drains all outstanding tasks
after the final submission.

Bench starts with `concurrency` reusable buffer slots. If every slot is still in
flight when the next interval expires, the pool grows instead of delaying the
submission. Completed tasks return their slots to the pool. Pool growth remains
subject to `limits.memory_max_bytes`.

With `--progress`, bench prints one measured progress line per second instead
of rewriting a terminal line. The final summary and report output are still
written after the run.

## Output

For commands other than `config check`, the result writer creates:

```text
<output.path or .>/
  index.html
  run-YYYYMMDD-HHMMSS/
    summary.txt
    summary.json
    report.html
```

Benchmark runs also create these files when samples are written:

```text
bench-realtime-0.csv
latency.csv
```

Runs with consistency failures create:

```text
consistency_errors.jsonl
```

Realtime CSV files roll when the next sample would exceed
`output.realtime_file_max_bytes`.

`report.html` is a self-contained HTML report for one run. It contains request
metadata, status, benchmark cards, an inline SVG bandwidth curve, latency
tables, consistency summary, and artifact links.

`index.html` is regenerated after each run and lists all `run-*` reports under
the same `output.path`.

For a Linux VM, one simple way to view the reports from a browser is:

```bash
cd ./kv-test-output
python3 -m http.server 8080 --bind 0.0.0.0
```

If the VM uses NAT networking, configure host-to-guest port forwarding for TCP
port `8080`, then open the forwarded host URL in the browser.

The terminal prints:

```text
kv-test: succeeded
command=<command>
config=<config path>
```

Failures print:

```text
kv-test: failed: <message> (exit_code=<code>)
```

Additional terminal summaries are printed for `exist`, enabled consistency
checks, and `bench`.

## Exit codes

The kv-test layer uses these explicit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Invalid argument or config/result-writer error. |
| `4` | Consistency check failed. |

ASU client failures are converted to `100 + static_cast<int>(kv::StatusCode)`
so they do not overlap the kv-test layer's exit codes.

## Examples

```bash
export KV_TEST_CONFIG=./ucm/transport/kv/kv-test/kv_test.conf

kv-test config check
kv-test connect
kv-test store --key hello --check
kv-test retrieve --key hello --check
kv-test exist --key hello
kv-test delete --key hello --check

kv-test store --prefix user- --key-start 100 --key-end 199 --check
kv-test retrieve --keys-file ./keys.txt --check

kv-test bench batch-store --batch-size 16 --duration 10 --concurrency 1
```
