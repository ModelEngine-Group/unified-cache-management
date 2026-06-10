# kv-test command line tool

`kv-test` is a command line validation tool for the ASU KV client path. The
current implementation supports local smoke testing, basic request validation,
consistency checks, and simple benchmark metrics.

The tool reads one key-value config file. It first loads the file through the
ASU client config parser, then reads kv-test-specific options from the same
file.

## Build and environment

`kv-test` is built from `ucm/transport/kv/kv-test/CMakeLists.txt` and links
against `asu_client` and the ASU Ascend dependency interface used by the ASU
module.

`kv-test` is included only when ASU support is enabled:

```bash
cmake -S . -B build-kv-test -DBUILD_UCM_ASU=ON -DBUILD_UCM_STORE=OFF -DBUILD_UNIT_TESTS=OFF -DRUNTIME_ENVIRONMENT=ascend
cmake --build build-kv-test --target kv-test
```

The same build can be started from any working directory with:

```bash
bash ucm/transport/kv/kv-test/build.sh
```

The example environment script is:

```bash
source ucm/transport/kv/kv-test/set_kvtest_env.sh
```

Use `source` if the exported `KV_TEST_CONFIG` and `PATH` must remain visible in
the current shell. Running the script as `./set_kvtest_env.sh` only updates the
script process and does not update the caller's shell.

The bundled example config is:

```text
ucm/transport/kv/kv-test/asu_kv_test.conf
```

The bundled example view file is:

```text
ucm/transport/kv/kv-test/asu_view.conf
```

The sample config uses relative paths such as `view.config_path`,
`local_store.path`, and `output.path`. They are resolved against the process
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
| `bench` | Runs a synchronous benchmark loop for `store`, `retrieve`, `batch-store`, `batch-retrieve`, or `mix`. |

## Common options

Options can use either `--option value` or `--option=value`.

| Option | Description |
| --- | --- |
| `--configpath <path>` | Config path for this run. Overrides `KV_TEST_CONFIG`. |
| `--help`, `-h` | Prints general or command-specific help. |
| `--version` | Prints tool version. Does not accept positional arguments. |
| `--check` | Enables consistency checking where supported. |
| `--timeout <ms>` | Overrides `default_wait_timeout_ms` for this run. |
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
client_id=kv-test-client-0
default_wait_timeout_ms=5000

asu.client.mode=fake_backend
local_store.path=./kv-test-local-store
fake_backend.path=./kv-test-fake-backend-store
fake_backend.latency_ms=1

view.config_path=./ucm/transport/kv/kv-test/asu_view.conf
hash_table.type=RING_HASH
ring_hash.virtual_node_count=128

transport.asu_ids=1,2,3
asu_info.1=protocol=TCP,local.comm_id=127.0.0.1,port=19001,local.phy_device_id=0
asu_info.2=protocol=TCP,local.comm_id=127.0.0.1,port=19002,local.phy_device_id=0
asu_info.3=protocol=TCP,local.comm_id=127.0.0.1,port=19003,local.phy_device_id=0

kv.key_prefix=kv-test-key-
kv.seed=20260530
kv.value_size=4096
kv.count=16

limits.memory_max_bytes=4294967296

bench.io_size=4096
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

These fields are parsed by the ASU client config parser:

| Field | Description |
| --- | --- |
| `client_id` | Client identifier. |
| `view_service_addrs` | View service addresses. |
| `view.config_path` | File used by the default `ConfigFileViewServer`. |
| `default_wait_timeout_ms` | Default wait timeout in milliseconds. |
| `transport.asu_ids` | ASU ids. |
| `asu_info.<id>` | Endpoint config for one ASU. |
| `hash_table.type` | Router hash table type. |
| `ring_hash.virtual_node_count` | Ring hash virtual node count. |
| `maglev.table_size` | Maglev table size. |
| `contiguous_block_affinity.*` | Contiguous block affinity options. |
| `batch_topk_affinity.*` | Batch top-k affinity options. |

When `view.config_path` is set, the loaded view is the ASU membership used by
`AsuClient`. Every ASU id in that view must have a matching `transport.asu_ids`
entry so Client can build a transport for it. Non-mocked transports also need
endpoint information such as `asu_info.<id>`. If `view.config_path` is omitted
and `view_service_addrs` is empty, the default view is derived from
`transport.asu_ids`.

### kv-test-only fields

These fields are parsed by `kv-test` itself:

| Field | Description |
| --- | --- |
| `asu.client.mode` | Set to `local` to use the file-backed local ASU transport. Set to `fake_backend` to use normal `AsuClient` and `AsuTransportImpl` paths with kv-test's mock backend. Any other value uses the default ASU client transport factory. |
| `local_store.path` | Local transport storage root. Defaults to `./kv-test-local-store` if local mode is enabled and this field is empty. |
| `fake_backend.path` | fake_backend storage root. If empty, fake_backend reuses `local_store.path`; if both are empty, it uses `./kv-test-fake-backend-store`. |
| `fake_backend.latency_ms` | Mock backend completion delay in milliseconds. Default is `1`. |
| `kv.key_prefix` | Prefix for count-based key generation. |
| `kv.seed` | Seed for deterministic value generation. |
| `kv.value_size` | Value size for normal commands. |
| `kv.count` | Default count for count-based generation. |
| `limits.memory_max_bytes` | Maximum value payload bytes held by kv-test. For normal commands this limits generated value bytes. For `bench`, this limits the reusable buffer pool. Default is 4 GiB. |
| `bench.io_size` | Bench value size for one key. Must not exceed the protocol 24-bit length limit `0xFFFFFF`. |
| `bench.concurrency` | Number of benchmark operations launched concurrently in one wave. |
| `bench.duration_sec` | Measured benchmark duration. Must be greater than zero. |
| `bench.warmup_sec` | Warmup duration. |
| `bench.read_ratio` | Read ratio used by `bench mix`. |
| `bench.write_ratio` | Write ratio used by `bench mix`. |
| `bench.batch_size` | Entries per batch operation. |
| `output.path` | Base output directory. Empty value uses `.`. |
| `output.realtime_file_max_bytes` | Maximum realtime CSV size before rolling. Default is 100 MiB. |
| `connection.timeout_ms` | Also overrides ASU client default wait timeout. |

Batch and sub-batch limits are left to Client and Transport. New kv-test configs
should not include older `limits.batch_store_max`, `limits.batch_retrieve_max`,
`limits.delete_max`, or `limits.exist_max` fields.

## Local mode

`asu.client.mode=local` creates a file-backed local ASU transport. It is useful
for early kv-test self-checks without connecting to a real ViewServer, Hcomm,
network, or ASU hardware.

Local mode still creates `AsuClient`, but injects kv-test's in-process
`LocalAsuTransport` factory instead of using the developed `AsuTransportImpl`.
It is intended for developing and debugging kv-test command parsing, config
loading, data generation, buffer allocation, consistency checks, report output,
and basic store/retrieve/delete/exist behavior with a local file-backed
transport.

Local mode still requires each ASU to have at least one endpoint in `asu_info`.
The endpoint's `local.comm_id` and `port` are validated so local tests catch
missing address config in the same place as real transport setup.

In the bundled local config, `protocol=TCP` is a convenient placeholder for ASU
config parsing and validation. It does not imply that local mode performs real
TCP IO. For real ASU/Hcomm testing, switch away from local mode and replace
`protocol`, `local.comm_id`, `port`, and `local.phy_device_id` with the target
hardware setup.

The local transport stores each key under:

```text
<local_store.path>/asu-<asuId>/<hex-encoded-key>.bin
```

The directory persists across separate `kv-test` commands. Delete removes the
corresponding file when it exists. Deleting a missing key is treated as success,
matching the current Delete result-buffer semantics. Exist checks whether the
file exists.

Because this mode replaces the ASU transport implementation, it can exercise
`AsuClient` routing and aggregation at a high level, but it does not validate
`AsuTransportImpl` request building, send-buffer submission, polling, or
CQE/result-buffer parsing paths.

## fake_backend mode

`asu.client.mode=fake_backend` is for the current `AsuClient` +
`AsuTransportImpl` software integration stage. kv-test creates the normal
`AsuClient` path and uses the developed `AsuTransportImpl`, but replaces the
missing backend send/completion side with a local mock backend.

Use this mode to validate:

- `AsuClient` routing and per-ASU subtask splitting
- `AsuClient` task aggregation, `Wait`, and result merging
- `AsuTransportImpl` async submit paths for store, retrieve, delete, and query
- SQE packing into send buffers
- flag buffer/CQE polling
- CQE status handling and result-buffer parsing
- current register/bind bookkeeping paths

Mocked or not covered in this mode:

- real device or network `Send`
- real ASU backend execution
- real CQE writeback by device
- real memory registration, `rkey`, and `lkey` semantics
- real connection failure, drain, and recovery behavior

The mock send path is installed through a temporary `AICPUTransProvider` send
hook while fake_backend mode is enabled. `AICPUTransProvider::CreateConnection`
also returns placeholder connection handles so `ConnectionManager` can create
channels during this software-only integration phase.

The fake backend stores each key under:

```text
<fake_backend.path>/asu-<kv_ns_id>/<fnv64-key-hash>.bin
```

If `fake_backend.path` is empty, fake backend reuses `local_store.path`. If both
are empty, it uses:

```text
./kv-test-fake-backend-store
```

During this temporary integration stage, fake backend maps each
`TransportConfig.asuId` into `TransportConfig.attrs["kv_ns_id"]` before
`Transport::Init`. The packed SQE carries `kv_ns_id`, so the mock backend can
recover a per-ASU namespace from the send buffer without changing the Transport
`Send` interface. This is a kv-test-only temporary semantic mapping, not a
long-term protocol statement that KVNS_ID is ASU ID.

The fake backend writes the CQE/flag buffer synchronously before `Send` returns.
Query returns CQE status `0` when every key exists and `0x732` with a
result-buffer payload when only some keys exist. Delete treats missing keys as
successful entries; a result-buffer byte value of `0` means success and `1`
means delete failed.

Mode differences:

| Item | local | fake_backend |
| --- | --- | --- |
| Main development stage | kv-test self-check | `AsuClient` + `AsuTransportImpl` software integration |
| Uses normal `AsuClient` routing | Yes | Yes |
| Uses developed `AsuTransportImpl` submit/poll/CQE code | No | Yes |
| Backend | kv-test `LocalAsuTransport` | kv-test `MockSend` + local CQE completion |
| Filesystem store layout | `<root>/asu-<asuId>/<hex-key>.bin` | `<root>/asu-<kv_ns_id>/<fnv64-key-hash>.bin` |
| Validates real device communication | No | No |

fake_backend smoke scripts live under:

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
kv-test connect --configpath ./ucm/transport/kv/kv-test/asu_kv_test.conf
```

The tool loads config, creates the ASU client, initializes it, opens the output
directory, writes a summary, shuts the client down, and exits.

### config check

```bash
kv-test config check --configpath ./ucm/transport/kv/kv-test/asu_kv_test.conf
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
- `bench.duration_sec` must be greater than zero.
- `bench.io_size` must be greater than zero.
- `bench.io_size` must be less than or equal to `0xFFFFFF`.
- `read_ratio` and `write_ratio` must each be in `0..100`.
- For `mix`, at least one of `read_ratio` or `write_ratio` must be greater than
  zero.
- For `mix`, `read_ratio + write_ratio` must equal `100`.

The benchmark runner launches one asynchronous task per operation in a wave.
Each wave contains up to `concurrency` operations.

Bench uses a fixed reusable buffer pool instead of pre-generating all data for
the whole run. The pool holds `entries_per_operation * concurrency` buffers of
`bench.io_size` bytes. Each new batch updates metadata for the selected slot and
reuses its value buffers.

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

ASU client failures are converted to `100 + static_cast<int>(UC::ASU::StatusCode)`
so they do not overlap the kv-test layer's exit codes.

## Examples

```bash
export KV_TEST_CONFIG=./ucm/transport/kv/kv-test/asu_kv_test.conf

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
