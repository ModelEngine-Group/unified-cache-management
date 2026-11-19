# M800 View Management

## Purpose and architecture

M800 view management publishes Ceph pool/namespace and ASU membership to workers:

```text
Ceph MON / DPU Agent -> Master -> HTTP /v1/views -> Worker/Slave -> ViewUniverse
```

The Master reads namespaces, in-state OSDs and epochs from the Agent and builds immutable multi-namespace snapshots. The Worker tries configured Masters sequentially, decodes the deterministic text protocol and publishes a complete snapshot through `ViewUniverse::PublishAll`. Permanent scheduling, KvClientImpl/IO routing and final `view.provider` factory integration are outside the current Shard 2/3/4/5 scope.

The Shard 1 Router overload accepting unavailable nodes is retained. Its default implementation preserves legacy routing by delegating to `RouteKeys(keys)`; real secondary-hash failover remains a later Shard 7 responsibility.

`ViewServer` remains unchanged with its original pure-virtual `GetGlobalView`; no base `GetAllViews` or default adapter is added. M800 Master and Slave return `UNSUPPORTED` from the legacy `GetGlobalView`; callers must use `GetAllViews`, and this method performs no Agent/HTTP fetch and does not modify the output view. Worker Master addresses are restricted to numeric IPv4/IPv6 through `inet_pton` and `AI_NUMERICHOST`; hostnames and embedded NULs are rejected.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `view.role` | `master` | `master` or `worker` |
| `view.listen_addr` | `0.0.0.0:9100` | Master HTTP listen address |
| `view.master_addrs` | empty | Comma-separated Worker Master list; takes precedence even when empty |
| `view.master_addr` | empty | Legacy single address, read only when `master_addrs` is absent |
| `view.ceph.mon_host` | empty | MON address required by Master |
| `view.ceph.client_name` | `admin` | Ceph client name |
| `view.ceph.keyring` | `/etc/ceph/ceph.client.admin.keyring` | Ceph keyring file |
| `view.ceph.agent_lib` | `/usr/lib/libkv_agent.so` | Agent shared library |
| `view.ceph.agent_fetch_timeout_ms` | `10000` | Agent fetch timeout; preferred over the compatibility key |
| `view.agent_fetch_timeout_ms` | `10000` | Compatibility form of the Agent timeout |
| `view.http_timeout_ms` | `2000` | Whole Worker fetch and per-connection Master deadline |
| `view.refresh_interval_ms` | `30000` | Agent refresh interval |
| `view.endpoint_protocol` | `ub` | Published ASU endpoint protocol |
| `view.endpoint_port` | `6000` | Published ASU endpoint port |
| `view.ceph.filter` | `in` | Only in-state filtering is supported |

See `examples/m800_view_master.conf` and `examples/m800_view_worker.conf` for deployment examples.

## HTTP and text protocol

`GET /v1/views?since_epoch=N` returns:

- `200`: complete text snapshot;
- `304`: the supplied non-empty caller object already has that epoch;
- `400`: malformed request line/version/input;
- `404`: path other than `/v1/views`;
- `405`: non-GET method, with `Allow: GET`;
- `500`: handler or response-generation failure.

An empty output object always sends `since_epoch=0`, even if the Worker previously fetched a view for another object. A non-empty object uses its own epoch. The Worker keeps no `cachedEpoch_` and performs no local epoch comparison. Every complete, decodable, non-empty Master response is passed directly to `ViewUniverse::PublishAll`, which alone decides whether to publish or return an empty/stale/equal no-op.

The text protocol uses a unified global epoch, `viewEpoch == globalEpoch`, `viewId = nsId`, and `KvNsId = XXH32(poolName, len, 5381)`. Optional `localKvNsId.N` immediately follows `asuInfo.N`; zero is omitted. Every text snapshot must end with LF; any CR or unknown field is rejected. Decoding is deterministic and transactional, and malformed numeric fields, truncation and oversized input are rejected without modifying the output. `PublishAll` emits per-namespace, unlimited INFO logs: `ns={} view published, epoch={}->{}, asu_count={}` or `ns={} view not published, epoch unchanged={}`. The summary is `ViewUniverse published, ns_changed={added+removed}, ns_total={snapshot namespace count}, changed_ns={changed retained namespaces}`.

## Agent ABI compatibility

ABI v1 exports 12 `kv_client_agent_*` symbols. The loader uses `RTLD_NOW | RTLD_LOCAL`, resolves every required symbol, checks `kv_client_agent_get_abi_version`, and passes both the ABI version and `sizeof(kv_client_agent_config)` to init. Missing libraries/symbols, version mismatch and init failure prevent Master startup; there is no silent fallback to an incompatible implementation. Master refresh is all-or-nothing: any namespace fetch failure or unusable endpoint rejects the entire candidate and preserves the complete last-known-good snapshot without backfilling failed namespaces. Output buffers are released before destroy and dlclose.

A third-party ABI implementation must respond to Stop. Permanent ABI calls that ignore cancellation require a supplier contract or process isolation; forcefully unloading the library would be unsafe.

## Capacity and deployment limits

- Up to **64 namespaces** per snapshot.
- Up to **4096 ASUs per namespace**.
- HTTP v1 handles accepted connections **serially**, with listen backlog **16**.
- Each connection has an absolute deadline, **2 seconds by default**; Stop shuts down active sockets.
- Slow clients therefore consume the single service slot. Large deployments must bound Worker count/poll rate per Master and distribute Workers across Masters. Higher concurrency requires a bounded thread pool and queue, not unbounded thread creation.

Multi-Master selection is sequential primary/backup, not a maximum-epoch election. Empty/stale/equal publication follows the documented no-op rules. All addresses and socket operations share one Worker fetch deadline.

## Verification

```bash
./build-m800-derived/tests/kv.test \
  --gtest_filter='ViewServerTest.*:ViewUniverseTest.*:M800View*:ViewHttpServerTest.*'
ctest --test-dir build-m800-derived --output-on-failure -j2
```

Coverage includes configuration, hashing, ABI/dlopen, first-view and partial failures, strict codec behavior, HTTP 400/404/405/500, empty-output full downloads, multi-Master failover, stale epochs, 304, timeouts, lifecycle races and immutable snapshots.

Real Ceph validation also covered MON/OSD, the real Agent shared library, Master→HTTP→Slave, bad-address failover and namespace creation/deletion. DPU Agent readiness is required for continuous OSDMap refresh; MON/OSD availability alone is insufficient.
