<!-- Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. -->
# HCCP KV-over-RoCE trans provider

`HCCP_KV_OVER_ROCE` is the Ascend HCCP KV-over-RoCE data path. This guide covers
building the provider, configuring it from `kv-test` and from ucm/vLLM, the
runtime library contract, a checklist for the operator library, and the sizing
rules. For the `kv-test` command workflow itself, see [kv-test](kv-test.md).

## Responsibilities

| Component | Responsibility |
| --- | --- |
| Operator library | Owns the HCCP/RDMA resources: creates QPs, registers MRs, executes `Send`. |
| Provider | Maps TransProvider calls to the ABI; owns ACL binding, HBM slots and MR handles. |
| Client | Selects the provider, registers buffers, opens one connection per ASU. |

The provider never allocates operator resources on its own: every handle it
returns is either a pointer to provider-owned memory or an index into a
provider-owned pool.

## Build

```bash
cmake -S . -B build -DRUNTIME_ENVIRONMENT=ascend -DBUILD_UCM_ASU=ON \
  -DBUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON -DKV_CLIENT_CANN_ROOT=/path/to/cann
```

The option is `OFF` by default and needs CANN (`acl/acl.h`, `libascendcl.so`),
found through `KV_CLIENT_CANN_ROOT`, `ASCEND_HOME_PATH`, or the configured
Ascend root. Enabling it defines `UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER`.

The operator library is loaded at runtime, not linked, so the provider binary
does not depend on it. When the provider is not built, creating a
`HCCP_KV_OVER_ROCE` transport fails with `UNSUPPORTED` and the message names the
CMake option.

## Configuration

### Transport keys (`kv-test`, standalone client config)

| Key | Notes |
| --- | --- |
| `transport.provider_type` | Set to `HCCP_KV_OVER_ROCE`. |
| `transport.device_id` | Device to bind; defaults to the current ACL device, then 0. |
| `transport.hccp_kv_roce_lib_path` | Library path; alias `hccp_lib_path`. |
| `transport.hccp_kv_roce_max_connections` | Default 3072; alias `hccp_max_connections`. |

The library path falls back to `UCM_HCCP_KV_ROCE_LIB_PATH` and then to
`libhccp_kv_over_roce.so` in the dynamic linker search path. There is no
environment variable for the capacity.

Minimal `kv-test` configuration:

```text
transport.provider_type=HCCP_KV_OVER_ROCE
transport.asu_ids=1
transport.device_id=0
transport.hccp_kv_roce_lib_path=/opt/hccp/libhccp_kv_over_roce.so
transport.hccp_kv_roce_max_connections=3072
asu_info.1=protocol=TCP,local.comm_id=127.0.0.1,port=19001
```

`asu_info.<id>` supplies the remote IP and port used for `CreateConnection`. The
IP comes from `local.comm_id` despite the `local.` prefix; an empty remote IP is
rejected by the provider.

### ucm / vLLM keys

| Key | Notes |
| --- | --- |
| `asu_trans_provider_backend` | `hccp_kv_over_roce`, or the alias `hccp`. |
| `asu_ips` | Required; one non-empty IP per ASU. |
| `asu_ports` | One port per IP, in `[1, 65535]`. |
| `asu_hccp_kv_roce_lib_path` | Optional; maps to `hccp_kv_roce_lib_path`. |
| `asu_hccp_kv_roce_max_connections` | Optional, in `[1, UINT32_MAX]`. |
| `asu_shared_provider` | `0` (default): one provider per transport. `1`: one shared provider. |

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "ASU"
      asu_trans_provider_backend: "hccp_kv_over_roce"
      asu_ids: [1]
      asu_ips: ["223.223.8.11"]
      asu_ports: [19001]
      asu_hccp_kv_roce_lib_path: "/opt/hccp/libhccp_kv_over_roce.so"
      asu_hccp_kv_roce_max_connections: 3072
```

When `asu_config_path` is set, the client loads the whole configuration from that
file and ignores the `asu_hccp_kv_roce_*` keys; use the `transport.*` keys shown
above inside the file instead. `AsuStore::Setup` rejects an HCCP configuration
without `asu_ips`, and a capacity outside `[1, UINT32_MAX]`.

## Device and context binding

The provider resolves its device in this order:

1. `TransportConfig::deviceId` when it is non-negative.
2. The `device_id`, `deviceId`, or `logical_device_id` attribute.
3. The caller's current ACL device.
4. Device 0, with a warning.

A device attribute that is not an integer in `[0, INT32_MAX]` is ignored with a
warning and treated as unconfigured, so resolution continues with the caller's
current ACL device (step 3), or device 0 (step 4) when there is none. This keeps
a typo from silently binding device 0.

The first binding happens during provider construction: it calls
`aclInit(nullptr)` and `aclrtSetDevice`, captures the resulting context, then
restores the constructing thread's device and context. Construction fails if no
context can be captured. A thread that had no ACL device when the binding was
captured stays on the provider device, because ACL has no API to unbind a
device. Every provider entry point only calls `aclrtSetCurrentContext` with the
captured context and restores the caller's context on exit, so the provider can
share a thread with other ACL users. `aclInit` may already have been called by
the process; `ACL_ERROR_REPEAT_INITIALIZE` is treated as success.

## HBM connection slots

The provider allocates one contiguous HBM region per provider instance:

- one slot per live QP, `HCCP_KV_ROCE_QPC_BYTES` bytes each (512 with the
  current header);
- `hccp_kv_roce_max_connections` slots, 3072 by default, so 1.5 MiB;
- the region is shared by every IP pair of that provider.

A slot holds exactly one QPC, with no padding. The slot size is derived from
`HCCP_KV_ROCE_QPC_BYTES` in `hccp_kv_over_roce_abi.h`, and a `static_assert` in
the provider enforces that it equals `sizeof(HccpKvRoceQpc)`.

Each slot is tracked as `FREE` or `IN_USE`. `Acquire` marks a slot `IN_USE` and
returns `RESOURCE_BUSY` when the pool is exhausted. `Release` only accepts an
`IN_USE` slot, and `Write`/`Clear` only accept an `IN_USE` slot; releasing twice
or writing a slot that was never acquired returns `INVALID_ARGUMENT` instead of
corrupting the region. `SlotAddress` never checks the state, because the provider
hands the address to the library during teardown paths as well.

The region is freed in the provider destructor, on the provider's own
device/context.

## Connections

`CreateConnection` builds one record per IP pair: the library's manager plus one
provider-owned QP entry per QP. The returned `ConnectionHandle` values are the
addresses of those QP entries and stay valid until the whole IP pair is deleted.
Only the remote IP is required; `qpNum == 0` is rejected with
`INVALID_ARGUMENT` and `qpNum` above the slot capacity with `RESOURCE_BUSY`
before the library is called.

Creation is all or nothing. A library error, a created count that differs from
`qpNum`, a null handle, a missing manager, or a failed slot write releases every
slot acquired by the call and asks the library to delete the manager that was
already reported. If that delete fails, the ABI keeps the manager valid, so the
provider retains the record and retries it on the next `CreateConnection`, the
next `DeleteConnections`, and provider destruction. The record is dropped only
after the library reports success.

`DeleteConnections` is manager based. The provider groups the requested handles by
IP pair and calls the library once per call with the manager array:

- the caller must pass every QP of an IP pair; a partial set returns
  `INVALID_ARGUMENT` for the given handles and deletes nothing of that pair;
- listing the same handle twice invalidates the whole pair in the same way;
- a null or unknown handle only fails that entry;
- if the library rejects the call, the records are put back so the caller can
  retry, and slots are only cleared and recycled after a successful call.

`DeleteConnections` also retries any manager retained after a failed creation
rollback, even when the handle list is empty.

`GetServerCapabilities` queries the manager of the IP pair that owns the handle
and copies the ABI structure field by field.

## Send

`Send` validates every batch on its own and hands all valid batches to the
library in a single call:

- a null or unknown connection handle fails only that batch with
  `INVALID_ARGUMENT`;
- the batch carries the HBM slot address of the QP (`slotAddress`), because the
  operator reads the QPC from that slot;
- a library call error fails every submitted batch with the mapped code
  (fallback `IO_ERROR`);
- a per-batch status from the library fails only that batch.

`kernelCount` and `quietCount` are forwarded unchanged. When the library does not
export `Send`, every batch fails with `UNSUPPORTED`.

## Memory registration

Only device memory is accepted: `RegisterMemory` and `BindMemory` reject
`MEM_HOST` descriptors with `INVALID_ARGUMENT`, because the operator cannot
address host buffers.

`RegisterMemory` sends the descriptors to the library and expects one
`HccpKvRoceMemoryInfo` per descriptor. A zero `rkey` is treated as a failure
(`INTERNAL_ERROR`); the whole batch is rolled back through `UnregisterMemory`
when the library exports it. The provider allocates its own `MRHandle` values
starting at 1, so zero always means invalid, and keeps the handle-to-info
mapping.

`BindMemory` never calls the library: it reuses a token id registered by another
provider and only creates a local handle. `GetMemTokenId` returns the stored
`rkey`. `UnregisterMemory` forwards only records created by `RegisterMemory` and
drops bind-only records locally. A duplicate handle in one call returns
`INVALID_ARGUMENT`, an unknown handle `BUFFER_NOT_REGISTERED`, and a library
failure returns the mapped code (fallback `INTERNAL_ERROR`) with the
operator-owned records restored.

`UnregisterMemory` detaches the selected records under the memory lock and calls
the library without holding it, so a slow unregister does not block
`RegisterMemory`, `BindMemory` or `GetMemTokenId`. A handle that is already
detached reports `BUFFER_NOT_REGISTERED` to a concurrent lookup until the call
finishes: it is dropped on success and restored on failure.

When the provider is destroyed, it deletes the connections it still owns and
unregisters the MRs it registered, both best effort.

## Threads

Thread bookkeeping is not part of the ABI: the operator owns no per-thread
resources. `AllocThread` returns success with no handles and `FreeThread` reports
success for every requested thread without calling the library.

## ABI reference

The header is
`src/trans_provider/hccp_kv_over_roce/hccp_kv_over_roce_abi.h`.

| Symbol | Required | Behavior when missing |
| --- | --- | --- |
| `CreateConnection` | yes | Provider construction fails with `NOT_FOUND`. |
| `DeleteConnections` | yes | Provider construction fails with `NOT_FOUND`. |
| `RegisterMemory` | yes | Provider construction fails with `NOT_FOUND`. |
| `GetAbiVersion` | no | Version check is skipped. |
| `UnregisterMemory` | no | Local records are dropped; rollback is skipped with a warning. |
| `Send` | no | `Send` fails with `UNSUPPORTED`. |
| `GetServerCapabilities` | no | The query fails with `UNSUPPORTED`. |

A library that exports `GetAbiVersion` must return `HCCP_KV_ROCE_ABI_VERSION`
(currently `1`); any other value fails provider construction with
`INTERNAL_ERROR` and the library is closed again. A `dlopen` failure or a missing
required symbol is reported by the load step as `NOT_FOUND`, but the factory
wraps every construction failure as `INTERNAL_ERROR` and keeps the original
message.

Return code mapping:

| ABI code | Transport status |
| --- | --- |
| `HCCP_KV_ROCE_OK` (0) | success |
| `HCCP_KV_ROCE_INVALID_ARGUMENT` (1) | `INVALID_ARGUMENT` |
| `HCCP_KV_ROCE_UNSUPPORTED` (2) | `UNSUPPORTED` |
| `HCCP_KV_ROCE_RESOURCE_BUSY` (3) | `RESOURCE_BUSY` |
| `HCCP_KV_ROCE_CONNECTION_ERROR` (4) | `CONNECTION_ERROR` |
| `HCCP_KV_ROCE_IO_ERROR` (5) | `IO_ERROR` |
| `HCCP_KV_ROCE_BUFFER_NOT_REGISTERED` (6) | `BUFFER_NOT_REGISTERED` |
| `HCCP_KV_ROCE_INTERNAL_ERROR` (7) | `INTERNAL_ERROR` |
| Any other value | `INTERNAL_ERROR`, or the operation fallback |

Two constraints are part of the contract: the provider pre-fills
`connectionHandles` with its own `HccpKvRoceQpc` addresses and the library fills
them in place, and the manager returned by `CreateConnection` is the only handle
accepted by `DeleteConnections` and `GetServerCapabilities`.

Each provider instance loads the library once and closes it on destruction.
`dlopen` reference counting makes several instances safe, including several
providers pointing at the same file.

## Operator library checklist

The header is the contract. The provider validates what it can and trusts the
library for the rest, so an implementation that satisfies these points avoids the
failure modes the provider cannot detect at runtime.

### Symbols and version

- [ ] Export the C ABI with `extern "C"` and the exact signatures from the
      header; never let a C++ exception cross the boundary.
- [ ] Export `CreateConnection`, `DeleteConnections` and `RegisterMemory`. A
      missing required symbol makes provider construction fail.
- [ ] If `GetAbiVersion` is exported, return `HCCP_KV_ROCE_ABI_VERSION`. Bump it
      only with an incompatible change, and build against the same header the
      provider uses.
- [ ] Return only `HccpKvRoceReturnCode` values; any other value becomes
      `INTERNAL_ERROR` or the operation fallback at the provider.

### CreateConnection

- [ ] Accept an empty `localIp`: the transport may not know the local address.
      The provider only validates that `remoteIp` is non-empty.
- [ ] Fill each `connectionHandles[i]` in place and keep the pointer: the
      provider passes addresses of its own `HccpKvRoceQpc` records and copies
      them into HBM afterwards.
- [ ] Write at most `HCCP_KV_ROCE_QPC_BYTES` bytes per QPC. The HBM slot is
      exactly one QPC, so a larger payload corrupts the next slot.
- [ ] Set `*connectionHandleNum` to `qpNum` on success and never write or report
      more than `qpNum` entries.
- [ ] Return a non-null, IP-pair-scoped `*connectionManager` on success. It must
      stay valid until `DeleteConnections` succeeds for it.
- [ ] On failure, return a nonzero code and leave no untracked QPs: either free
      everything internally and return a null manager, or return a manager that
      `DeleteConnections` can use to free the partial set.
- [ ] Do not keep the `connectionHandles` array pointer; copy what you need. The
      QPC pointers themselves stay valid until the IP pair is deleted.
- [ ] Support several live IP pairs at once, one manager each.

### DeleteConnections

- [ ] Accept several managers in one call: the provider batches every complete
      IP pair requested by the caller.
- [ ] Treat the array as all-or-nothing. Return `HCCP_KV_ROCE_OK` only if every
      manager was released; on a nonzero code the provider retries the same
      managers later, so none of them may be consumed.
- [ ] Free every QP owned by a manager, including QPs left behind by a partially
      failed `CreateConnection`.
- [ ] Keep the manager valid until the call returns success; the provider may
      still use it through `GetServerCapabilities` while the pair is live.

### RegisterMemory and UnregisterMemory

- [ ] Register device memory only. Host addresses and unaligned or otherwise
      unsupported ranges must fail with `HCCP_KV_ROCE_INVALID_ARGUMENT` instead
      of being registered silently.
- [ ] Fill every `memoryInfos[i]` (`addr`, `size`, `access`, `rkey`) and return a
      nonzero `rkey` for each entry on success.
- [ ] On a partial failure, report a nonzero `rkey` for the entries you did
      register and return a nonzero code: the provider rolls those entries back
      through `UnregisterMemory`.
- [ ] `UnregisterMemory` receives the `HccpKvRoceMemoryInfo` values exactly as
      reported; on a nonzero code keep them valid so the provider can retry.
- [ ] Treat `localAddr` as opaque input; the provider passes it through unchanged
      and may pass `0`.
- [ ] Do not keep pointers to `descs` or `memoryInfos`; they are caller arrays.

### Send

- [ ] Fill one `statuses[i]` per batch and never write past `batchNum` entries.
- [ ] Return `HCCP_KV_ROCE_OK` when only individual batches failed, and report
      each failure in its `statuses` entry. Reserve a nonzero return for a
      failure that affects the whole call.
- [ ] Read the QPC from `batch->slotAddress`, which is device HBM owned by the
      provider. Do not write it, free it, or cache it past `DeleteConnections`.
- [ ] Use `sendBuffer`, `flagBuffer` and `len` as provided; the caller owns the
      buffers and the provider only forwards the transport request.
- [ ] Map `kernelCount` and `quietCount` to your launch semantics; the provider
      forwards them unchanged.

### GetServerCapabilities

- [ ] Accept the IP-pair manager returned by `CreateConnection`.
- [ ] Fill every field: `queueNum`, `ioQueueDepth`, `ioQueueKeyConcurrency`,
      `connectionKeyConcurrency`, `singleValueMaxBytes`, `batchValueMaxBytes`,
      `batchStoreKeys`, `batchLoadKeys`, `deleteKeys`, `queryKeys`, `keyLength`
      and `kvCapabilities`. The provider copies them without further validation.

### Cross-cutting

- [ ] Tolerate concurrency: `Send` may run while another thread creates or
      deletes a different IP pair.
- [ ] Do not change the calling thread's ACL device or context; the provider
      binds its own context around each call and restores the caller's.
- [ ] Be safe to unload: the provider deletes live connections, unregisters its
      MRs, then closes the library. Do not leave threads, timers or callbacks
      that can run after `DeleteConnections` and `UnregisterMemory`.
- [ ] Report every failure through the return code. Unrecognized codes become
      `INTERNAL_ERROR`, which the caller sees as a failed operation.

### What the provider cannot check

- QPC payload size: only a compile-time assert on the provider side. A library
  built with a different header silently overflows the slot.
- Manager lifetime and delete atomicity: the provider only knows whether the
  call returned zero.
- Exceptions crossing the ABI: they are not caught on the library call path.
- Library-side thread and context hygiene, and the honesty of the reported
  capabilities.

## Sizing

The client opens `TransportConfig::qpNum` QPs per endpoint (`1` by default), so
the capacity must cover the sum of QPs over every live endpoint of that provider.
With the defaults, one provider covers 3072 QPs in 1.5 MiB of HBM.

`asu_shared_provider: 0` creates one provider per transport, so N transports
allocate N regions; `asu_shared_provider: 1` shares one provider and one region,
which is the cheaper option when the process has many transports.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `UNSUPPORTED` mentioning the CMake option | Provider not built into the transport library. |
| `INTERNAL_ERROR`, `dlopen ... failed` | Wrong library path, or a missing dependency. |
| `INTERNAL_ERROR`, `did not export required symbol` | Library implements an older ABI. |
| `INTERNAL_ERROR`, `ABI version ... does not match` | Library and header versions differ. |
| `INTERNAL_ERROR`, `must be greater than zero` | Capacity configured as `0`. |
| `INTERNAL_ERROR`, `aclrtMalloc ... failed` | Not enough device memory for the slot region. |
| `RESOURCE_BUSY` on connect | Capacity below the requested QP count, or slots not released. |
| `INVALID_ARGUMENT`, `remote IP is required` | Missing `asu_ips` entry for the ASU. |
| `INVALID_ARGUMENT` on registration | Host memory was registered instead of device memory. |
| `UNSUPPORTED` on `Send` | Library does not export `Send`. |
| `INVALID_ARGUMENT`, `incomplete connection set` | Only part of an IP pair was passed. |
| `INTERNAL_ERROR`, ACL error | Binding failed at construction; check `device_id` and driver. |

## Tests

The unit tests use a fake ABI library built from
`tests/support/fake_hccp_kv_over_roce_abi.cc`, with three variants: all symbols,
minimal (optional symbols missing), and missing `DeleteConnections`. All three
test files below live in `tests/transport/`.

| File | Coverage |
| --- | --- |
| `hccp_kv_over_roce_trans_provider_test.cc` | ABI, connections, send, memory, errors. |
| `hccp_hbm_slot_allocator_test.cc` | Slot state machine, capacity, invalid operations. |
| `hccp_acl_context_test.cc` | Device resolution, context capture/bind/restore. |

```bash
ctest --test-dir build -R "Hccp" --output-on-failure
```

The `kv.test` binary initializes a real NPU device in its global test
environment, so these tests need a working device even though the operator ABI
is faked.
