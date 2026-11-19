# CPU over RoCE trans provider

`CpuKvOverRoceTransProvider` moves KV blocks between a CPU process and a KV
appliance over RoCEv2. The data path uses the verbs API directly, while the
link is negotiated over a plain TCP connection (out of band) so that queue pair
numbers, GIDs and capability limits can be exchanged before the queue pairs
leave the INIT state.

The provider is registered as `kv::TransProviderType::CPU` and selected with
`asu_trans_provider_backend: cpu`.

## What is implemented

| Operation | State |
| --- | --- |
| `CreateConnection` | implemented: TCP out of band, capability negotiation, queue pair creation, INIT/RTR/RTS, connect-done. One handle per queue pair the two ends agreed on: at most `qpNum`, because the store may advertise fewer |
| `DeleteConnections` | implemented: one-way disconnect message, queue pair teardown, socket close |
| `GetServerCapabilities` | implemented: returns the limits the peer announced, field by field, with `queueNum` set to the number of queue pairs this link actually carries |
| `RegisterMemory` / `UnregisterMemory` | implemented with plain verbs (`ibv_reg_mr` / `ibv_dereg_mr`) |
| `BindMemory` | implemented for ranges some provider in the process registered: records the peer token and returns it unchanged |
| `Send` | not implemented yet (returns `UNSUPPORTED`) |
| `AllocThread` / `FreeThread` | no-op success: IO runs on the caller thread |
| `GetMemTokenId` | implemented, returns the `rkey` of the registered region |

## Link establishment

The sequence follows the product protocol tables. Multi-byte fields are big
endian, with one exception: the 128 byte private area of the two capability
messages carries little endian values. The header carries a CRC-32C
checksum.

1. **TCP**: the initiator (`asu_roce_oob_role: client`) connects, the peer
   (`server`) listens and accepts on the same port.
2. **Capability request** (table 1, `cmd=0`, 176 bytes): `crc|ver|cmd|pad|len`,
   `cap`, `rsv[24]`, `private_len=4`, then `major_version`, `minor_version`,
   `kato`. Everything outside the 128 byte private area is big endian;
   `kato` and every numeric field inside the private area are little endian.
3. **Capability reply** (table 2, `cmd=0`, 176 bytes): the same header,
   `private_len`, then queue depth, key batch (`store_key_batch`,
   `retrieve_key_batch`, `delete_key_batch`, `exist_key_batch`), MDTS and key
   length limits. The reply also carries
   `queue_num`, so the initiator creates as many queue pairs as the store
   accepts instead of insisting on the requested count. This build sends
   `private_len=38` and accepts anything of 36 bytes or more: 36 is the
   layout before `delete_key_batch` existed, and a longer value means the
   peer appends fields that the decoder ignores.
4. **Queue pair creation**: `ibv_create_cq` and `ibv_create_qp` for every queue
   pair of the connection.
5. **Connect request** (table 3, `cmd=1` from the initiator, `cmd=2` from the
   peer): `gid[16]`, `lid`, `mtu`, `total_qp_num`, `sl`, `traffic_class`,
   `rnr_timer`, `rnr_retry_cnt`, `timeout`, `retry_cnt`, `qp_rd_atom`,
   `start_psn`, then one 32 bit `qpn` per queue pair. The body is
   `32 + 4 * total_qp_num` bytes, so a peer only reads the queue pairs that
   were actually created; a body that does not match the announced count is
   rejected. `total_qp_num` is still bounded by the 32 entry table, so a
   larger request is rejected before anything is created. Both sides keep
   the smaller of the two announced counts and drop the surplus queue pairs.
6. **Queue pair state**: `ibv_modify_qp` INIT -> RTR -> RTS. RTR is built
   from the peer description, because it says how to reach and answer that
   peer: GID, LID, MTU, PSN, `sl`, `traffic_class`, the minimum RNR timer and
   the peer's RDMA read limit. RTS is local policy: `timeout`, `retry_cnt`,
   `rnr_retry` and `max_rd_atomic` come from this side's configuration, so the
   peer cannot make us retransmit longer than we intend to. The peer's MTU is
   an upper bound: `path_mtu` is the smaller of the two, because asking for
   more than the local port carries fails the transition. One PSN covers the
   whole link: it is generated once per connection, announced in every connect
   request, used as the send PSN of every queue pair, and the peer's value
   becomes the receive PSN.
7. **Connect done** (table 4, `cmd=3`, 16 bytes): header only.

The out-of-band socket is only used for this exchange. It stays open so that
teardown can be announced, but no data flows over it.

The header checksum is a reflected CRC-32C (Castagnoli, polynomial
`0x1EDC6F41`) over the frame from the `ver` byte onwards: the initial value
is zero, there is no final inversion, and the checksum of an empty buffer is
therefore zero.

Capability adoption is deliberate: the values the store reports become the
operating point of the connection, and a field the store leaves at zero keeps
the locally requested value. Nothing in the handshake requires both sides to
be configured identically.

## Disconnect

`DeleteConnections` sends the disconnect request (table 5, `cmd=4`, 16 bytes)
the first time the last handle of a link is released, then destroys the queue
pairs and closes the socket. The request is a one-way notification, so nothing
waits for an acknowledgement: a peer that is already gone does not fail the
call, and the local resources are released regardless.

## Relation to `link_protocol`

`kv_semantics/src/trans/proto/link_protocol.{h,cc}` describes an older RDMA-CM
style handshake (`gid`, `lid`, `qpn[32]`, `kato`) that nothing in this tree calls.
`roce_wire` is the frame format this provider speaks: big endian everywhere, and
little endian inside the 128 byte private area of the two capability messages.
Keep them apart when reading the code.

## Configuration

Every knob is a `asu_roce_*` key in the store configuration; the store forwards
them into `TransportConfig::attrs` as `roce_*`.

| Key | Default | Meaning |
| --- | --- | --- |
| `asu_roce_qp_backend` | `verbs` | `verbs` for real hardware, `simulated` for protocol tests without an HCA; any other value is rejected instead of silently meaning `verbs` |
| `asu_roce_oob_role` | `client` | `client` connects, `server` listens |
| `asu_roce_oob_bind_ip` | local IP | address the server binds |
| `asu_roce_oob_connect_timeout_ms` | `3000` | TCP connect/accept timeout |
| `asu_roce_handshake_timeout_ms` | `5000` | timeout of every message exchange |
| `asu_roce_hca_name` | first device | HCA to open, for example `mlx5_0` |
| `asu_roce_hca_port` | `1` | port of the HCA |
| `asu_roce_gid_index` | required | GID table index to use; there is no automatic selection |
| `asu_roce_sq_depth` / `asu_roce_rq_depth` / `asu_roce_cq_depth` | `256` / `64` / `512` | queue sizes |
| `asu_roce_inline_threshold` | `0` | inline data size per send |
| `asu_roce_sl`, `asu_roce_traffic_class` | `3`, `104` | service level and traffic class |
| `asu_roce_qp_timeout`, `asu_roce_retry_cnt`, `asu_roce_rnr_retry`, `asu_roce_min_rnr_timer` | `17`, `7`, `7`, `12` | retransmission timers |
| `asu_roce_qp_rd_atom` | `16` | outstanding read/atomic limit |

Numeric values are range checked against the field widths of the verbs
attributes and of the protocol tables. A value that is not a number, or that
does not fit, fails the connection setup with a message naming the key and the
maximum, instead of being truncated to whatever survives the cast: the check
runs on the full 64 bit value, so `4294967296` is reported as too large rather
than wrapping to zero. Zero itself is accepted wherever the field allows it -
`asu_roce_inline_threshold: 0`, `asu_roce_sl: 0` and `asu_roce_gid_index: 0` all
mean what they say - and only `asu_roce_hca_port` treats it as an error, because
ports start at one.

The TCP socket is tuned with a 1 MiB send and receive buffer and a 5 s
`TCP_USER_TIMEOUT`. The kernel doubles the requested buffer and clamps it to
`net.core.wmem_max` / `net.core.rmem_max`, so the effective size can be larger
(up to the sysctl) or smaller.

## Building

```bash
cmake -S kv_semantics -B build -G Ninja \
    -DRUNTIME_ENVIRONMENT=simu \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_KV_CLIENT_PROVIDER_ROCE=ON
cmake --build build -j
ctest --test-dir build -j8
```

`BUILD_KV_CLIENT_PROVIDER_ROCE=ON` requires `infiniband/verbs.h` and
`libibverbs` (package `libibverbs-dev`); configuration fails with a clear
message when they are missing. The provider sources are only compiled when the
option is on, so other builds are unaffected.

## Memory registration

`RegisterMemory` registers host or device buffers against the protection domain
of the HCA with `ibv_reg_mr`, using `LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ`;
the whole buffer is meant to be reachable by the peer. The handle that goes back
to the client is the address of the `ibv_mr`, and `GetMemTokenId` returns the
`rkey` of that region, which is what the peer needs in order to address it.

Three details are worth knowing:

- the call is all or nothing: if one descriptor fails, every region registered by
  that call is released again and `mrHandles` stays empty, so a partially
  registered batch can never be handed to the caller;
- device memory (`MemType::MEM_DEVICE`) takes the same path, because peer memory
  support lives in the HCA driver. When `ibv_reg_mr` is refused the error carries
  a note about `nvidia-peermem` / dmabuf support rather than a bare `errno`;
- `UnregisterMemory` reports one status per descriptor, so an unknown handle does
  not stop the rest of the batch. Regions still registered when the provider is
  destroyed are released then, before the protection domain goes away.

With `asu_roce_qp_backend: simulated` there is no protection domain to register
against, so the bookkeeping runs without verbs and the token is a synthetic key.
That keeps the client path testable on machines without an RNIC, which is how the
memory tests run in CI.

`BindMemory` is pure bookkeeping: the token in the descriptor is recorded
against a fresh handle and `GetMemTokenId` on that handle reports the very same
token. The client checks exactly that equality after the call, so passing the
peer token through unchanged is the whole contract. A bound handle carries no
`ibv_mr`, and releasing it therefore never touches the device.

The device and the registration table are process wide, because the store side
builds one provider per ASU node and they all share one HCA. Devices are kept
per (adapter, port, GID) triple: providers that resolve to the same triple share
one protection domain, while a deployment that drives several adapters - one per
NUMA node, say - gets one device per triple instead of fighting over a single
one. The GID belongs to the identity because the index of the local address ends
up in the global route header of every queue pair, so two providers that only
differ by `asu_local_ip` must not share a port. When the configuration
leaves `asu_roce_hca_name` empty the driver picks the adapter and the resolved
name goes into the key, so two providers that both asked for "any adapter" still
share one. A candidate that turns out to duplicate an open device is closed
again, since the GID can only be read once the port has been queried.
A region registered through one provider stays visible to the others: a second
provider can bind an address the first one registered, and `GetMemTokenId`
resolves the handle from either. The handle space is shared as well, so handles
of two providers never alias.
Sharing goes one way only: only the provider that registered a region may
release it, and a request from another provider fails with `INVALID_ARGUMENT`.
Each provider releases its own regions when it is destroyed, and the bindings
that pointed at them retire with them, so a token can never outlive the region
behind it. A region only ever belongs to the protection domain of its own
(adapter, port) pair, so binding it from a provider that talks to another pair
fails with `UNSUPPORTED`.
The binding range has to fall inside a range that some provider in the process
registered, otherwise the call fails with `UNSUPPORTED` instead of recording a
token that describes memory nobody owns. A provider that has not opened its
device yet adopts the device of the first region it binds, but only after every
descriptor of the batch passed: a batch that fails half way leaves the provider
unpinned, so the caller can retry with the region that did resolve even when it
lives on another adapter. The registration also has to come from
the same backend: a region registered with the simulated backend has no access
record for a `verbs` provider to address, and a real region must not have its
`rkey` replaced by a synthetic token, so a mismatch is refused with
`UNSUPPORTED` as well.

## Verification status

The `lint-and-unit-tests / cpp_gtest` job installs `libibverbs-dev` and
configures with `-DBUILD_KV_CLIENT_PROVIDER_ROCE=ON`, so everything below runs
on every pull request. No RDMA hardware is required: the queue pair backend
defaults to `simulated`, and the handshake tests talk over loopback TCP.

Verified:

- the frame codec, including the checksum covering header and body and the
  check vector of the zero based CRC-32C (`RoceWireTest`, fourteen cases);
- the full link handshake against a peer provider in the same process using the
  `simulated` queue pair backend (`CpuKvOverRoceCreateConnectionTest`);
- capability reporting, adoption of the store limits, and the queue pair
  count taken from the store;
- the disconnect sequence, including teardown of the socket and the queue
  pairs;
- clean failures when the peer is missing or the request exceeds the 32
  queue pair table;
- memory registration and release bookkeeping (`CpuKvOverRoceMemoryTest`), with
  and without a usable handle.

Not verified yet (needs two nodes with RoCE NICs):

- `ibv_reg_mr` of device memory through peer memory on a real HCA;

- the verbs path: `ibv_create_qp`, `ibv_modify_qp` up to RTS, and a first
  registered memory region;
- MTU negotiation against a real switch, and PFC/ECN behaviour;
- device memory registration (`ibv_reg_mr` with peermem/dmabuf);
- reconnect after a link failure. The client rebuilds a dropped channel, but
  this provider does not retry on its own.

## Troubleshooting

- `no RDMA device found (is the RNIC driver loaded?)` - no HCA is visible. On a
  host without hardware use `asu_roce_qp_backend: simulated`; in a container
  pass through the device and mount `/dev/infiniband`.
- `out-of-band TCP connect ... failed: Connection refused` - the peer is not
  listening, or `asu_roce_oob_role` is wrong on one side.
- `asu_qp_num` above the 32 entries of the queue pair table - the request is
  refused before a queue pair is created; a store that advertises a smaller
  `queue_num` is honoured and the surplus queue pairs are dropped.
- `checksum mismatch` - the peer does not speak this protocol version, or a
  middlebox rewrote the payload.
- `ibv_modify_qp(RTR) failed` - GID or MTU mismatch between the two hosts.
