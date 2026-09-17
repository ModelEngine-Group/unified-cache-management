# Confirmed-prefix capacity validation

## Source identity and delivery gate

- New UCM source branch: `codex/ucm-confirmed-budget-546f6a8`.
- Exact start: `546f6a8fb32eb7a6849b0928f6c9f9b92c22c671`.
- Functional A3/simu build and GLM test commit: `b54481f12396d85f06b1c440d372952820792278`.
- Implementation commits: `ab415d0` (bounded allocation/local confirmed charges),
  `e0abd1d` (shared confirmed-batch grants), `b54481f` (real binding rejection tests).
- Source worktree: `/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-confirmed-budget`.
- Original UCM worktree and branch were not modified; its HEAD remains
  `3ceda2b14556e4f1e52f79e436568b02a57595bc`. No descendant of `546f6a8`
  was cherry-picked or merged into this branch.
- Management repository is separate: `/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review`,
  branch `codex/ucm-a3-local-prefetch-perf-evidence`, commit
  `97dfba618fd8d680ac91050972aabbc08ded6166`. No management commit created;
  existing unrelated tracked changes were preserved.
- Push target, only when the smoke gate is satisfied:
  `review-origin = https://github.com/yhli-ml/glm51-vllm-ascend-ucm-clean.git`,
  new source branch only. `origin` is upstream ModelEngine and must not be pushed.

**Delivery gate: model output equivalence is not established; remote push is held.**
HTTP requests and resource protocol pass, but exact GLM output differs in ON/OFF
and even repeated OFF/OFF. Byte-exact NPU store-transfer checks pass. These
observations do not prove either model correctness or an Early regression.
No TTFT/performance improvement is claimed.

## New changes relative to 546f6a8

| Location | Change and necessity |
|---|---|
| `early_owner.py::OwnerBroker.compatible` | Domain limit is the actual native `reservable_nodes * shard_bytes`. Retains native load-exclusive area; removes hypothetical largest load/dump subtraction. A zero physical limit stays zero. |
| `OwnerBroker.dispatch / confirm_batch` | Dispatch charges zero bytes. Existing batches first confirm an L1/L2 continuous hit, then request an atomic unique-block grant on the existing broker control thread. Work-queue limits remain separate from byte credit. Duplicate/stale grants are idempotent/rejected. |
| `early_owner_worker.py`, `early_prefetch.py` | Generation/attempt scoped batch-grant exchange. Denied batches never enter the data queue; prior accepted I/O drains through the existing path. Existing reserve/read overlap, cross-request HRRN and bounded I/O window remain. |
| `reserved_localize_impl.h::NewGroup / ChargeConfirmed`, `reserved_prefetch.h::Reserve` | Local shared BlockPins claims are charged only after the continuous prefix is known, not at candidate localization. Last holder releases pins before the associated local claim. |
| `early_buffer_pins_impl.h::Acquire` | Existing friend uses the same native FindAt/Alloc/reference operations with the existing finite allocation attempt argument. No private visibility changes or new TransBuffer methods. Negative lookup is revalidated under native locking. |
| `cache_store.cc::EarlyCacheStore::Load / Dump / SubmitBounded` | Early-only subclass prepares references for one unchanged native task. Reuses reservation handles; missing handles get bounded native allocation. Rejects before queue/I/O submission if the complete task cannot be protected. |
| `early_task_pins.h::EarlyTaskPins` | RAII rollback marks abandoned allocation owners FAILED, then releases references. Does not falsely publish READY or leave an unowned LOADING node. Accepted handles move into native queues. |
| `trans_task.h` | Existing optional Early handle carrier gains rollback RAII. No native new method. |
| `dump_queue.cc::DumpOneTask` | Two-line optional prepared-handle selection; absent handles keep original Get. Entire task, stream, D2H, synchronization and backend dump remain native. This is necessary to avoid reacquiring references with an unbounded Get after admission. |
| `pipeline/cpy/early_bindings.h` | EarlyDump reuses private native MakeTaskDesc via the existing friend. EarlyLoad/EarlyDump classify pre-I/O NoSpace as a subtype of native StoreNotFoundError. Same native store/health wrapper, no second manager or task parser. |
| `pipeline/early_connector.py::_ReservedLoad`, `early_consumer.py::Consumer.rejected` | Route Early load/dump through bounded submission. Distinguish guaranteed pre-I/O rejection from unknown I/O failure; rejected receipts do not acknowledge consumption and cannot release other live native tasks. |
| `integration/vllm/early_connector.py::start_load_kv` | Direct-only optional attribute check fixes base 546f6a8's access to the layerwise-only `_failure_req_ids`; required for requested no-layerwise smoke. |

Only two native declarations/functions are newly changed in this round:
`TransTask`'s existing optional handle member and `DumpQueue::DumpOneTask`'s handle
selection. `cache_store.cc` changes are in the already-added Early subclass;
native CacheStore Setup/ParseConfig/Load/Dump are unchanged. No private variable
or function visibility change. No Scheduler, PENDING, LMCache, vLLM or
vLLM-Ascend source changes. No native task splitting, dependency upgrade,
lookup-thread increase, I/O concurrency tuning, or producer-wait restoration.

Relative to original upstream `5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1`, the branch
also contains all previously reviewed Early integration, native friend bridges,
read-lease deletion guard, native load handle carrier and owner protocol.
Those are inherited changes, not all new work in this round. Full two-base
patches and statistics are in the evidence directory (`from-start.patch`,
`from-native.patch`, corresponding `*-stat.txt`).

## Capacity and failure contract

The broker's unique confirmed-block union is a grant ledger, not a measurement
of all physical buffer use. A candidate probe may temporarily hold existing
native READY/LOADING references and an L2 read lease before credit; it allocates
no new data slots and submits no reads. One outstanding grant per lookup job
and finite lookup executors bound this temporary work. The native finite node
pool is the physical capacity boundary. A denied grant releases its probe;
accepted earlier batches drain normally. No partial-credit waiting loop.

Original load-exclusive slots remain unavailable to prefetch/dump. There is no
additional maximum-dump reserve. An oversized or contended dump may be rejected
before I/O and must not be reported as saved. Loads use native missing/error
handling after rejection: native `kv_load_failure_policy` determines failure
versus recomputation. **The tested configuration is `fail`, not `recompute`.**
This round does not change that policy or promise all native loads succeed
under arbitrary pressure.

Bounded complete-task acquisition shifts fallback index work before queue
submission. This trades possible submission latency/dump skips for finite
allocation and no partial-task allocation deadlock. Native successful task
granularity is unchanged; performance must be measured separately.

Owner credit is returned only after all required native consumers quiesce and
owner I/O drains. Unexpected submission/Wait errors still fence the shared
domain; process death is not treated as proof that native references can be
recovered. Native AIO persistence-order concerns are not redesigned here.
Read leases protect deletion, not producer write completeness.

`native_get_fallback=0` now counts only Get calls inside LoadQueue. Early bounded
submission may already have indexed/allocated a non-owner handle. It does not
prove cross-rank handle reuse, zero total index work, or zero SSD reads. Use
actual read counters separately.

## Executed checks and their evidence scope

Evidence root:
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/confirmed-budget-20260917`.
Runtime logs/requests:
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/runtime/confirmed-budget-20260917`.

| Check | Result | Boundary |
|---|---|---|
| Python scheduler/READY/owner/budget/batch tests | 146 passed | Mock protocol, request identity, missing/fallback, grants, cancellation, no new scheduling semantics; `python-patched-final.log`. Standard startup patch active. |
| C++ `*Early*:*ReadLease*` | 47 passed | Real shared memory, forked native LoadQueue with simulated H2D, real Posix leases/GC owner, short-hit, pressure, concurrent load/dump, rollback, drain/hints/LoadPlan; `cpp-final-b54481f.log`. One matching test is native health-breaker coverage. |
| CPython/C++ binding tests | 4 passed | Two routing mocks and two real Posix/simu transfer/lifetime/capacity tests; `binding-final-b54481f.log`. Oversized load/dump reject before file writes, later dump/reload byte equality. |
| A3 native build/install | Passed | Whole matched UCM library set, not borrowed .so; `a3-final-build.log`, manifests and runtime maps. Simu artifacts are separate and never installed over A3. |
| Static | Passed | AST parsing of changed Python modules, `git diff --check`; ruff not installed or claimed. |
| Byte-exact real NPU GLM-layout store probe | Native 3/3, Early 3/3 passed | 27 blocks, 24 tensor sizes `[131072,16384,32768]*8`, real SDMA Direct H2D/D2H, shared L1, AIO consumer. Cold L2, all-L1 without owner handles, dump/reload all 0 mismatched bytes. Single process/device; does not prove TP behavior or model quality. |
| GLM no-layerwise OFF / ON | Each 20/20 HTTP/token-count valid | Real TP4 NPU paths and owner protocol. Exact output matches 16/20; correctness gate not passed. |
| Identical OFF repeat | 20/20 HTTP/token-count valid | OFF/OFF only 17/20 exact output matches. Reproduces baseline nondeterminism; does not by itself exonerate ON. |
| GLM without KV connector | 20/20 HTTP/token-count valid | Same reduced model, framework and compatibility patches; repeated-token output also occurs without any UCM load/dump. Does not establish the cause of ON/OFF differences. |

The C++ capacity test fills the real native pool, attempts concurrent complete
load/dump tasks, checks finite rejection, no false persistence/READY and
recovery after release. Short-hit tests provide 64 candidates with only one
hit in a two-block protection budget. Protocol tests cover zero physical
capacity, high hypothetical native task maxima, duplicate shared blocks,
independent domains, delayed consumers, rejected receipts and late attempts.
Real GLM service requests did not force actual native NoSpace; that path's
pressure evidence is C++/binding, not a model/NPU failure-injection test.

Development failures are preserved: first GLM 2 GiB configuration was rejected
by native minimum 3 GiB validation; fixed matched tests use 3 GiB. Standalone
byte-probe setup iterations corrected a missing L2 directory, unaligned NumPy
O_DIRECT seed and a probe-only config error (Posix tensor_size passed to Cache
instead of tensor_size_list). These were harness failures. Only `buffer-v4-*`
results are valid NPU transfer evidence; no implementation changed to conceal
those failures. No EP/AICPU startup exception occurred in the final GLM runs.

## GLM effective matched configuration

- Image `quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`.
- vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, vLLM-Ascend
  `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`, both unchanged.
- Read-only `/data/model/GLM-5.1-w4a8`, **8-layer override**, TP4, DP/localDP1,
  EP on; devices 12–15; one container/serve, private IPC.
- BF16 model/KV, ascend W4A8, eager, APC/chunked prefill, context32768,
  max-seqs8, batch4096, HBM0.60, KV blocks256, seed20260725.
- Cache|Posix, native shared L1 3 GiB, L2 cap1024 GiB in independent task-owned
  directories. no-layerwise, AIO, O_DIRECT, Posix queue concurrencies1,
  timeout300000, GDR off, native SDMA Direct on, stream1; thresholds0.
- Event sync on, health breaker disabled for matched tests, same native
  `kv_load_failure_policy=fail`. Owner enabled; ON/OFF differ in Early switch
  and independent resource paths. Both receive byte-identical completed OFF
  seed files; each starts with fresh L1/HBM.
- Critical env: AIV, FUSED_MC2=0, HCCL_BUFFSIZE=200, OMP1,
  interface enp23s0f3/IP192.168.0.118; same UCM native compatibility patch.
- Import and every loaded UCM library path comes from `src/ucm-confirmed-budget`;
  `a3-fixed-library-environment.json`, plans and `/proc/maps` snapshots record it.

Native capacity handshake: 2234 nodes, 1210 reservable, 1,441,792 bytes/shard.
The old bound subtracts 1024 maximum load slots and 2048 maximum dump slots,
making the limit zero. New measured-domain limit is **1,744,568,320 bytes**.
The cold request has 27 candidates: owner dispatch charges **0**, confirmed
batch charges **38,928,384**, actual prefetch reads that amount. All 4 consumers
load with **0 subsequent SSD reads**. Warm all-L1 request reads **0 bytes** and
uses no data batch. All **20** generation owners return credit; final used=0,
read lease snapshots empty, task-owned services exit normally. ON outcomes:
one prefetched, one positive L1 fast path, 13 zero-hit, five native-HBM bypass.

Exact 16-token comparison fails for ON warm, miss and two eviction requests.
OFF/OFF also differs for HBM, miss and one eviction request. OFF full-compute
seed versus OFF L2 reload differs after the first token. These are preserved
in `smoke-summary.json` and per-case `*-requests.json`. A no-KV-connector control also produces the same repeated-token sequence on
19 requests. This isolates the degenerate output from any requirement for
Early or native UCM transfer, but does not explain the ON/OFF mismatches.
Eight-layer outputs are not a full GLM semantic-quality evaluation. Full78-layer TP8×DP2, stable-load
TTFT/throughput, NPU capacity-failure injection and real multi-DP remain untested.

## Build and restore

Use the pinned official A3 image/runtime and matching framework sources above.
Existing locked vendor sources are in the management recovery archive;
no downloads/upgrades or old compiled libraries are needed. Configure A3 and
simu in **separate** absolute build directories. A3 configuration used:
`RUNTIME_ENVIRONMENT=ascend-a3`, `CMAKE_BUILD_TYPE=Release`,
`BUILD_UCM_SPARSE=OFF`, `DOWNLOAD_DEPENDENCE=OFF`, `BUILD_UNIT_TESTS=OFF`;
install the entire `ucm` component into this source worktree. Simu uses
`RUNTIME_ENVIRONMENT=simu`, `BUILD_UNIT_TESTS=ON`, and is not installed into it.

Executable local entries are `build_fixed.py`, `build_simu.py`,
`verify_build.py`, `launch.py`, `sequence.py`, `analyze_smoke.py` in the evidence
root. Each launch verifies source hashes/library identities and device idleness;
never run it against a peer's active devices. Scripts require new case names
and never overwrite earlier results or clear peer caches. The tests in this
source tree are portable; `run_early_npu_buffers.py` requires an explicitly
allocated real NPU and a fresh output directory.

Restore source with the task's Git bundle (path and SHA recorded in the final
local evidence), or when authorized/pushed clone the new branch from
`review-origin`. Restore this full history, not only a management submodule
pointer. Check out the desired commit and rebuild **all** UCM libraries;
TransTask's C++ type and protocol VERSION=3 changed, so mixing old/new libraries
or workers is unsupported. Test/doc-only delivery commits may follow the
functional build commit; their production-source diff must be empty.
