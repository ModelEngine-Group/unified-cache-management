# Native allocation / protected prefix validation

## Identity and scope

UCM source branch `codex/ucm-native-allocation-prefetch` starts exactly at
`854f99fd2e171041e72aca4b8067496f7fe1c6e7` (`codex/ucm-confirmed-budget-546f6a8`).
Its isolated worktree is
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-native-allocation-prefetch`.
No later source commits were merged/cherry-picked. The original branch/worktree
was preserved. This task does not push.

Management is a separate repository at
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review`, HEAD
`97dfba618fd8d680ac91050972aabbc08ded6166`. Its existing changes are not ours.
No management commit is made.

Implementation commits:

- `f7a123e`: remove local byte charges; native allocation returns a typed
  capacity-truncated prefix after draining partial-block reads. Add explicit
  Early batch suffix discard; preserve actual backend failure.
- `09ba415`: remove broker capacity ledger and approval messages; propagate
  final prefixes through worker, owner result, metadata and release. Protocol v4.
- `9110e70`: binding/metadata regressions and accurate reference-byte metric;
  remove obsolete hypothetical dump-size handshake fields.
- `99c4550`: update handshake test expectations for the removed estimate.

No native private access attribute changed. Relative to this task's start,
no native Store/CacheStore/TransBuffer/queue method changed or was added.
`ucmstore_v1.h` only extends the already Early-added PrefetchResult structure.
The other implementation changes are Early components/friend adapters. Original
load/dump task granularity, Scheduler/PENDING and native load failure policy
remain unchanged. vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` and vLLM-Ascend
`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` are unchanged.

## Protocol and capacity boundaries

See EARLY_PREFETCH_DESIGN.md for state/sequence and inheritance/composition.
Reserve's lookup hit count is provisional. Batch completion returns the final
whole-block count and an explicit capacity_truncated boolean. Owner READY is
sent only after lookup has exited, retained batches complete, and discarded
suffix I/O drains. The broker checks owner/attempt, integer range and result
flag type. Scheduler uses the final count through existing native token/metadata
logic; it does not query a larger unprotected suffix. Zero is a valid READY hit.

A later failed allocation does not invalidate earlier complete blocks. A partial
block never contributes to READY. Later already-submitted batches are explicitly
discarded and polled to drain; actual I/O errors remain failures. Cancellation
uses the existing terminal/drain protocol; control loss fences the shared domain.
No message or error string is interpreted as proof of process-death recovery.

The allocator's finite attempts, native reference count, finite shared pool and
load-exclusive range enforce memory safety. There is no claims/used counter or
capacity approval queue. Only newly allocated native owner nodes submit reads; shared in-flight nodes
coalesce. Thus actual simultaneous destination bytes are also bounded by the
physical reservable pool, even when the sum of I/O windows is larger.
Per-worker 128-shard I/O windows and owner queued-work
limits remain; these are submission/work bounds, not speculative memory claims.
The physical reservable size can be zero. Step still executes to return a zero
capacity prefix instead of waiting forever for a zero I/O quota.

protected_reference_bytes is a per-reservation reference footprint, not a
shared physical-memory union. Repeated positions/references may be counted more
than once. The log is observational and never authorizes allocation. Read leases
can retain the whole candidate batch until reservation release; no new per-block
lease protocol is added. Shared pins protect each native consumer through H2D;
DRAM contents remain evictable/cacheable after the last reference is released.

Bounded native load/dump preparation from the starting commit is preserved.
A whole native task that cannot allocate is rejected before I/O; dump rejection
must not claim persistence. Native kv_load_failure_policy remains authoritative
(and is `fail` in these smokes). This is not a promise that arbitrary oversized
load/dump tasks always succeed under memory pressure.

## Evidence and reproduction

Evidence root:
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/native-allocation-20260917`.
Runtime root:
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/runtime/native-allocation-20260917`.
Build directories are `build/native-allocation-a3` and
`build/native-allocation-simu`; they never overwrite one another.
The pinned vendor source archive and wrapt 1.17.2 wheel are reused offline;
no dependencies upgraded and no old compiled UCM libraries reused.

Entries in the evidence root: build_fixed.py (whole A3 build/install),
build_simu.py (whole separate simu build), verify_build.py (source/library
manifest and actual imports/maps), launch.py/sequence.py/exercise.py (isolated
GLM requests, own-container stop only). The source test/python_with_build.py
selects simu extensions without installing them over A3. All scripts require
absolute workdirs and preserve prior case directories.

Use the locked official image
`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`.
A3: RUNTIME_ENVIRONMENT=ascend-a3, BUILD_UCM_SPARSE=OFF,
DOWNLOAD_DEPENDENCE=OFF, Release. Simu: RUNTIME_ENVIRONMENT=simu,
BUILD_UNIT_TESTS=ON. Rebuild/install the entire UCM component: PrefetchResult
and Early::Batch interfaces and protocol VERSION=4 changed; mixed libraries or
v3/v4 workers are unsupported.

Tests are run with pytest --noconftest for these isolated modules; the upstream
GPU benchmark conftest requires unavailable pynvml and is not their fixture.
No dependency is installed to work around that unrelated benchmark setup.
Initial host pytest failed on missing pygments; the locked container is used.
Initial binding-pressure setup was rejected by native minimum pool validation;
the corrected test keeps valid sizing and creates pressure via another actual
consumer's native references. These harness failures remain in evidence logs.

## Results and limitations

Historical output mismatch evidence in CONFIRMED_BUDGET_VALIDATION.md remains
valid as an unresolved gate: previous reduced-GLM ON/OFF matched 16/20 exact
outputs and OFF/OFF matched 17/20. Neither HTTP 200 nor a correct byte-transfer
test certifies model-output equivalence. No performance improvement is claimed.

All builds and runs below use `99c4550a251cc907b33e47dbec0d0d4c97d15b51`.
The final documentation-only descendant is not represented as a different
tested binary. Full source-file/library SHA256 manifests, ELF build IDs, import
paths and actual process maps are preserved in the evidence root.

| Verification | Result | Evidence boundary |
| --- | --- | --- |
| Whole A3 build/install and separate simu build | Pass, 16 libraries per manifest | Native aarch64 A3 compilation, not copied CUDA/x86 artifacts |
| Five Python protocol/scheduler/worker/metadata modules | 146 passed, 15 warnings | Mocks plus real threads and ZMQ processes; not all buffer transfers are native |
| C++ `*Early*:*ReadLease*` | 51 passed | Native buffer, fork/shared memory, native load queues, actual Posix lease/GC; H2D simulated and some backends mocked |
| Real CPython/C++ pipeline binding | 5 passed, 14 warnings | Actual Posix files and simulated H2D; includes capacity-truncated native load handoff |
| Python AST and `git diff --check` | Pass | Static syntax/whitespace only |
| GLM no-layerwise OFF / ON / repeat OFF | 20/20 transport-valid requests in each arm | Real NPU; output-equivalence gate fails, see below |
| GLM layerwise ON regression | 20/20 transport-valid requests | Real NPU; separate layerwise OFF seed, not a full paired layerwise OFF trial |
| Native and Early NPU byte-transfer probe | 3/3 cases each, zero byte differences | Real SDMA/H2D/D2H, one process/device, GLM 8-layer direct KV layout |

The C++ tests exercise partial-block allocation failure with delayed accepted
I/O, zero reservable pool with usable native load-exclusive nodes, duplicate
blocks without duplicate physical allocation, multi-consumer pressure, last
reference release followed by eviction, explicit tail discard, actual backend
failure, cancellation/drain, independent requests making progress, load/dump
preparation rejection and recovery, and shared consumer references after owner
release. The Python worker tests cover a later batch failing capacity while
earlier work survives and later active I/O drains; zero READY; cancellation;
invalid/late owner answers and unchanged native metadata/fallback.

## Actual GLM configuration and observations

This is a reduced-layer functional smoke, not the user's full workload.
Every launch was preceded by a device check and a displayed effective plan.
`plan.json`, `ucm.yaml`, seed-file hashes and `npu-before.txt` are saved per arm.

| Item | Effective value |
| --- | --- |
| Model/tokenizer/weights | Read-only `/data/model/GLM-5.1-w4a8`; 8-layer override and read-only model view recorded in each plan |
| Topology | Chips 12–15; TP=4, DP/local DP=1, EP enabled; one container/server |
| Compute | BF16, Ascend W4A8, BF16 KV; eager; APC and chunked prefill on |
| Limits | Context 32768; sequences 8; batch tokens 4096; HBM utilization 0.60; KV blocks overridden to 256; seed 20260725 |
| Store | Cache\|Posix; shared host DRAM L1=3 GiB; per-arm L2 limit=1024 GiB; native unique_id/owner |
| I/O | AIO, O_DIRECT; data/lookup/open/commit concurrency=1; timeout=300000 ms; GDR off; native SDMA direct on; stream count=1 |
| Connector | UCMConnector; layerwise false for OFF/ON/repeat OFF, true for separate regression; load/persist thresholds=0, event sync on |
| Early | OFF or ON as recorded; owner enabled; protocol v4; native allocator capacity, 128-shard submission window; no capacity claims/grants |
| Environment | AIV; FUSED_MC2=0; HCCL_BUFFSIZE=200; OMP=1; interface enp23s0f3; IP 192.168.0.118; ENABLE_UCM_PATCH=1; task-private IPC |
| Framework/runtime | Locked commits above; image Python 3.12.13, CANN 9.0.1, torch 2.10.0+cpu, torch_npu 2.10.0.post2 |

L1 here is host DRAM, distinct from each rank's HBM KV pool. The tested image
digest is authoritative; package-version banners are preserved rather than
inferred from earlier experiment names.

The direct OFF seed produced 27 KV files / 38,928,384 bytes. Each direct arm
received a separate, SHA256-identical seed copy and a fresh L1. The sequence was
cold L1/hot L2, repeated HBM hit, 12 task-local HBM replacement prompts, warm L1,
four concurrent repeated requests and an unrelated miss. Layerwise had its own
OFF seed because its shard layout differs. No peer/global L1, L2 or IPC was
cleared. Only task-labeled containers were stopped; all eight exited with code
0. Existing services were left running; Qwen was never started.

Direct shared-domain handshake: four workers verified the same native object
and layout, 2234 nodes, 1210 reservable nodes, 1,441,792-byte shards. Actual
reservable DRAM is 1,744,568,320 bytes; native load-exclusive space is
1,476,395,008 bytes. Layerwise: four workers, 17873 nodes, 16849 reservable,
180,224-byte shards, eight native layer ranges. These are allocator capacities,
not per-owner admission credits. See `shared-owner-final.json` for raw identities.

| Runtime observation | Direct ON | Layerwise ON |
| --- | ---: | ---: |
| Owner assignments / releases | 20 / 20 | 20 / 20 |
| Owner outcomes: prefetched / L1-ready / zero-hit | 1 / 3 / 13 | 1 / 5 / 13 |
| Native load-dispatch records | 8 | 64 |
| Actual native post-prefetch L2-read shards | 0 | 0 |
| Read locks held at final snapshot | 0 | 0 |

Outcome counts exclude jobs retired by the existing HBM-sufficient path before
an owner READY answer. Per-generation logs verify only the assigned owner rank
reserves. `native_get_fallback` is a native buffer-index lookup and does not mean
an SSD read; actual reads are counted separately above. Owner release receipts
show normal completion; they do not prove recovery from killed processes.
Shutdown retained Python multiprocessing resource-tracker warnings about
semaphores/shared-memory objects. Those warnings are not silently reclassified
as a proven leak or as a clean shutdown of every runtime resource.

### Output consistency remains unresolved

All 20 responses in each arm have HTTP 200 and the requested 16 integer token
IDs. This is transport validity only. Exact greedy token comparisons:

- OFF vs ON: **16/20** identical; `evict/2`, `evict/3`, `evict/4`, `miss/0` differ.
- OFF vs repeat OFF: **15/20** identical; `evict/2`, `evict/4`, `evict/6`,
  `evict/10`, `miss/0` differ.
- Repeat OFF vs ON: **15/20** identical.

The differences also occur without Early, but that does not identify the root
cause or exclude an Early contribution. Reduced-layer output quality remains
limited; repeated/degenerate tokens are preserved in the raw responses. Full
model semantic correctness and ON/OFF equivalence are **not passed**.

The additional NPU byte probe checked cold L1/hot L2, warm L1 without owner
handles, and bounded dump/reload. Native and Early returned exact tensor bytes
in all cases; Early read 27 shards / 38,928,384 bytes and reported 27 complete
blocks, `capacity_truncated=false`. This narrows the transfer evidence but does
not close the model-output gate.

No stable-backlog/full-78-layer TP8/DP2 performance experiment was run. Recorded
request durations are non-streaming response durations, not a TTFT study.
Capacity truncation under real multi-rank NPU pressure, NPU cancellation/death
injection and long-run resource stability remain unverified. There is no claim
of TTFT, throughput or output-correctness improvement.

## Reproduction and recovery notes

Keep the existing results immutable. Use new case directories/container names
for reruns. `launch.py` refuses an existing case and validates source/library
hashes before starting. It needs the small locked helper files
`evidence/owner-protocol-20260914/{glm.lock.json,model_entry.py}`, included in the
evidence recovery archive, plus the explicitly recorded read-only model/system
runtime. Its `--execute` path checks selected devices and service port first.
The 8-layer model view must be recreated at the recorded absolute path or the
launcher adjusted explicitly; it is not a copied set of weights.

Build scripts run inside the locked image with the new source as explicit
working directory. `build_fixed.py` performs configure/build/install and writes
the complete A3 manifest; `build_simu.py` builds separately; `verify_build.py`
checks imports/maps and installed ELF identities. For an exact replay of these
results restore `99c4550`; for the documentation-only descendant rebuild all
libraries and regenerate manifests before launching. Do not edit manifests to
bypass a commit or library mismatch.

Run these native test entries from the source workdir, with absolute paths to
the locked venv/build:

- `python -m pytest --noconftest`: `test_early_batch_pipeline.py`,
  `test_early_owner_protocol.py`, `test_early_owner_budget.py`,
  `test_early_native_scheduler.py`, `test_early_ready_reuse.py`.
- Simu `ucm/store/test/ucmstore.test --gtest_filter=*Early*:*ReadLease*`.
- `test/python_with_build.py --build <absolute-simu-build> --noconftest
  <absolute-source>/test/test_early_batch_binding.py` through the locked venv.
- `test/run_early_npu_buffers.py native|early <new-absolute-output-dir>` on an
  idle authorized NPU; do not run simultaneously with the GLM service.

The delivery includes a complete UCM Git bundle for this branch (no prerequisite
repository needed), both `854f99f..HEAD` and native
`5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1..HEAD` patches, and a small evidence
archive. SHA256s, bundle verification and the final documentation commit are in
`delivery.json` under the evidence root. No model weights, credentials, L2 KV
files, compiled libraries or large CANN logs are included. Pinned vendor source
archive `recovery/owner-protocol-vendor-sources.tar.gz` and the offline wrapt
wheel are existing rebuild inputs; both are recorded in the delivery manifest.
