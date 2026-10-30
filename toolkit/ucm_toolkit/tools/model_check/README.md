# UCM Model Compatibility Checker (`model-check`)

Toolkit tool: `ucm-toolkit run model-check`

Verifies a model's **native vLLM KV-cache layout and UCM dump/load compatibility
without loading checkpoint weights**:

1. Build the real `VllmConfig` (+ UCM connector configuration).
2. Construct the model structure on the `meta` device (no checkpoint weights).
3. Obtain the production `ModelRunner`'s `KVCacheSpec`.
4. Use vLLM's native grouping/config APIs to produce `KVCacheGroupSpec` /
   `KVCacheConfig` / layer-to-group maps / tensor shapes and strides.
5. Allocate a bounded KV-cache pool via `num_gpu_blocks_override`.
6. Create a real vLLM `Scheduler`, submit synthetic source/target requests, and
   let the Scheduler-owned `KVCacheManager` allocate group-aware block tables.
7. Fill source KV blocks with deterministic values, dump them through UCM, load
   them into distinct target blocks, and compare byte-for-byte.
8. Re-check the round trip without using any address the connector computed
   (see [Connector-independent checks](#connector-independent-checks)).

Running the checker requires only a model directory with `config.json` (+
tokenizer files for multimodal models) — no weights, no model download.

## Platform launchers

The checker ships one launcher per serving stack in this directory
(`adapter.py` auto-selects the launcher by the installed stack):

| launcher | vLLM stack | device | KV-cache layout produced |
|---|---|---|---|
| `ascend.py` | vLLM + vLLM-Ascend | `npu` | **Ascend-specific** layout (vLLM-Ascend patches: non-packed tuples, int8 SFA C8 pages, mamba-aligned blocks, …) |
| `cuda.py` | official vLLM (CUDA build) | `cuda` | **Official vLLM layout** |
| `cpu.py` | official vLLM (CPU build) | `cpu` | **CUDA layout simulation through the native GPUModelRunner allocator** |
| `ascend_sim.py` | vLLM + vLLM-Ascend, **no NPU** | `npu` platform, host memory | **Ascend layout from vLLM-Ascend's own spec/grouping/allocator**, with KV tensors in host memory |
| `cuda_sim.py` | CUDA build of vLLM, **no GPU** | `cuda` platform, host memory | **Official vLLM layout from the real CUDA attention backends**, with KV tensors in host memory |

**Layout-identity notes**

- `cpu.py` uses the installed official vLLM GPUModelRunner's spec, grouping,
  config and allocation code with real CPU tensors. It substitutes unexecuted
  attention/MoE computation dependencies and CUDA stream/event interfaces.
  Backend constraints must be checked against the target vLLM source; a CPU
  PASS does not prove CUDA hardware, kernel execution or full serving support.
  For Qwen4Exp it also skips only the construction-time QSA FlashAttention
  availability probe, keeping the native owner/spec/cache construction.
- The standalone checker publishes the resolved layout name into the worker
  KVCacheConfig, matching EngineCore's hand-off after native config creation.
- `ascend_sim.py` is the device-less counterpart of `ascend.py`: the same
  vLLM-Ascend model construction, `KVCacheSpec`, grouping, `KVCacheConfig` and
  `NPUModelRunner.initialize_kv_cache()`, with KV tensors allocated in host
  memory. It needs the CANN **driver libraries** present (mount
  `/usr/local/Ascend/driver` read-only) but **no** `/dev/davinci*` device, and
  UCM built with the `simu` runtime. What it substitutes, and why each is
  layout-neutral:
  | substitution | reason |
  |---|---|
  | device selection / streams / events / `synchronize` are host no-ops | they order execution, they do not shape tensors |
  | `TRITON_ASCEND_ARCH` pinned from the image's `SOC_VERSION` | triton's own documented override; vLLM-Ascend's import chain asks triton for the arch at module import, which reads the device |
  | custom ops reported unavailable (its own `_CUSTOM_OP_ENABLED` cache) | the extension load aborts the process without a device; vLLM-Ascend already takes this path on SoCs that disable runtime custom ops, and no kernel runs here |
  | pinned host memory reported unavailable | page-locking a CPU tensor goes through torch_npu's allocator (`aclInit`); pinning changes no shape, stride or dtype |
  | Engram `HostUvaBuffer` / `SharedUvaBuffer` kept on `meta` | they hold the codebook **weights** (`weight` / `weight_scale_inv`), which this checker never loads; the real buffers need `aclrtMallocHost` and tens of GiB of pinned memory |
  | `_cube_core_num()` placeholder | consumed only as a compressor kernel's `num_cores=` launch parameter — it allocates nothing and sets no shape |

  The SoC family is **not** simulated. `get_ascend_device_type()` reports the
  family the installed wheel was **built for**
  (`vllm_ascend._build_info.__device_type__`), so `--ascend-soc` only asserts a
  match: checking another family's layout needs a vLLM-Ascend built for it.
  A PASS here does not cover NPU copy kernels, ACL event ordering or
  performance.
- `cuda_sim.py` is the same idea for the official layout, and needs a **CUDA
  build** of vLLM (the one shipping `vllm.vllm_flash_attn`): vLLM's attention
  backends import their kernel entry points at module level behind
  `if current_platform.is_cuda()`, so the real backend classes -- the ones that
  compute the KV shapes and layouts -- are only importable there. The launcher
  refuses a CPU or `+empty` build with that explanation instead of failing deep
  inside vLLM. Two substitutions are specific to it:
  | substitution | reason |
  |---|---|
  | the CUDA platform probe is replaced by a declaration | vLLM picks its platform by asking NVML for a device count; with no GPU the CPU platform would win. This answers "is this hardware present", not "what does the layout look like", and it must run before anything touches `current_platform`, which resolves once and caches |
  | the compute capability is **declared** with `--cuda-capability` | it cannot be read without a device, and vLLM branches on it (FlashAttention version, fp8 support, MLA backend choice). The launcher refuses to guess -- the same rule `ascend_sim.py` applies to the SoC family. A PASS is therefore a statement about the layout vLLM produces *for that capability*, and the value appears in the log |

  Its kernel-availability probes are answered from that declaration rather than
  skipped, so backend **selection** stays native and the chosen backend shows
  up in the layout log. A PASS does not cover CUDA copy kernels, stream/event
  ordering or performance.
- The substitutions both device-less launchers share (host streams and events,
  pinned memory, UCM's host device) live in `sim_runtime.py`, so the two stay
  identical where they should be.
- The **Ascend layout is a different design**: page sizes/dtypes/shared-tensor
  structure differ (e.g. GLM-5.2: Ascend int8 packed 656 B/token vs official
  bf16 1,152 B/token; DSV4: Ascend bf16 non-packed tuples vs official
  fp8_ds_mla). Results from `ascend.py` must not be compared against
  `cpu.py`/`cuda.py` outputs as if they were the same layout.

## Environment

- **UCM** built from source with the matching platform:
  ```bash
  export PLATFORM=cuda     # or: ascend / ascend-a3 / musa / maca
  # CPU/simulated runtime (no accelerator libraries needed):
  export PLATFORM=simu     # any unset/unknown value also falls back to simu
  pip install -v -e . --no-build-isolation
  ```
- **toolkit** (provides the `ucm-toolkit` CLI):
  ```bash
  pip install -e ./toolkit --no-deps --no-build-isolation
  ```
  `--no-build-isolation` matters on the vLLM-Ascend images: they already ship
  setuptools, and an isolated build tries to download it and fails.
- **wrapt**, which UCM's patch helpers import and these images do not ship:
  ```bash
  pip install wrapt==1.17.2
  ```
- **Runtime library path**: UCM's C++ core (`ucmpipelinestore` and friends) is
  installed under the `ucm/` tree; add those directories to `LD_LIBRARY_PATH`
  while **keeping** the platform's own library dirs:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(find <ucm_repo>/ucm -name '*.so' -printf '%h\n' | sort -u | paste -sd:)
  ```
- **vLLM**: `ascend.py` needs vLLM-Ascend; `cuda.py` needs a CUDA vLLM build on
  an NVIDIA GPU (platform detection requires NVML-visible devices);
  `cpu.py` needs an official **CPU build** of vLLM (`vllm-...-cpu` wheels from
  the vLLM GitHub release; the CPU platform auto-activates from the `+cpu`
  build tag). Model directories need `config.json` + tokenizer files only.

## Usage

```bash
# Ascend
ucm-toolkit run model-check --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage --device-id 0
# GLM w8a8c8 additionally needs the C8 switches:
#   --additional-config '{"enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}'

# CUDA (same flags; needs an NVIDIA GPU)
ucm-toolkit run model-check --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage

# CPU (official vLLM CPU build + PLATFORM=simu UCM; no accelerator needed)
ucm-toolkit run model-check --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage

# Ascend layout with no NPU (vLLM-Ascend installed, CANN driver libraries
# mounted read-only, no /dev/davinci*; UCM built with PLATFORM unset)
ucm-toolkit run model-check --platform ascend-sim --ascend-soc A2 \
    --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage

# Official layout with no GPU (CUDA build of vLLM; UCM built with PLATFORM unset)
ucm-toolkit run model-check --platform cuda-sim --cuda-capability 9.0 \
    --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage

# Either device-less launcher can stop right after UCM parses the layout
ucm-toolkit run model-check --platform ascend-sim --layout-only \
    --model /path/to/model --storage-backends /path/to/ucm_storage

# Explicit connector-v2 validation (the legacy module remains the default)
ucm-toolkit run model-check \
    --connector-module-path ucm.integration.vllm.v2.ucm_connector \
    --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage
```

For connector v2, `--tokens-per-chunk` sets the token span of each UCM key,
independently of the native `--block-size`. Omit it to use the connector's
default. Explicit values must satisfy the connector's shared scheduler/group
alignment rules. The environment equivalent is `UCM_MODEL_CHECK_TOKENS_PER_CHUNK`.

On vLLM 0.30 / current Ascend, scheduler configuration conversion and manager
selection use the native APIs. The checker no longer installs the obsolete
Ascend UniformType manager-factory patch. The no-weight configuration uses the
native `dummy` load format, and the Ascend runner path registers custom kernels
through `enable_custom_op()` after selecting the device, as native Worker does.
For Ascend Engram construction, its native node-locality check executes with
real CPU control tensors and real Gloo collectives; model parameters remain
on meta. Cache specs, grouping, cache allocation, and scheduler block IDs are
obtained from the native runtime without replacing their values.
For v2 dump/load verification it
evicts source prefix hashes through the native manager while keeping source
blocks allocated, then poisons target IO ranges with `0xff` before loading.
The expected dump pattern only contains bytes in `[0,250]`; loaded bytes must
match that pattern in distinct scheduler-allocated destination blocks.

## Connector-independent checks {#connector-independent-checks}

The segment fill/compare above derives every address from the connector's own
`TransferBuilder`. That cannot see two classes of defect: an addressing error
applied **symmetrically** to dump and load cancels out, and a layer or group
the connector never transfers is never looked at. Three checks therefore use
only vLLM's tensors, specs and block tables:

| check | what it asserts | scope |
|---|---|---|
| **black-box** | no loaded byte holds a value UCM never stored: every byte of a poisoned target block is either the source byte at the same position (restored) or still `0xff` (not restored); a third value can only come from data placed at the wrong offset, block, layer or group | **every** persistent group, whatever it stores -- sliding-window and Mamba state groups included. Asserts nothing about *which* blocks a group stores, so a new group kind needs no adaptation. This is a **placement** check, not a completeness one: a group whose bytes all stayed `0xff` passes it. Each group's verdict says whether an independent required-restore set backed it, and groups without one are reported as `PARTIALLY VERIFIED` |
| **transfer ranges** | every range the connector is about to write lies inside a block vLLM allocated to this request | the allowed set comes from vLLM's block tables and views; the connector's addresses are verified against it, never added to it. Skipped, with a log line, when a layer's geometry is not addressable |
| **strict oracle** | target block *i* equals source block *i* byte-for-byte, through `kv_caches[layer]` and the scheduler's block tables | persistent full-attention groups (every block of the hit) and sliding-window groups whose window is exactly one block (the restore-boundary block). Other geometries are listed as not natively assertable instead of being guessed at |
| **stray writes** | the load changed no KV byte outside the ranges it was asked to fill | whole backing storage of every registered KV tensor |
| **hit length** (v2) | the dumped FA chunk count equals what the source request's scheduled range implies, **and** the loaded chunks are a contiguous prefix agreeing with the scheduler's external hit. Both expectations come from the request lengths and the chunk granularity, so a truncated dump cannot shrink its own expectation. The dump side is exact; the load side is exact for attention-only models and an upper bound for hybrid ones | connector v2 metadata |

The two are complementary: the strict check is narrow but demands equality, the
black-box check is wide but tolerates "not restored". Together they need no
knowledge of a group's storage policy to cover it.

Covered blocks are poisoned with `0xff` before the load, and the patterns are
position-dependent bytes written through a `uint8` alias, so a swapped
layer/component, a shifted offset and a low-precision rounding collision are
visible. A pattern is a sum of one arithmetic sequence per scale modulo 251, so
every shift below `PATTERN_PERIOD` (251**3, about 15.8 MB -- far beyond one KV
block) changes at least one byte; a single sequence would have hidden shifts
that are multiples of 251. Comparison is byte-wise, so NaN payloads compare
equal when their bits match.

The oracle is itself verified by fault injection: dropping one layer from
every v2 transfer keeps the connector-derived comparison green while the
native oracle fails with the layer name, block index and
`byte still poisoned (never loaded)` — for a full-attention layer and for a
sliding-window layer alike.

The native oracle runs on the connector-v2 path. The legacy/FAWA path already
compares native tensor blocks directly (it fills and snapshots them through
`kv_caches[layer]`), and the stray-write check applies to both.

Switches (both default on): `UCM_MODEL_CHECK_NATIVE_ORACLE=0`,
`UCM_MODEL_CHECK_STRAY_CHECK=0`. The wait between dump and the target lookup is
`UCM_MODEL_CHECK_DUMP_SETTLE_SECONDS` (default `0` for the synchronous v2 proxy,
`10` for the asynchronous legacy stores).

`--layout-only` stops once UCM has parsed the native layout and prints its
group classification — useful as a first step when adapting a new model or a
new vLLM version.

`adapter.py` picks the launcher automatically: `vllm_ascend` installed →
`ascend`; vLLM package version contains `+cpu` → `cpu`; plain vLLM → `cuda`.
All knobs are also available as environment variables (`UCM_MODEL_CHECK_*`,
see `config.py`).

Both implementations expose the same `UCMConnector` facade.  The module path
selects the implementation: `ucm.integration.vllm.ucm_connector` for legacy,
or `ucm.integration.vllm.v2.ucm_connector` for v2.  The v2 facade uses the
synchronous byte-range file Proxy and stores records below
`<storage_backends>/.ucm-v2/<device>-<dtype>-<policy>-b<scheduler-block>-c<chunk>/`,
so v2 files cannot be mistaken for legacy Store records or another platform's
layout.  The v2 check fills and compares
each selected `(key, offset, ptr, size)` range with a position-dependent byte
pattern; this catches sub-block offset errors that block-constant fills miss.

Success looks like:

```
[ucm-kv-check] PASS: Scheduler->UCM dump/load, rank=0, \
    compared_loaded_tensor_blocks=N, native_oracle_blocks=M, stray_write_check=on
```

## Verified runs

Checked on one host: openEuler 22.03 aarch64, vLLM 0.30.0 / vLLM-Ascend
0.30.0rc1 (`ascend_sim.py`, container with the CANN driver mounted read-only
and **no** `/dev/davinci*`) and official vLLM 0.30.0+cpu (`cpu.py`), UCM built
with `PLATFORM` unset (`simu`), `--tokens 1024` / `--block-size 128`
(Qwen: 8704 tokens, native `prefix_cache_retention_interval=1664`).

| case | launcher | connector | result |
|---|---|---|---|
| DeepSeek-V4.1-Flash, bulk | `ascend_sim` | v2 | PASS — 124 segments; black-box 11 groups / 176 blocks; strict 124 blocks |
| DeepSeek-V4.1-Flash, layerwise | `ascend_sim` | v2 | PASS — 124 segments; strict 124 blocks |
| DeepSeek-V4.1-Flash, bulk | `ascend_sim` | legacy (FAWA) | PASS — 124 blocks |
| DeepSeek-V4.1-Flash, `--layout-only` | `ascend_sim` | v2 | PASS — 12 groups: 1 MLA, 1 transient ring, 10 whole-block SWA |
| Qwen-Flash-Next, bulk | `cpu` | v2 | PASS — 12 segments; black-box 5 groups / 194 blocks, including 4 `MambaSpec` groups; strict 96 blocks |
| DeepSeek-V4.1-Flash | `cpu` | v2 | fails in construction (see *Known limitations*) |
| any | `cuda_sim` | — | **not yet run**: the available host is aarch64 with no CUDA vLLM wheel installed |

Every persistent group restored 100% of its poisoned bytes in these runs. The
four `MambaSpec` groups are the ones the strict oracle reports as not natively
assertable; the black-box check covers them.

Peak RSS of a full DeepSeek-V4.1-Flash run was 2.8 GB: weights stay on the
`meta` device and the KV pool is bounded to a few MB.

The device-less DeepSeek-V4.1 segment count (124) and the CPU Qwen count (12)
match the counts recorded for the same models on a real NPU and CPU in
`docs/connector-v2-layout-validation-20261016.md`.

Fault injection (not committed): dropping one layer from every v2 transfer
leaves the connector-derived comparison green (123 of 124 segments, PASS) and
fails the native oracle, for a full-attention layer and for a sliding-window
layer.

## Standard verification flow

Use this flow instead of writing ad-hoc scripts, so results stay comparable
across changes (this matters most for automated edits):

1. After changing `ucm/integration/vllm/`, run the component tests
   (`test/suites/Unit/test_ucm_connector_v2.py`, `test_ucm_proxy_v2.py`), the
   toolkit tests (`toolkit/tests/`), and this checker on the fixed model set.
2. For a new model: supply a directory with `config.json` (plus tokenizer
   files), run `--layout-only` first, then the full check in bulk and
   layerwise. Report the step that failed — construction / spec / grouping /
   allocation / scheduling / dump / load / comparison — with the native value
   and the value UCM expected.
3. For a new vLLM or vLLM-Ascend version, re-check the substitutions listed
   under *Layout-identity notes* against that version's source before trusting
   a PASS.
4. Never make a check pass by substituting a native value: KV specs, grouping,
   shapes, dtypes, strides, allocation, scheduler block IDs and lookup results
   must stay as the runtime produced them. Weights and compute kernels are out
   of scope and may be substituted, which is what the launchers already do.
5. If coverage is missing, extend the checker's fixed scenarios (and its unit
   tests in `toolkit/tests/`) rather than adding a separate script.

## Known limitations

Re-checked on vLLM 0.30.0 / vLLM-Ascend 0.30.0rc1 unless noted.

- **DeepSeek-V4.1 does not run on `cpu.py`**: model construction stops at
  `Failed to find a kernel that can implement the ScaledMM linear layer`, i.e.
  the fp8 quantized linear has no CPU kernel. (The earlier 0.26 note blamed
  vLLM's `_get_kv_cache_groups_uniform_groups` page-size assertion; that
  function no longer exists in 0.30, where grouping goes through
  `get_kv_cache_groups` / `_get_kv_cache_groups_uniform_page_size`.) Use
  `ascend_sim.py` for the Ascend layout without a device, or `cuda.py` on a GPU
  for the official layout.
- **`cuda_sim.py` has not been run yet.** It is written against the vLLM 0.30
  source read on the host (platform resolution in
  `vllm/platforms/__init__.py`, the module-level kernel import in
  `vllm/v1/attention/backends/fa_utils.py`), but no CUDA build of vLLM was
  available to execute it. Expect the same iterate-on-the-first-error work that
  `ascend_sim.py` needed: its substitutions are the set that source predicts is
  required, not a verified one.
- **`cpu.py` is currently the only launcher that covers Mamba state groups.**
  `ascend_sim.py` cannot construct Qwen4Exp (`NotImplementedError: Qwen4Exp QSA
  requires FlashAttention`, a vLLM-Ascend limitation), so do not drop the CPU
  launcher without another path for those groups.
- **Kimi-K3 was not registered in official vLLM 0.26** — only `ascend.py`
  supported it. Not re-checked on 0.30.
- **CPU external load works on 0.30 + connector v2**: `cpu.py` reaches
  `hit external` and compares loaded bytes (the earlier note that the
  scheduler → UCM block-hash bridge was missing no longer applies).
- Distributed init uses a fixed TCP port (29500); run checkers serially within
  one network namespace to avoid `EADDRINUSE`.
- These images ship no `wrapt`, which UCM's patch helpers import; install
  `wrapt` before running, and pass `--no-build-isolation` when installing the
  toolkit. The CPU image's optional `torchcodec` can fail to load (missing CUDA
  libraries) and must be moved aside for the run.
- **The declaration has to reach child interpreters.** vLLM inspects a model
  architecture in a fresh process
  (`python -m vllm.model_executor.models.registry`) which calls
  `load_general_plugins()` before anything else, so a monkey patch in the
  parent never reaches it and the child repeats the device query. `ascend_sim`
  therefore writes a `sitecustomize.py` and prepends its directory to
  `PYTHONPATH`; Python's `site` imports that name at startup, before the `-m`
  module runs. The same declaration then runs in every interpreter of the run,
  from one implementation in `sim_bootstrap.py`. It chains to any
  `sitecustomize` it shadows, does nothing when no simulated platform is
  declared, and warns on stderr rather than killing a child if it cannot
  declare -- replacing a clear error with an unrelated crash would be worse.
- **triton's Ascend arch lookup moved between images.** On
  `vllm-ascend:v0.30.0rc1` (triton-ascend 3.2.2) the `TRITON_ASCEND_ARCH`
  override is honoured. On `v0.30.0rc2` (3.6.0+dev) triton logs
  "host environment overrides are ignored" and queries the device, which
  without one fails as `cann_get_soc_name failed` /
  `get_arch returned NULL` during vLLM-Ascend's import chain -- before any KV
  layout work. `ascend_sim.py` therefore declares the arch on the driver's own
  lookup as well as through the environment, and fails with the triton version
  named if a future release moves that lookup again.
- Prefix caching: the CPU platform forces it off for MLA models; keep it off
  (UCM takes over prefix lookup; local HBM hits would bypass the external load,
  which is why `cpu.py` leaves it disabled).
- `w8a8`/`w4a8` model variants are Ascend-quantization exports: their configs
  carry no quantization fields and match their HF counterparts structurally,
  but the weights cannot be loaded on CUDA — official-layout results for these
  models are structural references only.
