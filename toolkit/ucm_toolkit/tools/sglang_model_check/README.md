# SGLang model compatibility checker

`sglang-model-check` checks model structure and UCM host-cache storage contracts
without starting an inference service or loading checkpoint weights. It does
not establish that a model can serve requests on the selected hardware.

Inspect the model structure and the storage methods actually overridden by the
installed integration:

```bash
ucm-toolkit run sglang-model-check --model /models/Qwen2.5-14B-Instruct --mode inspect
```

Run an actual UCM dump/load roundtrip through a selected SGLang host-pool layout:

```bash
ucm-toolkit run sglang-model-check \
  --model /models/Qwen2.5-14B-Instruct \
  --mode roundtrip \
  --storage-backends /mnt/test \
  --output result.json
```

Only `--model` is required; the default mode is `inspect`. Examples specify
`--mode` explicitly to distinguish the checks. Storage backends never change
the mode: writing test data requires `--mode roundtrip --storage-backends ...`.
Use a dedicated test storage directory.

Platform, model/cache dtype and page size are selected automatically where
supported; the default layout is `page_first`, with 2 test pages and a 512 MiB
host-buffer budget. This minimum probe does not exercise every sparse boundary
case; increase `--pages` when needed as described below.

`--help` shows common options. `--help-all` shows every advanced/model-specific
override. Existing explicit options remain supported; when matching a particular
service configuration, retain its non-default layout, dtype and backend options.

The checker intentionally does not instantiate `Engine`, `Scheduler` or an HTTP
server. Model modules are constructed on Torch's `meta` device. Roundtrip mode
allocates bounded CPU cache buffers using SGLang's production pool classes,
then calls the production `UnifiedCacheStore` API. Single-pool models exercise
a real SGLang host pool through v1. Native sparse/DSV4 providers also create
small CPU versions of the device buffer descriptions to derive physical formats;
they never load weights or execute device transfer kernels.

Hybrid roundtrip supports physical MHA KV anchors with conventional SWA and/or
Mamba side pools when the installed SGLang exposes a complete layer partition.
It reuses `get_num_kv_heads`, `get_swa_num_kv_heads`, `sliding_window_size`, and
`mambaish_config(...).mamba2_cache_params`; no model-name whitelist is needed.
SWA uses the actual MHA Host Pool with its own layers/head geometry and a
configuration-derived page window. Mamba uses the production `MambaPoolHost`
and heterogeneous temporal/conv tensors, with state dtypes from SGLang config.
NPU conv axes follow the standard persistent NPU layout. Replay/speculative
side states and CPU AMX-specific state layouts are outside this probe's scope.

Each side pool has its own page unit (Mamba checkpoint page_size=1).
The probe writes sparse checkpoint windows and validates both actual and
placeholder lookup keys, shorter requests and complete restorable-prefix sets.
Every persisted component is read into a separate location and compared.
Use at least `--pages 5` to exercise separated checkpoint boundaries; use more
pages than the reported SWA window to exercise missing older SWA pages.

```bash
ucm-toolkit run sglang-model-check --model /models/hybrid-model \
  --platform ascend --mode roundtrip --pages 5 --page-size 64 \
  --max-host-mib 2048 --storage-backends /mnt/hybrid-probe --verbose
```

Additional providers reuse native SGLang cache constructors and host builders:

| Cache family | Probe path |
| --- | --- |
| MHA/GQA/MQA and MLA | Physical KV host pages and v1 dump/lookup/load |
| MHA + SWA / Mamba | Configuration-derived layer partitions and specialized pools |
| SWA-only / recurrent-state-only | Production logical anchor plus real side pool |
| DSA | Native DSA cache descriptions, declared indexer builder and packed scale payloads |
| MiniMax sparse | Native main KV plus K-only sparse indexer, subject to production restrictions |
| DeepSeek V4 | Native CUDA/ROCm or Ascend pool producer; logical anchor; active compressed, indexer, scale, RoPE and checkpoint regions |

DSV4 pools are discovered from actual native regions, not all `PoolName` enum
members. Shared-source C1/C2 pools are included when exposed by SGLang; C128
state is not persisted when the production assembler does not register it.
Each pool retains its own physical page size (including NPU C128), and the
report states `host_storage_contract_only`: compressed-token translation,
controller transfer construction and device copies are outside this probe.
DSA indexer content integrated into a native primary host pool is checked
through the primary bytes and reported as `v1_integrated_payload`.

Inspect remains `inconclusive` for hybrid storage compatibility until roundtrip
runs. Native families defer actual pool discovery/allocation to roundtrip.
Unknown producer APIs, unsupported CPU construction, compressed heterogeneous
SWA, CPU AMX-specific states, and combined indexer + recurrent-state providers
not yet assembled return `CHECKER_RUNTIME_POOL_PROBE_REQUIRED` with the cause.
Explicit UCM pool-registration rejection returns `UCM_POOL_REGISTRATION_UNSUPPORTED`.
No production integration or SGLang code is modified.

Examples for sparse/DSV4 models on a deployment machine:

```bash
# Let the tool select a suitable default page size for the cache family.
ucm-toolkit run sglang-model-check --model /models/dsa-model \
  --mode roundtrip --pages 5 --max-host-mib 2048 \
  --storage-backends /mnt/dsa-probe --verbose

# DSV4 uses 256-token pages by default; --platform may be ascend/cuda/rocm.
ucm-toolkit run sglang-model-check --model /models/deepseek-v4 \
  --platform ascend --mode roundtrip --pages 5 --page-size 256 \
  --max-host-mib 2048 --storage-backends /mnt/dsv4-probe --verbose

# Test the CUDA/ROCm FP4 indexer format when supported by installed SGLang.
ucm-toolkit run sglang-model-check --model /models/deepseek-v4 \
  --mode roundtrip --fp4-indexer --dsv4-attn-backend flashmla \
  --max-host-mib 2048 --storage-backends /mnt/dsv4-fp4-probe --verbose
```

`--kv-cache-dtype` is independent of model `--dtype`. Native providers accept
`auto`, `bfloat16`, `float16`, `float32`, and `fp8_e4m3`, but the production
constructor may reject a format on a specific platform. Auto uses model dtype
for DSA/MiniMax and platform-specific DSV4 defaults (Ascend bf16, CUDA/ROCm fp8).
DSV4 backend selection uses `--dsv4-attn-backend`; `--c128-page-size` selects the
NPU C128 physical page size (default 16). Match these and relevant SGLang env
settings to the intended deployment; one default probe does not certify every
backend/quantization combination. Plain config-derived pools currently require
automatic KV dtype. `--skip-meta-model` can isolate Host-storage checks from
meta-constructor limitations; the JSON records that model construction was skipped.

Use `--skip-meta-model` only for a weaker configuration-only scan. A successful
result in that mode does not prove that the SGLang model class can be built.
In inspect mode this weaker result is `CHECKER_CONFIG_ONLY` / `inconclusive`.
Hybrid inspect uses `CHECKER_STORAGE_PROBE_REQUIRED` until storage is exercised.

Default JSON contains only model, mode, status and a scope-aware summary;
non-passing results also include code and suggestion. UCM INFO logs are suppressed
in the child process by default; warnings/errors remain visible. Add `--verbose` when debugging to
include detection sources, detection errors, full tracebacks, and roundtrip
details. `--output` writes the same summary or verbose representation shown on
standard output.

Use the `status` field for automation: `compatible`, `inconclusive`, or
`incompatible`. In particular, `CHECKER_RUNTIME_POOL_PROBE_REQUIRED` is
`inconclusive`, not evidence that the model is incompatible. The legacy
`supported` boolean remains available for backward compatibility.

Use `--platform auto|cuda|rocm|ascend|xpu|cpu` to select the accelerator family.
Auto prefers an existing Ascend environment or `torch_npu`; otherwise it selects
CUDA. The child process sets only the matching visibility variable, and the JSON
`environment` field records the detected SGLang version and platform.
Accelerator-specific quantization such as NVFP4 is rejected on a non-CUDA
platform before meta-model construction.

Only `page_first` and `page_first_direct` are accepted by the production UCM
integration. Asymmetric primary K/V head dimensions are rejected because v1
assumes two equal-size components. Host pool construction adapts to both the
older `device_pool` and newer `buffer_info` SGLang constructors. Unknown APIs
and meta/environment failures report `inconclusive`, not model incompatibility.

`--max-host-mib` (default 512) limits estimated probe buffers before allocation.
Reduce `--pages` / `--page-size` for larger models. This estimate is not a hard
process RSS limit and excludes Python, imported libraries and meta constructors.
The probe covers one rank, configured KV dtype and SGLang-configured Mamba state
dtypes. Quantized payloads/scales are covered only where discovered by native
providers. Model-weight quantization support is distinct from KV-cache format.
Config-derived hybrid memory is estimated across all pools before allocation;
native constructors have a cumulative common-factory CPU allocation budget,
reserving additional room for staging and snapshots. Explicit accelerator
requests through those factories are rejected rather than silently redirected.
This is not a complete allocation sandbox. Actual device transfers and
control-plane scheduling are not tested.

JSON schema v3 includes `validation`, which records checks actually performed.
`compatible` in inspect mode means structural checks passed; only roundtrip
mode establishes the tested host-storage IO contract. Neither validates GPU/NPU
copies, inference kernels, TP/PP rank consensus or cache-controller scheduling.
Non-passing results retain exit code 1, including `inconclusive`; consumers
must read `status` and `code`. Temporary UUID keys are written in roundtrip mode
and are not deleted; use a dedicated test backend directory.

See [需求与可行性分析](requirements_analysis_zh.md) for evidence, scope and the
hybrid-pool acceptance matrix.

Run dependency-free planning, sparse-window and transfer-orchestration tests:

```bash
cd toolkit
python -m unittest discover -s tests -p 'test_sglang_model_check*.py' -v
```

Native integration tests are opt-in and use both supported layouts. On a target
machine with Torch, SGLang and UCM installed, point them to a local supported
hybrid/sparse/DSV4 model and a dedicated backend:

```bash
export UCM_SGLANG_MODEL_CHECK_TEST_MODEL=/models/hybrid-model
export UCM_SGLANG_MODEL_CHECK_TEST_BACKENDS=/mnt/hybrid-probe
export UCM_SGLANG_MODEL_CHECK_TEST_MAX_HOST_MIB=2048
# Optional: override family defaults or probe a native quantized format.
# export UCM_SGLANG_MODEL_CHECK_TEST_PAGE_SIZE=256
# export UCM_SGLANG_MODEL_CHECK_TEST_KV_DTYPE=fp8_e4m3
python -m unittest discover -s tests -p 'test_sglang_model_check_pools.py' -v
```

Passing the dependency-free tests does not establish native pool or Posix
compatibility; the opt-in test must pass on the deployment environment.
