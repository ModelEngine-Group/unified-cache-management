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

Running the checker requires only a model directory with `config.json` (+
tokenizer files for multimodal models) — no weights, no model download.

## Platform launchers

The checker ships one launcher per serving stack in this directory
(`adapter.py` auto-selects the launcher by the installed stack):

| launcher | vLLM stack | device | KV-cache layout produced |
|---|---|---|---|
| `ascend.py` | vLLM + vLLM-Ascend | `npu` | **Ascend-specific** layout (vLLM-Ascend patches: non-packed tuples, int8 SFA C8 pages, mamba-aligned blocks, …) |
| `cuda.py` | official vLLM (CUDA build) | `cuda` | **Official vLLM layout** |
| `cpu.py` | official vLLM (CPU build) | `cpu` | **Official vLLM layout — identical to `cuda.py`** |

**Layout-identity notes**

- `cpu.py` and `cuda.py` run the *same* official vLLM code: the CPU build and
  the CUDA build are the same v0.26.0 sources compiled with different targets,
  and the KV-cache planning pipeline (`get_kv_cache_spec` → grouping →
  `_get_kv_cache_config_packed`) is platform-independent arithmetic. The two
  launchers differ only in the runtime platform: `cpu.py` needs no GPU (and no
  CUDA driver), `cuda.py` requires an NVIDIA GPU; both report the same layout.
  The only expected deltas are backend-name-driven fields (e.g. sparse-MLA
  `kv_cache_dtype` normalization) that follow the actually-selected GPU backend.
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
  pip install -e ./toolkit --no-deps
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

# Explicit connector-v2 validation (the legacy module remains the default)
ucm-toolkit run model-check \
    --connector-module-path ucm.integration.vllm.v2.ucm_connector \
    --model /path/to/model --tokens 1024 --block-size 128 \
    --storage-backends /path/to/ucm_storage
```

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
[ucm-kv-check] PASS: Scheduler->UCM dump/load, compared_loaded_tensor_blocks=N
```

## Known limitations (tested against vLLM 0.26.0 / vLLM-Ascend 0.26.0rc)

- **DSV4 multi-group layouts fail the official vLLM grouping assertion on
  non-GPU platforms** (`_get_kv_cache_groups_uniform_groups`: SWA sub-group page
  size > full-MLA group max page). Ascend is unaffected (vLLM-Ascend has its
  own grouping). Run DSV4's validator on `ascend.py` or on a CUDA GPU.
- **Kimi-K3 is not registered in official vLLM 0.26** — only `ascend.py`
  supports it.
- **CPU external-load bridge**: on the official vLLM CPU build the scheduler →
  UCM block-hash bridge is missing, so the target request reports
  `hit external: 0` and `verify()` stops at "no load_block_ids" (dump side and
  layout printing are complete). Ascend hits `hit external` normally; CUDA is
  expected to behave like Ascend (to be confirmed on a GPU machine).
- Distributed init uses a fixed TCP port (29500); run checkers serially to
  avoid `EADDRINUSE`.
- Prefix caching: the CPU platform forces it off for MLA models; keep it off
  (UCM takes over prefix lookup; local HBM hits would bypass the external load,
  which is why `cpu.py` leaves it disabled).
- `w8a8`/`w4a8` model variants are Ascend-quantization exports: their configs
  carry no quantization fields and match their HF counterparts structurally,
  but the weights cannot be loaded on CUDA — official-layout results for these
  models are structural references only.
