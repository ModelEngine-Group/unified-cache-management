# SGLang model compatibility checker

`sglang-model-check` checks whether the installed UCM SGLang integration can
serve a model without starting an inference service or loading checkpoint
weights.

Inspect the model structure and the storage methods actually overridden by the
installed integration:

```bash
ucm-toolkit run sglang-model-check --model /models/Qwen2.5-14B-Instruct
```

Run an actual UCM dump/load roundtrip through a selected SGLang host-pool layout:

```bash
ucm-toolkit run sglang-model-check \
  --model /models/Qwen2.5-14B-Instruct \
  --mode roundtrip \
  --layout page_first \
  --storage-backends /mnt/test \
  --output result.json
```

The checker intentionally does not instantiate `Engine`, `Scheduler` or an HTTP
server. Model modules are constructed on Torch's `meta` device. Roundtrip mode
allocates only a bounded CPU host-cache buffer using SGLang's production
`MHATokenToKVPoolHost` or `MLATokenToKVPoolHost`, then calls the production
`UnifiedCacheStore` production API. Single-pool models exercise a real SGLang
host pool through v1.

Hybrid models that require Mamba, SWA, indexer or DeepSeek-V4 sidecar pools are
reported as `UCM_STORAGE_API_UNSUPPORTED` until `UnifiedCacheStore` actually
overrides SGLang's v2 multi-pool methods. The checker requires no capability
declaration and does not modify the production integration. Once those methods
are implemented, inspect mode verifies the v2 API surface. Hybrid roundtrip
returns `CHECKER_RUNTIME_POOL_PROBE_REQUIRED` until specialized SGLang runtime
pools can be constructed; synthetic MHA/MLA pools are not accepted as evidence
of Mamba, SWA, DSA indexer, or DeepSeek-V4 compatibility.

Use `--skip-meta-model` only for a weaker configuration-only scan. A successful
result in that mode does not prove that the SGLang model class can be built.

Output is a compact JSON summary by default. Add `--verbose` when debugging to
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
