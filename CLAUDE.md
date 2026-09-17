# Unified Cache Management (UCM)

UCM persists LLM KVCache to external storage and replaces redundant computation via
retrieval mechanisms: prefix caching, cache blending, sparse attention (GSA/ESA/KVStar/ReRoPE),
and PD disaggregation. Integrates with vLLM (primary), SGLang, and MindIE.

Package: `uc-manager` (v0.5.0), Python >= 3.10. Hybrid Python + C++17 project —
C++ code is built via CMake through `setup.py` (`CMakeExtension`/`CMakeBuild`).

## Build / Install

Requires Linux. Builds target accelerators via the `PLATFORM` env var
(`cuda`, `ascend`, `ascend-a3`, `musa`, `maca`; unset -> `simu` fallback).

```bash
# Wheel build (CMake runs automatically inside setup.py)
export PLATFORM=cuda            # or ascend / ascend-a3 / musa / maca
export ENABLE_SPARSE=true       # optional: build sparse module
python -m build --wheel

# Editable install
pip install -e .

# MindIE build requires ABI flag
UCM_ENABLE_MINDIE=1 UCM_CXX11_ABI=0|1 python -m build -w

# CI-style packaging scripts
scripts/build_cuda.sh           # also build_ascend.sh / build_mindie.sh / build_sglang.sh
```

Key CMake options (see root [CMakeLists.txt](file:///d:/xiaojun/work/DramPool/unified-cache-management/CMakeLists.txt)):
`BUILD_UCM_STORE` (ON), `BUILD_UCM_DRAMPOOL` (ON), `BUILD_UCM_SPARSE` (OFF),
`BUILD_UCM_ASU` (OFF), `BUILD_UCM_DELEGATOR` (OFF), `BUILD_UCM_MINDIE` (OFF),
`BUILD_UNIT_TESTS` (OFF), `RUNTIME_ENVIRONMENT` (`simu|ascend|cuda|musa|maca`).

## Lint / Format

Python: black 24.4.2 + isort (profile=black). C++: Google style via `.clang-format`.
Run everything with:

```bash
pip install -r requirements-lint.txt
bash format.sh                  # pre-commit run --all-files (black, isort, codespell, actionlint)
```

## Tests

```bash
# C++ unit tests (gtest): configure with -DBUILD_UNIT_TESTS=ON, then
cmake --build . && ctest

# E2E / perf Python tests
cd test && pytest               # suites in test/suites/E2E; config in test/config.yaml
```

## Docs

```bash
cd docs && pip install -r requirements-docs.txt
make html && python3 -m http.server -d build/html/
```

## Architecture

```
ucm/
├── shared/        C++ infra: logger, metrics, buffer pools, router,
│                  trans (device/stream/buffer abstraction: cuda/ascend/maca/simu) + pybind11 wrappers
├── store/         KVCache storage backends (ucmstore.h/.py API + factory.py):
│   ├── cache/     in-process cache layer (dump/load queues, trans buffers)
│   ├── posix/     local POSIX-file store (AIO, sharding, GC)
│   ├── nfsstore/  NFS/local file store with hotness + space management
│   ├── pcstore/   prefix-cache store
│   ├── ds3fs/     object-store-backed (S3-like) store
│   ├── dram/      DramStore + DramPool daemon (see below)
│   ├── mooncakestore/  Mooncake connector
│   ├── asu/ delegator/ compress/ pipeline/  optional modules
│   └── test/      gtest cases + e2e python scripts
├── transport/     P2P HIXL transport: control channel, TCP channel,
│                  transport_manager; two-sided / one-sided (RDMA) endpoints
├── sparse/        Training-free sparse attention: gsa (with NPU/CUDA ops),
│                  esa, gsa_on_device, kvstar, blend, rerope; base.py + factory.py
└── integration/   Inference-framework glue + versioned patches:
    ├── vllm/      ucm_connector, blend_connector, hma_connector, device,
    │              patch/{0.9.2,0.11.0,v0110,v0180,v0191} (vllm & vllm_ascend override trees)
    ├── sglang/    ucm_connector + 0.5.5 patch
    └── mindie/    mempool, boot patches, hash ext
```

Other top-level dirs:
- `kv_semantics/` — ASU module: KV client library + `kv_test` benchmark (built when `BUILD_UCM_ASU=ON`)
- `test/` — E2E pytest suites (online/offline inference, sparse, performance, evaluator)
- `examples/` — `ucm_config_*.yaml` examples, offline inference scripts, `drampool.yaml`
- `docker/` — Dockerfiles per framework/platform combos
- `benchmarks/` — trace generation/replay, logging perf

## DramPool daemon

Standalone DRAM KVCache pool service in `ucm/store/dram/cc/drampool/`
(entry `main.cc`, server `drampool_server.cc`, health server `GET /health`,
eviction policies: TTL / POSITION / LRU, `metadata.periodic_eviction_policy` /
`deep_eviction_policy`). Configured by CLI flags (`--addr`, `--nics`,
`--pool-size-gb`, `--kvcache-block-sizes`, `--ttl-minutes`, `--config`) plus a
YAML file — see [examples/drampool.yaml](file:///d:/xiaojun/work/DramPool/unified-cache-management/examples/drampool.yaml)
(transport endpoints, queue depths, flag buffer, GC, eviction metadata, logger).
Log level follows `UC_LOGGER_LEVEL` env var.

## Conventions

- PR titles prefixed: `[Feat] [Bugfix] [Opt] [Build] [CI] [Doc] [Test] [Misc]`
- Python: PEP 8 (black/isort); C++: Google C++ Style Guide, C++17, `-Wall -Werror`
- Protected branches (develop/release): code-owner approval + CI required, squash merge
