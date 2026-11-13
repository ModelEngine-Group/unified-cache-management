# pynds_kv

`pynds_kv` provides C, C++, and Python interfaces for NDS KV semantics. The Python extension is built with pybind11 and supports multiple namespaces, multiple registered memory regions, and batched asynchronous I/O.

This source package targets NDS SDK v1.6. The Python package version is 1.7.3.

## Package contents

- `vllm-uc/`: NDS client SDK source, public headers, and agent adapters used to build `libndsclient.so`.
- `src/main.cpp`: pybind11 Python API.
- `vllm/NDS3.0/nds_wrapper.cpp`: NDS KV wrapper implementation.
- `vllm/NDS3.0/nds_c_adapter.cpp`: C API adapter.
- `vllm/NDS_include/nds_interface.h`: public C/C++ declarations.
- `vllm-uc/NDS_include/xxhash.h`: vendored xxHash header with its original BSD 2-Clause license notice.
- `vllm/Makefile`: builds the C++ shared library.
- `install.sh`: one-command SDK, wrapper, wheel, and Python extension installation.
- `setup.py`, `pyproject.toml`, `build.sh`: Python packaging and the compatibility build entry point.
- `nds_wrapper.conf`: default wrapper I/O configuration, alongside `nds_config.conf.example`.
- `key.txt`: 10,000 unique 16-digit block IDs used by the validation tests.
- `tests/test_read_write_correctness.py`: write/read data consistency validation.
- `tests/test_multithread_get.py`: multithreaded read/write bandwidth measurement.

Vendor agent binaries, authentication material, site-specific configuration, other internal tools, and build artifacts are intentionally not included.

## Prerequisites

- Linux on x86_64 or aarch64.
- Python 3.9 or later.
- `g++`, `make`, `setuptools`, `wheel`, and `pybind11>=2.10.0`, preinstalled in the target Python environment for offline builds.
- libibverbs, OpenSSL, and zlib development files.
- Only for cluster mode: vendor-provided `libkv_agent.so` and `libceph-common.so.2` in the same directory.

The NDS client library is built from the included `vllm-uc/` source. A separately supplied `libndsclient.so` is not required.

On systems with an externally managed Python environment (PEP 668), create and activate a virtual environment instead of bypassing the system package-manager protection with `--break-system-packages`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel pybind11
```

Pass the activated environment's interpreter explicitly if needed, for example `bash install.sh --python .venv/bin/python`.

## One-command build and installation

For non-cluster mode, no vendor agent library is required:

```bash
bash install.sh
```

For cluster mode, provide both vendor agent libraries:

```bash
bash install.sh --agent-lib-dir /path/from/vendor --python python3.12
```

By default, the script checks `vllm-uc/agent/release/`. If both agent libraries are present it builds cluster support; if neither is present it builds non-cluster support. Incomplete or explicitly specified library directories are rejected. Use `--no-agent` to force a non-cluster build. `--mock` is only for build validation and must not be used in production.

The script performs clean release builds, creates and installs a wheel with `--no-build-isolation`, and verifies `import pynds_kv` from an empty temporary directory with `PYTHONPATH` cleared, ensuring that validation loads the installed wheel rather than the source tree. Offline installation therefore does not download build dependencies. `build.sh` delegates to `install.sh` and accepts the same arguments.

All required native libraries are packaged in `_pynds_kv_native/` and resolved through relative `$ORIGIN` runpaths. The installation does not depend on source-tree absolute paths or `LD_LIBRARY_PATH`.

To inspect the installed link dependencies:

```bash
python -c "import pynds_kv; print(pynds_kv.__version__)"
ldd "$(python -c 'import pynds_kv; print(pynds_kv.__file__)')" | grep -E 'ndskv|ndsclient|kv_agent|ceph|not found'
```

## Runtime configuration

1. Copy `nds_config.conf.example` to `nds_config.conf` and replace the placeholders with site-specific network settings.
2. If the vendor SDK uses the agent configuration, copy `conf/agent.conf.example` to `conf/agent.conf` and provide the site-specific monitor addresses and credentials.
3. Keep `nds_wrapper.conf` beside `nds_config.conf` and adjust its values for the target KV pool. If `nds_config.conf` is moved elsewhere, copy `nds_wrapper.conf` to that directory as well. The wrapper does not search other directories: if the file is absent or misplaced, built-in defaults are used, including a 32 MiB `value_size`.

Real `nds_config.conf` and `conf/agent.conf` files are excluded by `.gitignore` and must not be published.

Important wrapper settings:

| Setting | Default | Meaning |
|---|---:|---|
| `read_req_size_mb` / `write_req_size_mb` | 32 | Maximum bytes per NDS request, in MiB; decimal values are supported. |
| `read_batch_size_kb` / `write_batch_size_kb` | 128 | Address-pair splitting granularity, in KiB. |
| `value_size_mb` / `value_size_kb` | 32 MiB | Default KV slot size. |
| `value_size_kb_nsid<N>` | unset | Per-namespace slot-size override. |
| `max_exist_addr_num` | 256 | Maximum keys per existence-query request. |
| `max_addr_num_per_req` | 4096 | Physical address-pair limit per request; cannot exceed the compiled ABI capacity. |
| `key_scatter` | 1 | Hash the complete 80-bit key before I/O. Changing this setting changes the key space. |
| `error_print` | 1 | Enable wrapper error messages. |
| `max_inflight_reqs` | 8 | Sliding-window size per calling thread. |

The recommended deployment uses the same slot size for the KV pool, `value_size_*`, and the actual write length.

## API overview

All functions return `int32_t`: zero means success and non-zero means failure.

| Function | Purpose |
|---|---|
| `init(buffer, length, nds_config_path=None)` | Initialize NDS and register one page-aligned memory buffer. |
| `init_multi(buffers, lengths, nds_config_path=None)` | Register multiple page-aligned buffers; list order determines `mr_id`. |
| `is_exists(block_ids, nsid=0)` | Return the consecutive prefix hit count, not a bitmap. |
| `get(block_id, buffer, offset, length, nsid)` | Read one block. |
| `put(block_id, buffer, offset, length, nsid)` | Write one block. |
| `batchGet(block_ids, buffers, offsets, lengths, nsids)` | Batch read with the GIL released. |
| `batchPut(block_ids, buffers, offsets, lengths, nsids)` | Batch write with the GIL released. |

Python block IDs are non-negative integers below `2^80`. Values below `2^64` remain compatible. The C adapter accepts 64-bit keys; the C++ and Python APIs accept the full 80-bit key.

For writes, `length > value_size` is rejected. For `length < value_size`, the wrapper writes the full slot and requires `offset == 0` plus sufficient registered buffer capacity. Reads return the requested bytes; the implementation may extend the final pair to satisfy SDK batch alignment. Callers must pass the correct namespace ID to every operation.

`init` and `init_multi` do not take ownership of the registered storage. Callers must keep every registered buffer alive at a fixed address and must not resize it while NDS may still use the registration. In particular, do not register temporary or resizable storage whose lifetime is not explicitly managed by the application.

## Validation tests

The two test scripts require NumPy and a reachable NDS deployment. Use a dedicated test namespace because they write KV data.

```bash
# Write/read consistency, including prefix existence and namespace isolation checks.
python tests/test_read_write_correctness.py \
  --nds_config_path ./nds_config.conf --nsid 1

# Write, read, and mixed-mode bandwidth measurements.
python tests/test_multithread_get.py \
  --nds_config_path ./nds_config.conf --nsids 1 --mode w --duration 30
python tests/test_multithread_get.py \
  --nds_config_path ./nds_config.conf --nsids 1 --mode r --duration 30
python tests/test_multithread_get.py \
  --nds_config_path ./nds_config.conf --nsids 1 --mode rw --duration 30
```

Run the scripts from the package root so they use the included `key.txt`. If the file is absent or contains too few IDs, they generate sequential IDs in memory. Neither script creates or overwrites `nds_wrapper.conf`.

## Diagnostics

Wrapper error logging remains enabled by default. For verbose request-level diagnostics, set:

```bash
export NDS_WRAP_DEBUG=1
```

Disable it for normal production use. Set `error_print=0` only when repeated SDK failures make the error output unusable; callers must still check every non-zero return value.
