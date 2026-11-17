# pynds_kv

`pynds_kv` provides C, C++, and Python interfaces for NDS KV semantics. The Python extension is built with pybind11 and supports multiple namespaces, multiple registered memory regions, and batched asynchronous I/O.

This source package targets NDS SDK v1.6. The Python package version is 1.7.1.

## Package contents

- `src/main.cpp`: pybind11 Python API.
- `vllm/NDS3.0/nds_wrapper.cpp`: NDS KV wrapper implementation.
- `vllm/NDS3.0/nds_c_adapter.cpp`: C API adapter.
- `vllm/NDS_include/nds_interface.h`: public C/C++ declarations.
- `vllm/NDS3.0/xxhash.h`: vendored xxHash header with its original BSD 2-Clause license notice.
- `vllm/Makefile`: builds the C++ shared library.
- `setup.py`, `pyproject.toml`, `build.sh`: build and install the Python extension.
- `nds_wrapper.conf`: default wrapper I/O configuration, alongside `nds_config.conf.example`.
- `key.txt`: 10,000 unique 16-digit block IDs used by the validation tests.
- `tests/test_read_write_correctness.py`: write/read data consistency validation.
- `tests/test_multithread_get.py`: multithreaded read/write bandwidth measurement.

Vendor libraries, authentication material, cluster-specific configuration, other internal tools, and build artifacts are intentionally not included.

## Prerequisites

- Linux on x86_64 or aarch64.
- Python 3.9 or later.
- `g++`, `make`, `setuptools`, and `pybind11>=2.10.0`.
- NDS SDK v1.6 `libndsclient.so`, obtained separately from the NDS vendor.
- System libraries required by the Makefile: `libibverbs`, `libcrypto`, and `librt`.

The minimum NDS ABI declarations needed for compilation are already contained in `nds_wrapper.cpp`; vendor C/C++ headers are not required.

## Build

Place the vendor library in `vllm/NDS_lib/`, then build the wrapper:

```bash
mkdir -p vllm/NDS_lib
cp /path/from/vendor/libndsclient.so vllm/NDS_lib/

cd vllm
make clean && make
# Output: vllm/NDS_bin/libndskv.so
```

Install the Python extension from the package root:

```bash
cd ..
bash build.sh
python -c "import pynds_kv; print(pynds_kv.__version__)"
```

`libndskv.so` has an `$ORIGIN/../NDS_lib` runpath, so it locates `libndsclient.so` relative to itself. The installed Python extension uses the absolute path of `vllm/NDS_bin`; reinstall the package after moving the source directory. Always run `make clean && make` after changing the NDS SDK library.

To inspect the link dependencies:

```bash
readelf -d vllm/NDS_bin/libndskv.so | grep -Ei 'needed|rpath|runpath'
ldd vllm/NDS_bin/libndskv.so | grep ndsclient
```

## Runtime configuration

1. Copy `nds_config.conf.example` to `nds_config.conf` and replace the placeholders with site-specific network settings.
2. If the vendor SDK uses the agent configuration, copy `conf/agent.conf.example` to `conf/agent.conf` and provide the site-specific monitor addresses and credentials.
3. Keep `nds_wrapper.conf` beside `nds_config.conf` and adjust its values for the target KV pool. If `nds_config.conf` is moved elsewhere, copy `nds_wrapper.conf` to that directory as well.

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
