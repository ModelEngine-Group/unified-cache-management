# dev-sandbox Developer Guide

> Developer reference: build, native sub-commands (`copy`/`trans`/`aio`), complete case table, and low-level parameters. For daily usage, see [README](../user/dev-sandbox.md).

CMake C++17 performance testing project. The toolkit handles project building, binary location, and sub-command argument forwarding; `copy`, `trans`, and `aio` business parameters are parsed by the underlying binary.

← Back to [UCM Toolkit](../index.md)

## Dependencies

| Feature | Dependencies |
| --- | --- |
| `dev-sandbox` build | CMake 3.18+, C++17 compiler. CUDA backend requires CUDA runtime; Ascend backend requires Ascend runtime; `copy` GDR case also requires `libibverbs` headers and library. |

## Build

Default build directory:

```text
toolkit/src/dev-sandbox/build
```

Common commands:

```bash
ucm-toolkit build dev-sandbox
ucm-toolkit build dev-sandbox --build-type Debug
ucm-toolkit build dev-sandbox --build-type Release --jobs 16
```

Specifying CUDA or Ascend runtime:

Via `--cmake-arg` (highest priority):

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda

ucm-toolkit build dev-sandbox \
  --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

Or via environment variables (same effect):

```bash
# CUDA: CUDA_HOME or CUDA_PATH
CUDA_HOME=/usr/local/cuda ucm-toolkit build dev-sandbox

# Ascend: ASCEND_HOME / ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest ucm-toolkit build dev-sandbox
```

If neither `--cmake-arg` nor environment variables are set, and no valid CUDA/Ascend runtime is detected on the machine, it automatically falls back to the CPU Simulation backend.

Specifying a build directory:

```bash
ucm-toolkit build dev-sandbox \
  --build-dir toolkit/build/dev-sandbox/release \
  --build-type Release \
  --jobs 16
```

After a successful `--build-dir` build, the `build_dir` field in the adapter is updated; subsequent `ucm-toolkit run dev-sandbox ...` will look for binaries in that directory.

Build parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--build-type` | `Release` | `CMAKE_BUILD_TYPE` passed to CMake. |
| `--jobs`, `-j` | Not set | Parallelism passed to `cmake --build`. |
| `--build-dir` | `toolkit/src/dev-sandbox/build` | Override the build output directory. |
| `--cmake-arg` | Empty | Additional CMake configure arguments; can be repeated. |

## Running Sub-features

View sub-features:

```bash
ucm-toolkit run dev-sandbox --help
```

Available sub-features:

| Sub-feature | Binary | Description |
| --- | --- | --- |
| `copy` | `module/copy/copy` | Device/host memory copy performance testing. Measures bandwidth between different memory types (normal host, pinned, anonymous, device) under different transport engines (CE, SM, GDR), for evaluating H2D/D2H/D2D path throughput. |
| `trans` | `module/trans/trans` | Host/device transfer matrix performance testing. Composes a matrix of direction (H2D/D2H) × host buffer type × transport method, batch-running all matching cases, for quickly scanning bandwidth distribution across all transfer paths. |
| `aio` | `module/aio/aio` | Async I/O disk write/read performance testing. Creates block files in a specified workspace, performs dump (write) and load (read) via Linux AIO, measuring disk bandwidth, for evaluating UCM POSIX store disk throughput. |

### Quick Mode (model-type / iodirect / sdma)

If you don't care about the underlying `copy` case names, you can use the three semantic parameters `--model-type`, `--iodirect`, and `--sdma` to select the corresponding Ascend copy interface via a mapping table. Parameters after these three are passed through to the underlying `copy` binary as-is (all native `copy` parameters `-s`/`-n`/`-i`/`-d`/`-f`, see [`### copy`](#copy) below). Native `copy`/`trans`/`aio` sub-command usage is not affected.

```bash
ucm-toolkit run dev-sandbox \
  --model-type gqa --iodirect false --sdma false \
  -s 16K -n 512 -i 128 -d 8
```

Mapping of `model-type` / `iodirect` / `sdma` to Ascend copy interfaces:

| model-type | iodirect | sdma | Copy Interface (Ascend) | Scenario |
| --- | --- | --- | --- | --- |
| `gqa` | `false` | `false` | `all_host_to_all_device_ce_multi_stream` | Per-card host buffers, 4-stream CE H2D. |
| `gqa` | `true` | `false` | `all_odirect_host_to_all_device_ce_multi_stream` | Per-card O_DIRECT mmap host buffers, 4-stream CE H2D. |
| `gqa` | `false` | `true` | `all_host_to_all_device_ffts_direct_h2d` | Per-card mapped host buffers, FFTS direct H2D SDMA. |
| `gqa` | `true` | `true` | `all_odirect_host_to_all_device_ffts_direct_h2d` | Per-card O_DIRECT mmap host buffers, FFTS direct H2D SDMA. |
| `mla` | `false` | `false` | `one_share_host_to_all_device_ce_multi_stream` | One shared memory host buffer fanned out to all cards, 4-stream CE. |
| `mla` | `true` | `false` | `one_share_host_to_all_device_ce_multi_stream` | MLA does not distinguish iodirect, same CE path as iodirect=false. |
| `mla` | `false` | `true` | `one_share_host_to_all_device_ffts_direct_h2d` | One shared memory host buffer, FFTS direct H2D SDMA distribution. |
| `mla` | `true` | `true` | `one_share_host_to_all_device_ffts_direct_h2d` | MLA does not distinguish iodirect, same SDMA path as iodirect=false. |

Quick mode only maps the three selectors to `-t <case>`; all other parameters follow `copy`'s native semantics (using `copy` defaults if not passed). For example, the following two commands are exactly equivalent:

```bash
# Quick mode
ucm-toolkit run dev-sandbox --model-type gqa --iodirect false --sdma false -s 16K -n 512 -i 128 -d 8
# Native mode
ucm-toolkit run dev-sandbox copy -t all_host_to_all_device_ce_multi_stream -s 16K -n 512 -i 128 -d 8
```

### copy

Examples:

```bash
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K -n 512 -i 128 -d 8
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -t device_to_host_ce -s 1M
```

Parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `-t <name>` | Required | Case name; can be repeated for multiple. |
| `-s <size>` | `512M` | Single data block size; only accepts `K/k` or `M/m` suffix, e.g. `16K`, `1M`. |
| `-n <count>` | `8` | Number of data blocks per buffer. |
| `-f`, `--frags`, `-frags <count>` | `0` | Number of fragments per IO/task for FFTS direct H2D. When set, `-n` represents IO/task count. |
| `-i <count>` | `128` | Number of iterations. |
| `-d <count>` | `8` | Number of devices. |

The native `copy` program does not currently register `-h/--help` as a successful help parameter. Running without arguments prints usage and exits non-zero; specifying a non-existent case lists all cases compiled into the current backend:

```bash
ucm-toolkit run dev-sandbox copy -t unknown
```

Common cases:

| Backend | Case | Description |
| --- | --- | --- |
| **CUDA / Ascend** | `host_to_device_ce` | Single-stream CE DMA from normal host memory to device memory. **Scenario**: Evaluate single-card H2D CE baseline bandwidth, suitable as H2D performance baseline. |
| | `host_to_device_batch_ce` | Batch CE DMA from host to device. **Scenario**: Evaluate whether batch H2D submission is more efficient than individual submission. |
| | `one_host_to_all_device_ce` | Same host data broadcast to all devices via CE. **Scenario**: Evaluate model loading performance when same parameters are distributed to multiple cards. |
| | `all_host_to_all_device_ce` | Per-card host buffers copied to respective devices via CE simultaneously. **Scenario**: Evaluate total throughput of concurrent multi-worker H2D. |
| | `device_to_device_ce` | Intra-card D2D CE copy (src and dst on same card). **Scenario**: Evaluate intra-card device memory transfer bandwidth. For cross-card D2D, use `one_device_to_all_device_ce`. |
| | `one_device_to_all_device_ce` | Single-card data broadcast to all devices (including itself) via CE, cross-card D2D. **Scenario**: Evaluate cross-card D2D transfer bandwidth. |
| | `anonymous_to_device_ce` | Anonymous pinned memory (mmap allocated but not explicitly registered) copied to device via CE. **Scenario**: Compare H2D performance between anonymous pinned and normal host memory. |
| **CUDA** | `device_to_host_ce` | Single-stream CE DMA from device memory to normal host memory. **Scenario**: Evaluate single-card D2H CE baseline bandwidth. |
| | `device_to_host_batch_ce` | Batch CE DMA from device to host. **Scenario**: Evaluate whether batch D2H is more efficient than individual readback. |
| | `host_to_device_sm` | SM kernel copies host data to device. **Scenario**: Compare SM kernel transport vs CE DMA bandwidth. |
| | `device_to_host_sm` | SM kernel copies device data to host. **Scenario**: Compare SM kernel readback vs CE readback bandwidth. |
| | `one_host_to_all_device_sm` | Same host data broadcast to all devices via SM. **Scenario**: Compare SM broadcast vs CE broadcast multi-card distribution. |
| | `device_to_anonymous_ce` | CE DMA from device to anonymous pinned memory. **Scenario**: Evaluate D2H bandwidth to anonymous pinned memory. |
| | `anonymous_to_device_sm` | SM kernel copies anonymous pinned memory data to device. **Scenario**: Evaluate H2D bandwidth with anonymous pinned + SM combination. |
| | `device_to_anonymous_sm` | SM kernel copies device data to anonymous pinned memory. **Scenario**: Evaluate D2H bandwidth with anonymous pinned + SM combination. |
| **Ascend** | `host_to_device_ce_multi_stream` | 4-stream concurrent CE DMA from host to device. **Scenario**: Evaluate whether multi-stream parallel transfer improves H2D throughput. |
| | `one_share_host_to_all_device_ce_multi_stream` | One POSIX shared memory host buffer fanned out to all devices via fork, 4-stream CE per card. **Scenario**: Simulate MLA model where multiple cards read the same shared host KV data and write to their own devices. |
| | `all_host_to_all_device_ce_multi_stream` | Per-card host buffers, fork fan-out with 4-stream CE per card. **Scenario**: Evaluate multi-process, multi-card, multi-stream H2D aggregate throughput. |
| | `all_odirect_host_to_all_device_ce_multi_stream` | Per-card UCM O_DIRECT-style mmap host buffers, fork fan-out with 4-stream CE to devices. **Scenario**: Closer to GQA model with O_DIRECT enabled, each card reading KV data from local host buffer simultaneously. |
| | `all_host_to_all_device_ffts_direct_h2d` | Per-card mapped `aclrtMallocHost` buffers, FFTS Plus direct H2D SDMA to devices. **Scenario**: Evaluate direct H2D SDMA with regular pinned host source. |
| | `one_share_host_to_all_device_ffts_direct_h2d` | One POSIX shared memory host buffer, mapped/pinned registered in child processes, distributed to all devices via FFTS Plus direct H2D SDMA. **Scenario**: Simulate MLA model where multiple cards read the same shared host KV data, verify FFTS direct H2D shared source read-in path. |
| | `all_odirect_host_to_all_device_ffts_direct_h2d` | Per-card UCM O_DIRECT-style mmap host buffers, mapped + pinned registered, FFTS Plus direct H2D SDMA to devices. **Scenario**: Closer to GQA model with O_DIRECT enabled, direct H2D path for each card reading KV from local host buffer simultaneously. |
| **CUDA + libibverbs** | `host_to_device_gdr` | GPUDirect RDMA direct host-to-single-card device memory. **Scenario**: Evaluate whether RDMA direct to GPU is faster than traditional CE. |
| | `one_host_to_all_device_gdr` | Same host data broadcast to all devices via GDR. **Scenario**: Evaluate multi-card RDMA direct distribution performance. |
| | `all_host_to_all_device_gdr` | Per-card host buffers simultaneously copied to respective devices via GDR. **Scenario**: Evaluate total throughput of concurrent multi-worker GDR. |
| **Simulation** | `host_to_anonymous_memcpy` | CPU memcpy from host to anonymous memory. **Scenario**: Simulate H2D transfer without GPU, for functional verification or CPU memcpy baseline. |
| | `shm_to_all_host_memcpy` | CPU memcpy from shared memory to all host buffers. **Scenario**: Evaluate shared memory to host copy bandwidth, simulating cross-process data distribution. |

GDR cases use `GDR_NICS` to specify device-to-RDMA NIC mapping; NIC count must match `-d`:

```bash
GDR_NICS=mlx5_0,mlx5_2,mlx5_4,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14 \
ucm-toolkit run dev-sandbox copy -t all_host_to_all_device_gdr -s 16K -n 512 -i 128 -d 8
```

### trans

`trans` composes a matrix of direction (H2D/D2H) × host buffer type × device buffer type × transport method, batch-running all matching cases. Unlike `copy` which focuses on detailed performance of a single path, `trans` focuses on quickly scanning bandwidth distribution across all combinations.

Host buffer types:

| Type | Description | Scenario |
| --- | --- | --- |
| `normal` | Normal pageable host memory (malloc allocated). May incur page faults during DMA transfer, typically lowest bandwidth. | Evaluate the most common malloc scenario, as a comparison baseline. |
| `anonymous` | Anonymous pinned memory (mmap allocated, not explicitly registered). Fewer page faults, higher DMA bandwidth than normal. | Evaluate the improvement of mmap pinned allocation on transfer. |
| `registered` | Explicitly registered pinned memory (cudaHostRegister / Ascend memory register). Most efficient DMA transfer, highest bandwidth. | Evaluate the transfer ceiling of registered pinned memory, for high-performance host buffer allocation strategy. |

Transport methods:

| Method | Backend | Description | Scenario |
| --- | --- | --- | --- |
| `ce` | CUDA / Ascend | Copy Engine DMA hardware transfer, does not occupy SM/compute resources. | Evaluate DMA baseline bandwidth, suitable for background data transfer. |
| `batch_ce` | CUDA / Ascend | Batch Copy Engine submission, reducing multiple launch overhead. | Evaluate whether batch DMA is more efficient than individual CE. |
| `sm` | CUDA | SM kernel transfer, occupies GPU compute resources. | Compare SM kernel vs CE DMA bandwidth. |
| `ms_48` | Ascend | 48-stream concurrent Copy Engine. | Evaluate whether multi-stream parallel DMA improves throughput. |
| `memcpy` | Simulation | CPU memcpy. | Simulate transfer without GPU, for functional verification or baseline. |

Examples:

```bash
ucm-toolkit run dev-sandbox trans -h
ucm-toolkit run dev-sandbox trans -t H2D -H normal -D normal -M ce -s 32768 -n 1024 -d 8 -i 1024
ucm-toolkit run dev-sandbox trans -M ce -M batch_ce
```

Parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `-H <host>` | All | Host buffer type; can be repeated. Common values: `normal`, `anonymous`, `registered`. |
| `-D <device>` | All | Device buffer type; can be repeated. Common values: `normal`. |
| `-M <method>` | All | Transport method; can be repeated. Common values: `ce`, `batch_ce`, `sm`, `ms_48`, `memcpy`, depending on backend. |
| `-t <type>` | `ANY` | Transfer direction, supports `H2D` or `D2H`; if not specified, runs matching cases in both directions. |
| `-s <size>` | `32768` | Single transfer size in bytes. |
| `-n <number>` | `1024` | Number of data items. |
| `-d <nDevice>` | `8` | Number of devices. |
| `-i <nIter>` | `1024` | Number of iterations. |
| `-h` | - | Show native help. |

If the filter conditions do not match any case, the program prints all available cases for the current backend.

### aio

`aio` creates/opens block files in a specified workspace and performs async writes (dump) and async reads (load) via Linux AIO to measure disk I/O bandwidth. Unlike `copy`/`trans` which measure device memory bandwidth, `aio` measures disk throughput.

Host buffer allocation strategies:

| Strategy | Description | Scenario |
| --- | --- | --- |
| `mmap` | Allocates host memory via `mmap`, page-aligned and directly referenceable by AIO. Suitable for large I/O, low allocation overhead. | Evaluate disk bandwidth with mmap allocation, suitable for large sequential read/write. |
| `alloc` | Allocates host memory via device-specific pinned allocation (CUDA: `cudaMallocHost`, Ascend: `aclrtMallocHost`). Memory is pinned, more efficient for DMA but higher allocation overhead. | Evaluate disk bandwidth with pinned memory, suitable for scenarios requiring direct DMA transfer of disk data to device. |

Examples:

```bash
mkdir -p /tmp/ucm-aio
ucm-toolkit run dev-sandbox aio --workspace /tmp/ucm-aio

ucm-toolkit run dev-sandbox aio \
  --workspace /tmp/ucm-aio \
  --io-type mmap \
  --io-size 1048576 \
  --io-number 512 \
  --device-id 0 \
  --epoch-number 32
```

Parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--workspace <path>` | Required | AIO test working directory. |
| `--io-type mmap\|alloc` | `mmap` | Host buffer allocation strategy. |
| `--io-size <bytes>` | `1048576` | Size of each I/O shard in bytes. |
| `--io-number <n>` | `512` | Number of I/O shards. |
| `--device-id <id>` | `0` | Device ID to use. |
| `--epoch-number <n>` | `32` | Number of epochs for write and read respectively. |
| `-h`, `--help` | - | Show native help. |

## FAQ

### Binary Not Found

Build first:

```bash
ucm-toolkit build dev-sandbox
ucm-toolkit doctor dev-sandbox
```

If `--build-dir` was previously used to specify a different directory, `run` will look for binaries in the build directory currently recorded in the adapter.

### Switching CUDA or Ascend Backend

Re-specify the runtime and rebuild:

```bash
ucm-toolkit build dev-sandbox --cmake-arg -DCUDA_ROOT=/usr/local/cuda
ucm-toolkit build dev-sandbox --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

You can also influence CMake detection via `CUDA_HOME`, `CUDA_PATH`, `ASCEND_HOME`, `ASCEND_TOOLKIT_HOME` environment variables.
