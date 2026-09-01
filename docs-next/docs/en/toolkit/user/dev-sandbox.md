# dev-sandbox

dev-sandbox measures Ascend host-to-device memory copy bandwidth, covering normal memory, O_DIRECT, shared memory allocation modes and multi-stream CE, FFTS direct H2D SDMA transport engines, supporting both GQA (per-card local memory) and MLA (single shared memory distributed to all cards) topologies.

← Back to [UCM Toolkit](../index.md)

## IO Flow and UCM KV Cache

UCM uses a tiered KV cache: device memory (HBM, fastest) → host memory (layer cache, second) → disk (POSIX store, slowest). When a request's required KV data misses device memory, it must be loaded from host memory (or disk) back to device memory. This **host-to-device (H2D) copy** step is exactly what dev-sandbox `copy` measures; the `aio` sub-feature separately measures disk dump/load bandwidth.

```text
┌──────────────────── UCM Tiered KV Cache ──────────────────┐
│                                                           │
│  ┌──────────────┐     H2D copy       ┌────────────────┐  │
│  │  Host Memory  │ ══════════════════▶ │ Device Memory  │  │
│  │ (layer cache) │   CE multi-stream  │  (HBM/KV cache)│  │
│  └──────┬───────┘   / SDMA            └────────────────┘  │
│         │ dump/load                                       │
│  ┌──────▼───────┐                                       │
│  │ Disk (POSIX)  │ ◀── aio sub-feature measures disk BW   │
│  └──────────────┘                                       │
└───────────────────────────────────────────────────────────┘
```

Different transport engines represent different hardware copy paths during the H2D stage:

| Transport Engine | Description | UCM Stage |
| --- | --- | --- |
| Multi-stream CE (`--sdma false`) | Copy Engine hardware DMA, 4 concurrent streams for higher throughput. | Standard H2D loading path: copying KV blocks from host layer cache back to device memory. |
| FFTS direct H2D SDMA (`--sdma true`) | SDMA direct H2D path with FFTS fast task scheduling, lower latency. | KV loading bandwidth evaluation for the direct H2D path. |

Host memory allocation modes correspond to different KV sources/topologies:

| Allocation Mode | Description | Topology |
| --- | --- | --- |
| Normal host memory (`--iodirect false`) | Pinned host memory, one copy per card. | GQA: each card loads its own KV from local host buffer. |
| O_DIRECT (`--iodirect true`) | O_DIRECT-style mmap host memory, bypasses page cache, closer to UCM's direct file read path. | GQA: local KV read-in path with O_DIRECT enabled per card. |
| Shared memory | A single POSIX shared memory host buffer, distributed to all cards via fork fan-out. | MLA: all cards share the same host KV data. |

Data flow for both topologies:

```text
GQA: each card has its own local host buffer → its own device memory (independent H2D per card)

  host buf(card0) ── multi-stream CE / SDMA ──▶ device HBM(card0)
  host buf(card1) ── multi-stream CE / SDMA ──▶ device HBM(card1)
  host buf(card2) ── multi-stream CE / SDMA ──▶ device HBM(card2)
   ...                                          ...
  host buf(cardN) ── multi-stream CE / SDMA ──▶ device HBM(cardN)

MLA: one shared host buffer fanned out to all cards

                 ┌──▶ device HBM(card0)
 shared host buf ┼──▶ device HBM(card1)   (fork fan-out, multi-stream CE / SDMA per card)
                 └──▶ device HBM(cardN)
```

## Quick Start

```bash
ucm-toolkit run dev-sandbox --model-type <gqa|mla> --iodirect <true|false> --sdma <true|false> [args...]
```

Three selector parameters:

| Parameter | Values | Description |
| --- | --- | --- |
| `--model-type` | `gqa` / `mla` | Model type. |
| `--iodirect` | `true` / `false` | Whether to enable O_DIRECT (direct IO, bypassing page cache). |
| `--sdma` | `true` / `false` | Whether to use SDMA (direct H2D SDMA transport path). |

Parameter combinations determine the test scenario:

| Model | IO-direct | SDMA | Test Scenario |
| --- | --- | --- | --- |
| `gqa` | `false` | `false` | Each card copies from its own local host memory to device memory using multi-stream CE. |
| `gqa` | `true` | `false` | Each card uses O_DIRECT to read from local host memory, multi-stream CE to device. |
| `gqa` | `false` | `true` | Each card copies from host memory to device using FFTS direct H2D SDMA. |
| `gqa` | `true` | `true` | Each card uses O_DIRECT from host memory, via FFTS direct H2D SDMA to device. |
| `mla` | `false` | `false` | A shared host memory is distributed across cards, each using multi-stream CE to its own device. |
| `mla` | `true` | `false` | Same as above (MLA does not distinguish iodirect, same CE path). |
| `mla` | `false` | `true` | A shared host memory is distributed to all device memories via FFTS direct H2D SDMA. |
| `mla` | `true` | `true` | Same as above (MLA does not distinguish iodirect, same SDMA path). |

## Run Parameters

After the three selector parameters, the following optional parameters can be used (defaults apply if omitted):

| Parameter | Description | Default | Example |
| --- | --- | --- | --- |
| `-s` | Single data block size | `512M` | `-s 16K`, `-s 1M` |
| `-n` | Number of data blocks | `8` | `-n 512` |
| `-i` | Number of iterations | `128` | `-i 128` |
| `-d` | Number of devices (cards) | `8` | `-d 8` |
| `-f` | Fragment count (SDMA scenarios only) | `0` | `-f 4` |

## Examples

```bash
# Test GQA + multi-stream CE: 16K blocks × 512 blocks × 128 iterations × 8 cards
ucm-toolkit run dev-sandbox --model-type gqa --iodirect false --sdma false \
  -s 16K -n 512 -i 128 -d 8

# Test MLA shared distribution + SDMA with default parameters
ucm-toolkit run dev-sandbox --model-type mla --iodirect false --sdma true
```

### Output Example

Below is actual output from two commands on an 8-card Ascend 910B3 (values vary by machine):

GQA + multi-stream CE (each card's own local host buffer → its own device memory):

```text
[[ all_host_to_all_device_ce_multi_stream ]] memcpy from all host to all device with ce using multi stream and fork submit
  From              To                Method    Size(KB)  Count   Submit(us)-(Min/Max/Avg/P50/P90)        Copy(us)-(Min/Max/Avg/P50/P90)              BW(GB/s)
  acl::host::all    acl::device::all  CE-MS-FORK16        4096    878 / 2576 / 1117 / 1112 / 1269         3155 / 4101 / 3736 / 3759 / 3905            16.729
```

MLA + multi-stream CE (one shared host buffer fanned out to all cards):

```text
[[ one_share_host_to_all_device_ce_multi_stream ]] memcpy from one shared host to all device with ce using multi stream and fork submit
  From              To                Method    Size(KB)  Count   Submit(us)-(Min/Max/Avg/P50/P90)        Copy(us)-(Min/Max/Avg/P50/P90)              BW(GB/s)
  acl::shm::0       acl::device::all  CE-MS-FORK16        4096    1056 / 1270 / 1162 / 1161 / 1209        3914 / 5024 / 4471 / 4424 / 4852            13.979
```

MLA uses `acl::shm::0` as the data source (shared memory), GQA uses `acl::host::all` (per-card host memory); both have `Count` = "card count × `-n`" (8 × 512 = 4096). The `--sdma true` FFTS direct H2D SDMA path requires the machine to support FFTS features; the output format is the same, only `Method`/`Copy` values differ.

Output field descriptions:

| Field | Description |
| --- | --- |
| Title line `[[ ... ]]` | Actual executed case name and summary; used to confirm which case `--model-type`/`--iodirect`/`--sdma` mapped to. |
| `From` / `To` | Source / destination memory type. `acl::host::all` = per-card host memory, `acl::shm::0` = shared memory, `acl::device::all` = all device memories. |
| `Method` | Transport engine, e.g. `CE-MS-FORK` = multi-stream CE + fork fan-out (column width is exactly filled by the method name, so in output `CE-MS-FORK` and the following `Size(KB)` value appear concatenated as `CE-MS-FORK16`). |
| `Size(KB)` | Single data block size, corresponds to `-s` (e.g. `-s 16K` → `16`). |
| `Count` | Total number of data blocks actually copied. In multi-card fork scenarios, this is "card count × `-n`" (e.g. 8 cards × 512 = 4096). |
| `Submit(us)` | Submit phase timing statistics (microseconds), given as Min/Max/Avg/P50/P90. |
| `Copy(us)` | Actual copy timing statistics (microseconds), given as Min/Max/Avg/P50/P90; basis for bandwidth calculation. |
| `BW(GB/s)` | Aggregate bandwidth = `Size × Count × 1e6 / Copy.avg`, in GB/s, higher is better. |

## First Use

Build the project before first run:

```bash
ucm-toolkit build dev-sandbox
```

For build details (backend selection, custom paths), native sub-commands (`copy`/`trans`/`aio`), and the complete case table, see the [Developer Guide](../developer/dev-sandbox.md).
