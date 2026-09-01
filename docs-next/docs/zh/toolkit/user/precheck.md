# precheck — UCM environment pre-check

`precheck` runs **locally on the UCM deployment host** before UCM starts and
verifies that the environment meets UCM's runtime requirements. It reports
`PASS` / `WARN` / `FAIL` for each item and prints **remediation advice** for
failing or borderline checks, so issues can be fixed before a deployment
attempt fails.

It is the first item of the toolkit RFC (#1208): *UCM 部署前预检工具 — verify
storage bandwidth, kernel version, driver version, shm-size, and other key
configuration before running UCM*.

← 返回 [UCM Toolkit 文档](../index.md)

```bash
ucm-toolkit run precheck                           # all checks (bandwidth skipped w/o a mount path)
ucm-toolkit run precheck --mount-path /mnt/ucm_cache
ucm-toolkit run precheck --skip-bandwidth --json
ucm-toolkit run precheck --only kernel --only accelerator_driver
```

## Checks

| Check | Severity | What it verifies | Threshold |
| --- | --- | --- | --- |
| `serving_stack` | INFO | installed `vllm` / `vllm-ascend` / `sglang` versions | display only |
| `uc_manager` | INFO | installed `uc-manager` version | display only |
| `accelerator_driver` | FAIL/WARN | NVIDIA `compute_cap` (via `nvidia-smi`) or Ascend `HDK` (via `npu-smi info`) | no driver → FAIL; CUDA `compute_cap >= 8.0`; Ascend `HDK > 25.2.0` (below → WARN) |
| `kernel` | **FAIL** | running kernel major.minor (`uname -r`) | `>= 5.10` series |
| `memory_shm` | WARN | `/dev/shm` size (`statvfs`); RAM is hidden | `/dev/shm` ≥ 512 GiB (below → WARN) |
| `aio_resources` | INFO | kernel AIO pool (`/proc/sys/fs/aio-max-nr`, `aio-nr`); computes max concurrent aio workers | display only |
| `bandwidth` | WARN | posix `psync`/`aio` matrix benchmarked against the real `ucm` C++ store | all metrics `>= 8 GB/s` |

### Severity and exit codes

- **INFO** — display only; never affects the exit code.
- **WARN** — soft threshold; does not fail unless `--strict` is set.
- **FAIL** — hard constraint; fails the run.

Exit codes: `0` all hard checks pass (warnings tolerated) · `1` a `FAIL`
(or a `WARN` under `--strict`) · `2` CLI/usage error. `--only`/`--skip`
select checks; `--skip-bandwidth` skips the (slow) benchmark.

## Bandwidth benchmark

The bandwidth check exercises the real `UcmPipelineStore`
(`store_pipeline="Posix"`) — the same code path UCM uses to dump/load KV cache
shards — across a configurable matrix and picks the best configuration.

Default matrix (all configurable via `--shard-sizes`, `--workers`, `--engines`,
`--modes`): shard sizes `{180 KiB, 8 MiB}` × worker counts `{1, 8, 16}` × IO
engines `{psync, aio}`, and for each combo runs the phases selected by
`--modes` (default `dump,read,mix`): a pure-dump (cold write) phase, a
pure-read phase, and a read-heavy mixed phase (per epoch `1 dump + rw_ratio`
loads, default `1:4`, mimicking the real KV-cache access pattern — write once,
fetch many). Per phase it reports `mean ± std` and `min..max` across epoch
samples plus the **aggregate** (per-worker × worker count). The combo with the
highest **aggregate mixed** bandwidth (falling back to aggregate `mean(dump,
load)`) is the recommended configuration; a warning fires when it is below
`--threshold` (default `8 GB/s`). Defaults are lean (8 epochs/phase, block 32);
`--quick` halves all epoch counts.

`numpy` and the `ucm` package (with its built C++ `posix` store) are
**lazy imports** — if `ucm` is not importable the bandwidth check is **skipped
with a warning** rather than failing, so the rest of the pre-check still runs.

## Configuration

Every tunable value — thresholds (kernel, CUDA compute cap, Ascend HDK,
`/dev/shm` min, bandwidth), the matrix (shard sizes, worker counts, engines,
modes, epochs, `rw_ratio`, block number, barrier timeout) — lives in a single
shipped file, `precheck.defaults.json` (next to this module). **A version
update is a data-only change**: edit that file, no code edit needed. Layering,
in increasing precedence: code constants (fallback if the JSON is absent) →
`precheck.defaults.json` → a `--config FILE` user override → CLI flags.

```json
{
  "mount_path": "/mnt/ucm_cache",
  "kernel_min": "5.10",
  "cuda_min_compute_cap": 8.0,
  "ascend_min_hdk": "25.2.0",
  "bandwidth": {
    "shard_sizes": ["180k", "1m"],
    "worker_counts": [1, 16],
    "engines": ["psync", "aio"],
    "block_number": 64,
    "shard_number": 1,
    "dump_epochs": 32,
    "load_epochs": 32,
    "threshold_gb": 8.0
  }
}
```

### Flags

| Flag | Description |
| --- | --- |
| `--config FILE` | load thresholds/matrix from a JSON or YAML file |
| `--mount-path PATH` | UCM storage mount point (required for the bandwidth benchmark) |
| `--shard-sizes LIST` | comma-separated shard sizes, e.g. `180k,8m` |
| `--workers LIST` | comma-separated worker counts, e.g. `1,8,16` |
| `--engines LIST` | comma-separated IO engines, subset of `psync,aio` (default both) |
| `--modes LIST` | comma-separated phases per combo: subset of `dump,read,mix` (default all; `mix` = read-heavy `1:rw_ratio`) |
| `--threshold GB` | minimum best aggregate bandwidth in GB/s |
| `--block-number` / `--dump-epochs` / `--load-epochs` / `--mixed-epochs` / `--rw-ratio` | matrix sweep parameters |
| `--quick` | halve all epoch counts for a faster run |
| `--skip-bandwidth` | skip the (slow) bandwidth benchmark |
| `--kernel-min` / `--cuda-min-compute-cap` / `--ascend-min-hdk` | threshold overrides |
| `--only CHECK` / `--skip CHECK` | select checks (repeatable) |
| `--strict` | treat warnings as failures for the exit code |
| `--no-color` / `--json` / `--verbose` | output control |

## Notes

- **`npu-smi info` HDK parsing** anchors on labels in priority order: an
  explicit `HDK Version` label → a `Driver Version` label → the banner
  `Version:` field. Validated against real openEuler firmware
  (`npu-smi 25.5.2   Version: 25.5.2`, where the banner `Version` *is* the
  HDK/driver version). The threshold is configurable via `--ascend-min-hdk`.
- **Kernel `>= 5.10` is on the major.minor series**: `5.10.0-216` (an openEuler
  5.10 LTS backport build) passes, `5.9` fails. Validated on a real UCM box.
  Adjust `--kernel-min` otherwise.
- **Bandwidth load reads can be inflated by the page cache** (the store default
  is `io_direct=false`): measured load BW reflects cached reads while the dump
  (cold write) is the disk-bound figure. The reported comprehensive
  `mean(dump, load)` still reflects usable throughput; for pure disk bandwidth
  enable `io_direct` in the store config.
- The bandwidth benchmark is Linux/posix-only (anonymous `mmap` + the ucm
  `.so`); it is skipped elsewhere with a note.
- The tool sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0` to prevent torch_npu from
  loading (the posix store uses libaio directly and does not need the NPU).
  This avoids FunctionLoader warnings and interference in container environments
  where the NPU driver is only partially mounted. Worker stderr is also
  redirected to `/dev/null` via `os.dup2`. A `combo_timeout` (default 300s,
  configurable) is a backstop that terminates any worker that blocks.
- Workers use **per-epoch barriers** (mirroring the original
  `posixstore_aio_test.py`): all workers sync after each epoch so the kernel
  aio queue is drained before the next burst. Without this, independent
  workers can overflow `fs.aio-max-nr` at high worker counts, causing
  `worker.wait(task)` to block in uninterruptible D state.

## Tests

```bash
cd toolkit
python -m unittest tests.test_precheck tests.test_precheck_toolkit -v
```

`test_precheck` covers the pure logic (version/size/SMI parsing, kernel
boundary, config, bandwidth selection, reporter exit codes) plus the
check-function decision logic with `subprocess`/`os` mocked, including the
RFC #1208 remediation advice for `FAIL`. `test_precheck_toolkit` covers toolkit
integration (registration, `list`, `run precheck` dispatch, `doctor`).
