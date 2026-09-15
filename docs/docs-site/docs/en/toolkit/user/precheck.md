# Deployment precheck

`precheck` inspects the host environment for a UCM deployment. It records software versions, checks accelerator drivers, the kernel, shared memory and AIO resources, and can benchmark storage bandwidth at a selected directory. This helps identify environment issues that may affect startup or cache I/O before running the service.

Use it when:

- Deploying for the first time or moving to a new machine to check the host against precheck requirements.
- Investigating engine startup failures or cache-loading problems that may involve drivers, shared memory or kernel resources.
- Evaluating a cache directory with an initial bandwidth check of its read/write path.

The tool reports each check with remediation advice and supports JSON output for saving and comparing environments. Informational checks record values such as versions; other checks produce `PASS`, `WARN` or `FAIL` against configured thresholds. Prepare the CLI through [toolkit installation](../index.md).

## Run checks

```bash
ucm-toolkit run precheck --skip-bandwidth
ucm-toolkit run precheck --mount-path /mnt/ucm-cache-test
ucm-toolkit run precheck --only kernel --only accelerator_driver
ucm-toolkit run precheck --skip-bandwidth --json
```

Bandwidth checks write and read the selected path. Use a dedicated test directory with space available. Without a mount path, that check is skipped. Environment checks use the Python standard library; bandwidth checks also require Linux, NumPy and loadable native UCM Posix Store.

## Interpret results

| Check | Current default | Meaning |
| --- | --- | --- |
| Engine and `uc-manager` versions | INFO, display only | Record the installed environment |
| Accelerator driver | Missing driver fails; CUDA compute capability at least 8.0; Ascend HDK above 25.2.0, otherwise warning | Read from device query commands |
| Kernel | At least the 5.10 series | Compare major/minor versions |
| `/dev/shm` | Warn below 512 GiB | A precheck threshold; actual UCM capacity depends on configuration and model |
| AIO resources | INFO, display only | Read kernel limits and current usage |
| Storage bandwidth | Warn when the best aggregate configuration is below 8 GB/s | Adjustable threshold, affected by caches and workload |

INFO does not change the exit code. Warnings are allowed by default; `--strict` treats them as failures. Codes are `0` for pass, `1` for failure or strict warnings, and `2` for CLI usage errors. Passing precheck still requires engine startup and [external-cache verification](../../user-guide/observability/verify-cache.md).

## Configure thresholds and load

Precedence is: fallback code values → `precheck.defaults.json` → `--config FILE` → CLI flags. JSON needs no extra parser; YAML requires PyYAML.

Example override file:

```json
{
  "mount_path": "/mnt/ucm-cache-test",
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

The override retains the original workload settings and leaves the shared-memory threshold at its configured default. Run without `--quick` to preserve the specified epoch counts:

```bash
ucm-toolkit run precheck --config /path/to/precheck.json
```

| Flag | Purpose |
| --- | --- |
| `--mount-path` | Bandwidth test directory |
| `--only` / `--skip` | Include or exclude checks, repeatable |
| `--skip-bandwidth` | Environment checks only |
| `--shard-sizes` / `--workers` / `--engines` | Shard sizes, worker counts and I/O engines |
| `--modes` | Select phases from `dump,read,mix` |
| `--threshold` | Bandwidth warning threshold in GB/s |
| `--block-number` / `--dump-epochs` / `--load-epochs` / `--mixed-epochs` / `--rw-ratio` | Test size and read/write ratio |
| `--kernel-min` / `--cuda-min-compute-cap` / `--ascend-min-hdk` | Environment threshold overrides |
| `--quick` | Reduce epoch counts |
| `--strict` / `--json` / `--no-color` / `--verbose` | Exit status and output formatting |

## Bandwidth interpretation

Defaults sweep 180 KiB and 8 MiB shards, 1/8/16 workers, and psync/aio. Each combination runs dump, read and read-heavy mixed phases. The mixed ratio is one write to four reads, with eight epochs per phase.

The tool reports sample means, variation and aggregates. Selection prefers mixed bandwidth and falls back to the dump/read mean when mixed data is absent. Non-Direct I/O reads may be served by the OS page cache; interpret results as throughput of that tested path. State I/O mode and cache state when measuring storage media.

Unavailable native dependencies produce a skipped bandwidth check with a warning. The implementation manages timeouts and worker synchronization; the current default `combo_timeout` is 120 seconds. Sources are under `toolkit/ucm_toolkit/tools/precheck/`; use `ucm-toolkit run precheck --help` for exact flags.

## Development checks

The existing tests cover threshold/configuration handling and Toolkit dispatch; they do not run a hardware bandwidth acceptance test.

```bash
cd toolkit
python -m unittest tests.test_precheck tests.test_precheck_toolkit -v
```
