# UCM Toolkit

`ucm-toolkit` is the unified tool entry point in the UCM repository, providing centralized access to performance testing, POSIX AIO testing, physical NIC traffic monitoring, metrics collection, and other auxiliary tools. It is a standalone Python package that is not automatically installed with the main UCM package.

## Tool List

| Tool | Alias | Type | Description | Documentation |
| --- | --- | --- | --- | --- |
| `precheck` | `pre_check` | Runnable | Runs environment pre-checks locally on the UCM deployment host before UCM starts, verifying serving-stack/uc-manager versions, accelerator drivers (CUDA compute capability or Ascend HDK), kernel version, `/dev/shm` and posix store bandwidth, outputting `PASS`/`WARN`/`FAIL` with remediation advice for failures (RFC #1208). | [Precheck](user/precheck.md) |
| `posix-aio` | `posix_aio` | Runnable | Runs `ucm/store/test/e2e/posixstore_aio_test.py` to test POSIX AIO store dump/load performance. | [POSIX AIO](user/posix-aio.md) |
| `metrics-view` | `metrics_view`, `terminal-metrics`, `terminal_metrics` | Runnable | Collects Prometheus/OpenMetrics samples to SQLite and queries aggregated metrics in the terminal. | [Metrics View](user/metrics-view.md) |
| `nic-monitor` | `nic_monitor` | Runnable | Monitors physical NIC real-time traffic, background sampling to disk, and generates phase statistics. | [NIC Monitor](user/nic-monitor.md) |
| `dev-sandbox` | `dev_sandbox` | Buildable, runnable | Measures host-to-device memory copy bandwidth and disk AIO throughput (C++ test program, requires building before use), includes `copy`, `trans`, and `aio` sub-features. | [Dev Sandbox](user/dev-sandbox.md) |
| KV Cache Calculator | - | - | Estimate KV cache memory usage for your model configuration. | [KV Cache Calculator](kv-cache-calculator.md) |

Dependencies, parameters, examples, and FAQs for each tool are documented in their respective pages.

## Installation

Editable install from the repository root is recommended:

```bash
python -m pip install -e toolkit
```

Verify the entry point is available after installation:

```bash
ucm-toolkit --help
ucm-toolkit list
```

To isolate the environment, create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e toolkit
```

## Dependencies

The base CLI only depends on the Python standard library and `setuptools`. Individual tools require additional system dependencies as summarized below (see each tool's documentation for details):

| Feature | Dependencies |
| --- | --- |
| `dev-sandbox` build | CMake 3.18+, C++17 compiler. CUDA backend requires CUDA runtime; Ascend backend requires Ascend runtime; `copy` GDR case also requires `libibverbs` headers and library. |
| `posix-aio` | Requires the UCM main package (unified-cache-management) and its native extensions installed, plus `numpy`. |
| `nic-monitor` | Linux, `bash`, `ethtool`, and root or sudo privileges to read NIC statistics. |
| `metrics-view` | Only depends on Python standard library (`sqlite3` built-in); collection requires an accessible Prometheus/OpenMetrics `/metrics` HTTP endpoint. |
| `precheck` | Core checks only depend on Python standard library; bandwidth benchmark requires the UCM main package (native extensions) and `numpy`, Linux only. |

For `dev-sandbox` backend detection priority and switching, see the [dev-sandbox Developer Guide](developer/dev-sandbox.md).

## Common Commands

The following commands are common to all top-level tools. For tool-specific `run` subcommands and parameters, see each tool's documentation.

List top-level tools:

```bash
ucm-toolkit list
ucm-toolkit list --verbose
```

Check tool environment:

```bash
ucm-toolkit doctor
ucm-toolkit doctor dev-sandbox
ucm-toolkit doctor posix-aio
ucm-toolkit doctor nic-monitor
ucm-toolkit doctor precheck
```

Build tools:

```bash
ucm-toolkit build TOOL [tool build args...]
```

Currently only `dev-sandbox` supports `build`.

Run tools:

```bash
ucm-toolkit run TOOL [tool args...]
```

Clean tool artifacts:

```bash
ucm-toolkit clean TOOL
ucm-toolkit clean TOOL --dry-run
```

Currently `clean dev-sandbox` removes the configured build directory; other tools have no cleanable artifacts by default.
