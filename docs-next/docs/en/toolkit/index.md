# UCM Toolkit

The UCM toolkit supports environment checks and capacity planning before deployment, plus metrics inspection and performance diagnosis during operation. Choose a tool to inspect the host, measure storage or device-copy bandwidth, or record serving metrics and NIC traffic.

Five command-line tools share the `ucm-toolkit` entry point, a standalone Python package that requires a separate installation. The KV Cache calculator runs directly in this documentation site and needs no CLI installation.

## Tool List

| Problem to investigate | Tool | What you get |
| --- | --- | --- |
| Check drivers, kernel and shared memory against precheck criteria before deployment | [precheck](user/precheck.md) | Environment results, remediation advice and an optional storage bandwidth report |
| KV saves or reads are slow; compare storage paths and I/O settings | [posix-aio](user/posix-aio.md) | UCM POSIX Store dump/load timings and bandwidth |
| Inspect cache hit rates, request latency and load bandwidth in a running service | [metrics-view](user/metrics-view.md) | Terminal metric snapshots or time-window queries over collected samples |
| Cross-node transfers are slow; inspect NIC load and traffic distribution | [nic-monitor](user/nic-monitor.md) | Per-NIC send/receive rates, utilization and CSV records for later review |
| KV loading is slow; measure the host-to-device copy step independently | [dev-sandbox](user/dev-sandbox.md) | Timings and bandwidth for selected copy, transfer or I/O test cases |
| Estimate KV memory needs before changing context length, batch size or parallelism | [KV Cache calculator](kv-cache-calculator.md) | KV capacity estimates from model and execution settings, and token capacity for a given KV memory budget |

Each tool page introduces its purpose and use cases before dependencies, parameters and examples. Start with an environment precheck before deployment. Once the service is running, use metrics and NIC traffic to narrow the investigation, then test storage or device copies independently.

## Installation

Install Toolkit independently or through the optional UCM extra. The following commands come from this documentation version's release manifest and pin RC pages to the corresponding RC version.

<div data-toolkit-install data-locale="en">Loading Toolkit installation commands for this version...</div>

The production PyPI installation forms also support combining a backend with Toolkit:

```bash
pip install ucm-toolkit
pip install 'uc-manager[toolkit]'
pip install 'uc-manager[cu130,toolkit]'
ucm-toolkit list
```

The `[toolkit]` extra does not select a CUDA/CANN backend. The package includes dev-sandbox sources, compiled on demand using your local SDK:

```bash
ucm-toolkit build dev-sandbox
```

For source development, `python -m pip install -e toolkit` remains supported from the repository root.

## Dependencies

The base CLI only depends on the Python standard library. Individual tools require additional system dependencies as summarized below (see each tool's documentation for details):

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
