# runtime-mode

`runtime-mode` switches the online runtime mode of UCM-enabled engines without restarting them. A UCM-enabled engine deployed with `kv_transfer_config` polls a small control file; writing a new mode to that file takes effect within the engine's poll interval (default 1s), so operators can disable or re-enable UCM, or run it in shadow mode, while the service keeps serving.

Use it when:

- UCM is suspected to hurt latency or throughput and you want to disable it without a restart.
- You want to keep measuring the cache hit rate UCM *would* achieve while KV load/dump is disabled (`lite` shadow mode).
- You want to re-enable UCM after a maintenance window or an incident, again without a restart.

The tool writes control files atomically (temporary file + rename), so an engine polling the file can never observe a half-written mode.

← Back to [UCM Toolkit](../index.md)

## Modes

| Mode | Behavior |
| --- | --- |
| `enabled` | Full behavior (default): external lookup on HBM miss, KV load on hit, KV dump after compute. |
| `lite` | Shadow: hashing and metadata lookups still run and as-if hit rates are recorded, but no KV data is loaded or dumped; the engine recomputes everything. |
| `disabled` | Bypass: external lookups return a miss immediately; no load or dump is planned; the engine keeps its original compute path. |

Mode changes only affect new scheduling decisions; loads and dumps already planned finish normally, so in-flight requests stay correct.

## Dependencies

- Python standard library only.
- The target engine must run a UCM connector version that supports runtime-mode control (the `UCMDirectConnector` family: direct / layerwise / CP / mock), and runtime control must not be disabled via `runtime_control_enabled: false`.

## Control-File Resolution

`set` and `get` resolve the control file in the same order as the engine:

1. `--control-file PATH` (explicit);
2. the `UCM_RUNTIME_CONTROL_FILE` environment variable;
3. the default per-engine path `<tempdir>/ucm_runtime_control_<engine_id>.json`.

`--engine-id` accepts the vLLM `kv_transfer_config` engine id; a trailing `_dpN` data-parallel suffix is ignored, matching the connector's convention. Engines configured with an explicit `runtime_control_file` in their UCM launch config require the same `--control-file PATH` here.

## Commands

### set

Write a mode to an engine's control file:

```bash
ucm-toolkit run runtime-mode set disabled
ucm-toolkit run runtime-mode set lite --engine-id engine-0
ucm-toolkit run runtime-mode set enabled --engine-id engine-0
ucm-toolkit run runtime-mode set disabled --control-file /var/run/ucm/control.json
```

Parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `mode` | required | `enabled`, `lite` or `disabled` (aliases such as `on`, `off`, `shadow`, `bypass` are accepted). |
| `--engine-id ID` | none | Engine id for the default control-file path; a trailing `_dpN` is ignored. |
| `--control-file PATH` | resolved | Explicit control-file path, overriding engine id and environment variable. |

The command reports the previous mode (if a readable control file existed), the new mode, and the file it wrote. Exit code 2 signals an invalid mode; 1 signals a write failure.

### get

Read the current mode from a control file:

```bash
ucm-toolkit run runtime-mode get --engine-id engine-0
```

Prints the file path and the parsed mode. Exit code 1 means the file is missing or does not contain a valid mode (an engine keeps its current mode in that case).

### list

Discover control files in the system temp directory and show their modes:

```bash
ucm-toolkit run runtime-mode list
```

```text
control files in /tmp:
  ucm_runtime_control_engine-0.json          disabled
  ucm_runtime_control_ucm.json               (unreadable)
```

A control file only exists after a mode was first set (or the engine was configured with an explicit `runtime_control_file`), so `list` can legitimately be empty on a fresh deployment.

## Typical Workflow

```bash
# Engine running with UCM enabled (engine id engine-0):

# 1. Disable UCM without restarting (bypass, full recompute):
ucm-toolkit run runtime-mode set disabled --engine-id engine-0

# 2. Later, evaluate what re-enabling would give (shadow mode, no KV IO):
ucm-toolkit run runtime-mode set lite --engine-id engine-0

# 3. Watch the as-if hit rate in metrics-view, then re-enable:
ucm-toolkit run runtime-mode set enabled --engine-id engine-0

# Check the effective mode at any time:
ucm-toolkit run runtime-mode get --engine-id engine-0
```

## FAQ

### The engine does not react to a mode change

- Confirm the engine's connector supports runtime-mode control; unsupported connectors (FAWA / hybrid linear attention) log a one-time warning at startup.
- Confirm the resolved file path matches the engine's `runtime_control_file` (or its default path) — use `get` to verify both sides see the same file.
- The engine observes changes only within its poll interval (`runtime_control_poll_interval_s`, default 1s).
- Check the `connector_runtime_mode_control_read_errors_total` metric: the file content may be invalid.

### Security on shared hosts

The default control directory is the system temp directory (inside a container this is usually the container-private `/tmp`). On multi-tenant hosts, configure engines with an explicit `runtime_control_file` under a controlled path, or disable online control with `runtime_control_enabled: false`.
