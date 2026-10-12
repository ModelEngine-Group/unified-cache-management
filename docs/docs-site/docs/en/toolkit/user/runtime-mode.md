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
| `--wait SECONDS` | `5` | Wait up to SECONDS for the engine to acknowledge the new mode; `0` disables verification. |

The command reports the previous mode (if a readable control file existed), the new mode, and the file it wrote.

**Effect verification.** Engines that support runtime-mode control maintain a status file next to the control file (`<control_file>.status`) reporting the last mode they observed and whether their connector actually gates on the mode. After writing, `set` waits for this acknowledgment by default, so an ineffective switch fails loudly:

- a hybrid-model connector (`UCMFAWAConnector`, `UCMHybridLinearAttention*`) does not gate on the mode → `set` fails immediately with `does NOT support runtime-mode control`;
- no status file appears within `--wait` seconds → the engine is not polling this file (not running, wrong path, `runtime_control_enabled: false`, another host, or an engine version without status reporting);
- the engine keeps reporting the old mode → the change was not observed.

On success the tool prints the acknowledging connector and pid (`verified: engine acknowledged in 1.2s (UCMDirectConnector, pid 12345)`). Exit codes: `0` written and verified; `1` written but not verified (or write failure); `2` invalid mode. Use `--wait 0` when pre-provisioning a control file before the engine starts.

### get

Read the current mode from a control file:

```bash
ucm-toolkit run runtime-mode get --engine-id engine-0
```

Prints the file path, the parsed mode, and the engine acknowledgment when a status file exists:

```text
file: /tmp/ucm_runtime_control_engine-0.json
mode: disabled
engine:  UCMDirectConnector (pid 12345), acknowledged mode: disabled
```

When the status file reports an unsupported connector, `get` prints a warning that mode changes have no effect. Exit code 1 means the control file is missing or does not contain a valid mode (an engine keeps its current mode in that case).

### list

Discover control files in the system temp directory and show their modes plus engine acknowledgments:

```bash
ucm-toolkit run runtime-mode list
```

```text
control files in /tmp:
  ucm_runtime_control_engine-0.json          disabled     [engine: UCMDirectConnector]
  ucm_runtime_control_h.json                 enabled      [engine: UCMFAWAConnector, UNSUPPORTED]
  ucm_runtime_control_ucm.json               (unreadable) [no engine ack]
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

`set` reports this at call time by default: it exits non-zero and prints the reason when the engine does not acknowledge the new mode. To diagnose further:

- `does NOT support runtime-mode control` — the engine runs a hybrid-model connector (FAWA / hybrid linear attention) that bypasses the mode gate; online toggling is not available for this deployment (a one-time warning also appears in the engine log at startup).
- `no engine acknowledged` — check that the engine is running, that the resolved file path matches the engine's `runtime_control_file` (or its default path), that `runtime_control_enabled` is not `false`, that the engine runs on the same host, and that its UCM version reports status.
- `still reports mode ...` — the engine is polling but did not observe the change within `--wait` seconds; retry with a larger `--wait` (the engine observes changes only within its poll interval, `runtime_control_poll_interval_s`, default 1s).
- Check the `connector_runtime_mode_control_read_errors_total` metric: the control-file content may be invalid.

### Security on shared hosts

The default control directory is the system temp directory (inside a container this is usually the container-private `/tmp`). On multi-tenant hosts, configure engines with an explicit `runtime_control_file` under a controlled path, or disable online control with `runtime_control_enabled: false`.
