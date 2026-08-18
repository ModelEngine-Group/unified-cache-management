# CLI Reference

Reference for the UCM command-line interface. This page is a placeholder —
populate it per the spec below.

## What to add

**Reader goal**: look up UCM CLI command syntax.

**Required content** (a command dictionary, not a tutorial) — full reference for:

- Installation & configuration (`config init`, `config validate`).
- Cache backend management (`store list`, `store inspect`, `store gc`,
  `store clear`).
- Metrics & diagnostics (`metrics`, `trace`).
- Engine integration helpers (`patch apply`, `patch status`, `doctor`).

For each command: syntax / arguments / example / exit code.

**Don't**:

- Don't write tutorials; this is a command dictionary only.

**Acceptance**:

- Covers every CLI subcommand present in the repository.
- Each command has a copy-pasteable example.

**Owner**: _(to be assigned)_

## Reference

- [Compatibility & Metrics](compatibility.md) — supported models and metrics.
- [API & Parameters](api-parameters.md) — `ucm_config_example.yaml` keys.
