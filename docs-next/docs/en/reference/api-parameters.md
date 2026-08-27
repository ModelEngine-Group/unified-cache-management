# API & Parameters

Authoritative reference for the OpenAI-compatible API, UCM configuration keys,
environment variables, and UCM-relevant engine parameters. This page is a
placeholder — populate it per the spec below.

## What to add

**Reader goal**: look up a parameter's default, valid values, and type when
running UCM.

**Required content** (the single source of truth for all parameters, maintained
as structured data):

- **OpenAI-Compatible API**: endpoints, request/response schema, UCM extension
  fields.
- **UCM configuration keys** (`ucm_config_example.yaml`): for each key — name /
  type / default / valid values / one-line meaning / which store or engine uses
  it.
- **Environment variables**: same fields as above (`ENABLE_UCM_PATCH`,
  `PLATFORM`, `MOONCAKE_CONFIG_PATH`, `HICACHE_ENGINE`, `UCM_CONFIG_FILE`,
  `ASCEND_RT_VISIBLE_DEVICES`).
- **Engine parameters**: vLLM / SGLang / MindIE parameters related to UCM / KV
  cache integration (not the engine's full parameter set).

**Don't**:

- Don't write task tutorials (that's User Guide's job).
- Don't repeat the parameter subsets already inlined in User Guide task pages;
  this page is the authoritative table only.

**Acceptance**:

- Covers every parameter already used across task pages.
- Each entry has type / default / valid values.
- No conflicts with task pages.

**First-draft approach**: extract parameters already in use from
`model-tour/*`, `capabilities/prefix-cache/*`, and `quick_start/*` into a table,
then back-validate task-page consistency.

**Owner**: _(to be assigned)_

## Reference

- [Compatibility & Metrics](api-parameters.md) — supported models and metrics.
- [Prefix Cache stores](../user-guide/capabilities/index.md) —
  backend config semantics.
- [GLM-5.1 4-node PD](../user-guide/model-tour/index.md) — a worked
  `ucm_config_example.yaml` + `vllm serve` example.
