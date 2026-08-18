# Deploy with Docker

Deploy UCM to production or evaluation environments using Docker. This page is
a placeholder — populate it per the spec below.

## What to add

**Reader goal**: deploy UCM via Docker to a production or eval environment.

**Required content**:

- Image selection (per engine/device, tags, registries).
- `docker run` parameters (device mappings, `--shm-size`, env vars).
- Data volumes (config, model weights, cache storage).
- Health checks.
- Hand-off to the [Installation](../installation.md) selector for the bootstrap
  command.

**Don't**:

- Don't repeat the install commands already in `installation.md`.
- Don't duplicate engine-specific launch details covered in the
  [Engines](../engines/vllm.md) guides.

**Acceptance**:

- One complete `docker run` that starts a serving endpoint.
- Data volume paths have placeholder prompts.

**Owner**: _(to be assigned)_

## Reference

- [Installation](../installation.md) — selector for the bootstrap command.
- [Engines](../engines/vllm.md) — engine-specific runtime config.
- [GLM-5.1 4-node PD](../model-tour/glm/glm-5-1.md) — a worked Docker PD setup.
