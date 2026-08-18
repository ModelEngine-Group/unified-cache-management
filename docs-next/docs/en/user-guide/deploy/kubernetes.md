# Deploy on Kubernetes (Helm + Kthena)

Deploy UCM on Kubernetes using the Helm chart, coordinated with Kthena. This
page is a placeholder — populate it per the spec below.

## What to add

**Reader goal**: deploy UCM on Kubernetes with Helm + Kthena.

**Required content**:

- Helm values reference (engine, device, architecture, cache backend, storage
  PVC).
- Kthena integration for heterogeneous clusters.
- Production rollout (rolling updates, scaling).
- PV / storage configuration, linking to
  [Prefix Cache stores](../capabilities/prefix-cache/index.md) for selection.

**Don't**:

- PyMotor has no trace in the repository — it is removed from the nav (the
  review line 267 TODO is not left dangling). Don't fabricate non-existent
  components.

**Acceptance**:

- Helm values can be filled in by对照.
- Storage configuration links to the corresponding store page.

**Owner**: _(to be assigned)_

## Reference

- [GLM-5.1 4-node PD](../model-tour/glm/glm-5-1.md) — a worked PD example.
- [PD Disaggregation](../capabilities/pd-disaggregation/index.md) — topology
  overview.
- [Compatibility](../../reference/compatibility.md) — supported platforms.
