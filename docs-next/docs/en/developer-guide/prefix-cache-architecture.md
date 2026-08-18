# Prefix Cache Architecture

This page holds the **design principles** behind UCM's Prefix Cache — the architectural rationale currently in `user-guide/capabilities/prefix-cache/index.md`. The User Guide keeps only the lightweight overview and the store how-to; the deep "why" lives here.

!!! info "Under construction"

    This page will be filled from `user-guide/capabilities/prefix-cache/index.md`.
    The outline below describes what to write.

## Planned content

- **What Prefix Cache accelerates and why**: KVCache hit rate as the core metric; bandwidth-bound IO profile; why larger cache capacity raises the hit rate (DeepSeek/Kimi sweet-spot data).
- **Storage media and multi-level hierarchy**: HBM to DRAM to local SSD to remote; the tiered cache design philosophy.
- **Centralized vs decentralized architecture**: the two directions; why UCM adopts DeepSeek's centralized approach over Dynamo's decentralized scheme; affinity scheduling trade-offs.
- **Relationship to other capabilities**: Prefix Cache as the foundation for PD disaggregation; the requirement it imposes on sparse algorithms.
- **Pointers**: link to the store how-to pages in User Guide; do not duplicate config or performance data here.

## Do not

- Do not duplicate store usage or config (that is User Guide).
- Do not list parameters here (Reference is the single source).

## Acceptance

- A reader can explain why centralized plus multi-tier is the design and how it relates to PD and sparse attention.
- Links to User Guide stores and Reference parameters are present.

Owner: ____
