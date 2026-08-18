# PD Disaggregation Design

Design principles behind Prefill/Decode disaggregation in UCM — the rationale currently in `user-guide/capabilities/pd-disaggregation/index.md`. The User Guide keeps only the deployment how-to.

!!! info "Under construction"

    This page will be filled from `user-guide/capabilities/pd-disaggregation/index.md`.
    The outline below describes what to write.

## Planned content

- **Why disaggregate Prefill and Decode**: the consensus for large-scale serving, especially for MoE; the three core components (independent P/D deployment, KV cache storage and transmission, scheduling).
- **KV cache storage and transmission strategies**: where the cache lives and how it moves P to D; the role of UCM stores vs transfer engines (e.g., Mooncake); centralized vs distributed transfer.
- **Scheduling strategies**: dependence on the cache layer; affinity and routing considerations.
- **Expert parallelism for MoE**: why large-scale EP needs data parallelism to distribute expert weights; how PD and EP combine.
- **Relationship to Prefix Cache**: PD builds on the Prefix Cache foundation; shared store assumptions.
- **Pointers**: link to the centralized/distributed/large-scale-EP deployment pages in User Guide.

## Do not

- Do not duplicate deployment commands (User Guide).
- Do not list store config parameters (Reference).

## Acceptance

- A reader can explain the three components and the design trade-offs.
- Links to the User Guide deployment pages are present.

Owner: ____
