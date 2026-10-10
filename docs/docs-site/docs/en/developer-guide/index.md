# Developer guide

Understand how UCM connects the engine's KV Cache lifecycle to external storage, then locate the interface and implementation needed for your change. Read the architecture and request flow before choosing a development task.

## Understand the system

1. [Architecture](architecture.md): engine, Connector and Store responsibilities and the storage layers traversed by data.
2. [How caching works](capability-principles.md): block matching, reuse conditions and the cost of layered storage.
3. [Request lifecycle](request-lifecycle.md): Scheduler/Worker metadata handoff and load/save completion.
4. [PD integration](pd-integration.md): how cross-request reuse combines with the current request's P/D handoff.

## Continue by task

| Development task | Guide |
| --- | --- |
| Prepare an editable runtime | [Build from source](build_from_source.md) |
| Choose and configure a backend | [Cache Configuration](cache-configuration/index.md) |
| Add storage or a Pipeline stage | [Extend Store](extending-store.md) |
| Add runtime measurements | [Metrics development](add-metrics.md) |
| Integrate a diagnostic or test tool | [Toolkit architecture and extension](../toolkit/developer/design.md) |
| Change native bandwidth tests | [Dev sandbox](../toolkit/developer/dev-sandbox.md) |
| Submit code or documentation | [Contribute](contribute.md) |

For deployment configuration, use the [user guide](../user-guide/index.md). These pages describe current source behavior; runtime verification requires compatible engines and device environments.
