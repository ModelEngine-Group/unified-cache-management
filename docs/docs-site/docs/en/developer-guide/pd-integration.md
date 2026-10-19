# PD integration

PD disaggregation divides a request into Prefill and Decode. Prefill produces KV needed by Decode, so separation requires request coordination and data handoff. UCM prefix reuse can reduce repeated Prefill computation; the selected deployment determines the handoff between phases.

## Shared-store handoff

`ucm/pd/toy_proxy_server.py` provides example coordination: send the prompt to Prefill, then to Decode. Prefill saves reusable blocks through UCM, while Decode uses its own UCM Connector to look up and load them from shared storage.

The shared object is the storage namespace. Instances require compatible models, tokenizers, cache identifiers and tensor layouts, and writes must be visible before Decode lookup. Incomplete writes or layout mismatches affect external hits. The proxy's request ordering does not replace write-completion and read verification.

Start with [Shared-store PD](../user-guide/capabilities/pd-disaggregation/centralized.md). The example does not provide production-router scheduling, fault tolerance or transparent request migration.

## Transport connector combined with UCM

The built-in Kubernetes PD configuration combines UCM with a Mooncake producer on Prefill. Prefill reuses available external prefixes, computes the remainder and transfers the current request's KV through the transport connector to the Decode consumer. Decode does not use UCM for handoff in this composition.

| Actor | Responsibility |
| --- | --- |
| Router | Select P/D request endpoints and coordinate the transport protocol |
| Prefill UCM Connector | Look up, load and save external prefixes reused across requests |
| Transport producer / consumer | Identify peers and hand off the current request's KV |
| Inference engine | Execute Prefill/Decode and manage device cache and model parallelism |

A cold request with no UCM hit still executes Prefill and P/D transfer. A repeated prefix may reduce Prefill work while preserving the transfer path. Measure external reuse, P/D transfer and client latency separately.

## How configuration reaches processes

The Chart describes roles and workers with `ModelServing`, declares the routing protocol through `ModelServer` and maps the client model name through `ModelRoute`. Resource details and supported combinations belong in the [Helm deployment guide](../user-guide/frameworks/kubernetes/deploy.md#pd-resources).

At Pod startup, `files/resolve-kv-transfer-config.py` resolves logical instance identifiers into transport configuration. It supplies connector initialization identifiers, not Router scheduling or per-request registration. Distinguish initialization identity, routing and actual transfer when diagnosing P/D behavior.

## Parallel layouts and completion

Replicas represent independent serving instances; workers represent execution processes within an instance. Engine settings control DP, TP and EP. More Pods do not automatically yield a compatible KV layout. Changes to P/D parallelism require verification of supported transport-layout conversions and resource budgets.

For shared storage, check Prefill save, Decode matching and loading. For transport, check producer/consumer handoff, then separately verify Prefill's UCM reuse. Both require output and request-continuity checks. See [PD Disaggregation](../user-guide/capabilities/pd-disaggregation/index.md) for operation and [request lifecycle](request-lifecycle.md) for cache completion semantics.
