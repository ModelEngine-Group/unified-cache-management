# PD deployment

Prefill computes prompt KV and Decode continues generation from it. Separating them requires routing the request to the correct instances and delivering compatible KV to Decode. Choose a deployment by its KV handoff path.

## Choose the handoff

| Path | How KV reaches Decode | Deployment guide |
| --- | --- | --- |
| Shared storage | Prefill saves blocks; Decode looks up and loads them from the same store | [Shared-store PD](centralized.md), starting with the repository's example proxy |
| Transport connector with UCM | A connector transfers the current request's KV; UCM supplies external prefix reuse to Prefill | [Transport composition](distributed.md), with Kubernetes routing and model deployment |

Shared-store blocks must be visible before Decode lookup. The transport path requires a working producer/consumer connection. See [PD integration](../../../developer-guide/pd-integration.md) for responsibilities and request ordering.

## Prepare and deploy

1. Establish the target model, engine and store using an [engine quickstart](../../quick_start/index.md).
2. Choose one handoff and start with one Prefill and one Decode instance. Check model, tokenizer, dtype, parallel settings and KV layout.
3. On Kubernetes, follow [Helm deployment](../../frameworks/kubernetes/deploy.md) for cluster preparation, image selection, resource configuration, and installation.
4. Send requests to the deployment's client entry point. Chart PD deployments use the kthena-router gateway. See [PD in the Helm guide](../../frameworks/kubernetes/deploy.md#pd-resources) for resources and supported protocol combinations.

## Verify before scaling

Use uncached prompts to verify P/D handoff and generated output, then repeated prefixes to verify UCM reuse. Check transfer and cache reuse separately using [external-cache verification](../../observability/verify-cache.md).

After one P/D group works, adjust replicas, workers, DP, TP or EP with the [multi-node guide](large-scale-ep.md). Record each change under the same workload.
