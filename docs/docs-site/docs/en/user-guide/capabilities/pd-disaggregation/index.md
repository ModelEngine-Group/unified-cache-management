# PD Disaggregation

PD Disaggregation runs Prefill and Decode on separate serving instances. Prefill computes the prompt's KV cache; Decode uses it to generate tokens. The deployment must route requests to the correct instances and deliver compatible KV to Decode. UCM provides external prefix reuse and, in the shared-store path, also supplies the store used for KV handoff. See [PD integration](../../../developer-guide/pd-integration.md) for responsibilities and request ordering.

## Choose the handoff and deployment size

First choose one of two KV handoff paths: shared storage or a transport connector. Then choose the deployment size. The scaling guide extends the transport path after a basic P/D group works.

| Guide | How KV reaches Decode | Deployment size |
| --- | --- | --- |
| [Shared-store PD](centralized.md) | Prefill saves blocks through UCM; Decode looks up and loads them from the same store | Start with one Prefill and one Decode instance and the repository's example proxy |
| [Transport with UCM](distributed.md) | A transport connector transfers the current request's KV; UCM supplies external prefix reuse to Prefill | Establish a working P/D group using manual Ascend deployment or Helm on CUDA/Ascend |
| [Scaling PD Deployments](large-scale-ep.md) | Uses the same transport handoff as Transport with UCM | Extend a working transport deployment with replicas, workers, DP, TP or EP |

Shared-store blocks must be visible before Decode lookup. The transport path requires a working producer/consumer connection. Scaling changes instance counts and model parallelism while retaining the chosen handoff.

## Prepare and deploy

1. Establish the target model, engine and store using an [engine quickstart](../../quick_start/index.md).
2. Choose one handoff and start with one Prefill and one Decode instance. Check model, tokenizer, dtype, parallel settings and KV layout.
3. On Kubernetes, follow [Helm deployment](../../frameworks/kubernetes/deploy.md) for cluster preparation, image selection, resource configuration, and installation.
4. Send requests to the deployment's client entry point. Chart PD deployments use the kthena-router gateway. See [PD in the Helm guide](../../frameworks/kubernetes/deploy.md#pd-resources) for resources and supported protocol combinations.

## Verify before scaling

Use uncached prompts to verify P/D handoff and generated output, then repeated prefixes to verify UCM reuse. Check transfer and cache reuse separately using [external-cache verification](../../observability/verify-cache.md).

After one transport-based P/D group works, continue with [Scaling PD Deployments](large-scale-ep.md). Record each change under the same workload.
