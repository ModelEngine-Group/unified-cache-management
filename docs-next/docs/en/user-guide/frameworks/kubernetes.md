# Kubernetes

UCM is distributed as a production Helm chart, `unified-cache-chart`, that
deploys one vLLM-UC inference engine per Helm release on Kubernetes. The chart
is **kthena-only**: every workload is rendered as kthena CRDs — `ModelServing`
plus, for PD or routed deployments, `ModelServer` / `ModelRoute` — with no
native Deployment / StatefulSet / Ray objects. The deployment shape (single
node, multi node, or PD disaggregation) is decided entirely by the shape of
`servingEngineSpec.modelSpec.roles[]`; install one release per model, or run
multiple releases for multiple models.

## Underlying components

The chart installs only the model-engine workload. The cluster-side components
below are **not** installed by the chart and must be provisioned before the
first release:

| Component | Why it is required | Reference | Check |
| --- | --- | --- | --- |
| Kubernetes ≥ 1.19 with Helm ≥ 3 | Runtime | — | `helm version` |
| kthena control plane | The controller-manager, kthena-router, and the `ModelServing` / `ModelServer` / `ModelRoute` CRDs (the chart renders CRs — something must reconcile them) | [kthena docs](https://kthena.volcano.sh/) | `kubectl get crd \| grep serving.volcano.sh` |
| Volcano scheduler | Gang scheduling and whole-group recovery for multi-node and PD runs; the root `values.yaml` defaults to `schedulerName: "volcano"` | [volcano.sh](https://volcano.sh/) | `kubectl get pod -n volcano-system` |
| Huawei CSI | PVC storage for model weights | [Huawei CSI docs](https://huawei.github.io/css-docs/docs/) | `kubectl get pod -A \| grep csi` |

!!! note "Provisioning kthena + Volcano"

    Neither kthena nor Volcano is installed by the chart and both must be
    deployed on the cluster beforehand. For a Helm-based installation on Ascend
    clusters, the mind-cluster community ships a scheduling installation guide:
    [mind-cluster: scheduling installation](https://gitcode.com/Ascend/mind-cluster/blob/master/docs/zh/scheduling/03_installation_guide/02_installation/00_helm_installation.md)

Additional cluster requirements:

- **Ascend Device Plugin** on NPU nodes when deploying Ascend models.
- **Node time synchronization** (chrony / ntpd) — distributed inference is
  sensitive to clock skew.
- **Engine image availability**: the vLLM-UC engine image
  (`images.image`, e.g. `registry.dev.huawei.com/flash_stor/ucm-vllm-cuda:v25.5.0`)
  must be pullable by the cluster — a private registry plus `imagePullSecret`
  is recommended.
- **Model weights** pre-placed on the nodes under `/mnt/model` by default
  (mounted at the same path in the container and passed to `vllm serve` via
  `modelSpec.modelPath`).

!!! tip "Networking"

    Multi-node and PD deployments need `hostNetwork: true` (and `hostIPC: true`
    for PD) so the engine can pick the right fabrics (`HCCL_*` on Ascend,
    `NCCL_*` on NVIDIA). The chart auto-detects the host NIC
    (`autoDetectInterface: true`); override variables per node with
    `nodeTopologyConfig` when the topology differs from the defaults.

## Install a model

1. **Get the chart** — download the `unified-cache-chart` archive from the
   [UCM releases](https://github.com/ModelEngine-Group/unified-cache-management/releases)
   matching your version, or build it from the chart sources with `helm package .`.

2. **Pick a model template** — the chart ships ready-made templates under
   `models/<chip>/`, one per platform and deployment shape:

   ```text
   models/
   ├── cuda/                    # NVIDIA GPU
   │   ├── values-qwen3-0p6b-1e1.yaml      # single node
   │   ├── values-qwen3-0p6b-1e2.yaml      # multi node
   │   ├── values-qwen3-0p6b-1p1-1d1.yaml  # PD 1P1D
   │   └── ...
   └── ascend/                  # Huawei Ascend NPU (huawei.com/Ascend910)
       └── ...
   ```

3. **Install** (from the chart directory):

   ```bash
   helm install qwen3-0p6b-1e1 -n <namespace> --create-namespace . \
     -f values.yaml \
     -f models/cuda/values-qwen3-0p6b-1e1.yaml
   ```

   The deployment shape is derived from `roles[]`:

   | Shape | `roles[]` | Rendered |
   | --- | --- | --- |
   | Single node | one role, `workerReplicas: 0` | `ModelServing` + Service (direct) |
   | Multi node | one role, `workerReplicas: ≥ 1` | `ModelServing` (entry + workers across nodes) + Service |
   | PD (xPyD) | two roles (`prefill` / `decode`) + `modelSpec.pd` | `ModelServing` + `ModelServer` + `ModelRoute`; traffic goes through the kthena router |

4. **PD-specific options** (PD deployments only):

   - KV transfer connector: `pd.kvTransfer.connector` is `MooncakeConnectorV1`
     or `MooncakeHybridConnector` (routerType `mooncake`), or `NixlConnector`
     (routerType `nixl`). NIXL cannot be combined with UCM.
   - UCM is enabled by default (`modelSpec.unifiedcacheConfig.enabled: true`);
     when enabled, the prefill role auto-stacks `UCMConnector` on top of the
     Mooncake producer while decode stays pure Mooncake.
   - Mooncake master: set `mooncakeMaster.enabled: true, create: true` to create
     one with the release, or reuse an external master via
     `mooncakeMaster.external.rpcAddress`.

## Verify the deployment

```bash
kubectl -n <namespace> get modelserving,modelserver,modelroute
kubectl -n <namespace> get pod -o wide

# single / multi node: forward the Service
# PD: forward the kthena-router gateway instead
kubectl -n <namespace> port-forward svc/<service-name> 8000:8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

Send a chat request — `"model"` must be the value of `modelSpec.modelName`
(it falls back to `modelPath` when left empty):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

When UCM is enabled, check the generated runtime config inside the engine pod:

```bash
kubectl -n <namespace> exec <pod-name> -- \
  cat /vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml
```

## Uninstall

```bash
helm uninstall <release> -n <namespace>
```

PVCs are kept or deleted depending on the StorageClass `reclaimPolicy`.

## References

The chart archive ships further context in `doc/`:

- `GET_START.md` — step-by-step deployment walkthrough and FAQ
- `uc-stack-kthena-values.md` — complete values schema reference
- `kthena-native-pd-multinode.md` — kthena CRD semantics for PD and multi-node deployments
- `SUMMARY.md` — architecture and feature overview