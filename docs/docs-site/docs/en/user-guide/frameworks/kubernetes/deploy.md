# Helm deployment

Use `unified-cache-chart` to deploy vLLM + UCM on Kubernetes. One Helm release manages one model; keep its model, device, storage, and topology settings in the selected model profile under `models/`. This guide starts with single-node Qwen3-0.6B and covers preparation, configuration, installation, and request verification. Multi-node and PD deployments use the same installation commands.

Helm creates `ModelServing` and supporting resources. An existing kthena controller creates engine Pods from the role configuration. PD deployments also create routing declarations and receive requests through an existing kthena-router.

## 1. Prepare the cluster {#prerequisites}

| Prerequisite | Requirement |
| --- | --- |
| Kubernetes and Helm | The Chart declares Kubernetes `>=1.19`; use Helm 3. The cluster version must also satisfy kthena and device-plugin requirements. |
| kthena | Install the controller and `ModelServing` CRD. PD also requires kthena-router and the `ModelServer` and `ModelRoute` CRDs. See the [kthena documentation](https://kthena.volcano.sh/) for installation. |
| Accelerators | Nodes need compatible GPU/NPU drivers and device plugins that expose the requested resources. Multi-node and PD deployments also need the corresponding communication network and RDMA. |
| Engine image | Release packages provide a default CUDA image using the latest stable vLLM in that release. All alternatives appear beside `images.image` in `values.yaml`. Ascend and source deployments require an explicit matching image. |
| Model and cache storage | Model weights must be available and the UCM cache directory writable. The example below uses an existing model PVC and a dynamically created cache PVC. |

The Chart does not install these cluster components. Check your current connection and resources:

```bash
kubectl config current-context
kubectl get nodes -o wide
kubectl get crd modelservings.workload.serving.volcano.sh
kubectl get storageclass
helm version
```

The base Chart enables the Volcano scheduler and `ServiceMonitor` by default. The configuration below uses the default scheduler and disables `ServiceMonitor`. If the cluster has Volcano or Prometheus Operator, you can set `schedulerName: volcano` or `serviceMonitor.enabled: true`, respectively.

Defaults include `hostNetwork: true` and `hostIPC: true`, with host port `8000` declared for each engine Pod. Provide an eligible node with that port available for every entry and worker Pod. Cluster admission policies must allow the Chart's network, IPC, and container security settings.

For PD, also check the routing CRDs before installation:

```bash
kubectl get crd \
  modelservers.networking.serving.volcano.sh \
  modelroutes.networking.serving.volcano.sh
```

## 2. Get the Chart {#get-chart}

This section reads the current documentation version’s release manifest and provides the matching Chart download and extraction commands. CUDA can use the packaged default image. For another version or Ascend, choose a complete address from the image comments in the packaged `values.yaml`.

=== "Release package"

    Download and extract the package below, then enter the Chart directory:

    <div class="ucm-install" data-chart-install data-locale="en">Loading Chart download information...</div>

=== "Source workspace"

    From the UCM repository root:

    ```bash
    cd charts/unified-cache-chart
    ```

Run subsequent commands in the Chart directory. Helm automatically reads its root `values.yaml`, which does not define model roles. Supply the model profile selected below.

## 3. Configure the model {#configuration-dependencies}

### Choose a platform and topology

For a first deployment, select a single-node profile, then edit and use that file directly:

=== "CUDA"

    ```bash
    export UCM_VALUES=models/cuda/values-qwen3-0p6b-1e1.yaml
    ```

=== "Ascend"

    ```bash
    export UCM_VALUES=models/ascend/values-qwen3-0p6b-1e1.yaml
    ```

For multi-node or PD deployment, point `UCM_VALUES` to the corresponding profile. Both platforms include these five Qwen3-0.6B profiles:

| Filename | Engine layout | Engine Pods | Request entry point |
| --- | --- | ---: | --- |
| `values-qwen3-0p6b-1e1.yaml` | 1 engine, no worker | 1 | Release Service |
| `values-qwen3-0p6b-1e2.yaml` | 1 engine with 1 worker | 2 | Release Service |
| `values-qwen3-0p6b-1p1-1d1.yaml` | 1 Prefill + 1 Decode | 2 | kthena-router |
| `values-qwen3-0p6b-2p1-2d1.yaml` | 2 Prefill + 2 Decode | 4 | kthena-router |
| `values-qwen3-0p6b-2p2-2d2.yaml` | 2 Prefill + 2 Decode, each instance with 1 worker | 8 | kthena-router |

Counts assume `modelSpec.replicas: 1` and exclude Mooncake master. CUDA also includes `values-deepseek-r1-awq-single.yaml` and `values-deepseek-r1-awq-multi.yaml`. Ascend also includes `values-deepseek-v3p1-multi.yaml` and `values-qwen3-235b-multi.yaml`. Find these in the corresponding platform directory under `models/`.

### Set the image, model, and storage

Edit the model profile selected by `UCM_VALUES` directly and merge the following fields into their corresponding locations. **Preserve the original `roles`, `env`, `unifiedcacheConfig`, and PD settings.** This is a fragment of fields to change, not a standalone replacement for the file. Edit existing keys instead of appending duplicate keys at the end.

```yaml
servingEngineSpec:
  schedulerName: ""
  serviceMonitor:
    enabled: false
  modelSpec:
    modelPath: /mnt/model/Qwen3-0.6B
    modelName: Qwen3-0.6B
    storage:
      extraStorage:
        - name: models
          mountPath: /mnt/model
          readOnly: true
          persistentVolumeClaim:
            claimName: model-weights
      unifiedcacheStorage:
        - name: model-data
          mountPath: /mnt/data
          dynamicPVC:
            storageClass: replace-with-your-rwx-storage-class
            pvcStorage: 1Ti
            pvcAccessMode: [ReadWriteMany]
```

- CUDA release deployments can inherit the default image without setting it in the model profile. For Ascend, source deployments, or another version, set `images.image` in the selected profile. The packaged `values.yaml` lists all alternatives, preferring Docker Hub and using GHCR when Docker Hub publication is disabled. These are complete addresses, so `images.registry` can stay empty. A configured `servingEngineSpec.modelSpec.image` takes precedence.
- `model-weights`: replace with an existing PVC in the release namespace. Its root must contain the `Qwen3-0.6B/` weights directory, making it available at `modelPath` after mounting. The Chart does not download weights automatically. For multi-node and PD deployments, the model volume must be mountable and readable on every engine node.
- `storageClass`: replace with a real cluster StorageClass supporting `ReadWriteMany`. `1Ti` is an example capacity; adjust it for your cache needs.
- Model and cache volumes also support `hostPath`, NFS, CSI, static PV/PVC, and existing PVC sources. See the Chart's `values.yaml` and `README.md` for their fields.

The profiles enabling UCM already contain `UcmPipelineStore` and `Cache|Posix`. The Chart generates `storage_backends` from the `unifiedcacheStorage` mount paths, so you do not need to create or mount a separate UCM configuration file.

### Adjust devices, capacity, and networking

| Configuration | What to check before deployment |
| --- | --- |
| `modelSpec.roles[].resources` | Bundled Qwen profiles request 1 accelerator, 1 `rdma/rdma_shared` resource, 64 CPU cores, and 256 GiB memory per engine Pod. Adjust `requests` and `limits` to node capacity. CUDA uses `nvidia.com/gpu`; Ascend uses `huawei.com/Ascend910`. Remove both RDMA resource entries when the single-node Posix example does not need RDMA. |
| `modelSpec.unifiedcacheConfig.config.ucm_connectors` | The bundled host cache is 64 GiB. When adjusting `cache_buffer_capacity_gb`, also account for the Pod memory budget and worker count. |
| `modelSpec.shmSize` and `roles[].vllmArgs` | Adjust shared memory, parallelism, context length, and GPU memory utilization for the model and image. The Chart generates the model path, served name, HTTP port, and KV connector arguments; do not repeat them in `vllmArgs`. |
| `servingEngineSpec.schedulerName` | For multi-node or PD deployments requiring Volcano gang scheduling, install Volcano and set this to `volcano`. An empty string uses the default scheduler. |
| `nodeTopologyConfig` or `forceInterface` | Interface detection uses the node IP by default. Specify the network explicitly when management and data networks differ. Non-empty `nodeTopologyConfig` also restricts engine scheduling to the listed nodes. |

Here, `modelSpec` is nested under `servingEngineSpec`. Helm replaces entire `roles[]`, `env[]`, and storage lists, so retain every required entry when editing. Installation loads this model profile through `-f "$UCM_VALUES"`.

For PD, also check `mooncakeMaster` resources, node labels, and client settings: CUDA uses `protocol: rdma`, while Ascend uses `protocol: ascend`. Bundled profiles create a master and reuse the engine image by default. The image must contain `mooncake_master`; alternatively, set `images.mooncakeMasterImage` to a compatible image. Include client shared segments and communication buffers in the memory budget.

## 4. Install the Helm release {#install}

Set the release name and namespace. With the model PVC example above, first ensure that `model-weights` exists in the `ucm` namespace.

```bash
export UCM_RELEASE=qwen3
export UCM_NAMESPACE=ucm

kubectl -n "$UCM_NAMESPACE" get pvc model-weights

helm lint --strict . -f "$UCM_VALUES"
helm template "$UCM_RELEASE" . \
  --namespace "$UCM_NAMESPACE" \
  -f "$UCM_VALUES" > /tmp/ucm-rendered.yaml
```

Inspect the rendered image, mounts, device resources, and scheduler. Once they match the cluster, install:

```bash
helm upgrade --install "$UCM_RELEASE" . \
  --namespace "$UCM_NAMESPACE" \
  --create-namespace \
  --reset-values \
  -f "$UCM_VALUES"

helm status "$UCM_RELEASE" --namespace "$UCM_NAMESPACE"
kubectl -n "$UCM_NAMESPACE" get modelserving
kubectl -n "$UCM_NAMESPACE" get pod,pvc,service -o wide
```

Helm reporting `deployed` means the release was submitted. Wait for kthena to create the engine Pods, PVC binding to complete, and the model to load and pass readiness probes. Read engine logs for this release:

```bash
kubectl -n "$UCM_NAMESPACE" logs \
  -l "helm-release-name=${UCM_RELEASE}" \
  -c vllm --tail=100 --prefix
```

??? info "Resources created by Helm"

    Helm creates `ModelServing`, Service, ConfigMaps, and any configured Secrets, PVC/PVs, and `ServiceMonitor`. kthena creates entry and worker Pods from `ModelServing`. PD adds `ModelServer` and `ModelRoute`; bundled PD profiles also create a Mooncake master Deployment and Service.

    [![Relationships between Helm resources, kthena, and engine Pods.](../../../../assets/images/kubernetes-helm-architecture.svg)](../../../../assets/images/kubernetes-helm-architecture.svg)

## 5. Access and verify the service {#verify}

### Single-node and multi-node

A regular deployment's Service is named `<release>-<modelSpec.name>` and uses port `80` by default. For the single-node Qwen profile, keep this port-forward running in one terminal:

```bash
kubectl -n "$UCM_NAMESPACE" port-forward \
  "svc/${UCM_RELEASE}-qwen3-0p6b-1e1" 8080:80
```

Run the complete request commands in another terminal:

```bash
export UCM_BASE_URL=http://127.0.0.1:8080
export UCM_MODEL_NAME=Qwen3-0.6B

curl --fail "$UCM_BASE_URL/health"
curl --fail "$UCM_BASE_URL/v1/models"
curl --fail-with-body "$UCM_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${UCM_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
    \"max_tokens\": 128
  }"
```

Multi-node profiles use the same access path. Replace the Service suffix with your `modelSpec.name`; the API model name is `modelSpec.modelName`.

### PD Disaggregation {#pd-resources}

After installation, check routing and Mooncake master resources:

```bash
kubectl -n "$UCM_NAMESPACE" get modelserver,modelroute
kubectl -n "$UCM_NAMESPACE" get deployment,service
```

**Send PD requests to the kthena-router gateway.** This Chart does not create the gateway Service; use the address provided by your existing kthena installation. The engine Service selects both Prefill and Decode entries for engine monitoring and cannot serve as the PD client entry point.

```bash
export UCM_BASE_URL='http://replace-with-kthena-router-host:port'
export UCM_MODEL_NAME=Qwen3-0.6B

curl --fail-with-body "$UCM_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${UCM_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
    \"max_tokens\": 128
  }"
```

`ModelServing` defines P/D roles, `ModelServer` selects the roles and declares the KV transfer protocol, and `ModelRoute` maps the API model name to `ModelServer`. All six bundled PD profiles use `MooncakeConnectorV1`. Prefill combines the transport connector with UCM through `MultiConnector`; Decode uses only the transport connector.

| `pd.kvTransfer.connector` | `routerType` | Chart constraint for UCM composition |
| --- | --- | --- |
| `MooncakeConnectorV1` | `mooncake` | UCM on Prefill only |
| `MooncakeHybridConnector` | `mooncake` | UCM on Prefill only |
| `NixlConnector` | `nixl` | UCM cannot be enabled at the same time |

These are Chart configuration constraints. When changing connectors, also verify compatibility with the engine image and router.

### Confirm that UCM is enabled

Choose an engine Pod from `kubectl get pods`. For PD, choose a Prefill Pod:

```bash
export UCM_ENGINE_POD='replace-with-engine-pod-name'
kubectl -n "$UCM_NAMESPACE" exec "$UCM_ENGINE_POD" -c vllm -- \
  cat /vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml
```

Confirm the file contains the selected connector and cache mount path. A successful chat request proves inference is available. To verify cache reuse, save a set of requests in a test release, restart the engine while retaining the cache PVC, and replay the same requests. Check hits, load logs, and outputs together. See [Verify external cache](../../observability/verify-cache.md) for the full procedure.

## 6. Update and uninstall {#update-and-uninstall}

After editing the selected model profile, repeat the rendering and `helm upgrade --install` commands in step 4. To upgrade the Chart version, get the new package from this page’s “Get the Chart” section, review and retain your cluster settings in the model profile in the new Chart directory, then run the same installation command. The command uses `--reset-values` to load the new package defaults and reapply the selected profile; keep all required custom settings in that profile. Leaving the image unset selects the new default CUDA image; an explicit `images.image` or `modelSpec.image` remains in effect.

```bash
helm uninstall "$UCM_RELEASE" --namespace "$UCM_NAMESPACE"
```

Uninstall requests deletion of PVCs and static PVs created by the Chart. Storage reclaim policies determine whether data survives. Existing PVCs referenced through `persistentVolumeClaim`, such as `model-weights`, are not created by the Chart and are not deleted with the release. Check reclaim policies or make a backup before uninstalling if you need to retain the cache.

## Common deployment problems {#troubleshooting}

| Symptom | Check first |
| --- | --- |
| `no matches for kind` | kthena CRDs are installed; `serviceMonitor.enabled` is disabled if Prometheus Operator is unavailable. |
| Pod stays `Pending` | Use `kubectl describe pod` to inspect events, then check the scheduler, accelerator/RDMA resource keys, CPU/memory, node labels, and host-port conflicts. |
| PVC stays `Pending` | StorageClass name, RWX support, and capacity. With `WaitForFirstConsumer`, also inspect Pod scheduling events. |
| `ImagePullBackOff` | Image address, architecture, registry connectivity, and pull credentials. |
| Model loading fails or the container exits | `modelPath` exists, the model volume is mounted, and vLLM logs do not report runtime, argument, or memory errors. |
| Regular requests work but PD fails | The client uses the router, `ModelRoute` has the right model name, and P/D connectors, Mooncake master, and the communication network are configured correctly. |

For other runtime issues, see [Troubleshooting](../../../reference/troubleshooting.md).
