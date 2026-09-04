# Unified Cache Stack Helm Chart

uc-stack 用一个 Helm release 在 Kubernetes 中部署一个 vLLM 模型。Chart 以 Kthena 为控制面，公开支持 CUDA 与 Ascend，覆盖单机、多机和 Prefill/Decode（PD）分离，并可按模型启用 Unified Cache（UCM）Posix 缓存。

Chart 只渲染 Kthena `ModelServing`，在需要路由时额外渲染 `ModelServer` 和 `ModelRoute`。集群级 Kthena controller、Kthena router、CRD、加速器驱动与 device plugin，以及可选的 Volcano、Prometheus Operator 均由使用者预先安装。

## 核心能力

- 以 `roles[]` 描述单机、多机和 PD 拓扑；每个 role 独立配置 vLLM 参数与资源。
- CUDA 与 Ascend 共用同一 Chart 接口，平台资源、运行时环境和 Mooncake 协议由对应 profile 明确声明。
- PD 支持 Mooncake 与 NIXL；Mooncake 可与 UCM 组合，NIXL 不与 UCM 组合。
- UCM 使用原生 `unifiedcacheConfig.config`，Chart 只从 `unifiedcacheStorage` 派生 `storage_backends`。
- 存储统一支持 `dynamicPVC`、`staticPVC`、`persistentVolumeClaim`、`hostPath`、`csi`、`nfs` 六种来源。
- Chart 不创建 Pod 身份或权限资源；`modelSpec.serviceAccountName` 只引用集群中已经存在的 ServiceAccount。
- 根 `values.yaml` 不绑定任何私有镜像、节点、网络地址或存储后端。

## 安装

### 前置条件

- Kubernetes 集群已安装 Kthena controller、router 和 `ModelServing` / `ModelServer` / `ModelRoute` CRD。
- CUDA 节点已安装兼容的 NVIDIA 驱动与 device plugin；Ascend 节点已安装兼容驱动与 device plugin，镜像提供与节点驱动匹配的 CANN toolkit、HCCL 和 vLLM Ascend。
- 多机或 PD 场景已经配置所需网络、RDMA 设备和调度器。
- `servingEngineSpec.serviceMonitor.enabled=true` 时，集群已安装 Prometheus Operator 的 `ServiceMonitor` CRD；否则应将其关闭。
- 已准备与所选平台匹配的 vLLM + UCM 镜像。项目不提供默认公共镜像，安装时必须设置 `images.image` 或 `modelSpec.image`。

先在 `local-overlay.yaml` 中提供模型挂载，并把 profile 中的 `replace-with-your-rwx-storage-class` 换成真实 StorageClass，再检查 Chart 是否能渲染：

```bash
helm lint . \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml

helm template qwen . \
  --namespace inference \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml
```

确认输出后再安装：

```bash
helm install qwen . \
  --namespace inference \
  --create-namespace \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml
```

Helm 渲染只能验证模板契约。StorageClass、RWX、CSI 挂载、GPU/NPU、CANN/HCCL、RDMA 与模型服务可用性仍需在目标集群验证。

## 模型示例

### CUDA

`models/cuda/` 保留七个 profile。它们都使用同一套公共接口，差异只落在模型、拓扑、资源和 PD 配置中。

| Profile | 形态 | 说明 |
| --- | --- | --- |
| `values-qwen3-0p6b-1e1.yaml` | 单机 | 一个 `engine` role，`workerReplicas: 0` |
| `values-qwen3-0p6b-1e2.yaml` | 双机 | 一个 `engine` role，entry + worker |
| `values-qwen3-0p6b-1p1-1d1.yaml` | PD 1P1D | 一个 prefill 实例和一个 decode 实例，Mooncake + UCM |
| `values-qwen3-0p6b-2p1-2d1.yaml` | PD 2P2D | prefill/decode 各两个单机实例，Mooncake + UCM |
| `values-qwen3-0p6b-2p2-2d2.yaml` | PD 2P2D 多机实例 | prefill/decode 各两个实例，每个实例由 entry + worker 组成 |
| `values-deepseek-r1-awq-single.yaml` | 单机 | DeepSeek-R1-AWQ 单节点 TP 示例 |
| `values-deepseek-r1-awq-multi.yaml` | 双机 | DeepSeek-R1-AWQ entry + worker 示例 |

### Ascend

`models/ascend/` 同样保留七个公开 profile，资源使用 Kubernetes device plugin 暴露的 `huawei.com/Ascend910` 键：

| Profile | 形态 | 说明 |
| --- | --- | --- |
| `values-qwen3-0p6b-1e1.yaml` | 单机 | 一个 `engine` role，`workerReplicas: 0` |
| `values-qwen3-0p6b-1e2.yaml` | 双机 | 一个 `engine` role，entry + worker |
| `values-qwen3-0p6b-1p1-1d1.yaml` | PD 1P1D | 一个 prefill 实例和一个 decode 实例，Ascend Mooncake + UCM |
| `values-qwen3-0p6b-2p1-2d1.yaml` | PD 2P2D | prefill/decode 各两个单机实例，Ascend Mooncake + UCM |
| `values-qwen3-0p6b-2p2-2d2.yaml` | PD 2P2D 多机实例 | prefill/decode 各两个 entry + worker 实例 |
| `values-deepseek-v3p1-multi.yaml` | 双机 | DeepSeek-V3.1 entry + worker 示例 |
| `values-qwen3-235b-multi.yaml` | 双机 | Qwen3-235B entry + worker 示例 |

例如：

```bash
IMAGE=example.com/your-vllm-ucm:tag
ASCEND_IMAGE=example.com/your-vllm-ascend-ucm:tag

helm install qwen-single . -n inference \
  --set images.image="$IMAGE" \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml

helm install qwen-multi . -n inference \
  --set images.image="$IMAGE" \
  -f models/cuda/values-qwen3-0p6b-1e2.yaml \
  -f local-overlay.yaml

helm install qwen-pd . -n inference \
  --set images.image="$IMAGE" \
  -f models/cuda/values-qwen3-0p6b-1p1-1d1.yaml \
  -f local-overlay.yaml

helm install qwen-ascend . -n inference \
  --set images.image="$ASCEND_IMAGE" \
  -f models/ascend/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml
```

示例中的 `modelPath` 指向容器内路径。根配置不会自动挂载模型盘；请用下文的任一存储来源将模型目录挂到对应路径，或把 `modelPath` 改为镜像能够直接加载的仓库 ID。两个平台的 profile 都不携带环境私有镜像、节点或存储地址。

## 配置模型

### 配置归属

- `images.image`：全局 vLLM + UCM 镜像；为空时必须由安装参数或 `modelSpec.image` 提供。
- `servingEngineSpec.configs`：所有模型共享的环境变量。
- `servingEngineSpec.modelSpec.env`：当前模型的环境变量。
- `nodeTopologyConfig`：按 Kubernetes 节点名覆盖网络变量。
- `modelSpec.roles[].vllmArgs`：当前 role 的原生 `vllm serve` 参数，只填写 flags。
- `modelSpec.roles[].resources`：原生 Kubernetes `ResourceRequirements`；CUDA profile 使用 `nvidia.com/gpu`，Ascend profile 使用公开 device plugin 资源键 `huawei.com/Ascend910`，RDMA 与 hugepages 键按集群实际配置。

部署形态由 `roles[]` 决定：

```yaml
servingEngineSpec:
  modelSpec:
    name: qwen
    modelPath: /mnt/model/Qwen3-0.6B
    modelName: Qwen3-0.6B
    roles:
      - name: engine
        replicas: 1
        workerReplicas: 0
        resources:
          requests:
            nvidia.com/gpu: "1"
          limits:
            nvidia.com/gpu: "1"
        vllmArgs: |
          --tensor-parallel-size 1
          --max-model-len 20000
```

Chart 会补充 `vllm serve <modelPath>`、`--served-model-name`、HTTP/DP 参数以及 PD 的 `--kv-transfer-config`。这些托管参数不要写进 `vllmArgs`。完整约束见 [Developer Guide](doc/DEVELOPER_GUIDE.md)。

### 平台运行时与网络

CUDA 镜像需要匹配节点驱动和 CUDA/NCCL；Ascend 镜像需要匹配节点上的 CANN，并包含 vLLM Ascend、UCM 与 HCCL 所需用户态组件。Chart 不安装或升级这些运行时，只把 profile 声明的资源、环境变量和挂载交给 Kthena。

`autoDetectInterface: true` 时，入口脚本根据 `HOST_IP` 选择宿主机网卡，并在内部为 Gloo、NCCL、HCCL 与 vLLM 设置对应网络变量。自动结果不适合管理网/数据网分离时，可在环境 overlay 的 `nodeTopologyConfig` 中按节点显式覆盖；显式值优先于自动探测。

### PD 与 KV 传输

PD 需要 prefill/decode 两个 role，并通过 `pd.kvTransfer` 选择 Mooncake 或 NIXL：

```yaml
servingEngineSpec:
  modelSpec:
    pd:
      prefill: prefill
      decode: decode
      kvTransfer:
        connector: MooncakeConnectorV1
        routerType: mooncake
        identity:
          engineIdBase: 0
          kvPortBase: 20001
          instanceStride: 100
      mooncake:
        master:
          enabled: true
```

能力边界如下：

| Connector | `routerType` | 身份字段 | 可与 UCM 组合 |
| --- | --- | --- | --- |
| `MooncakeConnectorV1` | `mooncake` | engine + port | 是 |
| `MooncakeHybridConnector` | `mooncake` | engine + port | 是 |
| `NixlConnector` | `nixl` | engine | 否 |

自建 Mooncake master 由顶层 `mooncakeMaster.enabled=true` 和 `create=true` 控制；`create=false` 时通过 `external.rpcAddress` 使用外部 master。CUDA profile 使用 `mooncakeMaster.client.config.protocol: rdma`，Ascend Mooncake profile 使用 `protocol: ascend`。master 是独立控制面，不申请 GPU/NPU；Ascend 协议仍要求目标节点和镜像具备匹配的 CANN/HCCL 与设备通信环境。

### Unified Cache

UCM 由 `unifiedcacheConfig.enabled` 与非空 `unifiedcacheConfig.config` 共同启用。`config` 是 UCM 原生配置，Chart 保持其余字段不变，只把每个 connector 的 `storage_backends` 设置为 `unifiedcacheStorage[].mountPath` 按顺序连接的结果。

```yaml
servingEngineSpec:
  modelSpec:
    unifiedcacheConfig:
      enabled: true
      config:
        log_level: INFO
        ucm_connectors:
          - ucm_connector_name: UcmPipelineStore
            ucm_connector_config:
              store_pipeline: "Cache|Posix"
    storage:
      unifiedcacheStorage:
        - name: ucm-cache
          mountPath: /mnt/ucm
          dynamicPVC:
            storageClass: replace-with-your-rwx-storage-class
            pvcStorage: 1Ti
            pvcAccessMode: [ReadWriteMany]
```

UCM 未启用时，`unifiedcacheStorage` 不会创建或挂载缓存卷。模型权重等与 UCM 无关的挂载应放在 `extraStorage`。

示例中的 `storageClass: replace-with-your-rwx-storage-class` 是公开占位符，部署前必须替换为目标集群真实支持所需访问模式的 StorageClass。

## 通用存储

`unifiedcacheStorage` 和 `extraStorage` 使用同一 schema。每一项都必须包含唯一的 `name`、`mountPath`，并且只能选择以下一种 source：

| Source | 用途 |
| --- | --- |
| `dynamicPVC` | 通过任意 StorageClass 动态创建 PVC |
| `staticPVC` | 由 Chart 创建静态 PV/PVC；`csi.driver` 与 `csi.volumeHandle` 必填 |
| `persistentVolumeClaim` | 复用同 namespace 中已经存在的 PVC |
| `hostPath` | 直接挂载节点路径 |
| `csi` | 使用 Kubernetes inline CSI volume；`driver` 必填 |
| `nfs` | 使用原生 NFS volume |

复用已有 PVC：

```yaml
servingEngineSpec:
  modelSpec:
    storage:
      extraStorage:
        - name: models
          mountPath: /mnt/model
          persistentVolumeClaim:
            claimName: existing-model-pvc
```

使用 inline CSI：

```yaml
servingEngineSpec:
  modelSpec:
    storage:
      extraStorage:
        - name: models
          mountPath: /mnt/model
          csi:
            driver: csi.example.io
            volumeAttributes:
              share: models
```

Chart 不假设 CSI 厂商，也不会替使用者选择 StorageClass、访问模式或回收策略。

## Pod 身份与权限

Chart 不创建 ServiceAccount，也不绑定权限。工作负载默认使用 namespace 的默认 Pod 身份；需要专用身份时，先由集群管理员创建，再通过 values 引用：

```yaml
servingEngineSpec:
  modelSpec:
    serviceAccountName: existing-inference-sa
```

该字段只透传为 PodSpec 的 `serviceAccountName`，不会创建同名对象或附加权限。

## 监控与验证

没有 Prometheus Operator 时关闭 ServiceMonitor：

```yaml
servingEngineSpec:
  serviceMonitor:
    enabled: false
```

部署后检查 Kthena 对象、Pod 和 Service：

```bash
kubectl get modelserving,modelserver,modelroute -n inference
kubectl get pods,svc -n inference
kubectl port-forward -n inference svc/<service-name> 8000:80
curl http://127.0.0.1:8000/v1/models
```

卸载 release：

```bash
helm uninstall <release> -n inference
```

PV/PVC 是否保留取决于所选来源与集群回收策略；卸载前应先确认数据保留要求。

## 进一步阅读

- [快速开始](doc/GET_START.md)
- [values 配置](doc/uc-stack-kthena-values.md)
- [模型适配与环境变量](doc/DEVELOPER_GUIDE.md)
- [Kthena PD 与多机原理](doc/kthena-native-pd-multinode.md)
- [功能与架构总结](doc/SUMMARY.md)
