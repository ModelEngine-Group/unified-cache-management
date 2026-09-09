# 快速一键部署 Unified Cache

本文从一个单机 profile 开始，说明如何在 CUDA 或 Ascend 集群准备镜像与存储，再部署多机或 PD 服务。uc-stack 不提供私有环境默认值：镜像、模型位置、StorageClass、网络和 Pod 身份都由使用者显式配置。

## 1. Helm 打包

直接从源码目录安装时可以跳过打包。需要分发 Chart 时执行：

```bash
helm lint . \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml

helm package . --destination ./dist
```

`helm lint` 和 `helm package` 只检查 Chart 的静态结构。实际的 GPU/NPU、CANN/HCCL、CSI、RDMA 和服务可用性需要目标集群验证。

## 2. 安装环境准备

### 2.1 基础组件要求

目标集群至少需要：

- Kubernetes 与 Helm 3；
- Kthena controller、router 与 `ModelServing` / `ModelServer` / `ModelRoute` CRD；
- CUDA 节点上的 NVIDIA 驱动与 device plugin，或 Ascend 节点上的兼容驱动与 device plugin；Ascend 镜像还需提供与节点驱动匹配的 CANN toolkit、HCCL 和 vLLM Ascend；
- 多机或 PD 场景所需的网络与 RDMA 设备；
- 使用 `schedulerName: volcano` 时已经安装 Volcano；
- 开启 ServiceMonitor 时已经安装 Prometheus Operator。

```bash
kubectl get crd modelservings.serving.volcano.sh
kubectl get crd modelservers.serving.volcano.sh
kubectl get crd modelroutes.serving.volcano.sh
kubectl get nodes -o wide
```

如果集群没有 Volcano，可在 overlay 中覆盖：

```yaml
servingEngineSpec:
  schedulerName: ""
```

如果集群没有 `ServiceMonitor` CRD，应关闭对应资源：

```yaml
servingEngineSpec:
  serviceMonitor:
    enabled: false
```

### 2.2 镜像准备

根 `values.yaml` 的 `images.image` 默认为空，Chart 不提供公共运行时镜像。安装时必须通过以下任一方式提供：

```yaml
images:
  image: example.com/your-vllm-ucm:tag
```

或只覆盖当前模型：

```yaml
servingEngineSpec:
  modelSpec:
    image: example.com/your-vllm-ucm:tag
```

镜像应与所选平台匹配，并包含当前 profile 所需的 vLLM、UCM connector 与传输依赖。CUDA 镜像依赖匹配的 CUDA/NCCL；Ascend 镜像提供 CANN toolkit/HCCL，并与节点驱动兼容。PD 使用自建 Mooncake master 时，`images.mooncakeMasterImage` 为空会复用 `images.image`；也可以显式提供单独镜像。

### 2.3 存储准备

Chart 不绑定任何 CSI 实现。根据集群能力选择以下一种方式提供模型或缓存存储：

- `dynamicPVC`：通过任意 StorageClass 动态创建 PVC；
- `staticPVC`：由 Chart 创建静态 PV 与 PVC，CSI 中必须提供 `driver` 和 `volumeHandle`；
- `persistentVolumeClaim`：复用当前 namespace 已存在的 PVC；
- `hostPath`：挂载节点路径；
- `csi`：使用 inline CSI volume，必须提供 `driver`；
- `nfs`：使用原生 NFS volume。

先确认集群中的可用对象：

```bash
kubectl get storageclass
kubectl get csidriver
kubectl get pvc -n inference
```

所有存储项都必须有 `name` 和 `mountPath`，且六种 source 只能选一种。模型权重通常放在 `storage.extraStorage`；UCM Posix 缓存放在 `storage.unifiedcacheStorage`。

复用已有模型 PVC：

```yaml
servingEngineSpec:
  modelSpec:
    modelPath: /mnt/model/Qwen3-0.6B
    storage:
      extraStorage:
        - name: models
          mountPath: /mnt/model
          persistentVolumeClaim:
            claimName: existing-model-pvc
```

通过 StorageClass 动态创建缓存 PVC：

```yaml
servingEngineSpec:
  modelSpec:
    storage:
      unifiedcacheStorage:
        - name: ucm-cache
          mountPath: /mnt/ucm
          dynamicPVC:
            storageClass: replace-with-your-rwx-storage-class
            pvcStorage: 1Ti
            pvcAccessMode: [ReadWriteMany]
```

`replace-with-your-rwx-storage-class` 是公开示例占位符。部署前必须替换成目标集群中实际存在、并支持所需访问模式的 StorageClass。

### 2.4 模型准备

`modelSpec.modelPath` 可以是镜像内路径、挂载卷中的目录或镜像能够访问的模型仓库 ID。CUDA 与 Ascend 的自带 profile 都使用 `/mnt/model/...`，因此安装前需要用 overlay 将模型存储挂载到 `/mnt/model`，或者修改 `modelPath`。

根 `values.yaml` 不默认挂载 NFS、节点目录或其他模型存储。

## 3. Helm 参数与配置说明

### 3.1 模型模板配置

CUDA 与 Ascend 各提供七个 profile。

CUDA：

| 文件 | 拓扑 | 主要用途 |
| --- | --- | --- |
| `values-qwen3-0p6b-1e1.yaml` | 单机，1 role / 0 worker | 最小部署与连通性检查 |
| `values-qwen3-0p6b-1e2.yaml` | 双机，1 role / 1 worker | entry + worker 多机服务 |
| `values-qwen3-0p6b-1p1-1d1.yaml` | PD 1P1D | Mooncake + UCM 基础 PD |
| `values-qwen3-0p6b-2p1-2d1.yaml` | PD 2P2D，单机实例 | P/D 横向扩展 |
| `values-qwen3-0p6b-2p2-2d2.yaml` | PD 2P2D，多机实例 | P/D 每个实例均由 entry + worker 组成 |
| `values-deepseek-r1-awq-single.yaml` | 单机 TP | DeepSeek-R1-AWQ 单节点示例 |
| `values-deepseek-r1-awq-multi.yaml` | 双机 DP × TP | DeepSeek-R1-AWQ 多节点示例 |

Ascend：

| 文件 | 拓扑 | 主要用途 |
| --- | --- | --- |
| `models/ascend/values-qwen3-0p6b-1e1.yaml` | 单机，1 role / 0 worker | 最小 Ascend 部署与连通性检查 |
| `models/ascend/values-qwen3-0p6b-1e2.yaml` | 双机，1 role / 1 worker | entry + worker 多机服务 |
| `models/ascend/values-qwen3-0p6b-1p1-1d1.yaml` | PD 1P1D | Ascend Mooncake + UCM 基础 PD |
| `models/ascend/values-qwen3-0p6b-2p1-2d1.yaml` | PD 2P2D，单机实例 | P/D 横向扩展 |
| `models/ascend/values-qwen3-0p6b-2p2-2d2.yaml` | PD 2P2D，多机实例 | P/D 每个实例均由 entry + worker 组成 |
| `models/ascend/values-deepseek-v3p1-multi.yaml` | 双机 DP × TP | DeepSeek-V3.1 多节点示例 |
| `models/ascend/values-qwen3-235b-multi.yaml` | 双机 DP × TP | Qwen3-235B 多节点示例 |

Profile 是可修改的起点，不包含集群私有的镜像和存储。先选择 `models/cuda/` 或 `models/ascend/`，再维护一个本地 overlay，只放环境相关值：

```yaml
images:
  image: example.com/your-vllm-ucm:tag

servingEngineSpec:
  serviceMonitor:
    enabled: false
  modelSpec:
    storage:
      extraStorage:
        - name: models
          mountPath: /mnt/model
          persistentVolumeClaim:
            claimName: existing-model-pvc
```

部署时按“根 values → 模型 profile → 环境 overlay”的顺序叠加：

```bash
helm install qwen . -n inference --create-namespace \
  -f values.yaml \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml
```

### 3.2 `modelSpec` 关键字段

| 字段 | 作用 |
| --- | --- |
| `name` | Kthena/Kubernetes 资源名片段 |
| `modelPath` | `vllm serve` 加载的模型路径或仓库 ID |
| `modelName` | OpenAI-compatible API 暴露的模型名 |
| `roles[]` | 描述 entry/worker、prefill/decode 和资源 |
| `roles[].vllmArgs` | 当前 role 的 vLLM flags，每行一项 |
| `replicas` | ServingGroup 数量 |
| `dataParallelMode` | `standard` 或 PD 多机实例使用的 `hybrid` |
| `pd` | PD role 配对、传输 connector、路由和身份参数 |
| `unifiedcacheConfig` | 可选 UCM Posix 配置 |
| `storage` | UCM 缓存和模型等公共挂载 |
| `serviceAccountName` | 引用集群中已经存在的 ServiceAccount |

Chart 不创建 ServiceAccount 或权限绑定。未填写 `serviceAccountName` 时，Pod 使用 namespace 的默认身份；填写后只把名称透传给 PodSpec。

### 3.3 部署形态由 `roles[]` 决定

单机：

```yaml
roles:
  - name: engine
    replicas: 1
    workerReplicas: 0
```

双机：

```yaml
roles:
  - name: engine
    replicas: 1
    workerReplicas: 1
```

PD：

```yaml
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

roles:
  - name: prefill
    replicas: 1
    workerReplicas: 0
  - name: decode
    replicas: 1
    workerReplicas: 0
```

`roles[].resources` 原样传给 Kubernetes。CUDA profile 申请 `nvidia.com/gpu`，Ascend profile 使用公开 device plugin 资源键 `huawei.com/Ascend910`；RDMA 与 hugepages 同样使用集群实际暴露的资源键。Chart 不负责安装设备插件，也不把一种资源键转换成另一种。

### 3.4 Chart 接管的 vLLM 参数

`roles[].vllmArgs` 只填写 flags。不要填写：

- 模型位置与 `--served-model-name`；
- `--host`、`--port` 和数据并行地址/排名参数；
- `--kv-transfer-config`；
- Mooncake master 的托管环境变量。

Chart 为每个 role 生成参数 ConfigMap；容器入口再结合 Kthena 注入的 entry/worker 身份，形成最终 `vllm serve` 命令。详细约束见 [Developer Guide](DEVELOPER_GUIDE.md)。

### 3.5 多机网络

`autoDetectInterface: true` 时，启动脚本根据 `HOST_IP` 选择宿主机网卡，并在内部为 Gloo、NCCL、HCCL 与 vLLM 设置相应网络变量。显式覆盖的优先级是：

`nodeTopologyConfig` > `forceInterface` > 自动探测 > 镜像默认行为。

当管理网和数据网分离，或自动探测选择错误时，按 Kubernetes 节点名显式覆盖：

```yaml
nodeTopologyConfig:
  accelerator-node-1:
    GLOO_SOCKET_IFNAME: ib0
    NCCL_SOCKET_IFNAME: ib0
    HCCL_SOCKET_IFNAME: ib0
    VLLM_NETWORK_INTERFACE: ib0
    VLLM_USE_NETIF: ib0
  accelerator-node-2:
    GLOO_SOCKET_IFNAME: ib0
    NCCL_SOCKET_IFNAME: ib0
    HCCL_SOCKET_IFNAME: ib0
    VLLM_NETWORK_INTERFACE: ib0
    VLLM_USE_NETIF: ib0
```

按实际平台只设置有意义的 NCCL 或 HCCL 变量；`nodeTopologyConfig` 中的显式值不会被自动探测覆盖。多机与 Mooncake 通常需要 `hostNetwork: true`，并要求所选端口和网卡在节点间可达。

## 4. Helm 服务部署

### 4.1 单机 CUDA

```bash
helm install qwen-single . \
  --namespace inference \
  --create-namespace \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml
```

### 4.2 多机 CUDA

```bash
helm install qwen-multi . \
  --namespace inference \
  --create-namespace \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e2.yaml \
  -f local-overlay.yaml
```

多机安装前确认 `workerReplicas` 对应的节点、GPU、共享模型目录和网络均已准备；Helm 不会替集群创建这些基础设施。

### 4.3 单机 Ascend

```bash
helm install qwen-ascend . \
  --namespace inference \
  --create-namespace \
  --set images.image=example.com/your-vllm-ascend-ucm:tag \
  -f models/ascend/values-qwen3-0p6b-1e1.yaml \
  -f local-overlay.yaml
```

安装前确认节点 allocatable 中存在 `huawei.com/Ascend910`，镜像中的 CANN 与节点驱动匹配，并且 HCCL 使用的网卡和设备通信路径可用。

### 4.4 PD 分离

```bash
helm install qwen-pd . \
  --namespace inference \
  --create-namespace \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1p1-1d1.yaml \
  -f local-overlay.yaml
```

Mooncake profile 会为 prefill/decode 生成独立传输身份，并通过 Kthena router 暴露统一入口。启用 UCM 时，producer 侧叠加 `UCMConnector`，decode 侧仍使用纯 Mooncake connector。NIXL connector 不允许同时启用 UCM。

CUDA Mooncake profile 的客户端协议是 `rdma`；Ascend Mooncake profile 使用 `ascend`。部署 Ascend PD 时，将 profile 路径切换到 `models/ascend/values-qwen3-0p6b-1p1-1d1.yaml`，并使用包含 Ascend Mooncake、CANN/HCCL 与 UCM 依赖的镜像。Mooncake master 不申请 NPU，但其 Ascend 通信路径仍必须在目标集群验证。

UCM 配置会写入 `<release>-ucm-config` ConfigMap。入口脚本把只读模板复制成可写的 `/vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml`，`UCM_CONFIG_FILE` 指向该运行时文件；启动过程中不会再根据集群存储对象修改配置。

## 5. 部署验证

### 5.1 渲染与对象状态

安装前：

```bash
helm lint . --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml -f local-overlay.yaml

helm template qwen . -n inference \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml -f local-overlay.yaml
```

安装后：

```bash
kubectl get modelserving,modelserver,modelroute -n inference
kubectl get pods,svc,pvc -n inference
kubectl describe modelserving -n inference <name>
```

### 5.2 模型 API

```bash
kubectl port-forward -n inference svc/<service-name> 8000:80
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 5.3 UCM 与 PD

启用 UCM 时检查 ConfigMap、挂载和运行时文件：

```bash
kubectl get configmap -n inference
kubectl exec -n inference <pod> -c vllm -- \
  cat /vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml
```

PD 还需要核对每个逻辑实例的 `engine_id` 和 Mooncake `kv_port` 唯一且节点间可达。成功渲染不代表数据面已经连通。

## 6. Helm 服务卸载

```bash
helm uninstall <release> --namespace inference
```

卸载前确认 PV/PVC 的回收策略。`persistentVolumeClaim` 只是引用已有 PVC，Chart 不创建也不删除该 PVC；动态或静态创建的资源遵循模板和集群回收策略。

## 7. 常见问题与建议

- **镜像为空**：通过 `images.image` 或 `modelSpec.image` 提供完整镜像引用。
- **模型路径不存在**：为 `/mnt/model` 配置 `extraStorage`，或修改 `modelPath`。
- **PVC Pending**：检查 StorageClass、访问模式、容量和 CSI controller/node 插件。
- **inline CSI 渲染失败**：确认 `csi` 为对象且包含 `driver`。
- **静态 CSI 渲染失败**：除 `driver` 外还需要 `volumeHandle`。
- **多 source 渲染失败**：每个 storage item 只能配置六种来源中的一种。
- **Pod 身份无权限**：Chart 不创建身份或权限；确认 `serviceAccountName` 引用的对象已存在并由集群管理员授予所需权限。
- **设备资源不足**：CUDA 检查 `nvidia.com/gpu`；Ascend 检查 `huawei.com/Ascend910`、device plugin 与节点 allocatable。
- **多机启动卡住**：检查 GPU/NPU、共享模型目录、`hostNetwork`、NCCL/HCCL 网卡和 RDMA 可达性。
- **Ascend 运行时失败**：检查节点驱动与镜像 CANN toolkit 是否兼容，并核对 HCCL 配置和设备文件。
- **PD KV 不通**：检查 connector/routerType 配对、身份跨度、端口冲突和 Mooncake master 地址。

## 8. 推荐阅读

- [README](../README.md)
- [values 配置](uc-stack-kthena-values.md)
- [开发者指南](DEVELOPER_GUIDE.md)
- [Kthena PD 与多机原理](kthena-native-pd-multinode.md)
- [功能与架构总结](SUMMARY.md)
