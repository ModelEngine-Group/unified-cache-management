# Helm 部署

使用 `unified-cache-chart` 在 Kubernetes 中部署 vLLM + UCM。一个 Helm Release 管理一个模型；模型、设备、存储和部署形态统一写在所选的 `models/` 模型配置中。下面从单节点 Qwen3-0.6B 开始，依次完成准备、配置、安装和请求验证，多节点与 PD 分离部署使用同一套命令。

Helm 创建 `ModelServing` 和配套资源，已有的 kthena controller 根据角色配置创建引擎 Pod。PD 分离部署还会创建路由声明，通过已有的 kthena-router 接收请求。

## 1. 准备集群 {#prerequisites}

| 准备项 | 要求 |
| --- | --- |
| Kubernetes 与 Helm | Chart 声明 Kubernetes `>=1.19`，使用 Helm 3；实际集群版本还需满足 kthena 和设备插件的要求。 |
| kthena | 已安装 controller 和 `ModelServing` CRD；PD 还需要 kthena-router、`ModelServer` 和 `ModelRoute` CRD。安装方法见 [kthena 文档](https://kthena.volcano.sh/)。 |
| 加速器 | 节点已安装匹配的 GPU/NPU 驱动和设备插件，能提供配置中的设备资源。多节点和 PD 还需准备对应通信网络与 RDMA。 |
| 引擎镜像 | 发布包提供本次最新稳定 vLLM 的默认 CUDA 镜像；在 `values.yaml` 的 `images.image` 旁查看全部候选。Ascend 或源码部署须显式指定匹配的镜像。 |
| 模型与缓存存储 | 模型权重已就位；UCM 缓存目录可写。下文使用已有模型 PVC 和动态创建的缓存 PVC。 |

Chart 不安装上述集群组件。先检查当前连接及资源：

```bash
kubectl config current-context
kubectl get nodes -o wide
kubectl get crd modelservings.workload.serving.volcano.sh
kubectl get storageclass
helm version
```

基础 Chart 默认启用 Volcano 调度器和 `ServiceMonitor`。下文配置使用默认调度器并关闭 `ServiceMonitor`；如果集群已经安装 Volcano 或 Prometheus Operator，可分别改回 `schedulerName: volcano` 和 `serviceMonitor.enabled: true`。

默认 `hostNetwork: true`、`hostIPC: true`，每个引擎 Pod 声明主机端口 `8000`。需为每个入口 Pod 和 worker Pod 准备端口空闲的合适节点，并确认集群准入策略允许 Chart 的网络、IPC 和容器安全配置。

PD 安装前还应确认路由 CRD 已存在：

```bash
kubectl get crd \
  modelservers.networking.serving.volcano.sh \
  modelroutes.networking.serving.volcano.sh
```

## 2. 获取 Chart {#get-chart}

本节直接读取当前文档版本的发布清单，提供对应的 Chart 下载链接和解压命令。CUDA 可使用包内默认镜像；需要其他版本或 Ascend 时，从包内 `values.yaml` 的镜像注释选择完整地址。

=== "发布包"

    下载下面的发布包并解压，然后进入 Chart 目录：

    <div class="ucm-install" data-chart-install data-locale="zh">正在加载 Chart 下载信息……</div>

=== "源码工作区"

    在 UCM 仓库根目录执行：

    ```bash
    cd charts/unified-cache-chart
    ```

后续命令均在 Chart 目录执行。Helm 自动读取根目录的 `values.yaml`，该文件没有定义模型角色，需要再提供下面选定的模型配置。

## 3. 配置模型 {#configuration-dependencies}

### 选择平台与部署形态

第一次部署先选择单节点配置，后续直接编辑并使用该文件：

=== "CUDA"

    ```bash
    export UCM_VALUES=models/cuda/values-qwen3-0p6b-1e1.yaml
    ```

=== "Ascend"

    ```bash
    export UCM_VALUES=models/ascend/values-qwen3-0p6b-1e1.yaml
    ```

需要多节点或 PD 时，将 `UCM_VALUES` 指向下表中的对应配置。两个平台都有这五份 Qwen3-0.6B 配置：

| 文件名 | 引擎布局 | 引擎 Pod 数 | 请求入口 |
| --- | --- | ---: | --- |
| `values-qwen3-0p6b-1e1.yaml` | 1 个 engine，无 worker | 1 | Release Service |
| `values-qwen3-0p6b-1e2.yaml` | 1 个 engine，带 1 个 worker | 2 | Release Service |
| `values-qwen3-0p6b-1p1-1d1.yaml` | 1 个 Prefill + 1 个 Decode | 2 | kthena-router |
| `values-qwen3-0p6b-2p1-2d1.yaml` | 2 个 Prefill + 2 个 Decode | 4 | kthena-router |
| `values-qwen3-0p6b-2p2-2d2.yaml` | 2 个 Prefill + 2 个 Decode，每个实例带 1 个 worker | 8 | kthena-router |

Pod 数按 `modelSpec.replicas: 1` 计算，不含 Mooncake master。CUDA 另有 `values-deepseek-r1-awq-single.yaml` 和 `values-deepseek-r1-awq-multi.yaml`；Ascend 另有 `values-deepseek-v3p1-multi.yaml` 和 `values-qwen3-235b-multi.yaml`，均位于对应平台的 `models/` 目录。

### 填写镜像、模型和存储

直接编辑 `UCM_VALUES` 指向的模型配置，将以下字段合并到对应位置，**保留原文件中的 `roles`、`env`、`unifiedcacheConfig`，以及 PD 配置**。下面是需要修改的字段片段，不能单独代替完整文件；已有的同名键直接修改，不要在文件末尾重复追加。

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

- 发布包的 CUDA 默认镜像无需在模型配置中重复填写。Ascend、源码部署或需要其他版本时，在所选模型配置中设置 `images.image`；候选地址位于包内 `values.yaml`，优先 Docker Hub，未发布到 Docker Hub 时使用 GHCR。地址已包含仓库域名，`images.registry` 可留空。若设置 `servingEngineSpec.modelSpec.image`，模型级镜像优先。
- `model-weights`：替换为与 Release 同命名空间的已有 PVC。该卷根目录应包含 `Qwen3-0.6B/` 权重目录，挂载后对应 `modelPath`；Chart 不会自动下载权重。多节点和 PD 分离部署需保证模型卷可被所有引擎节点挂载读取。
- `storageClass`：替换为集群真实支持 `ReadWriteMany` 的存储类。`1Ti` 是示例容量，可按缓存需求调整。
- 模型卷和缓存卷也支持 `hostPath`、NFS、CSI、静态 PV/PVC 或已有 PVC。具体字段见 Chart 中的 `values.yaml` 和 `README.md`。

启用 UCM 的模型配置中已经包含 `UcmPipelineStore` 和 `Cache|Posix`。Chart 根据 `unifiedcacheStorage` 的挂载路径生成 `storage_backends`，无需另外创建或挂载 UCM 配置文件。

### 调整设备、容量和网络

| 配置位置 | 部署前要改什么 |
| --- | --- |
| `modelSpec.roles[].resources` | 内置 Qwen 配置为每个引擎 Pod 申请 1 张卡、1 份 `rdma/rdma_shared`、64 核 CPU 和 256 GiB 内存。按节点实际容量调整 `requests` 与 `limits`。CUDA 使用 `nvidia.com/gpu`，Ascend 使用 `huawei.com/Ascend910`。单节点 Posix 示例不需要 RDMA 时，删除两处 RDMA 资源申请。 |
| `modelSpec.unifiedcacheConfig.config.ucm_connectors` | 内置主机缓存为 64 GiB，调整 `cache_buffer_capacity_gb` 时同步核对 Pod 内存预算及 worker 数量。 |
| `modelSpec.shmSize` 与 `roles[].vllmArgs` | 按模型及镜像调整共享内存、并行度、上下文长度和显存比例。模型路径、服务名、HTTP 端口和 KV connector 参数由 Chart 生成，不要重复写入 `vllmArgs`。 |
| `servingEngineSpec.schedulerName` | 多节点与 PD 若需要 Volcano gang scheduling，设为 `volcano` 并预先安装 Volcano；空字符串使用默认调度器。 |
| `nodeTopologyConfig` 或 `forceInterface` | 默认按节点 IP 探测网卡；管理网与数据网分离时显式指定。非空 `nodeTopologyConfig` 还会把引擎调度限制到列出的节点。 |

以上 `modelSpec` 均位于 `servingEngineSpec` 下。Helm 会整体替换 `roles[]`、`env[]` 和存储列表，编辑时保留该列表需要的全部条目。最终安装通过 `-f "$UCM_VALUES"` 加载这份模型配置。

PD 配置还需核对 `mooncakeMaster` 的资源、节点标签和客户端配置：CUDA 使用 `protocol: rdma`，Ascend 使用 `protocol: ascend`。内置配置创建 master，默认复用引擎镜像；镜像须包含 `mooncake_master`，也可通过 `images.mooncakeMasterImage` 指定兼容镜像。客户端共享内存及通信缓冲也要计入内存预算。

## 4. 安装 Helm Release {#install}

设置 Release 名称和命名空间；使用上面的模型 PVC 示例时，先确认 `model-weights` 已存在于 `ucm` 命名空间。

```bash
export UCM_RELEASE=qwen3
export UCM_NAMESPACE=ucm

kubectl -n "$UCM_NAMESPACE" get pvc model-weights

helm lint --strict . -f "$UCM_VALUES"
helm template "$UCM_RELEASE" . \
  --namespace "$UCM_NAMESPACE" \
  -f "$UCM_VALUES" > /tmp/ucm-rendered.yaml
```

检查渲染结果中的镜像、挂载、设备资源和调度器。确认与集群一致后安装：

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

Helm 显示 `deployed` 只表示 Release 已提交；还需等待 kthena 创建引擎 Pod、PVC 完成绑定、模型加载并通过就绪探针。查看当前 Release 的引擎日志：

```bash
kubectl -n "$UCM_NAMESPACE" logs \
  -l "helm-release-name=${UCM_RELEASE}" \
  -c vllm --tail=100 --prefix
```

??? info "Helm 创建哪些资源"

    Helm 创建 `ModelServing`、Service、ConfigMap，以及配置需要的 Secret、PVC/PV 和 `ServiceMonitor`。kthena 根据 `ModelServing` 创建入口 Pod 与 worker Pod。PD 另有 `ModelServer` 和 `ModelRoute`；内置 PD 配置还创建 Mooncake master 的 Deployment 和 Service。

    [![Helm 资源、kthena 与引擎 Pod 的关系。](../../../../assets/images/kubernetes-helm-architecture.svg)](../../../../assets/images/kubernetes-helm-architecture.svg)

## 5. 访问并验证服务 {#verify}

### 单节点和多节点

普通部署的 Service 名为 `<release>-<modelSpec.name>`，默认 Service 端口为 `80`。以单节点 Qwen 配置为例，在一个终端保持端口转发运行：

```bash
kubectl -n "$UCM_NAMESPACE" port-forward \
  "svc/${UCM_RELEASE}-qwen3-0p6b-1e1" 8080:80
```

在另一个终端执行完整的请求命令：

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

多节点配置使用相同访问方式，将 Service 名后缀换成自己的 `modelSpec.name`；API 请求中的模型名对应 `modelSpec.modelName`。

### PD 分离部署 {#pd-resources}

安装后检查路由和 Mooncake master：

```bash
kubectl -n "$UCM_NAMESPACE" get modelserver,modelroute
kubectl -n "$UCM_NAMESPACE" get deployment,service
```

**PD 请求发往 kthena-router 网关。** 此 Chart 不创建网关 Service，使用已有 kthena 安装提供的地址。引擎 Service 同时选择 Prefill 和 Decode 入口，只用于引擎监控，不能用作 PD 客户端入口。

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

`ModelServing` 定义 P/D 角色，`ModelServer` 选择角色并声明 KV 传输协议，`ModelRoute` 将 API 模型名映射到 `ModelServer`。六份内置 PD 配置均使用 `MooncakeConnectorV1`，Prefill 通过 `MultiConnector` 组合传输连接器与 UCM，Decode 只使用传输连接器。

| `pd.kvTransfer.connector` | `routerType` | Chart 的 UCM 组合约束 |
| --- | --- | --- |
| `MooncakeConnectorV1` | `mooncake` | UCM 仅用于 Prefill |
| `MooncakeHybridConnector` | `mooncake` | UCM 仅用于 Prefill |
| `NixlConnector` | `nixl` | 不允许同时启用 UCM |

这是 Chart 的配置约束；更换连接器时还需确认引擎镜像与路由器的兼容性。

### 确认 UCM 已启用

从 `kubectl get pods` 中选择本次部署的引擎 Pod；PD 场景选择 Prefill Pod：

```bash
export UCM_ENGINE_POD='replace-with-engine-pod-name'
kubectl -n "$UCM_NAMESPACE" exec "$UCM_ENGINE_POD" -c vllm -- \
  cat /vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml
```

确认文件包含所选 connector 和缓存挂载路径。聊天请求成功证明推理服务可用；验证缓存复用还需在测试 Release 中保存一组请求、保留缓存 PVC 重启引擎，再重放相同请求，结合命中、加载日志和输出判断。完整验证方法见[验证外部缓存](../../observability/verify-cache.md)。

## 6. 更新与卸载 {#update-and-uninstall}

修改所选模型配置后，重新执行第 4 步的渲染与 `helm upgrade --install`。升级 Chart 版本时，从本页的“获取 Chart”获取新发布包，在新 Chart 目录中检查并保留模型配置中的集群设置，再执行相同安装命令。命令使用 `--reset-values` 加载新包默认值，并重新应用所选模型配置；需要保留的设置应写在该配置中。未填写镜像时会采用新包默认 CUDA 镜像；显式设置 `images.image` 或 `modelSpec.image` 时继续使用该地址。

```bash
helm uninstall "$UCM_RELEASE" --namespace "$UCM_NAMESPACE"
```

卸载会请求删除 Chart 创建的 PVC 和静态 PV；实际数据是否保留取决于存储回收策略。通过 `persistentVolumeClaim` 引用的已有 PVC（例如 `model-weights`）不由 Chart 创建，不随 Release 删除。需要保留缓存时，先确认回收策略或完成备份。

## 常见部署问题 {#troubleshooting}

| 现象 | 先检查 |
| --- | --- |
| `no matches for kind` | kthena CRD 是否安装；没有 Prometheus Operator 时是否关闭了 `serviceMonitor.enabled`。 |
| Pod 一直 `Pending` | 用 `kubectl describe pod` 查看事件，核对调度器、设备/RDMA 资源键、CPU/内存、节点标签和主机端口占用。 |
| PVC 一直 `Pending` | StorageClass 名称、RWX 支持及容量；如果使用 `WaitForFirstConsumer`，结合 Pod 调度事件检查。 |
| `ImagePullBackOff` | 镜像地址、架构、仓库可达性和拉取凭据。 |
| 模型加载失败或容器退出 | `modelPath` 是否存在、模型卷是否挂载，以及 vLLM 日志中的运行时、参数或内存错误。 |
| 普通请求成功但 PD 失败 | 客户端是否访问 router，`ModelRoute` 的模型名是否匹配，以及 P/D connector、Mooncake master 和通信网络。 |

更多运行问题见[故障排查](../../../reference/troubleshooting.md)。
