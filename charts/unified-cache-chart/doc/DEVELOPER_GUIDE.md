# Developer Guide：Unified Cache 模型适配与环境变量管理

本指南面向维护模型 profile 和 Chart 模板的开发者。配置的核心边界是：根 `values.yaml` 只定义平台中立的公共接口，`models/cuda/` 与 `models/ascend/` 描述模型、平台运行时和拓扑，集群私有的镜像、节点、网络、存储和身份由部署 overlay 提供。

## 1. 目录与配置层级总览

### 1.1 推荐目录结构

```text
uc-stack/
├── Chart.yaml
├── values.yaml
├── models/
│   ├── cuda/
│   │   ├── values-qwen3-0p6b-1e1.yaml
│   │   ├── values-qwen3-0p6b-1e2.yaml
│   │   ├── values-qwen3-0p6b-1p1-1d1.yaml
│   │   ├── values-qwen3-0p6b-2p1-2d1.yaml
│   │   ├── values-qwen3-0p6b-2p2-2d2.yaml
│   │   ├── values-deepseek-r1-awq-single.yaml
│   │   └── values-deepseek-r1-awq-multi.yaml
│   └── ascend/
│       ├── values-qwen3-0p6b-1e1.yaml
│       ├── values-qwen3-0p6b-1e2.yaml
│       ├── values-qwen3-0p6b-1p1-1d1.yaml
│       ├── values-qwen3-0p6b-2p1-2d1.yaml
│       ├── values-qwen3-0p6b-2p2-2d2.yaml
│       ├── values-deepseek-v3p1-multi.yaml
│       └── values-qwen3-235b-multi.yaml
├── templates/
└── doc/
```

CUDA 与 Ascend 各有七个 profile。两边都覆盖 Qwen3 单机、双机和三种 PD 拓扑；CUDA 另有 DeepSeek-R1-AWQ 单机/双机，Ascend 另有 DeepSeek-V3.1 与 Qwen3-235B 双机。新增模型时先选择平台，再复制最接近的拓扑，不要把环境私有值写回公共 profile。

### 1.2 配置优先级与覆盖规则

推荐按下面顺序叠加：

```bash
helm install <release> . \
  -n <namespace> \
  -f values.yaml \
  -f models/<platform>/values-<model>-<topology>.yaml \
  -f local-overlay.yaml
```

后一个 values 文件覆盖前一个。每层职责如下：

1. `values.yaml`：公共默认值和 schema 说明，不含可识别私有环境的信息。
2. `models/cuda/*.yaml` 或 `models/ascend/*.yaml`：模型、平台运行时参数、资源量和拓扑。
3. `local-overlay.yaml`：实际镜像、模型盘、StorageClass、节点名、网卡、Secret 与 ServiceAccount。

Chart 不提供默认公共运行时镜像。`images.image` 为空时，部署必须在 overlay 中提供它，或者设置 `modelSpec.image`。

## 2. 模型适配开发流程

### 2.1 基础信息

一个 profile 至少要回答这些问题：

- `modelPath`：vLLM 从哪里加载模型；
- `modelName`：API 对外暴露什么名称；
- `roles[]`：单机、多机还是 PD；
- 每个 role 需要多少 GPU/NPU、CPU、内存和共享内存；
- 每个 role 使用哪些 vLLM flags；
- 是否启用 UCM Posix；
- PD 使用 Mooncake 还是 NIXL。

最小 CUDA 单机骨架：

```yaml
servingEngineSpec:
  modelSpec:
    name: qwen
    modelPath: /mnt/model/Qwen3-0.6B
    modelName: Qwen3-0.6B
    replicas: 1
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

### 2.2 平台边界

Chart 不通过 `chipType` 推断设备或运行时，也不在 CUDA 与 Ascend 之间转换配置。

CUDA profile 直接写：

- `roles[].resources` 中的 `nvidia.com/gpu`；
- 需要时的 `roles[].runtimeClassName`；
- CUDA/NCCL 环境变量；
- 镜像实际支持的 vLLM flags。

Ascend profile 直接写：

- `roles[].resources` 中公开 device plugin 资源键 `huawei.com/Ascend910`；
- 与镜像和目标集群匹配的 CANN/HCCL 环境变量；
- Ascend Mooncake 所需的 `mooncakeMaster.client.config.protocol: ascend`；
- vLLM Ascend 实际支持的 flags。

资源键与运行时类以目标集群真实配置为准。CUDA 镜像必须匹配节点驱动与 CUDA/NCCL；Ascend 镜像必须提供与节点驱动匹配的 CANN toolkit 和 HCCL 用户态组件。Chart 只原样传递 profile，不安装或升级设备运行时。

## 3. 参数归属规范

同一变量只应有一个清晰的所有者。开发者先判断变量的作用域，再决定放置位置。

| 位置 | 作用域 | 典型内容 |
| --- | --- | --- |
| `servingEngineSpec.configs` | 所有 role | 镜像公共环境变量 |
| `modelSpec.env` | 当前模型所有 role | 模型或框架级覆盖 |
| `roles[].env` | 单个 role | prefill/decode 差异 |
| `nodeTopologyConfig` | 指定节点 | 网卡和节点地址 |
| `roles[].vllmArgs` | 单个 role | vLLM CLI flags |
| `local-overlay.yaml` | 目标集群 | 镜像、存储、节点、Secret、Pod 身份 |

### 3.1 `servingEngineSpec.configs`：全局环境变量

根配置只保留两个平台都合理的值。CUDA、CANN 或 HCCL 专属开关必须进入对应 profile，模型特有开关继续下沉到模型：

```yaml
servingEngineSpec:
  configs: {}
```

如果一个变量只为了某个平台、模型或并行方式存在，把它下沉到对应 profile。这样根 values 保持平台中立，也避免把 CUDA 环境变量注入 Ascend Pod，或把 CANN/HCCL 变量注入 CUDA Pod。

### 3.2 `modelSpec.env` 与 `roles[].env`

模型级变量适合放在 `modelSpec.env`：

```yaml
servingEngineSpec:
  modelSpec:
    env:
      - name: VLLM_LOGGING_LEVEL
        value: INFO
```

只有 prefill 或 decode 使用的变量放在 role：

```yaml
roles:
  - name: prefill
    env:
      - name: EXAMPLE_PREFILL_MODE
        value: "1"
```

不要用环境变量重复表达 `roles[]`、`pd.kvTransfer` 或 `storage` 已拥有的结构化知识。

### 3.3 `nodeTopologyConfig`：节点级环境变量

`autoDetectInterface: true` 时，启动脚本根据 `HOST_IP` 自动选择宿主机网卡，并在内部为 Gloo、NCCL、HCCL 与 vLLM 设置相应网络变量。显式节点配置用于管理网和数据网分离、自动探测结果错误或需要固定注册地址的场景：

```yaml
nodeTopologyConfig:
  accelerator-node-1:
    GLOO_SOCKET_IFNAME: ib0
    NCCL_SOCKET_IFNAME: ib0
    HCCL_SOCKET_IFNAME: ib0
    VLLM_NETWORK_INTERFACE: ib0
    VLLM_USE_NETIF: ib0
    VLLM_HOST_IP: "<accelerator-node-1-data-ip>"
  accelerator-node-2:
    GLOO_SOCKET_IFNAME: ib0
    NCCL_SOCKET_IFNAME: ib0
    HCCL_SOCKET_IFNAME: ib0
    VLLM_NETWORK_INTERFACE: ib0
    VLLM_USE_NETIF: ib0
    VLLM_HOST_IP: "<accelerator-node-2-data-ip>"
```

这些名称和地址仅为文档占位，实际值必须来自目标集群。按平台只填写 NCCL 或 HCCL 相关字段；`nodeTopologyConfig` 显式值优先于 `forceInterface`、自动探测和镜像默认值。多机使用宿主网卡时应保持 `hostNetwork: true`。

## 4. vLLM 启动配置：`roles[].vllmArgs`

### 4.1 写法

`vllmArgs` 是逐行解析的 flags-only 文本。每个 role 可以独立设置，未设置时回退到 `modelSpec.vllmArgs`：

```yaml
roles:
  - name: engine
    vllmArgs: |
      --tensor-parallel-size 4
      --max-model-len 32768
      --gpu-memory-utilization 0.9
      --trust-remote-code
      --additional-config '{"example":true}'
```

支持空行、`#` 注释、单引号和双引号。引号只负责把一行中的 JSON 或带空格内容保持为一个 argv 元素。

### 4.2 Chart 接管的参数

Chart 从结构化配置和 Kthena 运行时身份生成下列内容，profile 不应重复填写：

- `vllm serve <modelPath>`；
- `--served-model-name`；
- `--host`、`--port`、`--headless`；
- 数据并行地址、端口、起始 rank 与 hybrid LB 参数；
- PD 的 `--kv-transfer-config`；
- Mooncake master 地址和配置路径。

`--config` 也不应放进 `vllmArgs`，否则 Helm 无法在渲染阶段检查托管参数与 P/D 并行布局。

### 4.3 UCM 配置文件

`unifiedcacheConfig.config` 是 UCM 原生配置对象。Chart 只把 `storage.unifiedcacheStorage[].mountPath` 按顺序拼接为每个 connector 的 `storage_backends`，其余字段保持用户输入。

Chart 将配置写入只读 ConfigMap 模板。容器启动时把模板复制到可写的 `/vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml`，`UCM_CONFIG_FILE` 指向该文件。复制后不再读取 PV/CSI 元数据或改写其他 UCM 字段。

启用条件：

- `unifiedcacheConfig.enabled` 未设为 `false`；
- `unifiedcacheConfig.config` 为非空对象；
- `storage.unifiedcacheStorage` 至少包含一个有效存储项。

UCM 未启用时，Chart 不创建也不挂载 `unifiedcacheStorage` 对应资源。模型权重和其他公共挂载应放入 `extraStorage`。

## 5. 多机与网络：开发者需要知道的最小集合

### 5.1 `roles[]` 决定多机形态

一个 role 表示一个逻辑引擎角色。`workerReplicas: 0` 时只有 entry；大于零时，每个实例由 entry 与 worker 共同组成。

```yaml
roles:
  - name: engine
    replicas: 1
    workerReplicas: 1
```

`modelSpec.replicas` 控制 ServingGroup 数量，`roles[].replicas` 控制 role 实例数。不要用 Pod 名推导传输身份；PD resolver 使用 Kthena 注入的 group/role 信息计算稳定身份。

### 5.2 PD 与传输 connector

PD profile 使用两个 role，并显式声明传输和路由：

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
```

- Mooncake connector 需要 `kvPortBase` 与 `instanceStride`，可与 UCM 组合。
- NIXL connector 只使用 engine 身份，不接受 Mooncake 端口字段，也不与 UCM 组合。
- `routerType` 必须与 connector 匹配。
- 多副本和多机并行时，`instanceStride` 必须覆盖 DP × TP × PP × CP 的端口跨度。
- Mooncake 客户端协议由平台 profile 声明：CUDA 使用 `rdma`，Ascend 使用 `ascend`。Ascend Mooncake 依赖目标节点与镜像中匹配的 CANN/HCCL 和设备通信环境。

### 5.3 `hostNetwork` 与调度

多机、RDMA 和 Mooncake 通常需要 `hostNetwork: true`。启用后，每个 vLLM 容器使用宿主端口，调度器必须避免同一节点上的端口冲突。

Profile 只描述公共资源需求。节点标签、toleration、affinity、queue 和实际网卡属于集群 overlay，不能写成公共默认值。Ascend profile 中的 `huawei.com/Ascend910` 是公开 Kubernetes 资源键，不代表 Chart 管理 device plugin 或 CANN 生命周期。

## 6. 新增一个模型模板的标准步骤

1. 选择目标平台，并在 `models/cuda/` 或 `models/ascend/` 选择最接近的现有 profile。
2. 修改 `name`、`modelPath`、`modelName`。
3. 按模型调整 `roles[]`、GPU/NPU、CPU、内存和 `vllmArgs`；资源键必须与平台一致。
4. 若为 PD，确认 connector、routerType、P/D role 名和 identity 字段一致。
5. 若启用 UCM，提供 UCM 原生 `config` 和 `unifiedcacheStorage`；不要把模型权重盘混入缓存盘。
6. 删除复制来源中不再适用的模型特有环境变量和注释。
7. 不写真实镜像、节点名、网络地址、StorageClass、Secret 或 ServiceAccount。
8. 对新 profile 运行 `helm lint` 与 `helm template`，并检查最终 argv、资源、卷、路由和 ConfigMap。
9. 在目标集群验证模型加载、GPU/NPU、CUDA/NCCL 或 CANN/HCCL、网络、存储、PD 数据面和 API。

示例验证：

```bash
helm lint . \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/<platform>/values-<model>-<topology>.yaml

helm template test . \
  --namespace inference \
  --set images.image=example.com/your-vllm-ucm:tag \
  -f models/<platform>/values-<model>-<topology>.yaml > /tmp/rendered.yaml
```

## 7. 通用存储开发约束

`unifiedcacheStorage` 和 `extraStorage` 共用 schema。每项必须包含唯一 `name`、`mountPath`，并在下列 source 中六选一：

```text
dynamicPVC | staticPVC | persistentVolumeClaim | hostPath | csi | nfs
```

关键校验：

- `csi` 必须是对象，且包含非空 `driver`；
- `staticPVC.csi` 还必须包含非空 `volumeHandle`；
- `persistentVolumeClaim.claimName` 必填；
- `nfs.server` 与 `nfs.path` 必填；
- 同一模型的两个存储列表不能使用重复 `name`；
- 一个存储项不能同时提供多种 source。

`dynamicPVC.storageClass` 接受集群任意 StorageClass。`staticPVC.csi` 和 inline `csi` 接受任意符合 Kubernetes 结构的 CSI driver，不应在模板中加入厂商字段或隐式默认值。

公开示例统一使用以下占位值，部署前必须替换为目标集群真实支持所需访问模式的 StorageClass：

```yaml
dynamicPVC:
  storageClass: replace-with-your-rwx-storage-class
  pvcStorage: 100Gi
  pvcAccessMode: [ReadWriteMany]
```

## 8. Pod 身份与权限边界

Chart 不拥有集群权限策略，也不创建 ServiceAccount。若运行时需要访问 Kubernetes API 或其他受保护资源，集群管理员应在 Chart 外创建专用身份并授权；profile 只引用名称：

```yaml
servingEngineSpec:
  modelSpec:
    serviceAccountName: existing-inference-sa
```

不要在模型 profile 中假定该对象存在，也不要通过模板为它附加权限。

## 9. 生命周期钩子

`modelSpec.hooks` 支持 `preStart`、`postReady` 和 `preStop`。`roles[].hooks` 按键覆盖模型级配置；显式 `null` 可关闭当前 role 的某个钩子。

```yaml
servingEngineSpec:
  modelSpec:
    hooks:
      preStart: |
        export VLLM_LOGGING_LEVEL=DEBUG
      postReady: |
        echo "ready: ${POD_NAME}"
      preStop: |
        echo "stopping: ${POD_NAME}"
```

- `preStart` 在入口脚本中通过 `source` 执行，导出的变量会传给 vLLM；失败会阻止容器启动。脚本应使用 `return`，不要使用 `exit`。
- `postReady` 在健康检查成功后执行一次；失败只记录告警。
- `preStop` 是 Kubernetes 容器生命周期钩子；需要在 `terminationGracePeriodSeconds` 中为它和 vLLM 退出都留出时间。
- 外部注册与摘流不能只依赖 `preStop`，因为容器异常退出时它可能不会执行。

## 10. 完成检查

提交模型或模板改动前确认：

- CUDA 与 Ascend 共十四个现有 profile 都能 lint 和 render；
- 文档中的字段来自当前模板，不依赖历史兼容接口；
- 公共文件没有真实镜像、域名、IP、节点、namespace、StorageClass 实例名或个人路径；
- UCM 只管理 `storage_backends`，并继续生成模板和 writable runtime 文件；
- 未启用 UCM 时没有缓存 PV/PVC 或挂载；
- ServiceAccount 名称只透传；
- 本地渲染结论没有被表述为真实集群验收。
