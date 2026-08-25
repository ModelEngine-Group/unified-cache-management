# Unified Cache 生产环境 Helm Chart

此 Helm Chart 用于在 Kubernetes 集群中部署单模型推理引擎（每次 release 一个模型）。

> ⚠️ **本 chart 现为 kthena-only**：只渲染 kthena `ModelServing`[+`ModelServer`/`ModelRoute`]，**已移除 native（Deployment/StatefulSet/Ray）+ ucm-vllm-router + `useKthena`/`chipType`/`replicaCount` 等旧 schema**。部署形态由 `servingEngineSpec.modelSpec.roles[]` 决定（单机/双机/PD）。
> 前置：集群需先装 kthena 控制面（controller-manager + kthena-router + CRDs）+ 可选 Volcano。

## 核心特性

- 支持单模型部署（每次 Helm release 一个模型，可通过多次 release 实现多模型）
- 直接从现有 PersistentVolumes 加载模型权重
- 跨平台部署，自动配置安全上下文
- 多节点分布式推理，支持双平面网络
- **多芯片支持**：同时支持 NVIDIA GPU 和华为昇腾 NPU，自动适配镜像、资源和环境变量


## 安装（kthena-only）

前置：集群已装 kthena 控制面 + CRDs（`kubectl get crd | grep serving.volcano.sh` 应见 modelservings/modelservers/modelroutes）+ 可选 Volcano。

部署一个模型：`helm install <release> -n <ns> --create-namespace . -f values.yaml -f models/<chip>/<file>.yaml`

## 模型示例快速开始

形态由 `modelSpec.roles[]` 决定（单机=1 role workerReplicas:0；双机=workerReplicas:1）；多引擎扩缩用 `modelSpec.replicas`。下列示例均已转为 kthena schema、保留 UCM 缓存。

```bash
# Ascend 单机 / 双机 / PD
helm install qwen3-0p6b-1e1      -n uc . -f values.yaml -f models/ascend/values-qwen3-0p6b-1e1.yaml      # 单机 TP1
helm install qwen3-0p6b-1e2      -n uc . -f values.yaml -f models/ascend/values-qwen3-0p6b-1e2.yaml      # 双机 DP2×TP1
helm install qwen3-0p6b-1p1-1d1  -n uc . -f values.yaml -f models/ascend/values-qwen3-0p6b-1p1-1d1.yaml  # PD 1P1D
helm install qwen3-0p6b-2p1-2d1  -n uc . -f values.yaml -f models/ascend/values-qwen3-0p6b-2p1-2d1.yaml  # PD 2P2D，单机实例
helm install qwen3-0p6b-2p2-2d2  -n uc . -f values.yaml -f models/ascend/values-qwen3-0p6b-2p2-2d2.yaml  # PD 2P2D，双机实例
helm install deepseek-v3p1  -n uc . -f values.yaml -f models/ascend/values-deepseek-v3p1-multi.yaml # 双机 DP2×TP8
helm install qwen3-235b     -n uc . -f values.yaml -f models/ascend/values-qwen3-235b-multi.yaml

# CUDA 单机 / 双机 / PD
helm install qwen3-0p6b-1e1      -n uc . -f values.yaml -f models/cuda/values-qwen3-0p6b-1e1.yaml
helm install qwen3-0p6b-1e2      -n uc . -f values.yaml -f models/cuda/values-qwen3-0p6b-1e2.yaml
helm install qwen3-0p6b-1p1-1d1  -n uc . -f values.yaml -f models/cuda/values-qwen3-0p6b-1p1-1d1.yaml
helm install qwen3-0p6b-2p1-2d1  -n uc . -f values.yaml -f models/cuda/values-qwen3-0p6b-2p1-2d1.yaml
helm install qwen3-0p6b-2p2-2d2  -n uc . -f values.yaml -f models/cuda/values-qwen3-0p6b-2p2-2d2.yaml
helm install ds-r1-awq      -n uc . -f values.yaml -f models/cuda/values-deepseek-r1-awq-single.yaml
helm install ds-r1-awq-multi -n uc . -f values.yaml -f models/cuda/values-deepseek-r1-awq-multi.yaml
```

> PD 分离（prefill/decode + Mooncake/NIXL + 可选 UCM MultiConnector）可在模型 values 中增加 `modelSpec.pd` 以及 prefill/decode 两个 role。

本地校验渲染：`helm template <rel> . -f values.yaml -f models/<chip>/<file>.yaml`。

## 芯片类型支持

Unified Cache 支持多种加速芯片，包括 NVIDIA GPU 和华为昇腾 NPU。

### 支持的芯片类型

- **nvidia**: NVIDIA GPU（默认）
- **ascend**: 华为昇腾 NPU（910、910B、310 等）

### NVIDIA GPU 配置（默认）

```bash
# 直接选 cuda 模型模板即可（资源键在 roles[].resources 显式写）
helm install qwen-nvidia . \
  -f values.yaml \
  -f models/cuda/values-qwen3-0p6b-1e1.yaml
```

### 华为昇腾 NPU 配置

```bash
# 使用昇腾专用模型配置
helm install qwen3-0p6b-1e1 . \
  -f values.yaml \
  -f models/ascend/values-qwen3-0p6b-1e1.yaml
```

### 芯片差异如何表达

本 chart **没有 `chipType` 自动适配**（该字段已移除）。芯片差异全部**显式写在各模型 values 文件**：
- 资源键：`roles[].resources` 直接写 `nvidia.com/gpu` 或 `huawei.com/Ascend910` 等
- 运行时类：按需在 `roles[].runtimeClassName` 显式设置（留空不设）
- 芯片/网络环境变量：放 `servingEngineSpec.configs`（全局）/ `modelSpec.env`（模型级）/ 顶层 `nodeTopologyConfig`（节点级网卡）

📖 **有关昇腾 NPU 的详细配置**，请参阅 `models/ascend/values-qwen3-0p6b-1e1.yaml`（单机）和 `models/ascend/values-qwen3-0p6b-1e2.yaml`（多机）。

## 监控与日志

### Prometheus ServiceMonitor（可选）

当集群安装了 Prometheus Operator（含 `ServiceMonitor` CRD）时，引擎指标可被自动采集。
本 chart **默认 `enabled: true`，因此目标集群必须已安装 `ServiceMonitor` CRD，否则 `helm install` 会失败**；
没有 Operator 的环境请显式关闭：

```yaml
servingEngineSpec:
  serviceMonitor:
    enabled: false
```

默认抓取配置（见 `values.yaml`）：

```yaml
servingEngineSpec:
  serviceMonitor:
    enabled: true
    path: "/metrics"
    interval: "5s"        # 高频抓取：存储/开销约为 30s 的 6 倍，稳态可回调到 10~15s
    scrapeTimeout: "4s"   # 必须 <= interval
    # 把 pod 维度的发现标签（__meta_kubernetes_pod_*）固化为普通指标标签
    relabelings:
      - sourceLabels: [ __meta_kubernetes_pod_host_ip ]
        targetLabel: host_ip
      - sourceLabels: [ __meta_kubernetes_pod_node_name ]
        targetLabel: node
      # …pod / pod_ready / pod_phase / controller_kind / controller_name 同理
    # 高频抓取兜底（0=不限制；超限则该次抓取整次丢弃）
    sampleLimit: 0
```

约束与优化要点：

- **`scrapeTimeout` 必须 ≤ `interval`**，否则 Operator/Prometheus 拒绝该 ServiceMonitor。
- **元标签固化**：`__meta_kubernetes_*` 是服务发现的临时标签，抓取后即被丢弃；要让指标带上 `host_ip`/`node` 等，必须用 `relabelings`（抓取前）将其 `replace` 成普通标签。Operator CRD 字段为驼峰式 `sourceLabels`/`targetLabel`，勿用原生 Prometheus 的 `source_labels`/`target_label`。
- **降本**：vLLM 的 `vllm:*_bucket`（histogram）基数最高，用不到分位数时可用 `metricRelabelings` 整类丢弃。
- **状态类标签**：`pod_ready`/`pod_phase` 会随 pod 状态翻转产生新 series（series churn），高频抓取下放大存储，稳态建议移除，生命周期状态交由 kube-state-metrics 表达。
- **防爆**：高频抓取可设 `sampleLimit` 兜底，但需留足余量——一旦超限，该次抓取会被整次丢弃而非截断。

## 卸载部署

运行 `helm uninstall uc-test`

## 配置部署

查看 `values.yaml` 获取更多详细信息。

### 配置归属速记

建议按下面的职责拆分配置：

* `servingEngineSpec.configs`：全局环境变量，适合放标准启动脚本里的通用 `export KEY=VALUE`
* `servingEngineSpec.modelSpec.env`：模型级环境变量，只对当前模型生效
* 顶层 `nodeTopologyConfig`：节点级网络变量，按 `nodeName` 分片注入
* `servingEngineSpec.modelSpec.roles[].vllmArgs`：**每个 role 独立**的一段原生 `vllm serve` 参数（flags-only），逐行解析

`vllmArgs` 启动链路如下：

1. Chart 为每个 role 渲染一份 ConfigMap（`<release>-<role>-vllm-args`，见 `templates/configmap-vllm-args.yaml`）
2. 入口完成 `nodeTopologyConfig` 加载、网卡探测，并在 `kvcsStoreIdAutoDetect=true` 时回填 `kvcs_store_id`
3. 入口按行解析该 role 的 `vllmArgs`，执行 `vllm serve <modelPath>`（`modelPath`、`--served-model-name`、6 个网络参数、`--kv-transfer-config` 由 chart 托管，详见 [vLLM 启动参数](#vllm-启动参数per-role-vllmargs)）

**网卡自动探测（默认开启）**：Chart 默认顶层 `autoDetectInterface: true`（与 `nodeTopologyConfig` 同级），启动脚本（`node-topology-setup.sh::apply_iface_env`）按“K8s 节点 IP(HOST_IP) → 归属网卡”自动探测互联网卡，扇出到 `GLOO_SOCKET_IFNAME` / `HCCL_SOCKET_IFNAME` / `HCCL_IF_NAME` / `NCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `VLLM_NETWORK_INTERFACE` / `VLLM_USE_NETIF`，并派生 `HCCL_IF_IP` / `VLLM_HOST_IP`，且置 `VLLM_DETECT_MULTI_IP=0`（已固定 IP，无需多 IP 探测）。探测用三信号递降：到 master 出口网卡 → HOST_IP 归属网卡 → 默认路由（兜底告警）。

优先级：`nodeTopologyConfig` 显式值 > `forceInterface` > 自动探测 > 不设置（交镜像/库默认）。

* 需 `hostNetwork: true` 才能在 Pod 内看到宿主机网卡；非 hostNetwork（`POD_IP≠HOST_IP`）会跳过探测并告警。
* 只要 Chart 接管了网卡（探测 / `forceInterface` / `nodeTopologyConfig` 任一），就注入 `UC_SKIP_IFACE_AUTO_DETECT=true`，让镜像内标准多机脚本不再重复探测；`autoDetectInterface: false` 且未配 `nodeTopologyConfig` 时注入 `false`，回退到镜像脚本自带探测。
* `forceInterface: "<网卡名>"` 可在“独立 fabric / 管理网与高速网分离”等探测会选错的场景强制指定；`interfaceEnvVars` 可自定义扇出的变量集合（留空用内置超集）。
* 覆盖范围：kthena `ModelServing` 的 entry（leader）容器与 worker 容器；非 hostNetwork 的容器仍会跳过探测并告警，多机场景仍建议在选错网卡时显式 `nodeTopologyConfig`。

## 生产环境 Helm Chart 配置参考

此表格记录了生产环境 Helm Chart 的所有可用配置值。

### 目录

- [核心特性](#核心特性)
- [安装（kthena-only）](#安装kthena-only)
- [模型示例快速开始](#模型示例快速开始)
- [芯片类型支持](#芯片类型支持)
- [监控与日志](#监控与日志)
- [卸载部署](#卸载部署)
- [配置部署](#配置部署)
- [生产环境 Helm Chart 配置参考](#生产环境-helm-chart-配置参考)
  - [目录](#目录)
  - [安全上下文配置](#安全上下文配置)
  - [推理引擎配置](#推理引擎配置)
    - [模型规格字段](#模型规格字段)
    - [vLLM 启动参数（per-role vllmArgs）](#vllm-启动参数per-role-vllmargs)
    - [Unified Cache 配置](#unified-cache-配置)
  - [路由（kthena ModelRoute/ModelServer）](#路由kthena-modelroutemodelserver)
  - [缓存服务器配置](#缓存服务器配置)

### 安全上下文配置

Chart 使用组件自己的 `securityContext` 和 `containerSecurityContext` 配置安全上下文。

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `servingEngineSpec.securityContext` | map | `{}` | 推理引擎 Pod 级安全上下文 |
| `servingEngineSpec.containerSecurityContext` | map | 见 values.yaml | 推理引擎容器级安全上下文 |

**示例**：

```yaml
# 推理引擎容器安全上下文
servingEngineSpec:
  containerSecurityContext:
    privileged: true
    seccompProfile:
      type: Unconfined
    capabilities:
      add: [ "ALL" ]
```

### 推理引擎配置

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `servingEngineSpec.enableEngine` | boolean | `true` | Whether to enable the serving engine deployment |
| `servingEngineSpec.labels` | map | `{environment: "test", release: "test"}` | Customized labels for the serving engine deployment |
| `servingEngineSpec.vllmApiKey` | string/map | `null` | (Optional) API key for securing vLLM models. Can be a direct string or `{secretName, secretKey}`; `secretName` 按 K8s metadata.name（小写字母/数字/`-`/`.`，起止为字母或数字） |
| `servingEngineSpec.modelSpec` | map | `{}` | Specification for configuring a single serving engine model deployment |
| `servingEngineSpec.containerPort` | integer | `8000` | Port the vLLM server container is listening on |
| `servingEngineSpec.servicePort` | integer | `80` | Port the service will listen on |
| `servingEngineSpec.serviceMonitor.enabled` | boolean | `true` | Create a Prometheus Operator ServiceMonitor for the engine service (requires the ServiceMonitor CRD installed in cluster) |
| `servingEngineSpec.serviceMonitor.namespace` | string | `""` | Namespace to create ServiceMonitor (defaults to release namespace) |
| `servingEngineSpec.serviceMonitor.path` | string | `"/metrics"` | Metrics path for scraping |
| `servingEngineSpec.serviceMonitor.interval` | string | `"5s"` | Scrape interval (5s is high-frequency; consider 10–15s for steady state) |
| `servingEngineSpec.serviceMonitor.scrapeTimeout` | string | `"4s"` | Scrape timeout (must be <= interval) |
| `servingEngineSpec.serviceMonitor.scheme` | string | `"http"` | Scrape scheme |
| `servingEngineSpec.serviceMonitor.honorLabels` | boolean | `false` | Keep original labels from scraped targets |
| `servingEngineSpec.serviceMonitor.sampleLimit` | integer | `0` | Per-scrape limit on scraped samples (0 = unlimited; the whole scrape is dropped if exceeded) |
| `servingEngineSpec.serviceMonitor.targetLimit` | integer | `0` | Per-ServiceMonitor limit on number of scraped targets (0 = unlimited) |
| `servingEngineSpec.serviceMonitor.labelLimit` | integer | `0` | Per-metric limit on number of labels (0 = unlimited) |
| `servingEngineSpec.serviceMonitor.additionalLabels` | map | `{}` | Extra labels to attach to ServiceMonitor |
| `servingEngineSpec.serviceMonitor.relabelings` | list | `[…]` | Target relabeling rules; defaults固化 pod/host_ip/node/controller 等发现标签为普通指标标签 |
| `servingEngineSpec.serviceMonitor.metricRelabelings` | list | `[]` | Metric relabeling rules (按需丢弃高基数指标降本) |
| `servingEngineSpec.configs` | map | `{}` | 全局环境变量；适合承接镜像标准脚本中的 `export KEY=VALUE` 默认项 |
| `servingEngineSpec.tolerations` | list | `[]` | Tolerations configuration for the serving engine pods (when there are taints on nodes) |
| `servingEngineSpec.runtimeClassName` | string | `""` | （已弃用，留空）运行时类改写在 `modelSpec.roles[].runtimeClassName`；留空不设 |
| `servingEngineSpec.schedulerName` | string | `"volcano"` | SchedulerName，写入 `ModelServing.spec.schedulerName`（设为 `""` 可回落 default-scheduler） |
| `servingEngineSpec.hostIPC` | boolean | `true` | Enable host IPC namespace for serving engine pods; can be overridden by `servingEngineSpec.modelSpec.hostIPC` |
| `servingEngineSpec.securityContext` | map | `{}` | Pod-level security context configuration for the serving engine pods |
| `servingEngineSpec.containerSecurityContext` | map | `{runAsNonRoot: false}` | Container-level security context configuration for the serving engine container |
| `servingEngineSpec.probes.healthPath` | string | `"/health"` | 三个探针共用的 HTTP 健康检查路径（exec 探 `http://127.0.0.1:<containerPort><healthPath>`） |
| `servingEngineSpec.probes.startup.{enabled,initialDelaySeconds,periodSeconds,failureThreshold}` | map | `{true, 300, 20, 180}` | startup 探针；默认给模型加载留足时间（上限 ≈ 300 + 180×20） |
| `servingEngineSpec.probes.liveness.{enabled,initialDelaySeconds,periodSeconds,failureThreshold}` | map | `{true, 60, 60, 5}` | liveness 探针 |
| `servingEngineSpec.probes.readiness.{enabled,initialDelaySeconds,periodSeconds,failureThreshold}` | map | `{true, 60, 20, 3}` | readiness 探针 |
| `servingEngineSpec.imagePullPolicy` |  string | `"Always"`| Image pull policy for serving engine |
| `servingEngineSpec.modelSpec.storage.unifiedcacheStorage` | list | `[]` | Main storage list. The first item's `mountPath` is used for `HF_HOME`; item `name` 作为 Volume/PV/PVC 名片段，仅支持小写字母、数字、`-`，起止为字母或数字 |
| `servingEngineSpec.modelSpec.storage.extraStorage` | list | `[]` | Additional storage list for extra mounts; item `name` 作为 Volume/PV/PVC 名片段，仅支持小写字母、数字、`-`，起止为字母或数字 |

> 探针说明：上述 `probes.*` 经 `chart.kthenaProbe` 渲染，**只挂在 entry（leader）容器**；worker 容器以 `--headless` 拉起、不配探针。可在 `modelSpec.probes` 下按模型覆盖（优先级 `modelSpec.probes` > `servingEngineSpec.probes` > 内置默认），如大模型调大 `startup.failureThreshold`。

#### Mooncake master（可选）

Mooncake master 是顶层共享基础服务，不是 kthena `ModelServing` role。现有 PD 示例默认使用自创建 master：

```yaml
mooncakeMaster:
  enabled: true
  create: true
  resources:
    limits:   { nvidia.com/gpu: "1", cpu: "2", memory: 4Gi }
    requests: { nvidia.com/gpu: "1", cpu: "1", memory: 2Gi }
  client:
    config: |
      {
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": "80GB",
        "local_buffer_size": "4GB",
        "protocol": "rdma",
        "device_name": ""
      }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `images.mooncakeMasterImage` | string | `""` | 自创建 master 镜像；空值默认复用 `images.image` |
| `mooncakeMaster.enabled` | boolean | `false` | 启用 Mooncake master 配置入口 |
| `mooncakeMaster.create` | boolean | `false` | `true` 时渲染独立 `Deployment` + `Service`；`false` 时使用外部 master |
| `mooncakeMaster.args` | string | 原生 flags | `mooncake_master` 原生启动参数块，每个非注释行按空白拆分 |
| `mooncakeMaster.service.rpcPort` / `metricsPort` | integer | `50088` / `9003` | Kubernetes Service/containerPort 暴露端口，需与 `args` 中端口保持一致 |
| `mooncakeMaster.external.rpcAddress` | string | `""` | 外部 master 地址；`enabled=true, create=false` 时必填 |
| `mooncakeMaster.client.config` | JSON string/map | `""` | Helm 生成 `mooncake.json` 的高级配置；推荐放在具体模型 values 里；`protocol` 按平台填写（CUDA/GPU 通常 `rdma`，Ascend/NPU 用 `ascend`）；`master_server_address` 由 chart 托管，用户不要写 |
| `mooncakeMaster.client.env` | list | `[]` | 追加到使用 master 的 prefill/decode Pod；不要覆盖 `MOONCAKE_MASTER` / `MOONCAKE_CONFIG_PATH` / `MOONCAKE_GLOBAL_SEGMENT_SIZE` |
| `mooncakeMaster.resources` | map | - | 自创建 master Deployment 的容器资源；放在具体 PD 模型 values 里。master 是纯 CPU 控制面，**不要写加速卡键**（`nvidia.com/gpu` / `huawei.com/Ascend910`）；可保留 `rdma/rdma_shared` 或改用 `nodeSelector`/`affinity` 钉节点 |
| `mooncakeMaster.nodeSelector` / `tolerations` / `affinity` | map/list/map | `{}` / `[]` / `{}` | master 的调度约束。Ascend 场景必须钉到 NPU 节点（内置的 driver/hccn.conf 宿主挂载仅 NPU 节点存在） |

自创建 master 在 `client.config.protocol=ascend` 时由模板内置只读宿主挂载 `/usr/local/Ascend/driver`（含 `tools/hccn_tool`）、`/usr/local/dcmi` 与 `/etc/hccn.conf`——昇腾构建的 `mooncake_master` 链接 CANN/驱动用户态库，仅需库文件可加载，不占 NPU 卡、无需 privileged；其他平台不渲染挂载。自创建 master 启动前会先 `export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"`，再 `exec mooncake_master`。`MOONCAKE_CONFIG_PATH` 固定由 Helm 注入为 `/vllm-workspace/UnifiedCache/mooncake/mooncake.json`。是否让某次 PD 使用 master，由 `servingEngineSpec.modelSpec.pd.mooncake.master.enabled` 单独控制。自创建 master 的镜像由 `images.mooncakeMasterImage` 覆盖，`imagePullPolicy` 统一使用 `images.pullPolicy`。

#### 模型规格字段

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `servingEngineSpec.modelSpec.annotations` | map | `{}` | (Optional) Annotations to add to the deployment, e.g., {model: "opt125m"} |
| `servingEngineSpec.modelSpec.podAnnotations` | map | `{}` | (Optional) Annotations to add to the pod, e.g., {model: "opt125m"} |
| `servingEngineSpec.modelSpec.name` | string | `""` | 模型实例名，作为 K8s/kthena 资源名片段；仅支持小写字母、数字、`-`，起止为字母或数字；`<release>-<name>` 总长需 <= 63 |
| `servingEngineSpec.modelSpec.image` | string | `""` | (Optional) Full model image, e.g., "vllm/vllm-openai:latest"; if empty, use `images.image` |
| `servingEngineSpec.modelSpec.imagePullSecret` | string | `""` | (Optional) Secret 名；按 K8s metadata.name（小写字母/数字/`-`/`.`，起止为字母或数字） |
| `servingEngineSpec.modelSpec.modelPath` | string | `""` | 模型加载路径或仓库 ID，作为 `vllm serve <modelPath>`，例如 "/mnt/model/Qwen3-0.6B" 或 "facebook/opt-125m" |
| `servingEngineSpec.modelSpec.modelName` | string | `""` | 对外模型名，统一注入 `--served-model-name` 并作为 `ModelRoute.modelName` / `ModelServer.model`；留空回退 `modelPath` |
| `servingEngineSpec.modelSpec.chatTemplate` | string | `null` | (Optional) Chat template (Jinja2) specifying tokenizer configuration |
| `servingEngineSpec.modelSpec.replicas` | integer | `1` | 引擎副本数（= ServingGroup 数；各自独立 gang/恢复单元）；写入 `ModelServing.spec.replicas` |
| `servingEngineSpec.modelSpec.roles` | list | `[]` | **部署形态来源**：每个 role 一组 entry+worker。单机 = 1 role + `workerReplicas:0`；双机 = 1 role + `workerReplicas:1`；PD = prefill/decode 两 role + `modelSpec.pd` 配对。详见下方逐字段 |
| `servingEngineSpec.modelSpec.roles[].name` | string | - | role 名（参与 ConfigMap/反亲和命名）；仅支持小写字母、数字、`-`，起止为字母或数字，最长 12；PD 时由 `pd.prefill`/`pd.decode` 引用 |
| `servingEngineSpec.modelSpec.roles[].replicas` | integer | `1` | 该 role 的副本数 |
| `servingEngineSpec.modelSpec.roles[].workerReplicas` | integer | `0` | >0 时渲染 worker 容器（多机一套引擎）+ `REPLICA_COUNT` + 默认按主机名反亲和 |
| `servingEngineSpec.modelSpec.roles[].vllmArgs` | string | `""` | **per-role** 原生 `vllm serve` 参数块（flags-only），逐行解析；缺省回退 `modelSpec.vllmArgs`。详见 [vLLM 启动参数](#vllm-启动参数per-role-vllmargs) |
| `servingEngineSpec.modelSpec.roles[].resources` | map | `{}` | 原生 k8s `ResourceRequirements`（chart 原样 toYaml 透传）；直接写资源键，如 `nvidia.com/gpu` / `huawei.com/Ascend910` / `rdma/...` / `hugepages-2Mi` |
| `servingEngineSpec.modelSpec.roles[].runtimeClassName` | string | - | 该 role 的 Pod `runtimeClassName`；留空不设 |
| `servingEngineSpec.modelSpec.roles[].affinity` / `nodeSelector` / `tolerations` | map/list | - | 该 role 的调度配置（`affinity` 覆盖默认反亲和） |
| `servingEngineSpec.modelSpec.pd` | map | `{}` | PD 分离配对（设置即启用 PD）；详见下方逐字段 |
| `servingEngineSpec.modelSpec.pd.kvTransfer.connector` | string | - | **必填**。vLLM 原始 connector 类名，精确且区分大小写：`MooncakeConnectorV1` / `MooncakeHybridConnector` / `NixlConnector` |
| `servingEngineSpec.modelSpec.pd.kvTransfer.routerType` | string | - | **必填**。Kthena `ModelServer.kvConnector.type`，Mooncake 两种 connector 必须为 `mooncake`，NIXL 必须为 `nixl` |
| `servingEngineSpec.modelSpec.pd.kvTransfer.identity.engineIdBase` | integer | - | **必填**，且 `>=0`；跨 ServingGroup 的每个逻辑 P/D 实例都在此基址上取得唯一 `engine_id` |
| `servingEngineSpec.modelSpec.pd.kvTransfer.identity.kvPortBase` | integer | - | Mooncake **必填**，范围 `1..65535`；NIXL 禁止填写 |
| `servingEngineSpec.modelSpec.pd.kvTransfer.identity.instanceStride` | integer | - | Mooncake **必填**；至少为 `max(100, DP×TP×PP×CP 的保守跨度)`，NIXL 禁止填写 |
| `servingEngineSpec.modelSpec.pd.prefill` / `decode` | string | - | 分别指向充当 producer/consumer 的 role 名 |
| `servingEngineSpec.modelSpec.pd.groupKey` | string | `modelserving.volcano.sh/group-name` | PD 组标签键 |
| `servingEngineSpec.modelSpec.pd.antiAffinity` | boolean | `true` | 是否启用 prefill/decode 主机名反亲和 |
| `servingEngineSpec.modelSpec.pd.mooncake.master.enabled` | boolean | `false` | 本次 PD 是否使用顶层 `mooncakeMaster`；仅 `kvTransfer.routerType=mooncake` 可启用 |
| `servingEngineSpec.modelSpec.router` | map | `{}` | kthena 路由开关；详见 [路由](#路由kthena-modelroutemodelserver) |
| `servingEngineSpec.modelSpec.recoveryPolicy` | string | - | 写入 `ModelServing.spec.recoveryPolicy` |
| `servingEngineSpec.modelSpec.gangPolicy` | map | - | 写入 `ModelServing` 模板的 `gangPolicy`（toYaml） |
| `servingEngineSpec.modelSpec.rdma.enabled` | boolean | `false` | true 时自动给 vllm 容器追加 `IPC_LOCK` capability（与 `containerSecurityContext.capabilities.add` 自动合并去重）。RDMA 设备资源仍写在 `roles[].resources` |
| `servingEngineSpec.modelSpec.shmSize` | string | `"128Gi"` | 共享内存 `/dev/shm` 大小（chart 据 `-tp` 决定是否挂载） |
| `servingEngineSpec.modelSpec.storage.unifiedcacheStorage` | list | `[]` | Main storage list. Each item requires `name`, `mountPath`, and exactly one source; item `name` 作为 Volume/PV/PVC 名片段，仅支持小写字母、数字、`-`，起止为字母或数字 |
| `servingEngineSpec.modelSpec.storage.extraStorage` | list | `[]` | Extra storage list with the same schema as `unifiedcacheStorage`; item `name` 同样仅支持小写字母、数字、`-`，起止为字母或数字 |
| `servingEngineSpec.modelSpec.storage.*.dynamicPVC.pvcStorage` | string | `"20Gi"` | Requested size when creating PVC dynamically |
| `servingEngineSpec.modelSpec.storage.*.dynamicPVC.pvcAccessMode` | list | `["ReadWriteOnce"]` | Access mode for dynamically created PVC |
| `servingEngineSpec.modelSpec.storage.*.dynamicPVC.storageClass` | string | `""` | Storage class for dynamically created PVC |
| `servingEngineSpec.modelSpec.storage.*.dynamicPVC.pvcMatchLabels` | map | `{}` | Selector labels for dynamically created PVC |
| `servingEngineSpec.modelSpec.storage.*.dynamicPVC.pvcLabels` | map | `{}` | Labels for dynamically created PVC |
| `servingEngineSpec.modelSpec.storage.*.staticPVC.pvcStorage` | string | `"20Gi"` | Requested size when creating a static PV+PVC pair |
| `servingEngineSpec.modelSpec.storage.*.staticPVC.pvcAccessMode` | list | `["ReadWriteOnce"]` | Access mode for the static PV+PVC pair |
| `servingEngineSpec.modelSpec.storage.*.staticPVC.storageClass` | string | `""` | Storage class for the static PV+PVC pair |
| `servingEngineSpec.modelSpec.storage.*.staticPVC.reclaimPolicy` | string | `"Retain"` | Reclaim policy for the generated static PV |
| `servingEngineSpec.modelSpec.storage.*.staticPVC.mountOptions` | list | `[]` | Mount options for the generated static PV |
| `servingEngineSpec.modelSpec.storage.*.staticPVC.csi` | map | `{}` | CSI source for the generated static PV; `driver` is required |
| `servingEngineSpec.modelSpec.storage.*.nfs.server` | string | `""` | NFS server address for native Kubernetes NFS volumes |
| `servingEngineSpec.modelSpec.storage.*.nfs.path` | string | `""` | Exported NFS path for native Kubernetes NFS volumes |
| `servingEngineSpec.modelSpec.storage.*.nfs.readOnly` | boolean | `false` | Whether to mount the native NFS volume read-only |
| `servingEngineSpec.modelSpec.pvcStorage` | string | `""` | Deprecated and ignored |
| `servingEngineSpec.modelSpec.pvcAccessMode` | list | `[]` | Deprecated and ignored |
| `servingEngineSpec.modelSpec.storageClass` | string | `""` | Deprecated and ignored |
| `servingEngineSpec.modelSpec.extraVolumes` | list | `[]` | Deprecated and ignored |
| `servingEngineSpec.modelSpec.extraVolumeMounts` | list | `[]` | Deprecated and ignored |
| `servingEngineSpec.modelSpec.serviceAccountName` | string | `""` | (Optional) ServiceAccount 名；按 K8s metadata.name（小写字母/数字/`-`/`.`，起止为字母或数字） |
| `servingEngineSpec.modelSpec.hf_token` | string/map | - | (Optional) Hugging Face token configuration; 引用 Secret 时 `secretName` 按 K8s metadata.name（小写字母/数字/`-`/`.`，起止为字母或数字） |
| `servingEngineSpec.modelSpec.env` | list | - | (Optional) 模型级环境变量；仅对当前模型生效，优先级高于 `servingEngineSpec.configs` |

PD connector 首版能力矩阵：

| `kvTransfer.connector` | `routerType` | 身份字段 | 与 UCM 组合 |
|---|---|---|---|
| `MooncakeConnectorV1` | `mooncake` | engine + port | 支持 |
| `MooncakeHybridConnector` | `mooncake` | engine + port | 支持；启动 vLLM 前校验 connector registry |
| `NixlConnector` | `nixl` | 仅 engine | 不支持；UCM 有效时 Helm 直接失败 |

`MultiConnector` 与 `UCMConnector` 是 Chart 组合出的内部 connector，不允许写入 `pd.kvTransfer.connector`。旧字段 `pd.connector`、`pd.mooncakePort`、`pd.ucm` 已直接删除；即使值为空或为 `false` 也会渲染失败并给出迁移提示。

#### RDMA / 设备资源说明

资源（含 GPU/NPU、RDMA 设备、Hugepages）一律以**原生 k8s `ResourceRequirements`** 写在 `servingEngineSpec.modelSpec.roles[].resources`，chart 原样 toYaml 透传；不再有 `requestCPU` / `requestGPU` / `extraResources` / `hugepages` / `rdma.resourceName` 等封装。

* RDMA 设备资源直接写键，如 `rdma/rdma_shared`、`rdma/hca_shared_devices_a`、`nvidia.com/hostdev`，放进 `roles[].resources.requests` / `.limits`。
* Hugepages 直接写 `hugepages-2Mi` / `hugepages-1Gi` 资源键；需要挂载时通过 `modelSpec.storage.extraStorage` 加 emptyDir/hostPath。
* `servingEngineSpec.modelSpec.rdma.enabled: true` 仍生效：唯一作用是自动给 vllm 容器追加 `IPC_LOCK` capability（与 `containerSecurityContext.capabilities.add` 自动合并去重）。其他 capability（`SYS_RESOURCE` / `NET_RAW` 等）请在 `containerSecurityContext.capabilities.add` 显式声明。
* 使用 **shared device plugin** 时通常需要 `hostNetwork: true`；根 `values.yaml` 已默认开启，若某模型曾关闭过，可在 `servingEngineSpec.modelSpec.hostNetwork` 显式设回 `true`。需要旁路挂载 `/dev/infiniband` 时在 `extraStorage` 加一项 `hostPath: /dev/infiniband`。

```yaml
servingEngineSpec:
  modelSpec:
    rdma:
      enabled: true        # 仅自动追加 IPC_LOCK
    hostNetwork: true       # shared device plugin 需要
    roles:
      - name: engine          # role 名：小写字母/数字/-，起止字母或数字，最长 12
        resources:
          requests:
            nvidia.com/gpu: "1"
            rdma/rdma_shared: "1"
            hugepages-2Mi: "4Gi"
          limits:
            nvidia.com/gpu: "1"
            rdma/rdma_shared: "1"
            hugepages-2Mi: "4Gi"
```

#### vLLM 启动参数（per-role vllmArgs）

引擎统一以 `vllm serve <modelPath> --served-model-name <modelName> …` 拉起。**参数只在 `servingEngineSpec.modelSpec.roles[].vllmArgs` 写**（每个 role 一段，flags-only，逐行解析；缺省回退 `modelSpec.vllmArgs`）。下列由 chart 托管、**勿写进 `vllmArgs`（写了会 `helm template` 直接报错）**：

* `modelPath`（来自 `modelSpec.modelPath`）
* `--served-model-name`（来自 `modelSpec.modelName`，空则回退 `modelPath`）
* 6 个网络参数：`--host` `--port` `--headless` `--data-parallel-address`（含 `-dpa`）`--data-parallel-rpc-port`（含 `-dpp`）`--data-parallel-start-rank`（含 `-dpr`）（entrypoint 按 kthena 注入的地址/序号自动填）
* `--kv-transfer-config`（PD 时由 Chart 为每个 role 生成 `kv-transfer.template.json` 与 `kv-transfer.meta.json`，Pod 启动时 resolver 根据 `group-name` / `role-id` 解析并追加）
* `--config`（禁止从运行时外部 YAML 注入参数，否则 Helm 无法校验 P/D 并行布局和托管参数）
* `--kv-offloading-size` / `--kv-offloading-backend`（仅当该 role 已生成 Chart KV template/meta 时禁止，避免 vLLM 在启动后静默替换 Mooncake/NIXL/Multi/UCM connector；无 Chart KV 的普通 role 仍可使用）
* Mooncake master 相关环境变量：`MOONCAKE_MASTER`、`MOONCAKE_CONFIG_PATH`、`MOONCAKE_GLOBAL_SEGMENT_SIZE`

`VLLM_ARGS_FILE` 也由 Chart 固定到当前 role 的投影文件，不允许通过 `modelSpec.env` 或 `preStart` 改写。role 启用 Chart KV 时，启动脚本还会在最终构造 argv 前清除 `VLLM_DP_SIZE/RANK/RANK_LOCAL/MASTER_IP/MASTER_PORT`，防止环境变量让实际 DP 布局偏离 Helm 已校验的端口跨度和 connector metadata。

`vllmArgs` 支持空行、`#` 注释与引号。示例：

```yaml
servingEngineSpec:
  modelSpec:
    modelPath: /mnt/model/Qwen3-0.6B
    modelName: Qwen3-0.6B          # 对外名；留空则回退 modelPath
    roles:
      - name: engine          # role 名：小写字母/数字/-，起止字母或数字，最长 12
        vllmArgs: |
          --tensor-parallel-size 1
          --max-model-len 20000
          --max-num-batched-tokens 13000
          --trust-remote-code
          --additional-config '{"torchair_graph_config":{"enabled":true}}'
```

> 各芯片/形态的真实写法见 `models/cuda/*.yaml` 与 `models/ascend/*.yaml`（单机 / 多机 / PD 均有样例）。

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `servingEngineSpec.modelSpec.roles[].vllmArgs` | string | `""` | per-role 原生 `vllm serve` CLI 参数块（flags-only），逐行解析；缺省回退 `modelSpec.vllmArgs` |
| `servingEngineSpec.modelSpec.vllmArgs` | string | `""` | 所有 role 的回退 `vllmArgs`（当某 role 未单独给时） |

#### Unified Cache 配置

| Field | Type | Default   | Description          |
|-------|------|-----------|----------------------|
| `servingEngineSpec.modelSpec.unifiedcacheConfig.enabled` | boolean | `true` | 显式开关（别名 `enable`）。`false` = 不使用 UCM：不渲染 ucm ConfigMap、不注 UCMConnector、不挂 UCM 卷/缓存盘、不建 SA/RBAC，等价于未填 `config`；显式 `true` 但 `config` 为空会渲染失败 |
| `servingEngineSpec.modelSpec.unifiedcacheConfig.kvcsStoreIdAutoDetect` | boolean | `false` | 是否自动从 PV CSI attributes 检测并回填 `kvcs_store_id` |
| `servingEngineSpec.modelSpec.unifiedcacheConfig.kvcsTlsEnable` | boolean | `false` | 写入 `config.ucm_connectors[*].ucm_connector_config.kvcs_tls_enable` |
| `servingEngineSpec.modelSpec.unifiedcacheConfig.config` | map | `{}` | UCM 主配置对象，渲染为 `ucm_config.template.yaml`；如需 `use_layerwise` 直接写在此处 |
| `servingEngineSpec.modelSpec.unifiedcacheConfig.config.ucm_connectors` | list | `[]` | UCM connector 列表（必填） |
| `servingEngineSpec.modelSpec.unifiedcacheConfig.config.ucm_connectors[*].ucm_connector_config.storage_backends` | string | `""` | 模板自动覆盖为 `storage.unifiedcacheStorage[*].mountPath` 以 `:` 连接 |

> 说明：`unifiedcacheConfig.config` 非空且 `enabled` 未设为 `false` 时 UCM 生效，Chart 会生成 `<release>-ucm-config` 作为模板文件，并以只读方式挂载到 `/vllm-workspace/UnifiedCache/config/ucm_config.template.yaml`。如果 `ucm_connectors[*].ucm_connector_config` 中存在 `kvcs_*` 字段，模板会在同一层补齐 `kvcs_instance_name` 和 `kvcs_tls_enable`。容器启动时会复制到 `/vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml`；当 `kvcsStoreIdAutoDetect=true` 时，还会按 PVC 对应 PV 的 CSI attributes 回填运行时配置里的 `kvcs_store_id`。标准启动脚本最终读取的是运行时文件路径 `ucm_config_yaml_path=/vllm-workspace/UnifiedCache/config/ucm_config.runtime.yaml`。PD + Mooncake 时该开关是唯一 UCM 开关：producer 自动生成 `MultiConnector{selected connector,UCMConnector}`，decode 保持纯传输 connector。
>
> 已废弃并忽略：`unifiedcacheConfig.dataDir/transferStreamNumber/enableGSA`、`kvcacheStoreConfig.*`、`unifiedcacheConfig.tlsSecret`、`unifiedcacheConfig.ucm_config`。（`unifiedcacheConfig.enabled` 已重新启用为显式开关，见上表；旧 values 若遗留 `enabled: true` 而无 `config` 会渲染失败，删掉该行或补 `config` 即可。）

### 路由（kthena ModelRoute/ModelServer）

本 chart **不再渲染任何 router 资源**（旧 `routerSpec` 已随 native + ucm-vllm-router 一并移除）。路由由 kthena 承载：

* **集群级 kthena-router 是前置控制面**，随 kthena 控制面预装，**不由本 chart 部署**。
* 是否暴露为 kthena 路由，由模型侧开关 `servingEngineSpec.modelSpec.router` 控制（`templates/kthena/modelserver.yaml` + `modelroute.yaml`）。开启时渲染 `ModelServer` + `ModelRoute`，由集群 kthena-router 接管流量；关闭（默认）时只渲染普通 `Service` 直连引擎。

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `servingEngineSpec.modelSpec.router.enabled` | boolean | `false` | 开启则渲染 `ModelServer` + `ModelRoute`（走 kthena-router）；否则只渲染普通 `Service` |
| `servingEngineSpec.modelSpec.router.inferenceEngine` | string | `"vLLM"` | `ModelServer.inferenceEngine` |
| `servingEngineSpec.modelSpec.router.workloadProtocol` | string | - | `ModelServer.workloadPort.protocol` |
| `servingEngineSpec.modelSpec.router.trafficPolicy` | map | - | `ModelServer.trafficPolicy`（toYaml） |
| `servingEngineSpec.modelSpec.router.loraAdapters` | list | - | `ModelRoute.loraAdapters` |
| `servingEngineSpec.modelSpec.router.parentRefs` | list | - | `ModelRoute.parentRefs` |
| `servingEngineSpec.modelSpec.router.rules` | list | - | `ModelRoute.rules`（缺省为单目标规则） |
| `servingEngineSpec.modelSpec.router.rateLimit` | map | - | `ModelRoute.rateLimit` |

> 完整路由写法见 `models/` 下的模型示例。

### 缓存服务器配置

> 本 chart 对 cacheserver **只渲染一个 `Service`**（`templates/service-cache-server.yaml`），**没有 Deployment/Pod**。渲染由 `cacheserverSpec` **对象是否存在**触发（`{{- if .Values.cacheserverSpec }}`，不读 `enableServer`）；Service `type` 硬编码为 `ClusterIP`（不读 `serviceType`）。真正生效的只有下面三个字段，其余 `cacheserverSpec.*`（`replicaCount`/`image`/`resources`/`strategy`/`probes`/PDB/`tolerations`/`runtimeClassName`/`schedulerName`/`securityContext`/`priorityClassName`/`affinity`/`serde` 等）当前不渲染任何资源，仅为向后兼容保留。

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cacheserverSpec.servicePort` | integer | `80` | Service 监听端口（`port`） |
| `cacheserverSpec.containerPort` | integer | `8000` | Service 目标端口（`targetPort`） |
| `cacheserverSpec.labels` | map | `{environment: "cache", release: "cache"}` | Service 标签与 selector（经 `chart.cacheserverLabels`） |
