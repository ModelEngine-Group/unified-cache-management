# 共享存储 PD 分离部署

本页介绍 [PD 分离部署](index.md)中的共享存储交接方案，从一个 Prefill 和一个 Decode 实例开始。

共享存储部署中，Prefill 和 Decode 通过 UCM 使用同一份外部缓存。请求先经过 Prefill，计算并保存提示词块，再携带原始提示词进入 Decode。Decode 通过查找发现可复用块，UCM 不会向它提供对端实例地址。

仓库中的 `ucm/pd/toy_proxy_server.py` 演示了这一顺序。内置 Kubernetes PD 配置采用另一条路径，详见[传输连接器与 UCM 组合](distributed.md)。

## 开始之前

先按 [vLLM 快速开始](../../quick_start/index.md#vllm)分别验证每个引擎，并从[安装](../../quick_start/index.md)选择引擎镜像或软件包。本页给出共享存储配置、两侧引擎、代理启动和请求验证步骤。

两个实例必须满足以下条件：

- 权重、tokenizer、模型目录名、dtype、块大小相同，KV 布局兼容。起步时使用相同的张量并行设置。
- 能访问同一批存储对象。两块本地磁盘上的路径字符串相同，不代表存储是共享的。
- 存储权限、容量和写后读可见性满足要求，让 Decode 能够看到 Prefill 已完成的写入。
- 使用不同的引擎 HTTP 端口，并显式分配设备。在错误设备上启动一个进程，不能算作第二个独立实例。

## 1p1d

以下沿用原 Qwen2.5-7B-Instruct 示例：TP1、20000-token 上下文和 32 GiB 主机缓存。Prefill、Decode 分别在各自主机启动，均使用本机设备 0；在同一主机运行时须分配不同设备。先准备匹配的引擎与 UCM 环境；这些命令不代表已经验证新引擎版本。

在两个引擎环境中将以下 YAML 保存为 `/etc/ucm/pd.yaml`。两侧 `/mnt/ucm-shared` 必须指向同一共享文件系统。

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-shared"
      cache_buffer_capacity_gb: 32
enable_event_sync: true
use_layerwise: false
```

### 启动 Prefill

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### 启动 Decode

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### 启动代理并调用

从 UCM 源码根目录运行，将带引号的主机占位符换成可访问的引擎 IP。保持代理运行，在另一终端发送请求。

```bash
python3 ucm/pd/toy_proxy_server.py \
+  --pd-disaggregation --host localhost --port 7802 \
+  --prefiller-hosts "<prefill-node-ip>" --prefiller-ports 7800 \
+  --decoder-hosts "<decode-node-ip>" --decoder-ports 7801
```

```bash
curl http://localhost:7802/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

### 验证交接

1. 使用专用缓存目录，提示词长度应覆盖多个配置块，并满足持久化阈值。
2. 通过代理发送一个确定性请求，查看两个引擎的日志，确认 Prefill 执行计算并提交缓存写入。
3. 检查 Decode 的 UCM 命中 token 和成功加载情况。用相同设置，与独立验证过的完整计算请求比较输出。
4. 修改提示词内容后重试，确认无关前缀不会表现为完整外部命中。
5. 保留缓存并重启两个引擎，再重放原始请求，以区分外部复用和引擎内存命中。

代理以 Prefill HTTP 响应作为阶段切换点，不会查询存储持久化回执。UCM dump 有独立的完成生命周期。如果 Decode 启动时块尚不可见，请求仍可能通过重算缺失前缀成功。宣称这条路径通过存储传递了全部提示词 KV 前，需要测量写入完成情况和 Decode 命中。`enable_event_sync` 协调设备事件，不是跨服务器提交协议。


## 不同平台的 1p1d

原异构示例使用 Ascend Prefill 和 CUDA Decode，两侧均设置 `--dtype bfloat16`。复用上面的 UCM YAML 和相同模型目录名。仅 dtype 一致不能证明 KV 布局兼容，需要在这一确切的引擎与设备组合上验证外部命中和输出。

### Ascend Prefill

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--dtype bfloat16 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### CUDA Decode

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--dtype bfloat16 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### 代理与请求

```bash
python3 ucm/pd/toy_proxy_server.py \
+  --pd-disaggregation --host localhost --port 7802 \
+  --prefiller-hosts "<prefill-node-ip>" --prefiller-ports 7800 \
+  --decoder-hosts "<decode-node-ip>" --decoder-ports 7801
```

```bash
curl http://localhost:7802/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

## XpYd

原双 Prefill、双 Decode 示例在同一主机分配 GPU 0–3。各代码块分别在独立终端运行，使用同一模型与共享 YAML。多主机部署按各自主机调整设备，并分别填写代理的主机列表。

### Prefill 0

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Prefill 1

```bash
export CUDA_VISIBLE_DEVICES=1
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Decode 0

```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=2
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--port 7802 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Decode 1

```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=3
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--port 7803 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### 代理与请求

```bash
python3 ucm/pd/toy_proxy_server.py \
+  --pd-disaggregation --host localhost --port 7805 \
+  --prefiller-hosts "<prefill-node-ip>" "<prefill-node-ip>" --prefiller-ports 7800 7801 \
+  --decoder-hosts "<decode-node-ip>" "<decode-node-ip>" --decoder-ports 7802 7803
```

```bash
curl http://localhost:7805/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

主机与端口列表长度必须一致。代理按轮询选择实例，不提供基于健康状态的调度或进行中请求恢复；每个 Decode 都必须能够读取各 Prefill 的缓存。

## 排查复用缺失

- **没有写入：**检查 Prefill 的持久化阈值、所选 connector 和存储错误。
- **有写入，但 Decode 没有命中：**检查共享挂载是否指向同一存储、写入可见性、模型与缓存兼容性，以及 Decode 实际生效的 UCM 配置。
- **有命中，但加载失败：**比较延迟前，先查看 UCM 加载错误和后端健康状态。
- **响应很快，但没有外部命中：**检查引擎内存命中和重计算量；仅凭延迟无法确定 KV 路径。

通过[指标](../../observability/metrics.md)分别采集各阶段的数据。
