# 多模态 Encoder Cache 复用

多模态请求处理中，视觉编码器（ViT）需要对每张图片、每段视频重复计算。启用 UCM 的 EC Connector 后，encoder 输出会存入 UCM 分层存储：热层使用共享内存，冷层写入本地或挂载盘。同一模型实例内，相同媒体内容的后续请求直接命中复用，跳过 encoder 重算，减少重复媒体的编码开销。

典型收益场景：同一批图片/视频被多个请求反复引用，例如多轮图文问答、重复素材审核、批量图片打标等。图片越大、重复次数越多，收益越明显。

本文介绍如何拉起服务、配置 YAML 并验证效果。内部实现（chunk 切分、key 编码、失败语义等）见 [UCM EC Connector 设计](../design-docs/ucm-ec-connector.md)。

## 前提条件

- vLLM 版本包含 EC Connector 调度接口，且已安装 UCM（含 `ucm.integration.vllm.ucm_ec_connector` 模块）；
- 多模态模型（示例：`Qwen3.8-27B-w8a8`）；
- 支持 EC 的设备环境。下文命令以 Ascend NPU + vllm-ascend 容器为参考，CUDA 平台命令结构相同，替换设备相关环境变量即可。

## 拉起服务（以 Qwen3.8-27B-w8a8 为例）

```bash
# 启用测试用缓存清理接口，并输出 EC 加载/保存日志。
export VLLM_SERVER_DEV_MODE=1
export UCM_LOG_LEVEL=debug

vllm serve /models/Qwen3.8-27B-w8a8/ \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 1 --tensor-parallel-size 2 \
  --quantization ascend \
  --served-model-name qwen3.8 \
  --max-num-seqs 32 --max-model-len 131072 --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --no-disable-hybrid-kv-cache-manager \
  --ec-transfer-config '{"ec_connector":"UCMECConnector","ec_connector_module_path":"ucm.integration.vllm.ucm_ec_connector","ec_role":"ec_both","ec_connector_extra_config":{"UCM_CONFIG_FILE":"/workspace/unified-cache-management/examples/ucm_ec_config.yaml"}}'
```

关键参数说明：

| 参数 | 说明 |
| --- | --- |
| `--ec-transfer-config` | EC 特性入口。`ec_role` 取 `ec_both` 时同一实例既写缓存也读缓存；`UCM_CONFIG_FILE` 指向 UCM YAML，示例直接使用仓库 `examples/ucm_ec_config.yaml` |
| `--speculative-config` / `--compilation-config` / `--additional-config` | qwen3_5_mtp 投机解码、decode 全量 CUDA Graph、CPU 绑定——均为引擎侧性能配置，与 EC 无关 |
| `--no-disable-hybrid-kv-cache-manager` / `--enable-prefix-caching` | hybrid 模型（线性注意力 + 全注意力）的 KV cache manager 与前缀缓存配置 |
| `--quantization ascend` | w8a8 量化模型在 vllm-ascend 上使用 `ascend` |

启动成功的标志日志（`Initialized UCM EC connector` 在 SCHEDULER 与 WORKER 各出现一条）：

```text
Initialized UCM EC connector: role=SCHEDULER, producer=True, consumer=True, width=..., dtype=..., rows_per_chunk=..., ...
Initialized UCM EC connector: role=WORKER, producer=True, consumer=True, ...
Creating UCM EC store UcmPipelineStore: unique_id=..., device_id=..., chunk_bytes=..., ...
```

## 配置 UCM YAML

示例命令中 `UCM_CONFIG_FILE` 直接指向仓库的 `examples/ucm_ec_config.yaml`。该 YAML 只有一个顶层段 `ucm_ec_connector`：

```yaml
ucm_ec_connector:
  ucm_connector_name: "UcmPipelineStore"
  ucm_connector_config:
    store_pipeline: "Cache|Posix"
    storage_backends: "/mnt/test"         # 冷层目录，按实际可写路径修改，冒号分隔可配置多个
    posix_capacity_gb: 10240
    posix_io_engine: "psync"
    posix_data_trans_concurrency: 128
    cache_buffer_capacity_gb: 32
  encoder_cache_config:
    chunk_size: 128              # 每个 chunk 的行数，需为正整数
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `ucm_connector_name` | 是 | 生产者模式当前要求 `UcmPipelineStore` |
| `store_pipeline` | 否 | 常用 `Cache\|Posix`：共享内存热层 + 磁盘冷层 |
| `storage_backends` | 否 | 冷层目录，冒号分隔可配置多个 |
| `cache_buffer_capacity_gb` | 否 | 热层容量，缺省值随 Cache 实现变化；建议显式配置，启动时校验共享内存容量 |
| `chunk_size` | 是 | 每个 chunk 的行数。过小会放大块数量与查找次数，过大会增加尾部填充浪费；示例取 128 |

宽度（`encoder_cache_hidden_dim`）、命名空间（`cache_namespace`）、本机 Cache 域标识（`store_unique_id`）均为可选，缺省自动推导；完整字段说明见[设计文档](../design-docs/ucm-ec-connector.md)。

## 测试效果

### 1. 快速验证：看日志确认命中

向服务发送两次相同图片请求（可用任意 OpenAI 兼容客户端，`/v1/chat/completions` 携带 `image_url`）。第一次请求完成后，先执行下文的缓存清理命令，再发送第二次请求，排除 KV 前缀缓存和本地 encoder cache 的收益。观察日志：

- 第 1 次（miss 后保存）：请求收尾输出 `EC request cache hits: req=..., mm_items=1, hits_total=0`，同时出现 `EC dump item ...` 日志；
- 第 2 次（EC 命中）：输出 `EC request cache hits: req=..., mm_items=1, hits_total=1`，`UCM_LOG_LEVEL=debug` 下可见 `EC load item ...` 逐项加载日志。

外存目录文件数不再增长可作为辅助观察；确认 UCM 复用应以实际 `EC load item ...` 日志为准。

### 2. Benchmark：同 seed 连跑两轮对比

同一条命令运行两遍，两轮之间清理 vLLM 本地缓存：第一轮建立 UCM 缓存，第二轮验证外部 EC 复用，并对比两轮 TTFT。第一轮 miss 基线需要待测媒体尚未存入 UCM。

```bash
vllm bench serve \
  --backend openai-chat --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --model qwen3.8 --tokenizer /models/Qwen3.8-27B-w8a8 \
  --dataset-name random-mm \
  --num-prompts 20 --seed 1000 \
  --random-input-len 8 --random-output-len 1 --random-range-ratio 0 \
  --random-mm-base-items-per-request 2 --random-mm-num-mm-items-range-ratio 0.5 \
  --random-mm-limit-mm-per-prompt '{"image":4,"video":0}' \
  --random-mm-bucket-config "{(5120,5120,1):1.0}" \
  --request-rate inf --max-concurrency 10 --num-warmups 0 \
  --ignore-eos
```

第一轮完成且服务没有其他在途请求后，执行以下命令；所有命令成功后，再原样执行第二轮 benchmark：

```bash
set -euo pipefail

# 清理本地 KV prefix cache；HTTP 200 仍可能返回 success=false。
curl -fsS -X POST http://127.0.0.1:8000/reset_prefix_cache \
  | python -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("success") is True else 1)'

# 清理多模态预处理缓存。
curl -fsS -X POST http://127.0.0.1:8000/reset_mm_cache

# 清理 Scheduler 和 Worker 的本地 encoder cache。
curl -fsS -X POST http://127.0.0.1:8000/reset_encoder_cache
```

启动服务前必须设置 `VLLM_SERVER_DEV_MODE=1` 才会开放这些接口。`reset_mm_cache` 不清理 encoder 输出，因此必须同时调用 `reset_encoder_cache`。若清理失败，先确认没有在途请求并重试，成功后再开始第二轮。

原理：固定 `--seed` 使两轮图片内容一致，第一轮保存成功的 encoder 输出仍保留在 UCM 中；清理 vLLM 本地缓存后，第二轮需要通过 UCM 加载这些输出。两轮之间不要重启服务、不要清理 UCM 热层或外存目录。

| 参数 | 作用 |
| --- | --- |
| `--dataset-name random-mm` | 合成多模态数据集，按 bucket 尺寸生成随机图片 |
| `--num-prompts 20 --max-concurrency 10 --request-rate inf` | 20 条请求、10 并发、无节流 |
| `--random-input-len 8 --random-output-len 1` | 文本压到最短、输出 1 token，TTFT 基本由图片编码决定 |
| `--random-mm-base-items-per-request 2 --random-mm-num-mm-items-range-ratio 0.5` | 每请求图片数围绕 2 张波动，实际在 1~3 张间采样 |
| `--random-mm-limit-mm-per-prompt '{"image":4,"video":0}'` | 仅用图片，单请求上限 4 张 |
| `--random-mm-bucket-config "{(5120,5120,1):1.0}"` | 单一图片尺寸 5120×5120（末位 1 表示单帧，即图片） |
| `--seed 1000` | 固定随机种子，两轮数据集完全一致 |

### 3. 结果解读

| 轮次 | 预期 TTFT | 含义 |
| --- | --- | --- |
| 第一轮（seed=1000） | 高 | 全部 EC miss：encoder 重算 + dump 写缓存 |
| 第二轮（同 seed，已清理本地缓存） | 预期降低 | 从 UCM 加载第一轮保存成功的输出，跳过 encoder 重算；以实际加载日志确认 |

命中证据：

- 第一轮请求收尾日志 `EC request cache hits: req=..., mm_items=N, hits_total=0`，`UCM_LOG_LEVEL=debug` 下有 `EC dump item ...`；
- 第二轮应出现 `EC load item ...`，证明实际发生 UCM 加载；无重复 identifier 或本轮本地共享时，预期 `hits_total=N`（N 为请求内不同媒体 identifier 数）；
- 外存目录文件数在第二轮不再增长仅是辅助证据，不能单独证明 UCM 命中。

## EPD 分离式部署

EPD 将媒体编码放在 E 节点，prefill 和 decode 放在 PD 节点。E 通过 UCM 保存 encoder 输出，原生 EPD Proxy 将图片请求改写为 metadata-only 输入并交给 PD。

以下示例使用 vLLM Ascend v0.30.0、`Qwen3.8-27B-w8a8`，E=TP2、PD=TP2。三个服务分别在独立终端或容器中启动；示例地址对应同机部署，跨机时替换 Proxy 的 E/PD 地址。

### 1. 准备环境和共享存储

- E、PD 安装同一版本的 vLLM、vllm-ascend 和 UCM，模型与预处理配置一致。
- 两侧均可访问共享目录 `/mnt/ucm-ec-shared`，使用相同 YAML。每个引擎使用自动生成的独立 Cache UUID。
- 示例 UCM 源码路径为 `/workspace/unified-cache-management`，vLLM 源码路径为 `/vllm-workspace/vllm`；按实际安装位置替换。

共享目录完成挂载后，在 E、PD 的运行环境创建根目录：

```bash
mkdir -p /mnt/ucm-ec-shared
```

### 2. 配置 UCM

在 E 和 PD 各自保存 `/workspace/ucm_ec_epd.yaml`：

```yaml
ucm_ec_connector:
  ucm_connector_name: UcmPipelineStore
  ucm_connector_config:
    store_pipeline: Cache|Posix
    storage_backends: /mnt/ucm-ec-shared
    cache_buffer_capacity_gb: 32
    posix_io_engine: psync
  encoder_cache_config:
    chunk_size: 128
    enable_epd_disaggregate: true
    epd_timeout_ms: 30000
```

| 配置 | 作用 |
| --- | --- |
| `enable_epd_disaggregate: true` | 开启 EPD，两端一致 |
| `epd_timeout_ms` | PD 从首次 Store miss 开始计算可见性等待期限 |
| `storage_backends` | E/PD 交接 EC 的共享目录 |
| `chunk_size` | 每个 chunk 的 embedding 行数，两端一致 |

示例使用 `max_model_len=1024`、每请求一张图片及 `max_pixels=262144` 作为基础验证预算。实际业务按输入规模和并发调整。

### 3. 启动 E

E 使用 `ec_producer`，通过 `cudagraph_mm_encoder` 开启 ViT 图模式。已有 HBM 输出和本次新编码输出都会经过 UCM 保存入口。E 运行 encoder-only，原生自动关闭 KV prefix caching。

```bash
export ENABLE_UCM_PATCH=1

ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve /data/model/Qwen3.8-27B-w8a8 \
  --host 0.0.0.0 --port 19634 \
  --served-model-name qwen3.8 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --compilation-config '{"cudagraph_mm_encoder":true}' \
  --max-model-len 1024 --max-num-seqs 2 \
  --gpu-memory-utilization 0.3 \
  --mm-processor-kwargs '{"max_pixels":262144}' \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --additional-config '{"enable_cpu_binding":false}' \
  --ec-transfer-config '{
    "ec_connector":"UCMECConnector",
    "ec_connector_module_path":"ucm.integration.vllm.ucm_ec_connector",
    "ec_role":"ec_producer",
    "ec_connector_extra_config":{"UCM_CONFIG_FILE":"/workspace/ucm_ec_epd.yaml"}
  }'
```

### 4. 启动 PD

PD 使用 `ec_consumer`，开启 decode 图模式和 HBM 中的 KV prefix caching。命中本地 EC 时直接复用；其余媒体等待 Store 可见后加载。

```bash
export ENABLE_UCM_PATCH=1

ASCEND_RT_VISIBLE_DEVICES=6,7 vllm serve /data/model/Qwen3.8-27B-w8a8 \
  --host 0.0.0.0 --port 19635 \
  --served-model-name qwen3.8 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --max-model-len 1024 --max-num-seqs 2 \
  --gpu-memory-utilization 0.7 \
  --mm-processor-kwargs '{"max_pixels":262144}' \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --additional-config '{"enable_cpu_binding":false}' \
  --ec-transfer-config '{
    "ec_connector":"UCMECConnector",
    "ec_connector_module_path":"ucm.integration.vllm.ucm_ec_connector",
    "ec_role":"ec_consumer",
    "ec_connector_extra_config":{"UCM_CONFIG_FILE":"/workspace/ucm_ec_epd.yaml"}
  }'
```

### 5. 启动 Proxy

E、PD 的 `/health` 返回 HTTP 200 后，启动原生 Proxy。`--prefill-servers-urls disable` 表示 prefill 与 decode 由同一个 PD 服务处理。

```bash
python /vllm-workspace/vllm/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py \
  --host 0.0.0.0 --port 11001 \
  --encode-servers-urls http://127.0.0.1:19634 \
  --prefill-servers-urls disable \
  --decode-servers-urls http://127.0.0.1:19635 \
  --decode-retries 0
```

客户端向 `http://127.0.0.1:11001/v1/chat/completions` 发送图片请求，模型名为 `qwen3.8`。每个服务终端可通过 Ctrl+C 停止对应进程。

### 6. 验证交接与热缓存

在部署容器中启用现有 EC 测试文件的模型用例，连续发送两次相同图片请求：

```bash
cd /workspace/unified-cache-management
export UCM_EPD_PROXY_URL=http://127.0.0.1:11001
export UCM_EPD_MODEL=qwen3.8
python -m unittest discover -s test -p test_ucm_ec_config.py -v
```

| 日志 | 观测 |
| --- | --- |
| Proxy：`as metadata references` | 图片已改写为 metadata-only 输入 |
| E：`EC dump item` | 输出已提交 UCM 保存 |
| E：`EC cached saves: candidates=1, submitted=1` | HBM 命中的输出再次提交保存 |
| PD：`EC load item` | 首次请求从 UCM 加载对应 identifier |
| PD：`EC local cache reuse` | 重复请求直接复用本地 EC |

首次请求观察 E 保存与 PD 加载；重复请求可命中 PD 的 KV prefix cache 或本地 EC。通过调度计数和逐 item 日志区分复用来源，参数与实现见[设计文档](../design-docs/ucm-ec-connector.md)。

当前 Cache2 对 DRAM READY 命中跳过后端写入。共享文件被删除后，冷 PD 可能在可见性等待处超时；热 PD 可继续复用本地 EC。


## 进阶：与 KV Connector 同时启用

UCM 支持同一实例同时启用 KV 连接器（如 HLA）与 EC 连接器，两者使用**独立的两份 YAML**，通过 `--kv-transfer-config` 与 `--ec-transfer-config` 分别指定。注意格式差异：KV 的顶层是 `ucm_connectors:` 列表，EC 的顶层是 `ucm_ec_connector:` 段。

```bash
vllm serve /home/models/Qwen3.8-27B-w8a8 \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --max-model-len 16384 --max-num-seqs 32 \
  --host 0.0.0.0 --port 8002 \
  --trust-remote-code --gpu-memory-utilization 0.9 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/home/yhq/dual-connector-test/ucm_kv_config.yaml"}}' \
  --ec-transfer-config '{"ec_connector":"UCMECConnector","ec_connector_module_path":"ucm.integration.vllm.ucm_ec_connector","ec_role":"ec_both","ec_connector_extra_config":{"UCM_CONFIG_FILE":"/home/yhq/dual-connector-test/ucm_ec_config.yaml"}}'
```

注意事项：

- KV 侧 `cache_buffer_capacity_gb` 按模型 KV 池校验，hybrid 27B 模型实测需 ≥160（设小直接报 `too small buffer`）；
- EC 侧外存目录应与 KV 侧分开（如 `store/kv` 与 `store/ec`）；
- 启动成功标志：`UCMHybridLinearAttentionConnector initialized` ×3 + `Initialized UCM EC connector`（SCHEDULER/WORKER 各一条）。

## 注意事项

- `/dev/shm` 容量需 ≥ `cache_buffer_capacity_gb`，启动前会自动校验，不足时启动失败；
- 前台服务通过 Ctrl+C 退出；容器服务使用正常的停止流程释放设备资源；
- load 失败会使实例进入错误状态，通过重启恢复服务；处理细节见[设计文档](../design-docs/ucm-ec-connector.md)第 7 节。
