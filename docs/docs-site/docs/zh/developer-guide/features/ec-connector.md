# 多模态 Encoder Cache 复用

多模态请求处理中，视觉编码器（ViT）需要对每张图片、每段视频重复计算。启用 UCM 的 EC Connector 后，encoder 输出会存入 UCM 分层存储：热层使用共享内存，冷层写入本地或挂载盘。同一模型实例内，相同媒体内容的后续请求直接命中复用，跳过 encoder 重算，显著降低重复素材场景的 TTFT。

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
| `--quantization ascend` | w8a8 量化模型在 vllm-ascend 上必须写 `ascend`，不能写 `w8a8_dynamic` |

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
| `cache_buffer_capacity_gb` | 否 | 热层容量，缺省 128；`/dev/shm` 不足时启动失败 |
| `chunk_size` | 是 | 每个 chunk 的行数。过小会放大块数量与查找次数，过大会增加尾部填充浪费；示例取 128 |

宽度（`encoder_cache_hidden_dim`）、命名空间（`cache_namespace`）、跨实例共享标识（`store_unique_id`）均为可选，缺省自动推导；完整字段说明见[设计文档](../design-docs/ucm-ec-connector.md)。

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
- 停止服务使用 `pkill -TERM`（或直接删容器）。对 NPU 环境使用 SIGKILL 可能泄漏驱动级显存，只能重启释放；
- 当前版本 load 失败为 fail-closed（引擎级错误），不支持自动重算；生产环境建议关注 Store 后端的可用性。设计权衡见[设计文档](../design-docs/ucm-ec-connector.md)第 7 节。
