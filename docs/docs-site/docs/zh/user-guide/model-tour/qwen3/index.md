# Qwen

Qwen 系列模型的 [vLLM Ascend 官方教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html)。接入 UCM 可直接查看下方的 [Qwen3.8-27B 部署与调用示例](#qwen38-27b-ucm)。

## vLLM Ascend 模型指南

| 模型 | vLLM Ascend latest 指南 |
| --- | --- |
| Qwen3-Dense(0.6B/1.7B/4B/8B/14B/32B) | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html) |
| Qwen-VL-Dense(8B/32B) | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen-VL-Dense.html) |
| Qwen3-30B-A3B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-30B-A3B.html) |
| Qwen3-235B-A22B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-235B-A22B.html) |
| Qwen3-VL-30B-A3B-Instruct | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-30B-A3B-Instruct.html) |
| Qwen3-VL-235B-A22B-Instruct | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-235B-A22B-Instruct.html) |
| Qwen3-Coder-30B-A3B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Coder-30B-A3B.html) |
| Qwen3-Embedding | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Embedding.html) |
| Qwen3-VL-Embedding | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-Embedding.html) |
| Qwen3-Reranker | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Reranker.html) |
| Qwen3-VL-Reranker | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-Reranker.html) |
| Qwen3-Next | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Next.html) |
| Qwen3-Omni-30B-A3B-Thinking | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.html) |
| Qwen3.5-27B & Qwen3.6-27B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html) |
| Qwen3.5-Dense (2B/4B/9B) | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-Dense.html) |
| Qwen3.5-397B-A17B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-397B-A17B.html) |
| Qwen3.6-35B-A3B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.6-35B-A3B.html) |
| Qwen3.8-2.4T-A95B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-2.4T-A95B.html) |
| Qwen3.8-27B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html) |
| Qwen3-ASR-1.7B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html) |
| Qwen2.5-Math-RM-72B | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen2.5-Math-RM-72B.html) |

## Qwen3.8-27B UCM 示例 { #qwen38-27b-ucm }

以下示例命令尚未在加速卡上实跑验证。

在单个 Docker 容器中运行 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，接入 UCM Prefix Cache。本例使用 BF16 权重，提供文本服务，上下文长度为 8,192 token，最多同时处理 8 个序列。

### 部署

先将完整模型下载到本地目录。Linux 主机需要安装 Docker，并准备 NVIDIA Container Toolkit，或与所选镜像匹配的 Ascend 驱动。下方命令分别使用两张 80 GB CUDA GPU，或一台配有八张 64 GB NPU 的 Atlas 800 A2；这些是示例资源配置，不代表最低要求。

打开[安装页](../../quick_start/index.md)，选择 **Image**、计算平台和主机架构，将完整镜像地址复制到 `UCM_IMAGE`。CUDA 示例选择已发布的 vLLM **0.28.0** 运行时，A2 示例选择 vLLM-Ascend **0.25.1rc0** 运行时。镜像已包含 UCM，直接使用选择器给出的完整地址。将 `UCM_MODEL_DIR` 改为下载的模型目录，再执行：

```bash
export UCM_MODEL_DIR=/srv/models/Qwen3.8-27B
export UCM_WORK_DIR="$PWD/ucm-qwen38"
export UCM_IMAGE='<copy the image reference from Installation>'
mkdir -p "$UCM_WORK_DIR/cache"
docker pull "$UCM_IMAGE"
```

每个标签先写入对应的 UCM 配置，再启动 Docker。`Cache` 负责设备与主机内存之间的 KV 搬运，`Posix` 将数据持久化到挂载目录。CUDA TP2 需要预留 **160 GiB** 主机缓存内存，Ascend TP8 需要预留 **256 GiB**，另外还要留出引擎内存。如果更换引擎布局后，初始化提示更大的最低缓冲容量，请按提示调大配置。两条命令通过 `--ipc=host` 使用主机共享内存。

Qwen3.8-27B 混合使用全注意力和 Gated DeltaNet。保留混合 KV Cache 管理器，使用 `--mamba-cache-mode align`，让 UCM 同时恢复注意力 KV 和对应的递归状态。`use_layerwise: false` 为本例选择批量加载方式。

=== "CUDA · vLLM"

    执行本标签前先选择 CUDA 镜像。命令向容器开放设备 0、1，使用 TP2。

    ```bash
    cat > "$UCM_WORK_DIR/ucm.yaml" <<'YAML'
    use_layerwise: false
    ucm_connectors:
      - ucm_connector_name: UcmPipelineStore
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          storage_backends: /mnt/ucm-cache
          cache_buffer_capacity_gb: 80
          posix_io_engine: psync
          io_direct: false
          timeout_ms: 30000
    YAML
    ```

    ```bash
    docker run -d --name ucm-qwen38 \
      --gpus '"device=0,1"' --ipc=host \
      -p 8000:8000 \
      -e ENABLE_UCM_PATCH=1 \
      -v "$UCM_MODEL_DIR:/models/Qwen3.8-27B:ro" \
      -v "$UCM_WORK_DIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORK_DIR/cache:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" serve /models/Qwen3.8-27B \
      --served-model-name qwen38-27b \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 2 \
      --dtype bfloat16 --max-model-len 8192 --max-num-seqs 8 \
      --block-size 128 --mamba-cache-mode align \
      --enable-prefix-caching --enforce-eager \
      --language-model-only --reasoning-parser qwen3 \
      --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"}}'
    ```

=== "Ascend · vLLM-Ascend"

    执行本标签前先选择兼容 A2 的镜像。命令向容器开放设备 0–7 和主机驱动，使用 TP8。驱动路径按 Atlas 800 A2 示例填写；A3 主机需要使用对应镜像及设备配置。

    ```bash
    cat > "$UCM_WORK_DIR/ucm.yaml" <<'YAML'
    use_layerwise: false
    ucm_connectors:
      - ucm_connector_name: UcmPipelineStore
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          storage_backends: /mnt/ucm-cache
          cache_buffer_capacity_gb: 32
          posix_io_engine: psync
          io_direct: false
          timeout_ms: 30000
    YAML
    ```

    ```bash
    docker run -d --name ucm-qwen38 \
      --network host --ipc=host \
      --device /dev/davinci0 --device /dev/davinci1 \
      --device /dev/davinci2 --device /dev/davinci3 \
      --device /dev/davinci4 --device /dev/davinci5 \
      --device /dev/davinci6 --device /dev/davinci7 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
      -v /usr/local/dcmi:/usr/local/dcmi:ro \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
      -e ENABLE_UCM_PATCH=1 \
      -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      -e HCCL_BUFFSIZE=200 \
      -v "$UCM_MODEL_DIR:/models/Qwen3.8-27B:ro" \
      -v "$UCM_WORK_DIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORK_DIR/cache:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" serve /models/Qwen3.8-27B \
      --served-model-name qwen38-27b \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 8 \
      --dtype bfloat16 --max-model-len 8192 --max-num-seqs 8 \
      --block-size 128 --mamba-cache-mode align \
      --enable-prefix-caching --enforce-eager \
      --language-model-only --reasoning-parser qwen3 \
      --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"}}'
    ```

使用 `docker logs -f ucm-qwen38` 查看启动日志。服务就绪后，按 Ctrl-C 退出日志查看，在同一主机终端调用服务。

### 调用

在主机准备 `curl` 和 `jq`。`/health` 应返回 HTTP 200；随后发送一次请求，并从 JSON 响应中读取 `choices[0].message.content`。

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-27b",
    "messages": [{"role": "user", "content": "Explain what a KV cache stores in two sentences."}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq -r '.choices[0].message.content'
```

官方资料：[模型与权重](https://huggingface.co/Qwen/Qwen3.8-27B)、[vLLM 配方](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)、[vLLM-Ascend 配方](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html)。
