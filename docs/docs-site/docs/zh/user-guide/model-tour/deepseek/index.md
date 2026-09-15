# DeepSeek

DeepSeek 系列模型的 [vLLM Ascend 官方教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html)。接入 UCM 可直接查看下方的 [DeepSeek-V4-Flash 部署与调用示例](#deepseek-v4-flash-ucm)。

## vLLM Ascend 模型指南

| 模型 | vLLM Ascend latest 指南 |
| --- | --- |
| DeepSeek-V3 & 3.1 | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.1.html) |
| DeepSeek-V3.2 | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html) |
| DeepSeek-V4-Flash | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html) |
| DeepSeek-V4-Flash-Vision-Exp (Experimental) | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash-Vision.html) |
| DeepSeek-V4.1-Flash | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) |
| DeepSeek-V4-Pro | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html) |
| DeepSeek-R1 | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-R1.html) |
| DeepSeek-OCR-2 | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeekOCR2.html) |

## DeepSeek-V4-Flash UCM 示例 { #deepseek-v4-flash-ucm }

在一个 Docker 容器中部署 DeepSeek-V4-Flash，并接入 UCM Prefix Cache。
按硬件选择下方标签；CUDA 和 Ascend 使用不同格式的权重。
这些启动命令需要在目标加速卡上验证模型输出与外部缓存。

### 部署

准备 Linux、Docker 和可用的宿主机加速卡驱动。CUDA 需要 NVIDIA Container Toolkit；
Ascend 的宿主机驱动、设备文件应与所选运行时匹配。
通过对应平台标签中的链接下载完整模型，包括 tokenizer 和配置文件。

打开[安装页](../../quick_start/index.md)，选择 **Image**，将已发布的 UCM 运行时镜像坐标
填入 `UCM_IMAGE`。下方命令分别使用 CUDA 上的 vLLM **0.28.0** 和 Ascend 上的
vLLM-Ascend **0.25.1rc0 / A2**。同时选择匹配的架构和后端；如果当前发布没有
该组合，请切换到提供该组合的版本。

在宿主机准备 UCM 配置：

```bash
export UCM_WORKDIR="$PWD/ucm-deepseek-v4"
mkdir -p "$UCM_WORKDIR/cache/cuda" "$UCM_WORKDIR/cache/ascend"
cat > "$UCM_WORKDIR/ucm.yaml" <<'YAML'
use_layerwise: false
load_tokens_threshold: 2048
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
      cache_buffer_capacity_gb: 128
      io_direct: false
YAML
```

为 UCM 预留 128 GiB 主机内存，平分给两个 64 GiB Store，并为引擎和模型加载
另留内存。容器共享内存上限设为 160 GiB，缓存保存在宿主机挂载目录中。
下方两个容器选择一个运行，避免占用同一服务端口。

=== "CUDA / vLLM"

    使用单节点 **8 张 B200 或 B300 GPU**，拓扑依据官方
    [单节点 TP 配方](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)。
    下载 [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)，
    将 `MODEL_DIR` 设为模型所在的绝对路径。这里使用原始 FP4 专家权重与 FP8 权重
    混合检查点；命令不覆盖 Hopper，也不适用于单独发布的 `0731` / DSpark 检查点。

    ```bash
    export UCM_IMAGE='<CUDA UCM image copied from Installation>'
    export MODEL_DIR=/absolute/path/DeepSeek-V4-Flash

    docker run -d --name ucm-deepseek-cuda \
      --gpus all --shm-size 160g \
      -p 127.0.0.1:8000:8000 \
      -e ENABLE_UCM_PATCH=1 \
      -v "$MODEL_DIR:/models/DeepSeek-V4-Flash:ro" \
      -v "$UCM_WORKDIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORKDIR/cache/cuda:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" \
      serve /models/DeepSeek-V4-Flash \
      --served-model-name deepseek-v4-flash \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 8 \
      --data-parallel-size 1 \
      --pipeline-parallel-size 1 \
      --distributed-executor-backend mp \
      --tokenizer-mode deepseek_v4 \
      --trust-remote-code \
      --kv-cache-dtype fp8 \
      --block-size 256 \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.9 \
      --enforce-eager \
      --kv-transfer-config '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
      }'

    docker logs -f ucm-deepseek-cuda
    ```

=== "Ascend A2 / vLLM-Ascend"

    使用一台 **Atlas 800 A2，配置 8 × 64 GB NPU**，下载量化后的
    [DeepSeek-V4-Flash-w8a8-mtp 权重](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)。
    将 `MODEL_DIR` 设为模型所在的绝对路径。权重包含 MTP head，本例不启用推测解码。
    命令采用 [Ascend 部署指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)
    中的标准设备和驱动路径。

    ```bash
    export UCM_IMAGE='<Ascend A2 UCM image copied from Installation>'
    export MODEL_DIR=/absolute/path/DeepSeek-V4-Flash-w8a8-mtp

    docker run -d --name ucm-deepseek-ascend \
      --network host --shm-size 160g --privileged \
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
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      -e ENABLE_UCM_PATCH=1 \
      -e OMP_PROC_BIND=false -e OMP_NUM_THREADS=10 \
      -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
      -e HCCL_BUFFSIZE=1024 -e HCCL_OP_EXPANSION_MODE=AIV \
      -e TASK_QUEUE_ENABLE=1 \
      -v "$MODEL_DIR:/models/DeepSeek-V4-Flash-w8a8-mtp:ro" \
      -v "$UCM_WORKDIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORKDIR/cache/ascend:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" \
      serve /models/DeepSeek-V4-Flash-w8a8-mtp \
      --served-model-name deepseek-v4-flash \
      --host 127.0.0.1 --port 8000 \
      --tensor-parallel-size 8 \
      --data-parallel-size 1 \
      --pipeline-parallel-size 1 \
      --enable-expert-parallel \
      --distributed-executor-backend mp \
      --tokenizer-mode deepseek_v4 \
      --trust-remote-code \
      --quantization ascend \
      --block-size 128 \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.9 \
      --enforce-eager \
      --kv-transfer-config '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
      }'

    docker logs -f ucm-deepseek-ascend
    ```

等待模型加载完成。UCM 启动日志应包含 `Init UCM FAWA connector`，以及
`FAWA FA` / `FAWA WA` 两套 Store 配置。
按 Ctrl+C 退出 `docker logs`；后台容器会继续运行。

### 调用

在同一宿主机检查服务就绪状态，再发送一次文本续写请求：

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "prompt": "A KV cache speeds up language model inference by",
    "max_tokens": 128,
    "temperature": 0
  }'
```

服务就绪时，`/health` 返回 HTTP 200。续写响应在 `choices[0].text` 中返回
生成文本，在 `usage` 中返回 token 用量。这条短请求只验证 API 调用成功，
不能证明发生了外部缓存命中。

其他硬件拓扑可参考 [vLLM 配方](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
和 [Ascend 模型指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)。
