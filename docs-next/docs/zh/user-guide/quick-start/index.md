# 快速开始

在您的推理引擎中快速启用 UCM。选择您的引擎并按照集成指南操作。

## 支持的引擎

UCM 与以下推理引擎集成：

### vLLM (CUDA)

**前置条件：** vLLM >= 0.9.1, device=cuda

**Docker 快速启动：**
```bash
docker pull unifiedcachemanager/ucm:latest
docker run --rm --gpus all --network=host --ipc=host \
    -v <模型路径>:/home/model \
    -v <存储路径>:/home/storage \
    -it unifiedcachemanager/ucm:latest
```

**详细指南：** 参见 [vLLM 集成指南](../getting-started/engines/vllm.md)

### vLLM (Ascend)

**前置条件：** vLLM-Ascend >= 0.18.0, device=NPU

**Docker 快速启动：**
```bash
docker pull unifiedcachemanager/ucm-ascend:latest
docker run --rm --network=host --ipc=host \
    -v <模型路径>:/home/model \
    -v <存储路径>:/home/storage \
    -it unifiedcachemanager/ucm-ascend:latest
```

**详细指南：** 参见 [vLLM-Ascend 集成指南](../getting-started/engines/vllm-ascend.md)

### SGLang

**前置条件：** SGLang >= v0.5.9, device=cuda

**Docker 快速启动：**
```bash
docker pull unifiedcachemanager/ucm-sglang:latest
docker run --rm --gpus all --network=host --ipc=host \
    -v <模型路径>:/home/model \
    -v <存储路径>:/home/storage \
    -it unifiedcachemanager/ucm-sglang:latest
```

**详细指南：** 参见 [SGLang 集成指南](../getting-started/engines/sglang.md)

### MindIE

**前置条件：** MindIE 环境, device=NPU

**快速启动：** 按照 MindIE 集成指南进行详细设置。

**详细指南：** 参见 [MindIE 集成指南](../getting-started/engines/mindie.md)

## 后续步骤

设置完引擎后：

1. **配置前缀缓存** - 设置存储后端（Pipeline、NFS、DS3FS 等）
2. **启用观测能力** - 配置 Prometheus 和 Grafana 进行监控
3. **运行 Trace 模式** - 使用跟踪分析验证部署效果

详细配置选项请参见[安装指南](installation.md)和[配置参考](../reference/api-parameters.md)。