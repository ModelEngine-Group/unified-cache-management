# Quick Start

Get started with UCM in your inference engine quickly. Choose your engine and follow the integration guide.

## Supported Engines

UCM integrates with the following inference engines:

### vLLM (CUDA)

**Prerequisites:** vLLM >= 0.9.1, device=cuda

**Quick Setup with Docker:**
```bash
docker pull unifiedcachemanager/ucm:latest
docker run --rm --gpus all --network=host --ipc=host \
    -v <path_to_models>:/home/model \
    -v <path_to_storage>:/home/storage \
    -it unifiedcachemanager/ucm:latest
```

**Detailed Guide:** See [vLLM Integration Guide](../getting-started/engines/vllm.md)

### vLLM (Ascend)

**Prerequisites:** vLLM-Ascend >= 0.18.0, device=NPU

**Quick Setup with Docker:**
```bash
docker pull unifiedcachemanager/ucm-ascend:latest
docker run --rm --network=host --ipc=host \
    -v <path_to_models>:/home/model \
    -v <path_to_storage>:/home/storage \
    -it unifiedcachemanager/ucm-ascend:latest
```

**Detailed Guide:** See [vLLM-Ascend Integration Guide](../getting-started/engines/vllm-ascend.md)

### SGLang

**Prerequisites:** SGLang >= v0.5.9, device=cuda

**Quick Setup with Docker:**
```bash
docker pull unifiedcachemanager/ucm-sglang:latest
docker run --rm --gpus all --network=host --ipc=host \
    -v <path_to_models>:/home/model \
    -v <path_to_storage>:/home/storage \
    -it unifiedcachemanager/ucm-sglang:latest
```

**Detailed Guide:** See [SGLang Integration Guide](../getting-started/engines/sglang.md)

### MindIE

**Prerequisites:** MindIE environment, device=NPU

**Quick Setup:** Follow the MindIE integration guide for detailed setup instructions.

**Detailed Guide:** See [MindIE Integration Guide](../getting-started/engines/mindie.md)

## Next Steps

After setting up your engine:

1. **Configure Prefix Cache** - Set up storage backend (Pipeline, NFS, DS3FS, etc.)
2. **Enable Observability** - Configure Prometheus and Grafana for monitoring
3. **Run Trace Mode** - Validate your deployment with trace analysis

For detailed configuration options, see the [Installation Guide](installation.md) and [Configuration Reference](../reference/api-parameters.md).