# NFS Store

此 connector 通过已有文件系统路径保存 KV 块。路径既可以位于本地磁盘，也可以位于 NFS 挂载上；UCM 不负责挂载 NFS 或运行 NFS 服务。因此，远端共享取决于为每个服务进程提供的挂载和权限。

本页使用当前 vLLM 的 V1 NFS Connector 配置。已有旧版后端部署应先核对接口版本，详见[实现说明](../extending-store.md#backend-entrypoints)。

## 选择文件系统路径

如果要沿用这个直接文件系统 connector 的已有配置，可以选择 `UcmNfsStore`。对于需要主机缓存、Posix 健康探针和基于容量的 GC 的新部署，按[存储流水线](pipeline.md)选择 `Cache|Posix`。两者都可以使用 NFS 挂载，挂载类型不会决定使用哪套 UCM 实现。

V1 NFS 包装器要求 `tensor_size_list` 中的张量大小相等。混合大小布局会报 `PcStore does not support different tensor sizes.`。该列表由引擎适配层提供，修改 YAML 来掩盖布局不匹配不会让模型变得兼容。

从[安装](../../user-guide/quick_start/index.md)获取匹配的 UCM 制品，或在目标引擎和设备环境中[从源码构建](../build_from_source.md)。`ucmpcstore` 由 `ucm/store/pcstore/CMakeLists.txt` 构建，使用 UCM 的设备传输实现；此 connector 不需要 NFS 专用 Python 包。

## 配置 UCM 前缀缓存

准备 `/mnt/ucm-nfs`，为服务用户提供读取、写入、目录遍历和删除权限。NFS 部署应在实际运行推理的容器内确认挂载正常，再创建缓存文件。

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmNfsStore"
    ucm_connector_config:
      storage_backends: "/mnt/ucm-nfs"
      io_direct: false
```

vLLM YAML 中的 `storage_backends` 是路径字符串，适配层将其转换为 Store 接受的列表。后端设置应放在 `ucm_connector_config` 内。初次验证使用一个专用目录，确保写入者和读取者使用同一命名空间。

### 可选参数

| 参数 | V1 默认值 | 作用 |
| --- | --- | --- |
| `io_direct` | `false` | 在挂载和对齐支持时，选择直接文件系统 I/O。 |
| `stream_number` | `8` | 设备传输流数量。 |
| `buffer_number` | `4096` | 中间传输缓冲数量。 |
| `timeout_ms` | `30000` | Store 传输的等待上限，单位毫秒。 |
| `shard_data_dir` | `true` | 将缓存文件分散到子目录。 |

缓冲数量不是 GiB 上限，内存占用还取决于张量传输大小。`cache_buffer_capacity_gb`、`posix_capacity_gb` 和 `store_health` 用于配置 Pipeline 阶段；此包装器不会将这些 key 映射到原生 PcStore 选项。

## 原服务启动与请求命令

在匹配的引擎环境中将上面的 YAML 保存为 `/etc/ucm/ucm.yaml`，使用下面原 Qwen2.5-14B-Instruct 启动参数。`--no-enable-prefix-caching` 为原存储测试保留；正常部署需要同时使用原生前缀缓存时移除此参数。

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
}'
```

服务就绪后，在另一终端运行原请求负载。这里只恢复测试输入，不表示复现了原性能结果：

```bash
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 12 \
--random-input-len 16000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```

## 启动推理

按 [vLLM 快速开始](../../user-guide/quick_start/index.md#vllm)或 [vLLM-Ascend 快速开始](../../user-guide/quick_start/index.md#vllm-ascend)启动，保持引擎侧选择 `UCMConnector`，并将 `UCM_CONFIG_FILE` 指向这份 YAML。模型、并行配置和块大小由引擎设置提供。

确认日志中选中了 `UcmNfsStore` 并初始化了 `ucmpcstore`。HTTP 健康检查只能说明模型服务已就绪，不能证明任何 KV 块已写入 NFS。

## 验证持久化复用

使用空的专用目录执行[外部缓存检查](../../user-guide/quick_start/index.md#vllm-verify-the-service-and-external-cache)。等待写入完成，保留文件，重启服务进程，再重放完全相同的提示词和模型几何配置。必须同时观察到外部缓存命中和已完成的加载。验证跨主机共享时，在第二台主机上用相同模型与配置重复重放，并确保能访问同一挂载。

排查时区分以下结果：

| 现象 | 下一步检查 |
| --- | --- |
| `Failed to initialize ucmpcstore` | 原生扩展与设备依赖、路径权限和实际生效配置。 |
| 只有一台主机上出现文件 | 实际容器挂载和后端 export；相同路径字符串可能指向不同文件系统。 |
| `PcStore does not support different tensor sizes.` | 引擎的 KV 张量布局和所选 connector。 |
| 文件存在，但没有外部命中 | Tokenize 后的前缀、模型修订、dtype、并行布局是否一致，以及块是否完整写入。 |
| 传输超时 | 增大超时或并发前，先检查挂载可达性和存储延迟。 |

评估性能时，应将冷外部读取与操作系统页缓存、引擎内存缓存区分开。报告 TTFT 时同时报告 NFS 流量和外部命中；仅凭重放延迟无法确定使用了哪一层缓存。

[实现与扩展说明](../extending-store.md#backend-entrypoints)。
