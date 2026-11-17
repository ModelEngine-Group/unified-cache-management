# Ds3fs Store

当部署环境已有可用的 3FS 客户端，并希望 UCM 通过原生客户端 I/O API 传输 KV 块时，使用 `Cache|Ds3fs`。Cache Store 在设备内存与主机缓冲之间暂存数据，Ds3fs Store 打开缓存文件，并通过 `hf3fs_usrbio` 提交读写。

普通本地磁盘或 NFS 挂载上的路径不足以支撑这个后端。如果只需要通过已有挂载执行文件系统 I/O，使用 [Cache|Posix](pipeline.md)。

## 准备客户端和原生库

启动 UCM 前，先部署并验证 3FS 服务和客户端。在服务容器内，以运行推理的用户执行客户端检查，包括通过客户端 I/O 接口完成小规模写入和读取。UCM 不负责部署 3FS 集群，也不会启动其客户端。

只有 CMake 同时找到以下依赖时，UCM 源码构建才会生成 `libds3fsstore.so`：

| 依赖 | 默认搜索位置 |
| --- | --- |
| `hf3fs_usrbio.h` | `/usr/local/3fs/src/lib/api` |
| `libhf3fs_api_shared` | `/usr/local/3fs/src/lib/rs/hf3fs-usrbio-sys/lib` |

直接配置 CMake 时，对应缓存变量为 `HF3FS_USRBIO_INCLUDE_DIR` 和 `HF3FS_USRBIO_LIBRARY`。缺少依赖时，构建会输出 `ds3fsstore: Skipping build`，但仍可能生成不含此后端的 UCM 软件包。使用前，确认已安装 `ucm/store/ds3fs/libds3fsstore.so`，并在目标 Linux 主机上用 `ldd` 检查其依赖能否解析。

已发布引擎制品见[安装](../../user-guide/quick_start/index.md)；缺少所需可选库时，按[源码构建](../build_from_source.md)处理。安装的 3FS 头文件和共享库必须与运行中的客户端匹配，本页不保证与任意 3FS 版本兼容。

## 配置前缀缓存

注册名称为 `Cache|Ds3fs`，大小写必须一致。将路径替换为准备好的 3FS 客户端路径后，保存以下 vLLM UCM YAML：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Ds3fs"
      storage_backends: "/mnt/3fs/ucm-cache"
```

初次检查使用单个客户端挂载：Ds3fs 根据 `storage_backends` 的第一个条目初始化 I/O 上下文。`ior_entries` 和 `ior_depth` 定义 3FS ring，两者默认均为 1。`numa_id` 默认是 -1。`stream_number` 控制 Ds3fs 传输 worker，默认是 32。Cache 的设备传输并发通过 `cache_stream_number` 单独配置。

客户端路径验证通过之前，保留默认 ring 设置。主机内存使用默认容量；调整时需遵循[流水线内存规划](pipeline.md#budget-host-memory)，包括最小分片数约束。vLLM 适配层提供块和分片几何信息，构建器用一个分片作为一次 Ds3fs 传输单元；手工替换为其他模型的大小值可能破坏文件布局。

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

## 启动并验证 3FS 路径

按 [vLLM 快速开始](../../user-guide/quick_start/index.md#vllm)或 [Ascend 对应指南](../../user-guide/quick_start/index.md#vllm-ascend)启动，将 `UCM_CONFIG_FILE` 指向上述 YAML。初始化时应能识别 Cache Store 和 Ds3fs Store，并显示预期存储路径。

按顺序验证：

1. 使用新的缓存目录，提交能够生成完整 KV 块的请求，观察 dump 完成以及目标 3FS 命名空间中的数据。
2. 保留文件并重启服务进程，重放完全相同的提示词、模型修订和 KV 几何配置。
3. 必须观察到外部命中 token、成功的 Ds3fs 加载和 3FS 客户端读取活动。Cache Store 的热缓存命中不会经过 3FS 读取路径。
4. 读写检查通过后，再测量 TTFT 和吞吐。

Ds3fs 当前继承 Store 基类不执行实际操作的健康探针。因此，此阶段的 Pipeline 健康状态不能证明 3FS 读写可用，应以实际传输完成情况和客户端观察为依据。

## 排查初始化和传输问题

| 症状 | 首先检查 |
| --- | --- |
| 无法加载 `libds3fsstore.so` | 是否构建了可选目标，以及 3FS 共享库是否可解析。 |
| `Failed to initialize worker context` | 客户端挂载、I/O vector/ring 创建、权限和 NUMA 选择。 |
| `Failed to register fd` | 文件是否属于预期的、正常工作的 3FS 客户端。 |
| I/O 准备、提交或等待出错 | 返回的 3FS 错误码，以及客户端和服务日志。 |
| 文件存在，但重放未命中 | 写入是否完成，提示词、模型、布局是否一致，以及是否访问同一命名空间。 |

不要用 Posix GC 参数管理 3FS 容量，Ds3fs 不解析这些设置。保留策略和容量需要结合 3FS 服务规划。Connector 层面的排查见[故障排查](../../reference/troubleshooting.md)。

[实现与扩展说明](../extending-store.md#backend-entrypoints)。
