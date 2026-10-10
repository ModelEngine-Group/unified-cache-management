# 故障排查

从失败发生的阶段开始排查，先看实际日志，再核对对应配置和依赖。

| 现象 | 检查入口 |
| --- | --- |
| 安装或初始化失败 | 当前页的安装、配置问题 |
| 服务可用但没有缓存复用 | [外部缓存验证](../user-guide/observability/verify-cache.md) |
| 存储操作被阻止 | [存储健康检查](../user-guide/observability/health-metrics.md) |
| 只有错误码 | [错误码参考](error-codes.md) |

## 常见问题

### 安装与构建

#### 未设置 `PLATFORM` 环境变量

**日志**：

```
WARNING: PLATFORM environment variable is not set!
Please set PLATFORM to one of: cuda, ascend, ascend-a3, musa, maca
```

**原因**：构建 UCM 时需要指定 `PLATFORM`；未设置时默认为仅用于 CI 测试的 `simu` 模拟模式。

**处理方法**：

```bash
export PLATFORM=cuda       # For CUDA
export PLATFORM=ascend     # For Ascend A2
export PLATFORM=ascend-a3  # For Ascend A3
pip install -v -e . --no-build-isolation
```

#### 构建出现 CMake 错误

**原因**：缺少构建依赖，如 `cmake`、`gcc`，或 CUDA toolkit 不在搜索路径中。

**处理方法**：

1. 用 `cmake --version` 确认已安装 `cmake >= 3.18`。
2. 用 `nvcc --version` 确认 CUDA toolkit 可用。
3. 昇腾平台确认 CANN toolkit 已正确安装。

#### `pip install` 构建 Wheel 失败

**原因**：C++ 扩展需要完整的编译环境。使用 `--no-build-isolation` 时，当前环境仍必须具备所需构建工具。

**处理方法**：从[安装](../user-guide/quick_start/index.md)选择已发布的运行时镜像，或包含对应 backend extra 的 Wheel。

---

### 配置问题

#### 错误码 -50000（InvalidParam）

**典型日志**：

```
invalid param ... InvalidParam(...)
```

**常见原因**：

- YAML 语法错误。
- 缺少必填参数。
- 参数类型错误，例如应为整数却提供字符串。

**处理方法**：

1. 用 `python -c "import yaml; yaml.safe_load(open('your_config.yaml'))"` 检查 YAML 语法。
2. 根据 [Pipeline Store](../developer-guide/cache-configuration/pipeline.md)或 [NFS Store](../developer-guide/cache-configuration/nfs.md)核对参数名。
3. 确保 `storage_backends` 是有效且非空的路径字符串。

#### 找不到 `UCM_CONFIG_FILE`

**日志**：

```
UCM config file not found: <path>
```

**原因**：`kv_connector_extra_config` 中的配置路径错误，或文件不存在。

**处理方法**：确认容器或服务进程能够访问该文件。使用 Docker 时，需要将文件挂载或复制到容器内。

#### 不支持的 connector 类型

**日志**：

```
Unsupported connector type: <name>
```

**原因**：`ucm_connector_name` 未注册。

**处理方法**：使用受支持的 connector 名称，如 `UcmPipelineStore` 或 `UcmNfsStore`。

#### 未知 Store 管线

**日志**：

```
unknown store pipeline: <name>
```

**原因**：`store_pipeline` 未在 PipelineStore 注册表中注册。

**处理方法**：使用已注册管线，如 `"Cache|Posix"`、`"Cache|Ds3fs"` 或 `"Cache|Compress|Posix"`。完整注册表见 `ucm/store/pipeline/connector.py`。

---

### 运行时问题

#### 错误码 -50001（OutOfMemory）

**常见原因**：

- `cache_buffer_capacity_gb` 超出可用 `/dev/shm` 容量。
- 同节点多个 DP 实例分别分配 CacheStore buffer。
- MLA 模型需要更多共享内存。

**处理方法**：

1. 用 `free -h` 检查主机内存。
2. 用 `df -h /dev/shm` 检查共享内存。
3. 根据主机容量显式设置正数 `cache_buffer_capacity_gb`。省略时，原生 Cache Store 的共享 buffer 默认为 256 GiB，非共享模式默认为每个 worker 32 GiB；vLLM 共享 buffer 路径会提供 128 GiB。非共享模式需计入主机上的每个 worker。
4. 同节点运行多个 DP 实例时，按实例数量分配容量预算。

#### 错误码 -50002（OsApiError）

**常见原因**：

- `storage_backends` 目录不存在。
- 存储目录权限不足。
- 磁盘 I/O 或文件系统故障。

**处理方法**：

1. 用 `ls <storage_backends_path>` 确认路径存在。
2. 用 `touch <storage_backends_path>/test_file && rm <storage_backends_path>/test_file` 检查写入和删除权限。
3. 对 NFS，用 `mount | grep <path>` 检查挂载。
4. 用 `dmesg | tail` 或 `journalctl -xe` 检查系统日志。

#### 错误码 -50009（NoSpace）

**常见原因**：

- `storage_backends` 所在磁盘已满。
- `posix_capacity_gb` 超出实际磁盘容量。
- 未启用 GC，或回收速度不足。

**处理方法**：

1. 用 `df -h <storage_backends_path>` 检查空间。
2. 将 `posix_capacity_gb` 设在本部署可用的容量预算内，为文件系统的其他用户预留空间。
3. `posix_capacity_gb > 0` 时自动启用 GC，确认设置符合预期。

#### 错误码 -50010（Timeout）

**常见原因**：

- 网络或存储带宽异常，如 NFS 挂载性能下降。
- CPU 或 I/O 负载过高。
- `timeout_ms` 对当前工作负载过小。

**处理方法**：

1. 检查存储延迟、I/O 错误和 UCM 任务耗时。需要压测时，使用专门的测试目录及已批准的存储负载。
2. 用 `top` 或 `htop` 检查系统负载。
3. 网络 NFS 场景用 `ping` 和 `iperf` 检查网络。
4. 实测操作确实需要更长时限时，在 `ucm_connector_config` 内调整 `timeout_ms`（默认 30000 毫秒）；YAML 根节点的同名参数不会设置 Store I/O 超时。
5. 对 NFS，先结合客户端、服务端日志和存储部署要求检查挂载参数，再决定是否更改缓存或一致性设置。

---

### vLLM 集成问题

#### 前缀缓存没有命中

**日志**：

```
request_id: xxx, total_blocks_num: N, hit hbm: 0, hit external: 0
```

**原因**：首次请求没有可复用的外部块。后续请求还要求先前写入成功、token 前缀和模型标识匹配、提示词包含足够多的完整块，并使用相同缓存块布局。

**处理方法**：

1. 核对 `UCM_CONFIG_FILE` 路径及启动日志中的生效 Store 配置。
2. 确认写入完成，保留存储并重启引擎，再发送完全相同的多块提示词。参见[外部缓存验证](../user-guide/quick_start/index.md#vllm-verify-the-service-and-external-cache)。
3. 分别检查 UCM 命中 token、load 活动和原生 HBM 缓存命中；原生前缀缓存与外部 UCM 复用是不同路径。

#### `Unsupported device platform for UCMDirectConnector`

**原因**：当前平台既不是 CUDA，也不是昇腾 NPU。

**处理方法**：此错误描述的是当前选用的 vLLM direct connector 路径。应选择与设备匹配的 connector 和引擎，不能据此认为所有 UCM 组件都仅支持这两个平台。核对[支持矩阵](../user-guide/support-matrix/index.md)及所选构建。

#### KV dump/load 错误

**典型日志**：

```
dump kv cache failed. <error>
wait for dump kv cache failed. <error>
submit dump task failed. <error>
```

**原因**：Store 后端（Posix/NFS）在 KV cache 传输中遇到 I/O 错误。

**处理方法**：

1. 检查存储空间、权限和网络等后端健康状态。
2. 设置 `UC_LOGGER_LEVEL=debug` 查看传输日志：
   ```
   [UC][D] Cache task(...) dispatching.
   [UC][D] Posix task(...) dispatching.
   ```
3. 按[错误码参考](error-codes.md)定位具体错误。

---

### 日志调试

查看详细 UCM 传输日志时，设置：

```bash
export UC_LOGGER_LEVEL=debug
```

日志记录任务 ID、操作类型、数据大小和耗时：

```
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```

| 字段 | 含义 |
| --- | --- |
| `task_id` | Store 任务唯一标识 |
| `operation` | Cache Store 的 `DUMP`（设备→主机）或 `LOAD`（主机→设备）；Posix Store 的 `Cache2Backend`（写入存储）或 `Backend2Cache`（从存储加载） |
| `subtask_number` | 本次操作的子任务数量 |
| `size` | 传输的数据总字节数 |
| `cost` | 执行耗时，单位为毫秒 |

自定义日志目录：

```bash
export UCM_LOG_PATH=my_log_dir
```

默认日志位于启动 vLLM 服务时所在目录下的 `log` 子目录。
