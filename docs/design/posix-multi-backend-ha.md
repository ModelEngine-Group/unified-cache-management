# Posix Store 多后端探测与高可用

多个 `storage_backends` 必须是**同一份数据的不同访问路径**。Posix Store 维护可用后端列表，业务 I/O 轮询选路，后台探测负责摘除故障路径、恢复正常路径。

**执行流程**

1. **初始化与选路**：检查各路径的数据目录，将检查成功的路径加入可用列表；至少一条可用即可启动。每次选择后端时，按列表顺序轮询；列表为空则返回 `StoreUnhealthy`。列表更新和选路通过同一把锁同步。
2. **逐路径探测**：配置多个不同路径时，每条路径启动独立监控线程，复用 `HealthCheckExecutor` 执行小 I/O。所有路径都持续探测，包括已摘除的路径；一条路径超时不会阻塞其他路径的探测。
3. **更新健康状态**：每条路径独立记录最近 N 次探测结果。窗口内失败次数达到阈值即摘除，无需等窗口填满；摘除后，必须取得一个完整、全部成功的窗口才重新加入。状态切换时重建可用列表、重置轮询位置并打印日志。
4. **对外健康检查**：多后端模式下，`check_health` 读取可用列表，任意一条可用即返回成功，全部不可用才返回 `StoreUnhealthy`。单后端模式保留调用时实际执行小 I/O 探测的行为。

**一次探测做什么**

在指定后端的数据分片目录内，用随机文件名创建临时文件：写入 **4 KiB** 固定内容 → 读回并逐字节校验 → 关闭、删除文件。探测沿用 `io_direct` 配置；开启时使用对齐缓冲区和 `O_DIRECT`，关闭时写完先执行 `fsync`。读写、校验、清理失败或执行超时，均记为一次失败。

健康窗口按**探测次数**统计，业务 I/O 的成败不会直接计入该窗口。例如窗口 8 次、失败阈值 2 次时，两次失败可以不连续；恢复则需要连续 8 次成功。

**配置与默认值**

后端探测与 Pipeline 健康检测共用 `StoreHealthConfig`，通过 `ucm_connector_config.store_health` 配置：

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: Cache|Posix
      storage_backends: "/mnt/path0:/mnt/path1:/mnt/path2"
      posix_io_engine: aio
      io_direct: true
      store_health:
        enabled: true
        health_check_interval_s: 10
        health_check_timeout_s: 3
        health_window_size: 8
        failure_threshold: 2
use_layerwise: true
```

上述健康参数均为默认值，可省略。超时必须小于探测间隔，失败阈值不得超过窗口大小，各数值必须为正。`enabled` 控制 Pipeline 外层的 `HealthBreakerStore`；即使设为 `false`，多后端内部探测仍然运行。

**行为边界**

选路使用最近一次探测形成的可用列表，故障发生到摘除之间存在检测延迟。已下发的业务 I/O 不会自动换路径重试；摘除影响后续选路。文件提交时，临时文件与最终文件使用同一条挂载路径完成重命名。

探测超时表示停止等待本次结果，不会取消已进入内核的 I/O。每次探测使用独立文件名，避免与尚未结束的探测冲突；执行器限制未结束任务数量，销毁时仍需等待这些任务退出。

**源码入口**

- [space_layout.cc](../../ucm/store/posix/cc/space_layout.cc)：选路、探测 I/O、后台循环及 `CheckHealth`。
- [health_window.h](../../ucm/store/detail/health_window.h)：滑动窗口、摘除与恢复条件。
- [health_check_executor.h](../../ucm/store/detail/health_check_executor.h)：探测执行与超时等待。
- [store_health_config.h](../../ucm/store/detail/store_health_config.h)：共用默认值与参数校验。
