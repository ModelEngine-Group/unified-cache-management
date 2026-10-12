# runtime-mode

`runtime-mode` 用于在不重启的情况下在线切换 UCM 接入引擎的运行模式。通过 `kv_transfer_config` 部署的引擎会轮询一个轻量控制文件；向该文件写入新模式后，在一个轮询间隔（默认 1s）内生效，运维人员可以在服务持续运行期间关闭、重新开启 UCM，或切换到影子模式。

适用场景：

- 怀疑 UCM 影响延迟或吞吐，需要不重启地临时关闭。
- 关闭 KV 加载/卸载的同时，持续观测"如果开启 UCM 命中率会是多少"（`lite` 影子模式）。
- 维护窗口或故障处置结束后，不重启地恢复 UCM。

工具写控制文件采用原子方式（临时文件 + rename），引擎轮询时不会读到写了一半的内容。

← 返回 [UCM Toolkit](../index.md)

## 模式说明

| 模式 | 行为 |
| --- | --- |
| `enabled` | 完整行为（默认）：HBM 未命中时查询外置存储，命中则加载 KV；计算后卸载新 KV。 |
| `lite` | 影子模式：仍执行哈希与元数据查询并记录 as-if 命中率，但不实际加载/卸载 KV 数据；引擎照常全量重算。 |
| `disabled` | 旁路：外置查询直接返回未命中，不规划任何加载与卸载；引擎走原有计算路径。 |

模式切换只影响新的调度决策，已规划的加载/卸载会正常执行完，在途请求的正确性不受影响。

## 依赖

- 仅依赖 Python 标准库。
- 目标引擎需运行支持运行模式控制的 UCM 连接器（`UCMDirectConnector` 家族：direct / layerwise / CP / mock），且未通过 `runtime_control_enabled: false` 关闭在线控制。

## 控制文件路径解析

`set` 与 `get` 按以下顺序解析控制文件，与引擎侧一致：

1. `--control-file PATH`（显式指定）；
2. 环境变量 `UCM_RUNTIME_CONTROL_FILE`；
3. 默认路径 `<临时目录>/ucm_runtime_control_<engine_id>.json`。

`--engine-id` 接受 vLLM `kv_transfer_config` 中的 engine id；结尾的 `_dpN` 数据并行后缀会被忽略，与连接器约定一致。若引擎在 UCM 启动配置里显式配置了 `runtime_control_file`，此处需传相同的 `--control-file PATH`。

## 命令

### set

向引擎的控制文件写入模式：

```bash
ucm-toolkit run runtime-mode set disabled
ucm-toolkit run runtime-mode set lite --engine-id engine-0
ucm-toolkit run runtime-mode set enabled --engine-id engine-0
ucm-toolkit run runtime-mode set disabled --control-file /var/run/ucm/control.json
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `mode` | 必填 | `enabled`、`lite` 或 `disabled`（接受 `on`、`off`、`shadow`、`bypass` 等别名）。 |
| `--engine-id ID` | 无 | 用于推导默认控制文件路径的引擎 id；结尾的 `_dpN` 会被忽略。 |
| `--control-file PATH` | 按规则解析 | 显式控制文件路径，优先级高于 engine id 与环境变量。 |

命令会输出旧模式（若存在可读的控制文件）、新模式以及写入的文件路径。退出码 2 表示模式非法；1 表示写入失败。

### get

读取控制文件中的当前模式：

```bash
ucm-toolkit run runtime-mode get --engine-id engine-0
```

输出文件路径与解析出的模式。退出码 1 表示文件不存在或内容非法（此时候擎保持当前模式不变）。

### list

发现系统临时目录中的控制文件并显示各自模式：

```bash
ucm-toolkit run runtime-mode list
```

```text
control files in /tmp:
  ucm_runtime_control_engine-0.json          disabled
  ucm_runtime_control_ucm.json               (unreadable)
```

控制文件只在首次设置模式后（或引擎配置了显式 `runtime_control_file` 时）才会存在，全新部署时 `list` 为空是正常现象。

## 典型工作流

```bash
# 引擎以 UCM enabled 模式运行（engine id 为 engine-0）：

# 1. 不重启关闭 UCM（旁路，全量重算）：
ucm-toolkit run runtime-mode set disabled --engine-id engine-0

# 2. 之后评估重新开启的收益（影子模式，无 KV IO）：
ucm-toolkit run runtime-mode set lite --engine-id engine-0

# 3. 结合 metrics-view 观察 as-if 命中率，确认后重新开启：
ucm-toolkit run runtime-mode set enabled --engine-id engine-0

# 随时查看当前生效模式：
ucm-toolkit run runtime-mode get --engine-id engine-0
```

## FAQ

### 写入模式后引擎没有反应

- 确认引擎连接器支持运行模式控制；不支持的连接器（FAWA / 混合线性注意力）启动时会打印一次性告警。
- 用 `get` 确认工具与引擎解析到了同一个文件路径（检查引擎的 `runtime_control_file` 或默认路径）。
- 引擎在轮询间隔（`runtime_control_poll_interval_s`，默认 1s）内才会观察到变化。
- 查看 `connector_runtime_mode_control_read_errors_total` 指标：控制文件内容可能非法。

### 共享宿主机上的安全提示

默认控制目录是系统临时目录（容器内通常即容器私有 `/tmp`）。多租户宿主机上，建议为引擎配置受控路径下的显式 `runtime_control_file`，或以 `runtime_control_enabled: false` 关闭在线控制。
