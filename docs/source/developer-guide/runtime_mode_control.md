# UCM 在线运行模式切换（Runtime Mode Control）设计文档

## 1. 背景与需求

UCM 通过 `kv_transfer_config` 以 KV Connector 插件形式接入 vLLM / vllm-ascend 等推理引擎，实现 KV Cache
向外置存储的卸载（dump）与回填（load）。当前一旦在部署时启用了 UCM，就无法在服务运行期间关闭——要绕开
UCM 必须修改配置并重新拉起服务。

本特性目标：**部署时启用了 UCM 的推理引擎，可以在运行时由用户"在线"切换 UCM 的工作模式，无需重启**。

| 模式 | 行为 |
| :--- | :--- |
| `enabled` | （默认，现有行为）HBM 未命中时查询外置存储，命中则加载 KV 到 HBM；计算后卸载新 KV |
| `disabled` | 完全旁路：HBM 未命中后**直接返回未命中**，继续引擎原有计算逻辑；不查询外置存储、不规划任何加载与卸载 |
| `lite` | 影子模式：仍然执行哈希与元数据查询（lookup），**记录"假命中"统计**（as-if hit rate），但不实际执行 KV 数据的卸载与加载；引擎照常全量重算 |

`lite` 模式回答了"如果 UCM 一直开着，命中率会是多少"——可用于在关闭缓存的同时持续评估重新开启的收益。

## 2. 可行性分析

### 2.1 vLLM V1 Connector 的双角色架构

vLLM v1 的 KV Connector 在两个角色上各有一个实例（`KVConnectorRole.SCHEDULER` / `WORKER`）：

- **调度侧（决策点）**：`get_num_new_matched_tokens(request, num_computed_tokens)` 在 HBM 前缀未命中后调用，
  返回 `(external_hit_tokens, load_async)`。调度器把 `external_hit_tokens` 计入 `num_computed_tokens`，
  并据此跳过这些 token 的计算；`build_connector_meta` 随后生成 load/dump 计划（`RequestDispatchMeta`）下发给
  worker。
- **工作侧（执行点）**：`start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` 严格按
  connector metadata 执行 KV 数据搬运，不做任何自主决策。

### 2.2 三个关键推论

1. **"关闭"必须发生在调度侧决策点。** 若调度侧返回 `(0, False)`，vLLM 视为外置存储未命中，全量重算，
   输出正确性与未部署 UCM 时一致。反之，若只在 worker 侧拦截（调度侧仍上报命中但 worker 不加载），
   被跳过计算的 token 会读到未初始化的 KV 块，产生**错误输出**。
2. **模式切换只需影响"新的调度决策"。** worker 侧由 metadata 驱动：调度侧已经规划并计入
   `num_computed_tokens` 的 load 必须继续执行，否则同样会产生错误输出。因此设计上 **worker 侧行为完全不变**，
   切换天然只对切换之后新调度的请求生效，在途请求自然收敛，不存在"半执行"的一致性问题。
3. **调度侧与 worker 侧（以及 TP 各 rank）是多个进程。** 开关信号必须跨进程可见。对比三种机制：

   | 机制 | 评估 |
   | :--- | :--- |
   | 信号（SIGUSR1/2） | 多进程投递繁琐，Windows 不可用，语义扩展困难 |
   | HTTP API | 需要打通 api_server → engine core 的 IPC，侵入 vLLM，sglang/MindIE 无法复用 |
   | **控制文件轮询** | 文件系统天然跨进程共享；对引擎零侵入；对各引擎（vLLM/vllm-ascend/sglang/MindIE）通用；代价是切换延迟 ≤ 轮询间隔（默认 1s） |

   选择控制文件轮询。

### 2.3 风险分析与对策

| 风险 | 对策 |
| :--- | :--- |
| 切换瞬间在途请求的 load 被跳过 | 不可能发生：worker 侧不感知模式，metadata 驱动（见 2.2-2） |
| 抢占（preemption）后重查请求残留旧 `RequestMeta`，规划出错误的 load/dump | `disabled` 分支在返回 miss 前主动清除该请求的调度侧状态（`requests_meta`、异步 load 派发表） |
| 控制文件写一半被读到 | 解析失败保持当前模式，且不缓存失败签名，下一轮询自动重试；官方写入工具使用临时文件 + `os.replace` 原子替换 |
| 控制文件被误写非法值 | 同上，保持当前模式并限频告警、计入 `connector_runtime_mode_control_read_errors_total` |
| `lite` 模式"假装命中"导致输出错误 | 不会：`lite` 仅在 UCM 自身指标中记录 as-if 命中，对引擎仍返回 `(0, False)`，引擎全量重算保证正确性 |
| 异步 dump / 异步 load 在途任务 | 由既有 `get_finished` / `_poll_pending_*` 机制自然回收，模式无关 |
| 多 rank / 多进程模式视图短暂不一致 | 每个进程独立轮询同一文件，1 个轮询间隔内收敛；仅调度侧进程做决策，worker 侧视图不影响行为 |

## 3. 方案设计

### 3.1 模块与状态机

新增引擎无关模块 `ucm/runtime_mode.py`：

```
UCMRuntimeMode(str, Enum): enabled | lite | disabled
mode_gauge_value(): 0.0 | 1.0 | 2.0   # 供 gauge 指标使用

RuntimeModeController
  ├── from_launch_config(launch_config, engine_id)   # 解析配置构造
  ├── refresh(now=None, force=False) -> UCMRuntimeMode  # 节流轮询，永不抛异常
  ├── mode / control_file / poll_interval_s
  └── 内部: (mtime_ns, size) 签名去重、双次读取容错、线程锁、
            切换/错误限频日志、模式切换指标
```

- **控制文件内容**：JSON `{"mode": "disabled"}`（允许多余键）或裸字符串 `"disabled"`；大小写不敏感；支持别名
  （`on/off/true/false/enable/disable/1/0/full/shadow/trace/bypass`）。
- **路径解析优先级**：`runtime_control_file`（配置，显式 `""` 表示关闭）> 环境变量
  `UCM_RUNTIME_CONTROL_FILE` > 默认 `<tempdir>/ucm_runtime_control_<engine_id>.json`；
  `runtime_control_enabled: false` 可整体关闭轮询并将模式固定为启动值。
- **轮询节流**：默认间隔 `runtime_control_poll_interval_s: 1.0`；签名（mtime+size）未变化不重复读文件；
  文件从缺失到出现、从出现到消失均能被检测（消失时保持最后观测模式）。
- **指标依赖惰性导入**：`ucm.shared.metrics`（C++ 扩展）在函数内延迟导入，保证模块可独立于构建产物被测试
  与复用（sglang / MindIE 后续可复用同一控制器）。

### 3.2 门控点（`ucm/integration/vllm/ucm_connector.py`）

```
UCMDirectConnector.__init__
  └── self._runtime_mode_controller = RuntimeModeController.from_launch_config(...)
      （启动时打印 initial_mode / control_file / poll_interval_s）

get_num_new_matched_tokens          ← 唯一行为门控点（调度侧）
  ├── DISABLED → 清理该请求调度侧残留状态（含抢占残留）
  │             计 connector_runtime_mode_bypassed_requests_total
  │             return (0, False)                       # 引擎原路重算
  ├── LITE     → 正常哈希 + lookup（元数据查询）
  │             计 lite_shadow_* 影子指标
  │             记录 total_hit==hbm_hit、token_processed==num_token_ids 的
  │             RequestMeta（⇒ build_connector_meta 生成的 load/dump 均为空）
  │             return (0, False)                       # 引擎原路重算
  └── ENABLED  → 现有行为不变

get_finished                         ← 空闲保活刷新（每调度步调用，节流）
worker 侧（start_load_kv / wait_for_save / get_finished 轮询任务等）
  └── 不做模式判断（见 2.2-2）
```

`lite` 的"假命中"通过两个精心构造的 `RequestMeta` 字段实现空计划：

- `total_hit_block_num == hbm_hit_block_num` → `_generate_dispatch_meta` 的 load 区间为空；
- `token_processed == num_token_ids` → dump 区间为空。

### 3.3 连接器支持声明

`UCMDirectConnector` 增加类属性 `supports_runtime_mode_control = True`；`UCMLayerWiseConnector` /
`UCMCPConnector` / `UCMMockConnector`（经 super() 委托）自动继承。重写了
`get_num_new_matched_tokens` 且不经过基类的 `UCMFAWAConnector`（hma_connector.py）与
`UCMHybridLinearAttentionConnector`（hla_connector.py）显式置 `False`。门面 `UCMConnector` 在初始化末尾
调用 `_log_runtime_mode_control_support()`：当运行时控制开启而内层连接器不支持时打印一次性告警，
避免"改了文件却无效果"的无声失败。

### 3.4 切换时序语义

```
时间 ────────────────────────────────────────────────────────►
          [mode=enabled]      [写控制文件 disabled]      [mode=disabled]
请求 A:  ── 调度(命中100) ── load执行 ── dump执行 ────          （在途，按已规划执行）
请求 B:                        ── 调度 ──────── 全量重算 ──      （新决策，返回 miss）
```

- 模式切换 ≤ `poll_interval_s`（默认 1s）内被调度侧观察到；
- 切换只影响其后**首次进入** `get_num_new_matched_tokens` 的调度决策；
- 重新 `enabled` 后，此前因 `disabled` 而未持久化的请求不会补 dump（缓存本就是尽力而为语义），
  新请求恢复正常的查询/加载/卸载。

### 3.5 可观测性

新增默认指标（`ucm/default_metrics_config.py` 与 `examples/metrics/metrics_configs.yaml` 同步维护，
Grafana 前缀 `ucm:`）：

| 类型 | 指标 | 含义 |
| :--- | :--- | :--- |
| counter | `connector_runtime_mode_bypassed_requests_total` | 因 disabled 被回答为全量未命中的请求数 |
| counter | `connector_runtime_mode_lite_requests_total` | 以 lite 影子模式评估的请求数 |
| counter | `connector_runtime_mode_transitions_total` | 观测到的模式切换次数 |
| counter | `connector_runtime_mode_control_read_errors_total` | 控制文件读取/解析失败次数 |
| counter | `lite_shadow_lookup_blocks_total` | lite 模式下发起的块哈希查询数 |
| counter | `lite_shadow_external_hit_tokens_total` | lite 模式记录的 as-if 命中 token 数 |
| gauge | `connector_runtime_mode` | 当前模式（0=enabled，1=lite，2=disabled） |

日志：启动配置行、模式切换行（含旧/新模式与文件路径）、读错误限频告警、不支持连接器的一次性告警。

### 3.6 配置与使用

```yaml
# UCM launch config（YAML 文件或 kv_connector_extra_config 内联）
runtime_mode: "enabled"            # 启动初始模式: enabled | lite | disabled
runtime_control_file: ""           # 空 = 关闭轮询；缺省 = /tmp/ucm_runtime_control_<engine_id>.json
runtime_control_enabled: true      # false = 固定为启动模式
runtime_control_poll_interval_s: 1.0
```

在线切换（无需重启）：

```bash
echo '{"mode": "disabled"}' > /tmp/ucm_runtime_control_<engine_id>.json   # 关闭
echo lite           > /tmp/ucm_runtime_control_<engine_id>.json           # 影子模式
echo enabled        > /tmp/ucm_runtime_control_<engine_id>.json           # 重新开启
```

完整示例见 `examples/ucm_config_example.yaml`。

## 4. 范围与限制

| 范围 | 状态 |
| :--- | :--- |
| UCMDirectConnector 家族（direct / layerwise / CP / mock） | 支持（单测覆盖） |
| UCMFAWAConnector（DeepSeek-V4 混合注意力） | 暂不支持，启动告警；扩展点为其 `get_num_new_matched_tokens` 顶部加同款门控 |
| UCMHybridLinearAttention*（mamba/混合线性注意力） | 同上 |
| 部署期 `use_lite`（UCMLiteConnector 轨迹模式）/ Monitor 连接器 | 不适用（本身即无 IO 行为） |
| sglang / MindIE 集成 | 控制器已引擎无关，可后续接入 |

安全提示：默认控制文件位于系统临时目录（容器内通常即容器私有 `/tmp`）。多租户宿主机上建议通过
`runtime_control_file` 指定受控路径，或以 `runtime_control_enabled: false` 关闭在线控制。

## 5. 测试策略

- **纯单元**（`test/test_ucm_runtime_mode.py::TestRuntimeModeParsing/TestRuntimeModeController`）：
  别名/JSON/裸字符串解析、非法值、路径与环境变量优先级、轮询节流、签名去重、坏文件容错、文件出现/消失、
  原子写入、并发安全。
- **连接器门控**（`TestConnectorRuntimeModeGating`）：enabled 命中路径回归、disabled 全量 miss 且不查
  外置存储且清理抢占残留、lite 影子指标 + 空计划（经 `build_connector_meta` 验证）、三模式往返切换、
  `get_finished`/`request_finished` 语义不变。
- **门面与注册**（`TestFacadeRuntimeModeSupport` / `TestRuntimeModeMetricsRegistration`）：
  支持声明、告警触发条件、新指标已注册于默认配置。
- 现有套件（`test/test_ucm_connector_metrics.py` 71 项）全量回归。

详细结果见[自验证报告](runtime_mode_control_verification.md)。
