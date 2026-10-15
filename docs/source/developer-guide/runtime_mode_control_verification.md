# UCM 在线运行模式切换——自验证报告

对应设计文档：[runtime_mode_control.md](runtime_mode_control.md)

## 1. 验证环境

| 项 | 值 |
| :--- | :--- |
| 主机 | Windows 11（无 GPU / NPU，无 C++ 构建产物） |
| Python | 3.14.7（仓库 `.venv`） |
| pytest | 9.1.1（本次新装） |
| 测试方式 | 沿用 `test/test_ucm_connector_metrics.py` 的 sys.modules stub 方案：伪造 `vllm.*` / `torch` / `prometheus_client` / `ucm.shared.metrics` 等模块后加载真实 connector 源码 |

> 说明：本机为开发机，无 GPU 与已编译的 C++ 扩展，依赖真实引擎/存储的用例（E2E、save_load、kv_cache_layout、
> mooncake、request_hasher）无法本地执行，需在 CI / 部署环境完成（见第 6 节）。本地可运行集合已全部执行。

## 2. 变更清单

| 文件 | 变更 |
| :--- | :--- |
| `ucm/runtime_mode.py` | **新增**。`UCMRuntimeMode` 枚举、`RuntimeModeController` 节流轮询器、配置解析、原子写入工具（约 400 行，引擎无关） |
| `ucm/integration/vllm/ucm_connector.py` | `UCMDirectConnector`：类属性 `supports_runtime_mode_control`、`__init__` 创建控制器、`_refresh_runtime_mode()`、`get_num_new_matched_tokens` 三模式门控（disabled 旁路 + lite 影子）、`get_finished` 保活刷新；门面 `UCMConnector`：选择逻辑重构为 if/elif/else（语义等价）、`_log_runtime_mode_control_support()` |
| `ucm/integration/vllm/hma_connector.py` | `UCMFAWAConnector.supports_runtime_mode_control = False`（不支持声明） |
| `ucm/integration/vllm/hla_connector.py` | `UCMHybridLinearAttentionConnector.supports_runtime_mode_control = False`（不支持声明） |
| `ucm/default_metrics_config.py` | 新增 6 counter + 1 gauge |
| `examples/metrics/metrics_configs.yaml` | 与上同步（现有测试 `test_default_metrics_config_matches_example_yaml` 校验两者一致） |
| `examples/ucm_config_example.yaml` | 新增 runtime-mode 配置段（含注释说明） |
| `test/test_ucm_runtime_mode.py` | **新增**。31 个单元测试（见第 4 节） |
| `docs/source/user-guide/metrics/metrics_list.md` | 指标清单与数量更新（84 C / 15 G / 64 H） |
| `docs/source/developer-guide/runtime_mode_control.md` | **新增**。设计文档 |
| `docs/source/developer-guide/runtime_mode_control_verification.md` | **新增**。本报告 |
| `docs/source/index.md` | Developer Guide 导航加入上述两篇 |

## 3. 预存问题修复（非本特性引入，为使回归可运行）

| 问题 | 根因 | 修复 |
| :--- | :--- | :--- |
| `test/conftest.py` 在 Windows 上所有用例报错 `No module named 'fcntl'` | autouse fixture 在 marker 检查前无条件 `import fcntl`（POSIX-only） | 将 import 移至 marker/NPU 早退分支之后，Linux 行为不变 |
| `test_ucm_connector_metrics.py` 收集错误：stub 缺 `get_current_device_id` | 6c7c1cd7 在 connector 引入该 import 后 stub 未同步 | stub 补充 `get_current_device_id` |
| 同文件 3 用例失败：stub 缺 `MLAAttentionSpec` | hla_connector 新增 import 后 stub 未同步 | stub 补充 `MLAAttentionSpec` |
| 同文件 1 用例失败：fake 模块缺 `UCMFAWALiteConnector` | use_lite 分支新增该 import 后 fake 未同步 | fake namespace 补充 |

## 4. 单元测试结果

命令：`.venv\Scripts\python.exe -m pytest <files> -q`

| 套件 | 结果 |
| :--- | :--- |
| `test/test_ucm_runtime_mode.py`（新增） | **31 passed** |
| `test/test_ucm_connector_metrics.py`（回归） | **71 passed** |
| 两文件同会话（正序 / 反序各一次，验证共享 stub 一致性） | **102 passed / 102 passed** |
| `test/test_logger.py` | 4 passed |
| `test/test_yuanrong_pipeline_builder.py` + `test/test_tier_observability_source.py` | 11 passed |
| 合计（本次验证本地可运行全集，最终一次全量执行） | **117 passed, 0 failed** |

### 需求场景 ↔ 用例映射

| 需求场景 | 用例 | 关键断言 |
| :--- | :--- | :--- |
| disabled：HBM 未命中直接返回未命中、走原计算逻辑 | `test_disabled_answers_miss_without_lookup` | 返回 `(0, False)`；`lookup_on_prefix` 零调用；`requests_meta` 清空（含抢占残留）；bypassed 计数器 =1 |
| disabled → enabled 重新开启 | `test_mode_transition_roundtrip` | 三模式往返：48 命中 → 0 → 0（lite）→ 48 命中恢复，每次切换查表行为与指标正确 |
| lite：仅记录元数据、假装命中、不实际加载/卸载 | `test_lite_records_shadow_hit_and_plans_nothing` | lookup 仍执行（1 次）；`lite_shadow_external_hit_tokens_total=48`；`build_connector_meta` 产出 `load=([],[])`、`dump=([],[])`；`_async_dump_req_ids` 为空 |
| enabled 行为不变（回归） | `test_enabled_returns_external_hits`、`test_ucm_connector_metrics.py` 全量 | 命中 48 tokens、meta/指标与改造前一致 |
| 控制文件容错（半写/非法值/缺失） | `test_invalid_content_keeps_mode_and_retries`、`test_file_appearance_and_disappearance`、`test_missing_file_keeps_initial_mode` | 模式保持 + 错误计数 + 修复后即时生效；文件消失保持最后模式 |
| 轮询节流与 mtime 去重 | `test_mode_switch_observed_after_poll_interval`、`test_unchanged_file_is_not_reread` | 间隔内不切换；签名不变不重读（`_read_mode` 仅调用 1 次） |
| 配置解析（优先级/别名/关闭开关） | `TestRuntimeModeParsing` 7 项 | 配置 > 环境变量 > 默认路径；`runtime_control_enabled=false` / 空路径关闭轮询；非法 `runtime_mode` 启动即报错 |
| 写入原子性与往返 | `test_write_runtime_control_file_is_atomic_json` | JSON 落盘 `{"mode":"disabled"}`；无 `.tmp` 残留；非法模式拒绝 |
| 并发安全 | `test_concurrent_refresh_is_safe` | 4 线程 × 50 次强制刷新无异常 |
| 引擎空闲时可观测 | `test_get_finished_refreshes_mode_in_all_modes` | `get_finished` 三模式下正常返回并刷新模式 |
| 不支持连接器的可发现性 | `TestFacadeRuntimeModeSupport` 5 项 | 支持者不告警；不支持者一次性告警（含类名）；控制关闭/配置非法时行为明确 |
| 指标注册完整性 | `test_runtime_mode_metrics_are_in_default_config` | 7 项新指标均在默认配置中（yaml/py 一致性由既有测试覆盖） |
| 控制器缺失时降级 | `test_no_controller_falls_back_to_enabled` | 删除控制器属性后回落 enabled 行为 |

### 过程中测试发现并修复的缺陷

1. `resolve_runtime_control_settings` 使用 `IntEnum` 数值默认值导致配置缺省键时抛 `ValueError`（`parse_runtime_mode(0)` 不识别）——改为显式 None 判断；枚举随之改为字符串值 `str, Enum`，gauge 改用独立映射 `mode_gauge_value()`，同时修复了 JSON 往返（`{"mode": 2}` 无法解析回模式）问题。
2. `_apply_mode` 在赋值前上报 gauge，导致切换瞬间上报旧模式——调整上报顺序。

## 5. 代码质量检查

| 检查 | 结果 |
| :--- | :--- |
| `python -m py_compile`（4 个改动 py 文件） | 通过 |
| `black --check`（8 个相关文件，版本对齐 .pre-commit-config 24.4.2） | 通过（已格式化） |
| `isort --profile=black --check-only` | 通过 |
| pytest 重复运行（格式化后重跑全量） | 117 passed |

## 6. 限制与后续验证项

| 项 | 说明 |
| :--- | :--- |
| 真机 E2E | 需在 GPU/NPU 环境验证：真实 vLLM 服务在线 `echo disabled` 后 TTFT 恢复无缓存水平、输出与基准一致（含切换瞬间在途请求）、`/metrics` 出现 `ucm:connector_runtime_mode` 且随切换变化 |
| TP 多进程一致性 | 单测以单进程模拟；真机需确认各 rank + scheduler 均在轮询间隔内观察到同一控制文件变化 |
| FAWA / HLA 连接器 | 本期不支持（启动告警）；扩展方式见设计文档 3.3 |
| 本机不可运行用例 | `test_ucm_request_hasher.py`（需 C++ `ucmlogger`）、`test_mooncake.py`（需 torch）、`suites/Unit/test_kv_cache_layout.py`（需 numpy，本机网络原因安装失败）、`test_ucm_connector_save_load.py`（需真实 vllm + GPU）。均与本特性无代码交集（本特性未触碰 KVCacheLayout / hasher / 存储后端），留待 CI |
| codespell | 本机网络不稳未能安装；改动文案为常规英文技术用语 |

## 7. 结论

需求所述三种运行时行为（在线关闭返回未命中、lite 元数据影子模式、在线重新开启）已按设计实现并通过
117 项本地单元测试验证（含 71 项既有套件全量回归）；切换语义保证只影响新调度决策，在途请求安全；
不支持的场景以显式告警与文档声明兜底。建议后续在 GPU 环境补充第 6 节所列集成验证。
