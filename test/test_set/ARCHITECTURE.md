# Test Set Runner - 架构文档

> 本文档面向 Code Agent，用于理解系统设计以便后续开发和维护。

## 系统概览

```
┌──────────────────────────────────────────────────────────┐
│                    run_test_set.py                        │
│                                                          │
│  CLI (argparse)                                          │
│       │                                                  │
│       ▼                                                  │
│  Config Loader ──── run_config.yaml                      │
│       │                                                  │
│       ▼                                                  │
│  Test Set Loader ── test_set_config/test_set.yaml        │
│       │                                                  │
│       ▼                                                  │
│  Image Resolver                                          │
│   ├─ override_image (直接使用完整镜像 URL)                │
│   └─ get_image_url() (通过 Harbor API 查询)              │
│       │                                                  │
│       ▼                                                  │
│  Parameter Builder                                       │
│   ├─ PLATFORM_MAP (平台映射)                              │
│   └─ build_pipeline_params() (参数构建)                   │
│       │                                                  │
│       ▼                                                  │
│  Executor (parallel/sequential)                          │
│   ├─ trigger builds (失败时 abort 已触发的 build)         │
│   ├─ wait & collect results (支持 Ctrl+C 中断)           │
│   └─ abort_all() (异常/中断时中止所有 build)              │
│       │                                                  │
│       ▼                                                  │
│  Result Reporter (汇总表含 build number)                  │
└──────────┬───────────────────────────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐    ┌──────────────────┐
│   jenkins_sdk.py     │    │   Harbor API      │
│                      │    │   (镜像查询)      │
│  JenkinsPipelineClient│    └──────────────────┘
│  PipelineParameters  │
│  BuildInfo           │
│  BuildStatus         │
└──────────────────────┘
           │
           ▼
     Jenkins REST API
```

## 数据流

```
1. CLI args + run_config.yaml
        │
        ▼ (merge, CLI 优先)
2. Merged Config
   ├─ jenkins: {url, username, api_token, job_name}
   ├─ harbor: {url, auth_token, project}
   ├─ test_set_config: path
   ├─ parallel: bool
   └─ test_build: {package_name, override_image}
        │
        ▼
3. Load test_set_config yaml → List[test_set]
        │
        ▼ (--test 筛选)
4. Filtered test_sets
        │
        ▼ (for each test_set)
5. Image Resolution:
   ├─ override_image[platform] 存在 → 直接使用完整 URL
   └─ 否则 → get_image_url(platform, package_name, harbor_cfg) → 从 Harbor 查询
        │
        ▼
6. Parameter Building:
   ├─ platform (docker suffix) → PLATFORM_MAP → Jenkins PLATFORM
   ├─ image URL → OVERRIDE_IMAGE
   └─ test_set fields → build_pipeline_params() → PipelineParameters
        │
        ▼
7. Print all parameters (dry-run 到此结束)
        │
        ▼
8. JenkinsPipelineClient.trigger(params) → build_number
   (并行触发, 任一失败则 abort_all 已触发的 build 并退出)
        │
        ▼ (ThreadPoolExecutor if parallel)
9. JenkinsPipelineClient.wait_for_completion(build_number, shutdown_event) → BuildInfo
   (支持 Ctrl+C 中断, 通过 shutdown_event 即时响应)
        │
        ▼
10. print_results() → 汇总表 (含 Build #, Status, Duration) + exit code
```

## 核心函数

### run_test_set.py

| 函数 | 职责 |
|---|---|
| `main()` | 入口，编排完整流程，注册 SIGINT handler |
| `parse_args()` | CLI 参数解析 |
| `load_run_config(path)` | 加载 run_config.yaml |
| `load_test_sets(path, base_dir)` | 加载 test_set 配置 |
| `get_image_url(platform, filter_string, harbor_cfg)` | 通过 Harbor API 查询匹配的镜像 URL |
| `get_override_image(platform, overrides, package, harbor_cfg)` | 确定镜像 URL（优先 override，其次 Harbor 查询） |
| `build_pipeline_params(test_set, platform, override_image_url)` | test_set 配置 → PipelineParameters |
| `abort_all(triggered_builds)` | 中止所有已触发的 Jenkins build |
| `print_results(results)` | 打印结果汇总表（含 Build #） |
| `print_trigger_results(triggered)` | 打印触发结果（--no-wait 模式） |
| `format_duration(ms)` | 毫秒 → 可读时间格式 |

### jenkins_sdk.py

| 类/方法 | 用途 |
|---|---|
| `JenkinsPipelineClient(url, user, token, job_name)` | Jenkins 客户端（普通 Pipeline 模式） |
| `client.trigger(params, **overrides)` | 触发 build，返回 build number |
| `client.wait_for_completion(build_number, shutdown_event=)` | 等待 build 完成，支持 shutdown_event 中断 |
| `client.abort_build(build_number)` | 中止运行中的 build |
| `client.get_build_info(build_number)` | 获取 build 详细信息 |
| `client.get_build_status(build_number)` | 获取 build 当前状态 |
| `client.get_console_log(build_number)` | 获取控制台日志 |
| `client.stream_console_log(build_number)` | 流式获取控制台日志 |
| `client.list_artifacts(build_number)` | 列出构建产物 |
| `client.download_artifacts(build_number, output_dir)` | 下载构建产物 |
| `PipelineParameters` | Jenkins 参数 dataclass |
| `BuildInfo` | Build 结果 dataclass（含 number, status, duration_ms 等） |
| `BuildStatus` | 状态枚举 (SUCCESS, FAILURE, UNSTABLE, ABORTED, ...) |

## 配置文件 Schema

### run_config.yaml

```yaml
jenkins:                          # Jenkins 连接配置
  url: str                        # 必填
  username: str                   # 必填
  api_token: str                  # 必填
  job_name: str                   # 必填, folder/pipeline 路径

harbor:                           # Harbor 镜像仓库配置
  url: str                        # Harbor 地址
  auth_token: str                 # Basic auth token
  project: str                    # Harbor 项目名

test_set_config: str              # test_set 配置文件路径 (相对或绝对)
parallel: bool                    # 是否并行执行, 默认 true

test_build:
  package_name: str               # 包名, 用于从 Harbor 查询镜像
  override_image:                 # 可选, 按平台直接指定完整镜像 URL
    <platform>: str               # key = Jenkins PLATFORM, value = 完整镜像 URL
```

### test_set_config (test_set.yaml)

```yaml
test_sets:
  - name: str                     # 唯一标识
    platform: str                 # docker 后缀: vllm_gpu | vllm_npu | mindie_llm | sglang_gpu

    server_start_config:
      gpu_count: int
      node_count: int             # >1 表示多节点 (DEPLOY_MODE=multi)
      server_port: int
      master_start_command: str
      slave_start_command: str    # 可选, 多节点时使用
      ucm_config: str             # YAML 字符串
      environment:                # 可选
        - KEY: "value"

    pytest_config:
      api_model_name: str
      test_params: str            # pytest 参数
```

## test_set → PipelineParameters 映射

```
test_set field              → PipelineParameters field
─────────────────────────────────────────────────────
name                        → BUILD_NAME
platform (经 PLATFORM_MAP)  → PLATFORM
(resolved image URL)        → OVERRIDE_IMAGE
gpu_count                   → GPU_COUNT
node_count > 1 ? multi      → DEPLOY_MODE
server_port                 → SERVER_PORT
master_start_command        → VLLM_COMMAND_MASTER
slave_start_command         → VLLM_COMMAND_WORKER
ucm_config                  → UCM_CONFIG_YAML
api_model_name              → API_MODEL_NAME
api_model_name              → MODEL_FOLDER_NAME
test_params                 → TEST_PARAMS
```

注意：`OVERRIDE_IMAGE` 优先级高于 `ImageTag`，当前 run_test_set.py 始终使用 `OVERRIDE_IMAGE` 传递完整镜像 URL。

## 平台映射 (PLATFORM_MAP)

```python
{
    "vllm_gpu":    "vllm-cuda",
    "vllm_npu":    "vllm-ascend",
    "mindie_llm":  "mindie",
    "sglang_gpu":  "sglang-cuda",
}
```

来源：`docker/Dockerfile.<suffix>` 的后缀。新增平台需同时更新此映射。

## 镜像解析流程

```
override_image[platform] 存在?
    ├─ 是 → 直接使用完整镜像 URL
    └─ 否 → 调用 get_image_url(platform, package_name, harbor_cfg)
             ├─ 拼接 Harbor API URL: /api/v2.0/projects/{project}/repositories/ucm-{platform}/artifacts
             ├─ 用 package_name 做大小写不敏感的 tag 过滤
             └─ 返回第一个匹配的完整镜像 URL
```

## 错误处理与中断机制

### Ctrl+C (SIGINT) 处理

1. 注册 `signal.SIGINT` handler，设置 `threading.Event` (shutdown_event)
2. 第一次 Ctrl+C：设置 shutdown_event，触发优雅关闭
3. 第二次 Ctrl+C：`os._exit(1)` 强制退出
4. `wait_for_completion()` 使用 `shutdown_event.wait(timeout=poll_interval)` 替代 `time.sleep()`，可被即时唤醒

### 触发阶段错误

- 并行触发时，单个 build 触发失败不会阻塞其他 future
- 任一触发失败后，设置 `trigger_failed` 标志
- 遍历完所有 future 后，调用 `abort_all()` 中止所有已触发的 build

### 等待阶段错误

- 异常或 shutdown_event 被设置时，调用 `abort_all()` 中止所有已触发的 build

### Jenkins 错误响应

- `_extract_error_message()` 从 HTML/JSON 响应中提取可读错误信息
- `_raise_for_status()` 替代 `resp.raise_for_status()`，避免输出大段 HTML

## 扩展点

### 1. 新增平台

1. 在 `docker/` 下添加 `Dockerfile.<new_suffix>`
2. 在 `run_test_set.py` 的 `PLATFORM_MAP` 中添加映射
3. 在 Jenkins Pipeline 中确认对应的 PLATFORM 参数值

### 2. 新增 test_set 配置字段

如果 Jenkins Pipeline 新增参数：
1. 在 `jenkins_sdk.py` 的 `PipelineParameters` 中添加字段
2. 在 `run_test_set.py` 的 `build_pipeline_params()` 中添加映射
3. 在 test_set_config yaml 中添加对应配置项

### 3. environment 字段

当前 test_set 中的 `environment` 字段尚未映射到 PipelineParameters。
Jenkins Pipeline 如需传递环境变量，需在 `PipelineParameters` 中增加对应字段，
并在 `build_pipeline_params()` 中完成映射。
