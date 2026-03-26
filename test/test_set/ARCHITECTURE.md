# Test Set Runner - 架构文档

> 本文档面向 Code Agent，用于理解系统设计以便后续开发和维护。

## 系统概览

```
┌─────────────────────────────────────────────────────┐
│                  run_test_set.py                     │
│                                                     │
│  CLI (argparse)                                     │
│       │                                             │
│       ▼                                             │
│  Config Loader ──── run_config.yaml                 │
│       │                                             │
│       ▼                                             │
│  Test Set Loader ── test_set_config/test_set.yaml   │
│       │                                             │
│       ▼                                             │
│  Parameter Builder                                  │
│   ├─ PLATFORM_MAP (平台映射)                         │
│   ├─ resolve_image_tag() (镜像标签解析)              │
│   └─ build_pipeline_params() (参数构建)              │
│       │                                             │
│       ▼                                             │
│  Executor (parallel/sequential)                     │
│   ├─ trigger builds                                 │
│   └─ wait & collect results                         │
│       │                                             │
│       ▼                                             │
│  Result Reporter                                    │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   jenkins_sdk.py     │
│                      │
│  JenkinsPipelineClient│
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
   ├─ jenkins: {url, username, api_token, job_name, branch}
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
5. Parameter Building:
   ├─ platform (docker suffix) → PLATFORM_MAP → Jenkins PLATFORM
   ├─ override_image[platform] → extract_image_tag() → ImageTag
   │   OR package_name → resolve_image_tag() → ImageTag
   └─ test_set fields → build_pipeline_params() → PipelineParameters
        │
        ▼
6. JenkinsPipelineClient.trigger(params) → build_number
        │
        ▼ (ThreadPoolExecutor if parallel)
7. JenkinsPipelineClient.wait_for_completion(build_number) → BuildInfo
        │
        ▼
8. print_results() → 汇总表 + exit code
```

## 核心函数

### run_test_set.py

| 函数 | 职责 |
|---|---|
| `main()` | 入口，编排完整流程 |
| `parse_args()` | CLI 参数解析 |
| `load_run_config(path)` | 加载 run_config.yaml |
| `load_test_sets(path, base_dir)` | 加载 test_set 配置 |
| `build_pipeline_params(test_set, platform, tag)` | test_set 配置 → PipelineParameters |
| `get_image_tag(platform, overrides, package)` | 确定 image tag（优先 override） |
| `resolve_image_tag(platform, package_name)` | **占位函数** - 根据平台和包名解析 tag |
| `extract_image_tag(image)` | 从 `repo/name:tag` 中提取 tag |
| `print_results(results)` | 打印结果汇总表 |

### jenkins_sdk.py（已有，仅引用）

| 类/方法 | 用途 |
|---|---|
| `JenkinsPipelineClient(url, user, token, job, branch)` | Jenkins 客户端 |
| `client.trigger(params, **overrides)` | 触发 build，返回 build number |
| `client.wait_for_completion(build_number)` | 等待 build 完成，返回 BuildInfo |
| `PipelineParameters` | Jenkins 参数 dataclass |
| `BuildInfo` | Build 结果 dataclass |
| `BuildStatus` | 状态枚举 (SUCCESS, FAILURE, ...) |

## 配置文件 Schema

### run_config.yaml

```yaml
jenkins:                          # Jenkins 连接配置
  url: str                        # 必填
  username: str                   # 必填
  api_token: str                  # 必填
  job_name: str                   # 必填, folder/pipeline 路径
  branch: str                     # 可选, 默认 "jenkins-dev-yhq"

test_set_config: str              # test_set 配置文件路径 (相对或绝对)
parallel: bool                    # 是否并行执行, 默认 true

test_build:
  package_name: str               # 包名, 用于 resolve_image_tag()
  override_image:                 # 可选, 按平台覆盖完整镜像地址
    <platform>: str               # key = docker 后缀, value = 完整镜像地址
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
(resolved image tag)        → ImageTag
gpu_count                   → GPU_COUNT
node_count > 1 ? multi      → DEPLOY_MODE
server_port                 → SERVER_PORT
master_start_command        → VLLM_COMMAND_MASTER
slave_start_command         → VLLM_COMMAND_WORKER
ucm_config                  → UCM_CONFIG_YAML
api_model_name              → API_MODEL_NAME
test_params                 → TEST_PARAMS
```

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

## 扩展点

### 1. resolve_image_tag(platform, package_name) → str

当前为占位实现（直接返回 package_name）。后续需实现：
- 查询制品库获取对应平台和包名的 image tag
- 输入：platform（docker 后缀）、package_name
- 输出：image tag 字符串

### 2. 新增平台

1. 在 `docker/` 下添加 `Dockerfile.<new_suffix>`
2. 在 `run_test_set.py` 的 `PLATFORM_MAP` 中添加映射
3. 在 Jenkins Pipeline 中确认对应的 PLATFORM 参数值

### 3. 新增 test_set 配置字段

如果 Jenkins Pipeline 新增参数：
1. 在 `jenkins_sdk.py` 的 `PipelineParameters` 中添加字段
2. 在 `run_test_set.py` 的 `build_pipeline_params()` 中添加映射
3. 在 test_set_config yaml 中添加对应配置项

### 4. environment 字段

当前 test_set 中的 `environment` 字段尚未映射到 PipelineParameters。
Jenkins Pipeline 如需传递环境变量，需在 `PipelineParameters` 中增加对应字段，
并在 `build_pipeline_params()` 中完成映射。
