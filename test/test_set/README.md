# Test Set Runner

通过 Jenkins Pipeline 批量执行测试套件的独立工具。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置连接信息

编辑 `run_config.yaml`，填写 Jenkins 和 Harbor 连接信息：

```yaml
# Jenkins 连接配置
jenkins:
  url: "https://your-jenkins.com"
  username: "your-username"
  api_token: "your-api-token"
  job_name: "your-folder/your-pipeline"

# Harbor 镜像仓库配置（通过 package_name 自动查找镜像时使用）
harbor:
  url: "https://registry.dev.huawei.com"
  auth_token: ""  # Basic auth token
  project: "ai-dailybuild"
```

### 3. 运行

```bash
# 使用默认配置运行所有测试
python run_test_set.py

# 先 dry-run 确认参数
python run_test_set.py --dry-run
```

## 配置文件

### run_config.yaml

运行时配置，包含本次执行的版本信息和连接信息。

#### Jenkins 配置

| 字段 | 说明 |
|---|---|
| `jenkins.url` | Jenkins 服务地址 |
| `jenkins.username` | Jenkins 用户名 |
| `jenkins.api_token` | Jenkins API Token |
| `jenkins.job_name` | Pipeline 的 folder/job 路径 |

#### Harbor 配置

当未配置 `override_image` 而是通过 `package_name` 自动查找镜像时，需要配置 Harbor 连接信息。

| 字段 | 说明 |
|---|---|
| `harbor.url` | Harbor 仓库地址 |
| `harbor.auth_token` | Harbor Basic Auth Token |
| `harbor.project` | Harbor 项目名称 |

#### 构建与测试配置

| 字段 | 说明 |
|---|---|
| `test_set_config` | test_set 配置文件路径（相对或绝对） |
| `parallel` | 是否并行执行测试 (true/false) |
| `test_build.package_name` | 包名，用于从 Harbor 自动查找匹配的镜像 |
| `test_build.override_image` | 按平台直接指定完整镜像地址（优先级高于 package_name） |

#### 镜像解析优先级

1. `override_image` 中指定了该平台的完整镜像地址 → 直接使用
2. 否则通过 `package_name` + Harbor API 查询匹配的镜像

### test_set_config (如 test_set_config/test_set.yaml)

定义测试集合，与版本无关。每个 test_set 包含：

| 字段 | 说明 |
|---|---|
| `name` | 测试名称（唯一标识） |
| `platform` | 平台，对应 Docker 后缀（如 `vllm_gpu`, `vllm_npu`） |
| `server_start_config.gpu_count` | GPU 数量 |
| `server_start_config.node_count` | 节点数（>1 则为多节点部署，DEPLOY_MODE=multi） |
| `server_start_config.server_port` | 服务端口 |
| `server_start_config.master_start_command` | 主节点启动命令 |
| `server_start_config.slave_start_command` | 从节点启动命令（多节点时需要） |
| `server_start_config.ucm_config` | UCM 配置 YAML 内容 |
| `server_start_config.environment` | 环境变量列表 |
| `pytest_config.api_model_name` | API 模型名称 |
| `pytest_config.test_params` | pytest 参数 |

#### test_set 配置示例

```yaml
test_sets:
  - name: online_inference_pc_nfs_store
    platform: vllm_npu

    server_start_config:
      gpu_count: 1
      node_count: 1
      server_port: 9527
      master_start_command: |
        vllm serve /models/Qwen3-1.7B
        --served-model-name Qwen3-1.7B
        --block-size 128
        --tensor-parallel-size 1
        --gpu-memory-utilization 0.87
        --trust-remote-code
      ucm_config: |
        ucm_connectors:
          - ucm_connector_name: "UcmNfsStore"
            ucm_connector_config:
              store_pipeline: "Cache|Posix"
              storage_backends: "/mnt/kvcache-local"
        enable_event_sync: false
      environment:
        - ENABLE_UCM_PATCH: "1"

    pytest_config:
      api_model_name: Qwen3-1.7B
      test_params: "--feature=fvt_test"
```

## CLI 参数

| 参数 | 说明 |
|---|---|
| `-c, --config` | run_config.yaml 路径（默认: `run_config.yaml`） |
| `--test-set-config` | 覆盖 test_set_config 路径 |
| `--package-name` | 覆盖 package_name |
| `--override-image PLATFORM=IMAGE` | 覆盖指定平台镜像（可多次使用） |
| `-t, --test NAME` | 只运行指定名称的测试（可多次使用） |
| `--no-parallel` | 串行执行 |
| `--no-wait` | 只触发不等待结果 |
| `--dry-run` | 仅打印参数，不实际触发 |
| `--list` | 列出所有可用测试名称 |

## 使用示例

```bash
# 列出所有测试
python run_test_set.py --list

# 指定包名运行（通过 Harbor 自动查找镜像）
python run_test_set.py --package-name 20260326-release

# 覆盖镜像运行（直接指定完整镜像地址）
python run_test_set.py --override-image vllm_npu=registry.example.com/ai/ucm-vllm_npu:v2.0

# 只运行某个测试
python run_test_set.py -t online_inference_pc_layerwise

# 运行多个指定测试
python run_test_set.py -t online_inference_pc_nfs_store -t online_inference_pc_pp

# Dry-run 查看参数
python run_test_set.py --dry-run -t online_inference_pc_layerwise --package-name test-pkg

# 串行执行
python run_test_set.py --no-parallel

# 只触发不等待
python run_test_set.py --no-wait
```

## 平台映射

| Docker 后缀 (test_set platform) | Jenkins PLATFORM |
|---|---|
| `vllm_gpu` | `vllm-cuda` |
| `vllm_npu` | `vllm-ascend` |
| `mindie_llm` | `mindie` |
| `sglang_gpu` | `sglang-cuda` |

平台名来自 `docker/` 目录下的 Dockerfile 后缀（如 `Dockerfile.vllm_gpu` → `vllm_gpu`）。

## 运行行为

- 默认并行触发所有测试，等待全部完成后打印汇总结果表
- 如果部分测试触发失败，会立即 abort 所有已触发的 build 并退出
- 运行过程中按 Ctrl+C 会自动 abort 所有已触发的 Jenkins build，再按一次强制退出
- 等待过程中发生异常，同样会自动 abort 已触发的 build
- 退出码：全部通过返回 0，有失败返回 1
