# Test Set Runner

通过 Jenkins Pipeline 批量执行测试套件的独立工具。

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml requests
```

### 2. 配置 Jenkins 连接

编辑 `run_config.yaml`，填写 Jenkins 连接信息：

```yaml
jenkins:
  url: "https://your-jenkins.com"
  username: "your-username"
  api_token: "your-api-token"
  job_name: "your-folder/your-pipeline"
  branch: "jenkins-dev-yhq"
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

运行时配置，包含本次执行的版本信息。

| 字段 | 说明 |
|---|---|
| `jenkins.url` | Jenkins 服务地址 |
| `jenkins.username` | Jenkins 用户名 |
| `jenkins.api_token` | Jenkins API Token |
| `jenkins.job_name` | Multibranch Pipeline 的 folder/job 路径 |
| `jenkins.branch` | Pipeline 对应的 Git 分支 |
| `test_set_config` | test_set 配置文件路径（相对或绝对） |
| `parallel` | 是否并行执行测试 (true/false) |
| `test_build.package_name` | 包名，用于解析 image tag |
| `test_build.override_image` | 按平台覆盖镜像地址（优先级高于 package_name） |

### test_set_config (如 test_set_config/test_set.yaml)

定义测试集合，与版本无关。每个 test_set 包含：

| 字段 | 说明 |
|---|---|
| `name` | 测试名称（唯一标识） |
| `platform` | 平台，对应 Docker 后缀（如 vllm_gpu, vllm_npu） |
| `server_start_config.gpu_count` | GPU 数量 |
| `server_start_config.node_count` | 节点数（>1 则为多节点部署） |
| `server_start_config.server_port` | 服务端口 |
| `server_start_config.master_start_command` | 主节点启动命令 |
| `server_start_config.slave_start_command` | 从节点启动命令（多节点时需要） |
| `server_start_config.ucm_config` | UCM 配置 YAML |
| `server_start_config.environment` | 环境变量列表 |
| `pytest_config.api_model_name` | API 模型名称 |
| `pytest_config.test_params` | pytest 参数 |

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

# 指定包名运行
python run_test_set.py --package-name 20260326-release

# 覆盖镜像运行
python run_test_set.py --override-image vllm_gpu=myrepo.io/vllm:v2.0

# 只运行某个测试
python run_test_set.py -t vllm_cuda_qwen3_pc

# 运行多个指定测试
python run_test_set.py -t vllm_cuda_qwen3_pc -t vllm_ascend_qwen3_pc

# Dry-run 查看参数
python run_test_set.py --dry-run -t vllm_cuda_qwen3_pc --package-name test-pkg

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
