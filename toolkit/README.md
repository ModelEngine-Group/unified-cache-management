# UCM Toolkit 用户文档

`ucm-toolkit` 是 UCM 仓库里的统一工具入口，用来集中调用性能测试、POSIX AIO 测试、物理网卡流量监控、指标采集等辅助工具。它本身是一个独立 Python 包，不会随主 UCM 包自动安装。

## 工具列表

| 工具 | 别名 | 类型 | 功能 | 详细文档 |
| --- | --- | --- | --- | --- |
| `dev-sandbox` | `dev_sandbox` | 需构建、可运行 | 测量主机内存到设备显存的拷贝带宽及磁盘 AIO 吞吐（C++ 测试程序，使用前需先构建），包含 `copy`、`trans`、`aio` 三个子功能。 | [dev-sandbox README](ucm_toolkit/tools/dev_sandbox/README.md) |
| `posix-aio` | `posix_aio` | 可运行 | 运行 `ucm/store/test/e2e/posixstore_aio_test.py`，测试 POSIX AIO store 的 dump/load 性能。 | [posix-aio README](ucm_toolkit/tools/posix_aio/README.md) |
| `nic-monitor` | `nic_monitor` | 可运行 | 监控物理网卡实时流量、后台采样落盘，并生成阶段统计。 | [nic-monitor README](ucm_toolkit/tools/nic_monitor/README.md) |
| `metrics-view` | `metrics_view`, `terminal-metrics`, `terminal_metrics` | 可运行 | 采集 Prometheus/OpenMetrics 样本到 SQLite，并在终端查询聚合指标。 | [metrics-view README](ucm_toolkit/tools/metrics_view/README.md) |
| `precheck` | `pre_check` | 可运行 | 在 UCM 部署前于宿主机本地运行环境预检，校验 serving-stack/uc-manager 版本、加速卡驱动（CUDA 算力或昇腾 HDK）、内核版本、`/dev/shm` 及 posix store 带宽，输出 `PASS`/`WARN`/`FAIL` 并对失败项给出修复建议（RFC #1208）。 | [precheck README](ucm_toolkit/tools/precheck/README.md) |
| `ttft-analyze` | `ttft_analyze` | 可运行 | 估算 KV Cache 命中 UCM SSD/DRAM Prefix Cache 时的 TTFT，输出增益/损失与瓶颈拆分。 | [ttft-analyze README](ucm_toolkit/tools/ttft_analyze/README.md) |

各子工具的依赖、参数、示例与常见问题都在各自 README 中说明。

## 安装

推荐在仓库根目录使用 editable 安装：

```bash
python -m pip install -e toolkit
```

安装后确认入口可用：

```bash
ucm-toolkit --help
ucm-toolkit list
```

如果希望隔离环境，可以先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e toolkit
```

## 依赖

基础 CLI 只依赖 Python 标准库和 `setuptools`。不同工具还需要额外系统依赖，概览如下（详见各子工具 README）：

| 功能 | 依赖 |
| --- | --- |
| `dev-sandbox` 构建 | CMake 3.18+、C++17 编译器。CUDA 后端需要 CUDA runtime；Ascend 后端需要 Ascend runtime；`copy` 的 GDR case 还需要 `libibverbs` 头文件和库。 |
| `posix-aio` | 需先安装 UCM 主软件包（unified-cache-management）及其 native 扩展，并需要 `numpy`。 |
| `nic-monitor` | Linux、`bash`、`ethtool`，并且需要 root 或 sudo 权限读取网卡统计。 |
| `metrics-view` | 仅依赖 Python 标准库（`sqlite3` 内置）；采集需要可访问的 Prometheus/OpenMetrics `/metrics` HTTP 接口。 |
| `precheck` | 核心检查仅依赖 Python 标准库；带宽基准需要已安装 UCM 主软件包（native 扩展）与 `numpy`，且仅 Linux 可运行。 |

`dev-sandbox` 的后端探测优先级与切换方式见 [dev-sandbox README](ucm_toolkit/tools/dev_sandbox/README.md#依赖)。

## 通用命令

以下命令对顶层工具通用。具体工具的 `run` 子命令与参数见各自 README。

列出顶层工具：

```bash
ucm-toolkit list
ucm-toolkit list --verbose
```

检查工具环境：

```bash
ucm-toolkit doctor
ucm-toolkit doctor dev-sandbox
ucm-toolkit doctor posix-aio
ucm-toolkit doctor nic-monitor
ucm-toolkit doctor precheck
```

构建工具：

```bash
ucm-toolkit build TOOL [tool build args...]
```

目前只有 `dev-sandbox` 支持 `build`。

运行工具：

```bash
ucm-toolkit run TOOL [tool args...]
```

清理工具产物：

```bash
ucm-toolkit clean TOOL
ucm-toolkit clean TOOL --dry-run
```

目前 `clean dev-sandbox` 会删除配置的 build 目录；其他工具默认没有可清理产物。
