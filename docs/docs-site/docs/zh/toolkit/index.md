# UCM Toolkit 用户文档

UCM 工具集用于部署前的环境检查与容量规划，以及运行中的指标观察和性能排查。可以根据当前问题，分别检查宿主机、测量存储和设备拷贝带宽，或记录服务指标与网卡流量。

其中五个命令行工具通过 `ucm-toolkit` 统一调用。它是独立 Python 包，需要单独安装；KV Cache 计算器直接在文档页面使用，无需安装 CLI。

## 工具列表

| 要解决的问题 | 工具 | 能获得什么 |
| --- | --- | --- |
| 部署前需要检查驱动、内核和共享内存是否满足预检条件 | [precheck](user/precheck.md) | 环境检查结果、异常处理建议和可选的存储带宽报告 |
| KV 保存或读取慢，需要比较存储路径与 I/O 参数 | [posix-aio](user/posix-aio.md) | UCM POSIX Store 的 dump/load 耗时与带宽 |
| 服务已启动，需要查看缓存命中率、请求延迟和加载带宽 | [metrics-view](user/metrics-view.md) | 终端指标快照，或持续采集后按时间窗口查询的结果 |
| 跨节点传输慢，需要观察网卡负载与流量分布 | [nic-monitor](user/nic-monitor.md) | 各物理网卡的收发速率、利用率和可回看的 CSV 记录 |
| KV 加载慢，需要单独测量主机与设备间的拷贝环节 | [dev-sandbox](user/dev-sandbox.md) | 选定拷贝、传输或 I/O 测试场景的耗时与带宽 |
| 调整上下文长度、批大小或并行度前，需要估算 KV 内存需求 | [KV Cache 计算器](kv-cache-calculator.md) | 按模型与执行参数估算的 KV 容量，以及给定 KV 内存预算下的 token 容量 |

各工具页面先介绍用途与适用场景，再说明依赖、参数和使用示例。部署前可先做环境预检；服务运行后结合指标与网卡流量缩小排查范围，再对存储或设备拷贝环节做独立测试。

## 安装

Toolkit 可独立安装，也可以通过 UCM 的可选插件入口安装。下面的命令来自当前文档版本的发布清单，RC 页面会固定到对应 RC 版本。

<div data-toolkit-install data-locale="zh">正在加载本版本的 Toolkit 安装命令……</div>

正式 PyPI 包的安装形式如下；同时选择计算后端时可以组合 extras：

```bash
pip install ucm-toolkit
pip install 'uc-manager[toolkit]'
pip install 'uc-manager[cu130,toolkit]'
ucm-toolkit list
```

`[toolkit]` 不会自动安装 CUDA/CANN 后端。`dev-sandbox` 随包提供源码，使用本机 SDK 按需编译：

```bash
ucm-toolkit build dev-sandbox
```

源码开发仍支持在仓库根目录执行 `python -m pip install -e toolkit`。

## 依赖

基础 CLI 只依赖 Python 标准库。不同工具还需要额外系统依赖，概览如下（详见各子工具文档）：

| 功能 | 依赖 |
| --- | --- |
| `dev-sandbox` 构建 | CMake 3.18+、C++17 编译器。CUDA 后端需要 CUDA runtime；Ascend 后端需要 Ascend runtime；`copy` 的 GDR case 还需要 `libibverbs` 头文件和库。 |
| `posix-aio` | 需先安装 UCM 主软件包（unified-cache-management）及其 native 扩展，并需要 `numpy`。 |
| `nic-monitor` | Linux、`bash`、`ethtool`，并且需要 root 或 sudo 权限读取网卡统计。 |
| `metrics-view` | 仅依赖 Python 标准库（`sqlite3` 内置）；采集需要可访问的 Prometheus/OpenMetrics `/metrics` HTTP 接口。 |
| `precheck` | 核心检查仅依赖 Python 标准库；带宽基准需要已安装 UCM 主软件包（native 扩展）与 `numpy`，且仅 Linux 可运行。 |

`dev-sandbox` 的后端探测优先级与切换方式见 [dev-sandbox 开发者文档](developer/dev-sandbox.md)。

## 通用命令

以下命令对顶层工具通用。具体工具的 `run` 子命令与参数见各自文档。

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
