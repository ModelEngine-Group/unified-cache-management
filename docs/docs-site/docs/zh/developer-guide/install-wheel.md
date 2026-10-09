# 接入 wheel 安装脚本

使用 UCM 单文件 Python 安装器 `scripts/install_ucm.py`，在用户环境或接入项目的镜像构建中安装已发布 wheel。环境探测和选包逻辑由 UCM 维护，接入项目只负责获取脚本、传递参数和安排调用位置。

脚本需要 Python 3.10 及以上，以及该解释器中的 pip。它复用 pip 随附的 packaging，不需要克隆仓库、预装 UCM、GPU/NPU 或 Docker。安装目标使用 Linux 和 glibc，CPU 架构是否可用由**实际发布的兼容 wheel 标签**决定，不另设架构白名单。安装检查覆盖 Ubuntu、openEuler 的 AMD64/ARM64 环境。例如，包声明 `Requires-Python: >=3.10`，不代表它的 `cp312` 后端 wheel 可以用于 Python 3.10。

## 获取单个脚本

脚本版本与要安装的 UCM 包版本分别固定。将 `INSTALLER_REF` 替换为已评审脚本的完整 commit SHA。固定脚本版本后，构建时仍可以自动选择最新发布的兼容包。

```bash
INSTALLER_REPOSITORY=ModelEngine-Group/unified-cache-management
INSTALLER_REF='<full-commit-sha>'
curl --fail --location \
  "https://raw.githubusercontent.com/${INSTALLER_REPOSITORY}/${INSTALLER_REF}/scripts/install_ucm.py" \
  --output install_ucm.py
python3 install_ucm.py
```

默认参数为 `--version latest --extra auto`。`latest` 指当前环境具有兼容元包、后端 wheel 且 Toolkit 发布坐标精确匹配的最高已发布 UCM 版本，包含 RC 等预发布版本；排除 dev/nightly 版本和撤回文件。版本按 PEP 440 排序，例如 `0.7.0 < 0.8.0rc1 < 0.8.0rc2 < 0.8.0`。源码标签不决定包是否可用，脚本始终不进行源码构建。

## 使用参数

```bash
# 使用指定解释器及其 pip 配置。
/path/to/python3.12 install_ucm.py

# 固定 UCM 版本和已发布的后端 extra。
python3 install_ucm.py --version 0.7.0 --extra cann910-a2

# 只解析，不安装任何包。
python3 install_ucm.py --resolve > ucm-selection.json
```

精确版本可以是已发布的预发布版本；不可用时不会换成其他版本。即使精确指定版本，也会排除撤回的文件。`--extra` 接受单个后端名称，例如 `cann910-a3`、`cu129`，不接受 extra 列表或仅安装 `toolkit` 等辅助功能。

直接用 `python3` 或 `/path/to/python3.12` 等指定解释器路径运行文件。环境探测和安装始终使用该解释器，脚本通过 `sys.executable -m pip install --only-binary=:all:` 执行固定 requirement，继承 pip 的软件源、代理、证书等配置；版本发现始终访问官方 PyPI。如果配置的镜像源缺少选中的包，pip 会失败，脚本不会更换软件源或退回旧版本掩盖问题。普通依赖解析仍由 pip 完成。

`--resolve` 向 stdout 输出一个 JSON 对象，固定字段如下：

| 字段 | 含义 |
| --- | --- |
| `version` | 选中的 UCM 元包版本 |
| `extra` | 选中的后端 extra |
| `requirement` | 固定的 `uc-manager[extra]==version` |
| `backend_requirement` | 发布依赖声明中的后端包名和版本 |
| `wheel` | 匹配当前环境的后端 wheel 文件名 |

诊断输出到 stderr，失败返回非零退出码。JSON 用于记录选包结果，不是全部传递依赖的锁文件，也不代表硬件兼容性已经验证。

## 环境识别与选包规则

脚本将实际 Python/ABI、CPU 架构和 glibc 与 wheel 标签、`Requires-Python` 比较。操作系统名称用于诊断，不作为允许安装的白名单。版本、extras、依赖和 wheel 元数据均来自官方 PyPI，不使用 UCM 的 vLLM-Ascend 支持声明过滤。

Ascend 环境应先配置活动 Toolkit。`ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 用于定位它；未设置时检查标准 Toolkit 位置。CANN 版本从 Toolkit 安装元数据读取，不采用镜像 tag、目录名称、`CANN_VERSION`、驱动元数据或 `npu-smi`。`SOC_VERSION` 用于区分 A2（`ascend910b…`）和 A3（`ascend910_93…`）；SoC 缺失或无法识别时，需要显式指定 extra。

CUDA 通过 `CUDA_HOME`、`CUDA_PATH` 定位活动 Toolkit；未设置时检查 `/usr/local/cuda` 和 `PATH` 中 `nvcc` 所属的 Toolkit。优先读取 Toolkit 版本元数据，必要时执行该 Toolkit 的 `nvcc --version`，不使用 `nvidia-smi` 或驱动宣称支持的 CUDA 版本。

自动模式下，无法识别 Toolkit、元数据冲突或同时存在两类加速器运行时，都会报错。显式 extra 跳过自动 Toolkit/SoC 探测，但仍检查 Python、架构、glibc 和 wheel 是否可用，适用于调用方已明确目标后端的无设备构建环境。

对于每个候选 UCM 版本，自动模式要求 Toolkit 发布坐标**精确匹配**，Ascend 还必须属于相同 A2/A3 家族。CANN 比较主、次、补丁版本；CUDA 比较主、次版本，不比较 Toolkit 构建后缀。没有对应后端时，`latest` 继续检查较旧的 UCM 版本；精确指定 UCM 版本则报错。不会选择相邻档位或跨 CUDA 主版本代替。

`AcceleratorRuntime.extra` 按发布协议将坐标编码为 `cann910-a2`、`cu129`，并与 PyPI 声明的 extra 比较，不反向猜测紧凑名称里的版本位数。元包的 `Requires-Dist` 决定实际后端包名及版本。Toolkit 路径、SoC 家族映射和 extra 编码是接口约定；新发布版本无需新增版本白名单。

精确选包仍不等于原生库、引擎接口或硬件运行验收。脚本不安装或改变 Toolkit、驱动、torch 或推理引擎；普通包依赖仍由 pip 解析。

## 多架构使用同一次选择

先在每个目标镜像中，用最终安装 UCM 的 Python 导出环境：

```bash
# 分别在 ARM64、AMD64 目标环境中执行，保存各自的文件。
python3 install_ucm.py --probe > runtime-arm64.json
python3 install_ucm.py --probe > runtime-amd64.json
```

`--probe` 不联网、不安装。JSON 包含完整 PEP 508 `markers`、按本机优先级排列的 wheel `tags` 字符串数组，以及 `accelerator` 对象，例如 `{"family":"a2","toolkit_version":[9,1,0]}`。使用显式 `--extra` 时跳过 Toolkit 检测，`accelerator` 为 `null`。

在构建协调端，将同一后端的目标环境一起解析：

```bash
python3 install_ucm.py --resolve \
  --runtime runtime-amd64.json --runtime runtime-arm64.json > ucm-selection.json
```

`--runtime` 可重复，只允许与 `--resolve` 一起使用，不会探测或安装协调端环境。安装器从最高版本开始，选出全部目标环境均有兼容 wheel 的同一个 UCM 版本和后端 extra；任何目标缺少该版本的 wheel 都不会被单独降级。A2 和 A3 属于不同后端，不能合成同一组自动选择。

单目标输出保持上面的五个字段；多目标用 `wheels` 替代 `wheel`，其余四个字段相同。`wheels` 按输入顺序列出 `architecture`（如 `aarch64`）、`python`（完整版本，如 `3.12.4`）及 `wheel`（文件名）。调用方把所选的 `version` 和 `extra` 传给各目标环境安装：

```bash
python3 install_ucm.py --version 0.8.0rc1 --extra cann910-a2 --report pip-report.json
```

`--report` 仅用于安装，直接透传给 `pip --report`，输出原生 pip JSON。后端条目的 `metadata.name/version`、`download_info.url` 和 `download_info.archive_info.hashes.sha256` 可用于核对实际产物。已安装同版本时 pip 可能不产生新的 `install` 条目；此时应核对已安装包版本，不能把缺失的下载记录当成本次完成了 hash 校验。脚本不会为生成报告而强制重装。

## 在 Docker 构建中调用

构建前下载已评审版本的脚本，将该单文件放入构建上下文：

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG UCM_VERSION=latest
ARG UCM_EXTRA=auto
COPY install_ucm.py /tmp/install_ucm.py
RUN python3 /tmp/install_ucm.py --version "${UCM_VERSION}" --extra "${UCM_EXTRA}" \
    && rm /tmp/install_ucm.py
```

基础镜像应提供 Python/pip；自动模式还需要可读的 Toolkit 元数据，以及 Ascend 目标的 `SOC_VERSION`。如果镜像需要初始化 Toolkit 环境，先按镜像既有方式完成初始化。探测和安装不需要挂载设备。

Motor 等接入项目继续负责自己的构建参数、缓存策略、各架构构建和 manifest 合并，不复制选包规则。不同环境独立执行 `latest` 可能得到不同版本；需要一致结果时先收集 `--probe` 输出，用重复 `--runtime` 得到共同版本，再把固定版本和 extra 传入构建。命中 Docker 缓存的安装层不会重新发现新版本，缓存更新由调用项目控制。

UCM 根目录的 `install.sh` 仍是安装 wheel 后执行的钩子，与这个独立入口无关。

## 分层验收

脚本测试不需要安装 UCM 或接入加速卡：

```bash
python3 -m py_compile scripts/install_ucm.py
python3 -m pytest -q scripts/tests
```

安装器 CI 另外在 AMD64/ARM64 的 Ubuntu/openEuler Ascend 环境及 CUDA Toolkit 环境中安装真实发布的 wheel，不挂载加速卡，并核对安装后的包版本、pip 报告中的 wheel 与解析结果，还使用两种架构的 probe 验证共同版本选择。这些检查不等于原生库加载或推理验收。

在准备好的目标环境中，可以显式执行版本与导入检查：

```bash
python3 -c 'from importlib.metadata import version; print(version("uc-manager"))'
python3 -c 'import ucm; from ucm.store.pipeline import ucmpipelinestore; print(ucm.__file__)'
```

还应单独检查 `backend_requirement` 对应的后端包版本。原生链接检查应加载实际部署使用的库，例如：

```python
import ctypes
import os
from pathlib import Path
import ucm

for relative in ("cache/libcachestore.so", "posix/libposixstore.so"):
    ctypes.CDLL(str(Path(ucm.__file__).parent / "store" / relative),
                mode=os.RTLD_NOW | os.RTLD_LOCAL)
```

原生链接检查需要对应的运行库。真实 GPU/NPU 验收还需要硬件、驱动、目标引擎，以及 UCM 缓存读写或推理工作负载。脚本测试、包安装、原生加载和硬件运行应分别报告。

## 发布流水线接入

正式发布在先发布 backend wheel、再发布元包后，使用现有动态 PyPI 安装验证矩阵调用同一个安装器。此检查固定本次发布版本，由安装器探测目标 Toolkit/SoC；选出的版本、extra、backend 和 wheel 必须与构建计划及发布回执一致，随后继续执行已有原生库与 Toolkit 冒烟检查。

Fork 发布保留隔离的 TestPyPI 包名及已有安装验证。发布元数据测试构建真实元包 wheel，再验证单文件安装器能够读取其依赖声明。独立安装器流水线另外在两种架构测试当前公开发布的包，不发布新包，也不替代现有 release manifest。
