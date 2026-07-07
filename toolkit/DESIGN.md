# UCM Toolkit 设计文档

## 背景

UCM 是一个基于 vLLM KV connector 构建的 KV cache 卸载插件。`toolkit`
目录将作为 UCM 的工具子项目，用来统一管理多种测试、诊断和辅助工具，提供一致的入口、CLI、
文档、构建和打包方式。

当前计划纳入的工具包括：

- 网卡负载监控工具：当前接入 `toolkit/src/nic_monitor/nic_monitor_pro.sh`，用于被动物理网卡流量监控。
- `dev-sandbox`：当前已经存在于 `toolkit/src/dev-sandbox` 下，是一个基于 CMake 的 C++17
  测试项目，用于测试 CUDA / Ascend 平台的 D2D、D2H、H2D 等拷贝带宽。
- POSIX AIO 测试脚本：代码仓中的 POSIX AIO 脚本需要统一纳入 toolkit 的 CLI 入口。

## 设计目标

- 提供一个统一命令入口，例如 `ucm-toolkit`。
- 提供一致的 `--help` 输出和参数风格。
- 提供统一的工具列表、构建、运行、诊断和清理命令。
- 对需要编译的工具提供统一构建入口，但不破坏原工具自身的构建语义。
- 将不同工具隔离在 adapter 后面，方便后续新增工具。
- 将 toolkit 的使用说明集中放在 `toolkit` 目录下。
- 支持工具原生命令参数透传，避免统一 CLI 阻断高级用法。

## 非目标

- 不把所有工具实现合并到一个巨大的 Python 文件中。
- 不立即重写 `dev-sandbox`。第一阶段只包装它现有的 CMake 构建和二进制运行方式。
- 不在 toolkit 层强行干预 `dev-sandbox` 的后端探测逻辑。
- 不在第一版引入重量级 Python CLI 依赖。
- 不要求所有原生工具参数都在第一版转换成 toolkit 的标准参数。

## 推荐目录结构

```text
toolkit/
  pyproject.toml
  README.md
  DESIGN.md
  ucm_toolkit/
    __init__.py
    cli.py
    registry.py
    runner.py
    errors.py

    commands/
      __init__.py
      build.py
      run.py
      list.py
      doctor.py
      clean.py

    tools/
      __init__.py
      dev_sandbox/
        __init__.py
        adapter.py
        README.md
      posix_aio/
        __init__.py
        adapter.py
        README.md
      nic_monitor/
        __init__.py
        adapter.py
        README.md

  src/
    dev-sandbox/
      ...

  build/
    dev-sandbox/            # 可选统一构建目录，不作为 dev-sandbox 默认目录

  dist/
```

其中：

- `ucm_toolkit/` 是 Python CLI 和工具编排层。
- `ucm_toolkit/tools/*/adapter.py` 是各工具的接入层。
- `toolkit/src/` 放置各工具的实际源码。
- `toolkit/src/dev-sandbox/` 保持为现有 CMake 项目。
- `toolkit/build/` 和 `toolkit/dist/` 是生成目录，建议加入 `.gitignore`。
- `toolkit/build/` 可以作为用户显式指定的统一构建输出目录，但对 `dev-sandbox` 这种已有原生项目，
  默认应保持原项目习惯，也就是使用 `toolkit/src/dev-sandbox/build`。

## 总体架构

toolkit 使用一个轻量 Python 包作为统一编排层。Python 负责：

- CLI 参数解析。
- 工具注册和发现。
- 构建命令编排。
- 子进程执行。
- 错误信息整理。
- 文档入口和帮助信息。

具体工具通过 adapter 接入。CLI core 不直接关心每个工具内部如何构建、如何运行，而是通过 registry
找到对应工具对象，再调用该对象的 `build`、`run`、`doctor` 等方法。

工具路径、默认 build 目录、二进制相对路径、脚本路径等都属于工具对象自身的配置，不再拆出单独的
`config.py`。这样 registry 是工具事实来源，避免 registry 与 config 两套配置重复。

概念结构如下：

```text
ucm-toolkit
  |
  +-- commands/build.py
  |     |
  |     +-- registry.get("dev-sandbox").build(args)
  |
  +-- commands/run.py
  |     |
  |     +-- registry.get("dev-sandbox").run(args)
  |     +-- registry.get("posix-aio").run(args)
  |     +-- registry.get("nic-monitor").run(args)
  |
  +-- commands/doctor.py
        |
        +-- registry.get(...).doctor(args)
```

## 工具 Adapter 接口

每个工具 adapter 建议提供统一接口，并将路径作为类字段或实例字段集中在工具类内部：

```python
class ToolAdapter:
    name: str
    aliases: list[str]
    description: str
    buildable: bool

    source_dir: str | None
    build_dir: str | None
    binary_relpath: str | None
    script_path: str | None
    subcommands: dict[str, str]

    def add_build_args(self, parser): ...
    def build(self, args): ...

    def add_run_args(self, parser): ...
    def run(self, args): ...

    def doctor(self, args): ...
```

这样新增工具时，只需要新增一个 adapter 并注册到 registry，不需要大改 CLI 主流程。工具需要的路径也跟随
adapter 本身，不再放到全局 config 文件中。

示例：

```python
class DevSandboxTool(ToolAdapter):
    name = "dev-sandbox"
    aliases = ["dev_sandbox"]
    buildable = True
    source_dir = "toolkit/src/dev-sandbox"
    build_dir = "toolkit/src/dev-sandbox/build"
    subcommands = {
        "copy": "module/copy/copy",
        "trans": "module/trans/trans",
        "aio": "module/aio/aio",
    }


class PosixAioTool(ToolAdapter):
    name = "posix-aio"
    buildable = False
    script_path = "ucm/store/test/e2e/posixstore_aio_test.py"
```

如果执行：

```bash
ucm-toolkit build dev-sandbox --build-dir toolkit/build/dev-sandbox/release
```

则 build adapter 在构建成功后更新 `DevSandboxTool.build_dir`。这里的“更新”不是只修改当前进程内存，
而是写回定义该工具类的 Python 源文件，使下一次执行 `ucm-toolkit run dev-sandbox copy/trans/aio`
时也能读取到新的 build 目录。`copy/trans/aio` 作为 `dev-sandbox` 的子功能，从
`DevSandboxTool.build_dir` 读取 build 目录。

## 核心模块设计

### `cli.py`

`cli.py` 是 `ucm-toolkit` 的唯一命令入口，负责解析用户输入并分发到具体 command 或 tool adapter。
它不实现具体工具逻辑，也不直接拼接各工具的底层命令。

主要职责：

- 构建顶层 argparse parser。
- 解析顶层命令，例如 `list`、`doctor`、`build`、`run`、`clean`。
- 处理 `ucm-toolkit --help`、`ucm-toolkit run --help` 等通用帮助。
- 将 `build dev-sandbox` 分发给 registry 中的 `dev-sandbox` 工具对象。
- 将 `run dev-sandbox copy ...` 分发给 registry 中的 `dev-sandbox` 工具对象。
- 捕获 `ToolkitError`，统一转换为命令行错误输出和退出码。

建议函数：

```python
def main(argv: list[str] | None = None) -> int:
    ...


def build_parser() -> argparse.ArgumentParser:
    ...


def handle_list(args: argparse.Namespace) -> int:
    ...


def handle_doctor(args: argparse.Namespace) -> int:
    ...


def handle_build(args: argparse.Namespace) -> int:
    ...


def handle_run(argv: list[str]) -> int:
    ...


def handle_clean(args: argparse.Namespace) -> int:
    ...
```

`run` 命令需要特殊处理。`cli.py` 只解析出一级工具名，一级工具名后面的参数应原样交给对应工具对象。
对于 `dev-sandbox`，`copy/trans/aio` 是它自己的子功能，由 `DevSandboxTool.run()` 继续解析。

示例：

```bash
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K
```

`cli.py` 只应拆出：

```text
tool = "dev-sandbox"
tool_args = ["copy", "-t", "host_to_device_ce", "-s", "16K"]
```

`tool_args` 的一级子命令由 `DevSandboxTool.run()` 解释；子命令之后的参数完全由对应二进制决定。

### `registry.py`

`registry.py` 是工具事实来源，负责保存所有工具对象、工具别名、工具路径和工具元信息。由于不再设计
`config.py`，工具路径、脚本路径、二进制路径、build 目录等都应由 registry 中的工具对象持有。

主要职责：

- 定义工具基类或协议。
- 定义各工具对象的注册方式。
- 提供工具名和 alias 查询。
- 提供工具列表。
- 持有工具路径字段。
- 提供受控字段更新能力，例如更新 `DevSandboxTool.build_dir`。

建议结构：

```python
class ToolAdapter:
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    buildable: bool = False

    source_dir: str | None = None
    build_dir: str | None = None
    binary_relpath: str | None = None
    script_path: str | None = None
    subcommands: dict[str, str] = {}

    def build(self, args: argparse.Namespace) -> int:
        raise ToolNotBuildableError(self.name)

    def run(self, tool_args: list[str]) -> int:
        raise NotImplementedError

    def doctor(self) -> int:
        return 0
```

建议 registry API：

```python
def register(tool: ToolAdapter) -> None:
    ...


def get(name: str) -> ToolAdapter:
    ...


def list_tools() -> list[ToolAdapter]:
    ...


def update_tool_field(tool_name: str, field_name: str, value: str) -> None:
    ...
```

`list_tools()` 只返回顶层工具，用于 `ucm-toolkit list`。例如只列出 `dev-sandbox`、`posix-aio`、
`nic-monitor`，不列出 `copy/trans/aio`。

内置工具建议：

```text
dev-sandbox  可构建工具，持有 source_dir 和 build_dir
posix-aio    Python 脚本工具，持有 script_path
nic-monitor     被动物理网卡流量监控工具
```

`copy/trans/aio` 不作为 registry 顶层工具注册，只作为 `dev-sandbox` 的子功能暴露。

`update_tool_field()` 应是受控更新，不应成为任意文件写入工具。建议限制：

- 只允许更新白名单字段，例如 `build_dir`。
- 只允许更新 registry/adapter 中明确注册的工具类字段。
- 更新前校验字段值是普通字符串路径。
- 更新失败时抛出 `ConfigUpdateError` 或更名后的 `RegistryUpdateError`。

### `runner.py`

`runner.py` 负责统一执行外部命令。它封装 `subprocess`，让 build、run、doctor 等逻辑不用重复处理
命令打印、cwd、env、退出码和异常转换。

主要职责：

- 检查外部命令是否存在。
- 执行命令并保留 stdout/stderr 原样输出。
- 命令失败时抛出统一错误。
- 规范化命令显示格式，方便用户复制复现。

建议函数：

```python
def command_exists(command: str) -> bool:
    ...


def format_command(cmd: list[str]) -> str:
    ...


def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    ...


def check_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    ...
```

使用示例：

```python
check_command(["cmake", "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"], cwd=source_dir)
check_command(["cmake", "--build", build_dir, "-j", str(jobs)], cwd=source_dir)
run_command([copy_binary, *tool_args])
```

`runner.py` 不应该知道 `dev-sandbox`、`copy`、`posix-aio` 这些具体工具，只做通用命令执行。

### `errors.py`

`errors.py` 定义 toolkit 的统一异常类型。这样各模块只负责抛出明确错误，最终由 `cli.py` 统一打印。

建议异常层次：

```python
class ToolkitError(Exception):
    exit_code: int = 1


class UnknownToolError(ToolkitError):
    ...


class ToolNotBuildableError(ToolkitError):
    ...


class BuildDirNotFoundError(ToolkitError):
    ...


class BinaryNotFoundError(ToolkitError):
    ...


class ScriptNotFoundError(ToolkitError):
    ...


class CommandNotFoundError(ToolkitError):
    ...


class CommandFailedError(ToolkitError):
    ...


class RegistryUpdateError(ToolkitError):
    ...
```

`cli.py` 统一捕获：

```python
try:
    return dispatch(args)
except ToolkitError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return exc.exit_code
```

错误信息应尽量可操作，例如：

```text
error: dev-sandbox binary not found: toolkit/src/dev-sandbox/build/module/copy/copy
hint: run `ucm-toolkit build dev-sandbox` first
```

模块关系：

```text
cli.py
  -> registry.py
      -> tool adapter
          -> runner.py
          -> errors.py
```

`errors.py` 可以被所有模块引用，但不依赖其他 toolkit 模块。

## CLI 设计

安装后提供统一入口：

```bash
ucm-toolkit --help
```

顶层命令建议如下：

```bash
ucm-toolkit list
ucm-toolkit doctor [TOOL]
ucm-toolkit build TOOL [options]
ucm-toolkit run TOOL [options]
ucm-toolkit clean [TOOL]
```

示例：

```bash
ucm-toolkit list
ucm-toolkit doctor
ucm-toolkit doctor dev-sandbox
ucm-toolkit build dev-sandbox --build-type Release --jobs 16
ucm-toolkit build dev-sandbox --cmake-arg -DCUDA_ROOT=/usr/local/cuda
ucm-toolkit build dev-sandbox --build-dir toolkit/build/dev-sandbox/release
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K -n 512 -i 128 -d 8
ucm-toolkit run dev-sandbox trans --help
ucm-toolkit run dev-sandbox aio --help
ucm-toolkit run posix-aio --help
ucm-toolkit run nic-monitor --help
```

`build`、`run`、`doctor`、`clean` 等一级命令后只跟顶层工具名称。`copy/trans/aio` 不作为顶层工具暴露，
而是作为 `dev-sandbox` 的 run 子功能暴露。对于 `dev-sandbox copy/trans/aio` 这类已经有原生命令行参数的
编译产物，子功能名后面的参数应默认原样透传给底层二进制。

## dev-sandbox 构建设计

### 当前 dev-sandbox 的构建行为

`dev-sandbox` 当前是一个独立 CMake 项目，其 README 中的构建方式为：

```bash
cmake -B build
cmake --build build -j
```

也就是说，如果在 `toolkit/src/dev-sandbox` 目录下执行原命令，默认 build 目录就是：

```text
toolkit/src/dev-sandbox/build
```

构建完成后，`copy` 可执行文件通常位于：

```text
toolkit/src/dev-sandbox/build/module/copy/copy
```

其他 CMake 产物也位于同一个 build 目录下，例如：

```text
toolkit/src/dev-sandbox/build/module/trans/trans
toolkit/src/dev-sandbox/build/module/aio/aio
```

它的 `cmake/DetectRuntime.cmake` 会自动探测运行后端。当前逻辑大致为：

1. 尝试探测 CUDA。
2. 如果 CUDA 不可用，再尝试探测 Ascend。
3. 如果 CUDA 和 Ascend 都不可用，则回退到 CPU simulation 模式。

当前 CMake 配置中没有发现类似下面这样的显式平台选项：

```text
-DPLATFORM=cuda
-DUCM_TOOLKIT_PLATFORM=cuda
-DRUNTIME_BACKEND=ascend
```

也就是说，`dev-sandbox` 原项目目前并不支持直接传入 `cuda` 或 `ascend` 来选择后端。

不过它支持通过 CMake cache 变量影响 runtime root 探测，例如：

```text
-DCUDA_ROOT=/usr/local/cuda
-DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

这些参数不是显式平台选择器，但可以作为原项目已有 CMake 参数透传给 CMake。

这些 cache 变量的读取位置在 `toolkit/src/dev-sandbox/cmake/DetectRuntime.cmake`：

- `CUDA_ROOT`：优先读取 `CACHE{CUDA_ROOT}`，然后读取 `CUDA_HOME`、`CUDA_PATH` 和默认路径
  `/usr/local/cuda`。
- `ASCEND_ROOT`：优先读取 `CACHE{ASCEND_ROOT}`，然后读取 `ASCEND_HOME`、
  `ASCEND_TOOLKIT_HOME` 和默认路径 `/usr/local/Ascend/ascend-toolkit/latest`。

因此，toolkit 中的 `--cmake-arg -DCUDA_ROOT=...` 不是新增的业务参数，而是把用户传入的 CMake
cache 参数原样追加到 CMake configure 命令中。

### toolkit 对 dev-sandbox 的原则

toolkit 不应该默认干预 `dev-sandbox` 的后端选择。默认情况下，toolkit 应执行与原项目等价的 CMake
命令，只是提供统一 CLI 入口。

默认构建行为：

```bash
cd toolkit/src/dev-sandbox
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

这保持了 `dev-sandbox` 原本的自动探测行为。

默认情况下，如果用户不传 `--build-dir`，toolkit 应使用：

```text
toolkit/src/dev-sandbox/build
```

如果用户显式传入 `--build-dir`，toolkit 再使用用户指定目录，例如：

```bash
ucm-toolkit build dev-sandbox \
  --build-dir toolkit/build/dev-sandbox/release
```

toolkit 不保存“上一次构建目录”这类外部状态文件。默认 build 目录应作为 `dev-sandbox`
工具类字段放在 adapter 中，例如：

```python
class DevSandboxTool(ToolAdapter):
    build_dir = "toolkit/src/dev-sandbox/build"
```

`ucm-toolkit build dev-sandbox` 不传 `--build-dir` 时，使用该默认值。用户传入 `--build-dir`
时，构建命令使用用户指定路径，并同步写回定义 `DevSandboxTool` 的 Python 文件，更新
`DevSandboxTool.build_dir`。之后用户运行 `ucm-toolkit run dev-sandbox copy/trans/aio` 时不需要再次传入 build
目录，统一从工具类字段读取。

注意：只在内存中修改类变量无法跨 CLI 进程持久化，因此这里需要提供一个明确的源码更新函数，例如：

```python
def update_tool_field(tool_name: str, field_name: str, value: str) -> None:
    ...
```

该函数只允许更新 registry/adapter 中受控的简单字符串字段，例如 `build_dir`，避免任意改写 Python 文件。

如果用户需要传入原 CMake 项目支持的参数，可以使用通用透传参数：

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda
```

或：

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

第一版不建议在 `ucm-toolkit build dev-sandbox` 中强制要求 `--platform cuda|ascend`，因为这会让
toolkit 暗示它能控制 `dev-sandbox` 的后端选择，但当前原 CMake 项目并没有这样的显式能力。

### 可选 CLI 参数

`dev-sandbox` build adapter 建议支持：

```bash
ucm-toolkit build dev-sandbox \
  --build-type Release \
  --jobs 16 \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda
```

参数含义：

- `--build-type`：传入 `-DCMAKE_BUILD_TYPE=...`，默认 `Release`。
- `--jobs`：传给 `cmake --build ... -j`。
- `--build-dir`：覆盖 `DevSandboxTool.build_dir`；不传时使用该字段当前值。
- `--cmake-arg`：可重复传入，原样传给 CMake configure 阶段。

如果后续 `dev-sandbox` 自己增加了显式后端参数，例如：

```text
-DUCM_TOOLKIT_PLATFORM=cuda|ascend|simu
```

那么 toolkit 再增加：

```bash
ucm-toolkit build dev-sandbox --platform cuda
```

并将它翻译为原项目支持的 CMake 参数。当前阶段不做这个翻译。

## dev-sandbox 运行设计

`dev-sandbox` 构建完成后，toolkit 应定位并运行其二进制产物，例如：

- `module/copy/copy`
- `module/trans/trans`
- `module/aio/aio`

其中 `copy` 是第一阶段重点接入对象。

对于这些编译产物，toolkit 的运行策略应是“顶层工具分发 + 子功能纯代理”：

- toolkit 只负责定位二进制。
- toolkit 只负责在二进制不存在时给出构建提示。
- toolkit 不重新定义二进制已有参数。
- toolkit 不把 `copy` 的 `-t/-s/-n/-i/-d` 二次包装成 `--case/--size/...`。
- `ucm-toolkit run dev-sandbox copy` 中 `copy` 后面的参数由 `copy` 二进制自己解析。
- `copy/trans/aio` 只在 `ucm-toolkit run dev-sandbox --help` 中作为子功能展示，不出现在 `ucm-toolkit list` 中。

二进制查找规则：

1. `ucm-toolkit run dev-sandbox copy/trans/aio` 不提供 `--build-dir` 参数。
2. run 阶段统一读取 `DevSandboxTool.build_dir`。
3. 如果用户希望切换 build 目录，应先执行 `ucm-toolkit build dev-sandbox --build-dir <path>` 更新工具类字段。

toolkit 不需要把构建产物 install 到单独的 `bin` 目录，也不需要复制二进制。构建产物在哪里生成，运行时就从哪里查找。

示例：

```bash
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K -n 512 -i 128 -d 8
```

等价于执行：

```bash
toolkit/src/dev-sandbox/build/module/copy/copy \
  -t host_to_device_ce \
  -s 16K \
  -n 512 \
  -i 128 \
  -d 8
```

### dev-sandbox help 设计

`ucm-toolkit run dev-sandbox copy --help` 应尽量展示底层二进制的原生帮助，而不是维护一份容易过期的 toolkit
侧参数说明。

建议策略：

- `ucm-toolkit run dev-sandbox --help`：展示 `dev-sandbox` 可用子功能，例如 `copy/trans/aio`。
- `ucm-toolkit run dev-sandbox copy --help`：定位 `copy` 二进制并触发它的原生 help 输出。
- `ucm-toolkit run dev-sandbox trans --help`：定位 `trans` 二进制并触发它的原生 help 输出。
- `ucm-toolkit run dev-sandbox aio --help`：定位 `aio` 二进制并触发它的原生 help 输出。
- `ucm-toolkit run --help`：只展示 toolkit 的通用运行规则和顶层可运行工具列表。

当前 `trans` 和 `aio` 已有 `-h` 或 `--help` 风格的帮助参数。当前 `copy` 内部有
`ArgsParser::Help()` 函数，但没有把 `-h/--help` 注册成正式成功返回的帮助参数；`--help`
会落入未知参数分支，打印 Usage 后以失败码退出。因此第一版可以通过执行 `copy` 的 help 路径展示原生
Usage；更理想的后续改进是给 `copy` 二进制补充原生 `-h/--help`，这样
`ucm-toolkit run dev-sandbox copy --help` 就能完全等价地转发到 `copy --help`，并获得成功退出码。

## POSIX AIO 接入设计

POSIX AIO 脚本路径固定为：

```text
ucm/store/test/e2e/posixstore_aio_test.py
```

POSIX AIO 通过稳定命令接入：

```bash
ucm-toolkit run posix-aio
```

当前脚本参数以文件开头的全局变量形式存在：

```python
worker_number = 1
shard_size = 8 * 1024 * 1024
shard_number = 1
block_number = 64
dump_epoch_number = 32
load_epoch_number = 32
storage_backends = ["./build/data"]
```

toolkit 接入时应保留这些默认值，同时提供 CLI 参数覆盖入口：

```bash
ucm-toolkit run posix-aio \
  --worker-number 1 \
  --shard-size 8388608 \
  --shard-number 1 \
  --block-number 64 \
  --dump-epoch-number 32 \
  --load-epoch-number 32 \
  --storage-backend ./build/data
```

参数设计：

- `--worker-number`：覆盖 `worker_number`。
- `--shard-size`：覆盖 `shard_size`，单位为 bytes。
- `--shard-number`：覆盖 `shard_number`。
- `--block-number`：覆盖 `block_number`。
- `--dump-epoch-number`：覆盖 `dump_epoch_number`。
- `--load-epoch-number`：覆盖 `load_epoch_number`。
- `--storage-backend`：覆盖 `storage_backends`，可重复传入多次。

为了实现这个能力，建议对 `posixstore_aio_test.py` 做一个小的兼容性改造：

- 保留文件开头的默认值。
- 增加 `parse_args()`，默认值使用现有全局变量。
- 增加 `main()`，根据 argparse 结果更新运行配置。
- 保持直接执行 `python ucm/store/test/e2e/posixstore_aio_test.py` 时仍使用原默认行为。

这样 `ucm-toolkit run posix-aio` 可以通过 subprocess 调用该脚本，并传入用户指定参数；原脚本直接运行的习惯也不会被破坏。

如果脚本路径后续发生变化，只需要修改 `posix_aio/adapter.py`，不改变用户面对的命令入口。

## 网卡负载监控接入设计

网卡负载监控工具当前阶段只支持被动监控，不支持主动压测。当前接入脚本为：

```text
toolkit/src/nic_monitor/nic_monitor_pro.sh
```

稳定命令入口：

```bash
ucm-toolkit run nic-monitor
```

脚本支持两种模式：

- `fg [interval_sec]`：前台动态展示，默认 2 秒刷新。
- `bg [duration_hours] [interval_sec]`：后台守护监控，默认 12 小时。

示例：

```bash
ucm-toolkit run nic-monitor fg
ucm-toolkit run nic-monitor fg 5
ucm-toolkit run nic-monitor bg
ucm-toolkit run nic-monitor bg 24 5
ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log
ucm-toolkit run nic-monitor bg 24 5 --stat-cycle-seconds 600
```

adapter 使用 `bash` 调用该脚本并转发参数。该工具不设计主动打流、主动压测或流量生成能力。底层脚本使用
`ethtool` 读取网卡信息和统计计数，运行时需要 root 或 sudo 权限。

后台日志默认写到当前执行命令目录下的 `net_log` 目录。可通过 `--log-dir` 覆盖。后台阶段统计周期默认
3600 秒，可通过 `--stat-cycle-seconds` 覆盖。

## 打包设计

`ucm-toolkit` 应作为独立 Python 包安装，不随主 `uc-manager` 包默认安装，避免干扰主包安装、依赖和发布流程。

`toolkit/pyproject.toml` 建议定义独立包：

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ucm-toolkit"
requires-python = ">=3.10"
dynamic = ["version"]

[project.scripts]
ucm-toolkit = "ucm_toolkit.cli:main"
```

第一版建议只使用 Python 标准库：

- `argparse`
- `pathlib`
- `subprocess`
- `os`
- `sys`
- `shutil`

这样可以减少在 CUDA、Ascend、容器和 CI 环境中的额外依赖。

## 文档设计

`toolkit/README.md` 作为主入口文档，建议包含：

- 项目简介
- 安装方式
- 快速开始
- 通用 CLI 模式
- 可构建工具
- `dev-sandbox` 构建与 copy 带宽测试
- POSIX AIO 测试
- 网卡负载监控
- 常见问题

每个工具 adapter 目录下保留自己的 `README.md`，用于记录该工具的原生命令、参数和示例。

CLI 的 `--help` 与 README 中的示例应保持同步。新增 CLI 参数时，应同步更新对应文档。

## 错误处理

toolkit 应尽早失败，并给出可操作的错误信息：

- 工具名不存在。
- 对不可构建工具执行 build。
- `cmake` 不存在。
- 构建目录不存在。
- 构建产物不存在。
- 原生脚本路径不存在。
- 子进程执行失败。

子进程失败时，应打印失败命令和退出码，方便用户复现。

## 实施顺序

1. 新增 Python 包骨架和 `ucm-toolkit --help`。
2. 新增 registry 和 adapter 接口。
3. 新增 `list`、`doctor`、`build`、`run`、`clean` 命令。
4. 接入 `dev-sandbox` 默认 CMake 构建，保持原自动探测行为。
5. 为 `dev-sandbox` 构建增加 `--cmake-arg` 透传。
6. 接入 `dev-sandbox` run 子功能，支持 `ucm-toolkit run dev-sandbox copy/trans/aio ...`。
7. 接入 POSIX AIO，并为脚本补充保留默认值的参数覆盖入口。
8. 接入网卡负载被动监控脚本。
9. 新增 `toolkit/README.md` 和各工具 README。

## 已确认决策

- POSIX AIO 脚本路径为 `ucm/store/test/e2e/posixstore_aio_test.py`。
- POSIX AIO 保留脚本现有默认值，同时提供 CLI 参数覆盖入口。
- `dev-sandbox` 构建产物不 install 到单独 bin 目录；构建在哪里，运行时就从哪里查找。
- toolkit 不引入额外状态文件记录 build 目录。
- 不再设计 `config.py`；工具路径、脚本路径、二进制路径、build 目录都由 registry 中的工具对象持有。
- `DevSandboxTool.build_dir` 默认值为 `toolkit/src/dev-sandbox/build`。
- 只有 `ucm-toolkit build dev-sandbox` 支持 `--build-dir`；指定并构建成功后，同步写回 Python 源文件中的
  `DevSandboxTool.build_dir`。
- `ucm-toolkit run dev-sandbox copy/trans/aio` 不支持 `--build-dir`，只从 `DevSandboxTool.build_dir` 读取 build 目录。
- `ucm-toolkit list` 只展示顶层工具，不展示 `copy/trans/aio`；这些子功能通过 `ucm-toolkit run dev-sandbox --help` 暴露。
- 当前不在 toolkit 中新增 `--platform cuda|ascend` 语义；除非未来 `dev-sandbox` 原项目自己支持显式后端选择参数。
- 网卡负载监控当前阶段只支持被动监控，不支持主动压测。
- `ucm-toolkit` 作为独立包安装，不随主 `uc-manager` 包默认安装。
