# Toolkit 架构与扩展

UCM Toolkit 为预检、指标查看和原生性能测试提供统一命令入口。各工具依赖不同，CLI 负责分派命令，Adapter 负责工具自己的参数、资源路径和运行方式。Toolkit 是独立 Python 包，安装与使用见[工具集](../index.md)。

## 一条命令如何执行

以 `ucm-toolkit run dev-sandbox copy ...` 为例：

1. `cli.main()` 初始化内置工具注册表，识别 `run` 命令。
2. `commands/run.py` 用工具名或别名查询 Registry，将剩余参数交给 Adapter。
3. `DevSandboxTool` 从 `build_dir` 和 `subcommands` 找到 `copy` 二进制，透传原生参数。
4. `runner.py` 启动子进程并返回退出码。CLI 将 `ToolkitError` 转为可读错误和对应退出码。

顶层 CLI 不需要理解 copy 的测试类型或设备参数。对于 `dev-sandbox` 的便捷模式，Adapter 将 `--model-type`、`--iodirect`、`--sdma` 映射为已有 Ascend copy case，再透传其余参数。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `cli.py` 与 `commands/` | 解析顶层动作，选择工具并返回结果 |
| `registry.py` | 保存工具对象及别名，定义 `ToolAdapter`，维护工具注册关系 |
| `tools/<tool>/` | 保存该工具的参数、依赖检查、构建、运行与清理规则 |
| `runner.py` | 检查命令、运行子进程、传递失败 |
| `errors.py` | 为 CLI 错误定义可读信息和退出码 |

`list` 展示顶层工具，`doctor` 检查各工具环境。只有声明可构建的工具进入 `build`；当前为 `dev-sandbox`。工具内部子命令由该工具的 `--help` 展示。

## 构建目录与运行资源

`DevSandboxTool` 从安装包读取 CMake/C++ 源码，执行 configure 和 build。构建产物默认放在用户缓存目录，按 Toolkit 安装位置和版本隔离。显式 `--build-dir` 相对当前工作目录解析，并仅在构建成功后保存为绝对路径。

`run`、`doctor` 和 `clean` 读取同一份用户状态，不再修改 Adapter 源文件。`clean --dry-run` 显示目标；清理前验证该目录确实属于当前 dev-sandbox 的 CMake 构建。

## 接入一个工具

1. 在 `tools/` 中实现 `ToolAdapter`，定义唯一名称、别名和说明；将工具特有路径和规则保留在该模块内。
2. 实现 `run(tool_args)` 与 `doctor()`。需要编译时声明 `buildable=True` 并实现构建参数及 `build()`；只为实际生成的产物实现清理。
3. 在 `init_builtin_tools()` 中导入并注册实例。重复名称或别名会被 Registry 拒绝。
4. 使用参数列表调用 `runner`；工具自行处理参数语义，不在顶层 CLI 增加与某个工具耦合的分支。
5. 更新用户指南，验证 list、help、doctor、运行退出码，以及需要时的构建和清理。

可参考 `tools/nic_monitor/` 的脚本调用或 `tools/dev_sandbox/adapter.py` 的原生构建。新增工具不要求它拥有所有生命周期动作，未支持的操作应按现有错误约定处理。

## 维护与验证

单元验证优先覆盖命令分派、参数透传和错误语义。原生构建或带宽结果需要对应设备与运行环境，CLI 分派通过不能替代这些验证。`precheck` 的检查结果与阈值含义见[用户文档](../user/precheck.md)，原生测试开发见[开发沙箱](dev-sandbox.md)。
