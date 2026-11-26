# testTool

kv_semantics 测试工具（C++17），开发中。

## 目录结构

- `src/` 源码（根目录为顶层编排 `main` / `scheduler`；模块目录 `log` / `request` / `utils` / `worker`）
- `conf/` 运行时配置（两件套 `test_config.conf` / `kv_client.conf`，simu 环境缺省直接可用；RoCE / HCCP 真机验证复制副本修改，模板段已内置，见 `sop/`）
- `sop/` 建链断链 SOP（`0.build-and-run-sop.md` 原理与排障 / `1.cpu-connect-disconnect-sop.md` / `2.npu-connect-disconnect-sop.md`）
- `input/` 流量模型输入文件（`input.txt` 为驱动源，格式见 `input/readme.md`）
- `scripts/` 辅助脚本（`build-kv-link.sh` 建链测试一键编译，CPU/NPU 双模式；`clean-build.sh` 清理构建目录与编译缓存）

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 缺省 Release：产物随时可用于性能测试；调试构建显式改 -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
./build/testTool <test config path> <kv client config path> [input file path, default input.txt]
# 执行模式（onlyconnect=off）：input 可缺省，缺省取启动目录下 input.txt（相对当前工作目录查找）；例：./build/testTool conf/test_config.conf conf/kv_client.conf input/input.txt
# 只建链模式（onlyconnect=on）：input 不使用（传入也会被忽略）
```

## 说明

- `src/` 下源文件由 GLOB 自动收集（CONFIGURE_DEPENDS），新增/删除 `.cc`、`.cpp` 文件无需修改 CMakeLists。
- 第三方依赖（fmt / spdlog）复用父工程 `cmake/vendor/dep-*.cmake`，默认 FetchContent 在线拉取，版本与 kv_semantics 一致；离线时 `-DDOWNLOAD_DEPENDENCE=OFF` 并在本目录放置 `fmt/`、`spdlog/` 源码。
- 编译选项与主工程一致：`-Wall -Werror`，Debug 为 `-O0 -g`，Release 为 `-O2` + 栈保护。
- 构建产物在 `build/`（已 gitignore）。

## 编码规范

> 本节与 `AGENTS.md`（agent 指令版）内容对应，修改任一处须保持同步。

- 遵循 C++ 工程原则，模块化设计：`src/` 根放置顶层编排（`main`、`scheduler`），功能按 `src/` 下模块目录（`log` / `request` / `utils` / `worker`）划分，职责单一。
- **版权标记**：`src/` 下所有 C/C++ 源码文件（`.h` / `.cc` / `.cpp`）顶部必须包含与仓库统一的完整 MIT License 声明（与 `kv_semantics` 现有源文件同形态，见 `testTool/AGENTS.md` 中的完整模板）；CMakeLists.txt、conf 等非 C/C++ 文件不加。
- **clang-format 风格（硬性）**：全部 C/C++ 源码必须与 git 仓库一级根目录（`unified-cache-management/`）下的 `.clang-format`（Google base、4 空格缩进、100 列行宽）保持一致；改动文件交付前执行 `clang-format -i`（CI 对 PR 改动行做 clang-format 检查，任何 diff 均导致 PR 失败）。宿主机先安装 clang-format（如 `apt-get install clang-format`），再以 `clang-format --style=file:<仓库根>/.clang-format -i <文件>` 执行。
- 头文件统一使用 `#pragma once` 防止重复包含，不使用传统 include 守卫。
- 方法（函数）注释按需写：签名与命名已能自明（如一行 getter / setter）时不写；仅职责无法从签名直观理解时写一句话；参数较多或含义不直观时补充参数与返回值说明；方法体内只在关键逻辑处注释，不逐行注释。
- **注释保持精简（防膨胀）**：注释只写"看代码看不出来的东西"（坑、顺序约束、异常契约等关键语义）；工作笔记——历史沿革、决策过程、方案对比、"不再 / 沿用某做法"等过程性叙述一律不写；单条注释超过两行即视为膨胀，应精简注释而不是扩写。
- 默认不使用 namespace，类名直接全局命名；确需 namespace 时另行约定。
- 类名与文件名保持一致：CamelCase 类名对应 snake_case 文件名（如 `LoggerManager` → `logger_manager.h`），仅允许大小写与下划线差异。
- 新增源文件放入 `src/` 对应模块目录（扩展名优先 `.cc`，兼容 `.cpp`）；GLOB 自动收集（CONFIGURE_DEPENDS），无需修改 CMakeLists。
- 头文件引用以 `src/` 为根，形如 `#include "log/xxx.h"`，保持模块前缀语义清晰。
- CMake 文件分段编写，每段以注释标明该段内容的作用。

## 可移植性约束（重要）

代码后续可能在 **CPU / GPU / NPU** 上执行，架构覆盖 **ARM / x86**，现阶段遵循：

- **不引入难以适配的第三方库**；新增依赖前必须确认同时支持 arm64 与 x86_64（首选纯 C++ 实现的库）。
- 优先使用 C++17 标准库等可移植能力，不直接调用平台相关接口。
- 不写死架构假设：不使用无编译开关保护的 SIMD / 内联汇编等架构相关特性；确需优化时通过预处理宏隔离并提供通用回退实现。
- GPU / NPU 设备相关能力一律通过接口抽象隔离，实现可替换，业务代码不感知具体设备。
- 构建选项（编译器、FLAGS）保持架构无关，由目标平台原生工具链编译，不做交叉编译特殊处理。
