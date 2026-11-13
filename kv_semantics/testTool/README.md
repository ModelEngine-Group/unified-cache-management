# testTool

kv_semantics 测试工具（C++17），开发中。

## 目录结构

- `src/` 源码（`log` / `request` / `utils` / `worker` 模块；`test/` 为临时测试目录，开发完成后删除）
- `conf/` 运行时配置
- `scripts/` 辅助脚本

## 构建（Docker 开发容器内）

```bash
docker exec -it testtool-dev bash   # 进入容器，工作目录即 /workspace/testTool
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
./build/testTool
```

## 说明

- `src/` 下源文件由 GLOB 自动收集（CONFIGURE_DEPENDS），新增/删除 `.cc`、`.cpp` 文件无需修改 CMakeLists。
- 第三方依赖（fmt / spdlog）复用父工程 `cmake/vendor/dep-*.cmake`，默认 FetchContent 在线拉取，版本与 kv_semantics 一致；离线时 `-DDOWNLOAD_DEPENDENCE=OFF` 并在本目录放置 `fmt/`、`spdlog/` 源码。
- 编译选项与主工程一致：`-Wall -Werror`，Debug 为 `-O0 -g`，Release 为 `-O2` + 栈保护。
- 构建产物在 `build/`（已 gitignore）。

## 编码规范

> 本节与 `AGENTS.md`（agent 指令版）内容对应，修改任一处须保持同步。

- 遵循 C++ 工程原则，模块化设计：功能按 `src/` 下模块目录（`log` / `request` / `utils` / `worker` / `test`）划分，职责单一；`test/` 为临时测试目录，开发完成后整体删除。
- **版权标记**：`src/` 下所有 C/C++ 源码文件（`.h` / `.cc` / `.cpp`）顶部必须包含版权声明 `Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.`（多行注释形式）；CMakeLists.txt、Dockerfile、conf 等非 C/C++ 文件不加。
- 头文件统一使用 `#pragma once` 防止重复包含，不使用传统 include 守卫。
- 方法（函数）前需有中文注释说明职责，一句话即可；仅参数较多或含义不直观时补充参数与返回值说明，简单参数不写；方法体内只在关键逻辑处注释，不逐行注释。
- 默认不使用 namespace，类名直接全局命名；确需 namespace 时另行约定。
- 类名与文件名保持一致：CamelCase 类名对应 snake_case 文件名（如 `LoggerManager` → `logger_manager.h`），仅允许大小写与下划线差异。
- CMake 文件分段编写，每段以注释标明该段内容的作用。

## 可移植性约束（重要）

代码后续可能在 **CPU / GPU / NPU** 上执行，架构覆盖 **ARM / x86**，现阶段遵循：

- **不引入难以适配的第三方库**；新增依赖前必须确认同时支持 arm64 与 x86_64（首选纯 C++ 实现的库）。
- 优先使用 C++17 标准库等可移植能力，不直接调用平台相关接口。
- 不写死架构假设：不使用无编译开关保护的 SIMD / 内联汇编等架构相关特性；确需优化时通过预处理宏隔离并提供通用回退实现。
- GPU / NPU 设备相关能力一律通过接口抽象隔离，实现可替换，业务代码不感知具体设备。
- 构建选项（编译器、FLAGS）保持架构无关，由目标平台原生工具链编译，不做交叉编译特殊处理。
