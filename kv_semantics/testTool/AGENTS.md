# AGENTS.md — testTool

kv_semantics 测试工具（C++17）子树的 agent 约定。仓库根 `AGENTS.md` 与 `kv_semantics/AGENTS.md` 的通用规则同样适用；本文件只覆盖 testTool 特有内容，冲突时以本文件为准。

## 变更范围（硬性）

- **只允许修改 `testTool/` 目录内的内容**。其余目录（`src/`、`tests/`、`metrics/`、`cmake/` 等）一律只读：构建脚本可按需引用（如 `../cmake/vendor/dep-*.cmake`），但不得改动其中任何文件。

## 构建环境与命令

在 testTool 目录下编译（Linux 环境，工具链与主工程一致）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release # 缺省 Release（产物随时可用于性能测试）；调试构建显式改 -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j                        # 增量编译
./build/testTool
```

- fmt / spdlog 经 FetchContent 在线拉取，版本由父工程 `cmake/vendor/dep-*.cmake` 锁定，与主工程保持一致；首次配置需网络，失败时先检查网络再查脚本。
- 依赖缓存位于 `build/_deps/`，删除整个 `build/` 目录即可重置（或用 `bash scripts/clean-build.sh` 统一清理构建目录）。

## 验证流程约定（硬性）

- agent 自测需要改配置时，**复制默认配置为独立测试配置文件**（如 `conf/test_config_agent_verify.conf`），只改副本、用副本运行验证，结束后删除副本；**不得原地修改 `conf/test_config.conf` 等交付给使用方的默认配置**。验证用临时 input 文件同样遵循"新建独立文件、用完删除"模式，不改 `input/` 下已有样例。
- 配置文件的修改一律使用 edit 工具或新建文件完成；**禁止用 sed 原地改写**（PowerShell 传参会破坏 sed 表达式，已发生过 conf 损坏事故）。

## 编码规范

- **版权标记**：`src/` 下所有 C/C++ 源码文件（`.h` / `.cc` / `.cpp`）文件顶部必须包含与仓库统一的完整 MIT License 声明（与 `kv_semantics` 现有源文件如 `src/common/utils/parser_common.cc` 的形态一致；其后接空行再接头文件守卫或代码正文；CMakeLists.txt、conf 等非 C/C++ 文件不加）：

  ```c
  /**
   * MIT License
   *
   * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in all
   * copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   * SOFTWARE.
   * */
  ```

- **clang-format 风格（硬性）**：全部 C/C++ 代码必须与 git 仓库一级根目录（`unified-cache-management/`）下的 `.clang-format`（Google base、IndentWidth 4、ColumnLimit 100）保持一致，改动的源文件交付前必须过 `clang-format -i`（CI 对 PR 改动行做 clang-format 检查，任何 diff 均导致 PR 失败）。环境未预置该工具时先 `apt-get install clang-format`，以 `clang-format --style=file:<仓库根>/.clang-format -i <文件>` 执行。
- 头文件统一使用 `#pragma once` 防止重复包含，不使用传统 include 守卫。

- 模块化设计：`src/` 根放置顶层编排（`main`、`scheduler`），功能按 `src/` 下模块目录（`log` / `request` / `utils` / `worker`）划分，职责单一；模块间通过头文件公开接口交互，不直接依赖其他模块内部实现细节。
- 方法（函数）前需有中文注释说明职责，一句话即可；仅参数较多或含义不直观时补充 `@param` / `@return`，简单参数不写；方法体内只在关键逻辑处注释，不逐行注释。
- CMake 文件分段编写，每段以注释标明该段内容的作用。
- 默认不使用 namespace，类名直接全局命名；确需 namespace 时另行约定。
- 类名与文件名保持一致：CamelCase 类名对应 snake_case 文件名（如 `LoggerManager` → `logger_manager.h`），仅允许大小写与下划线差异。
- 新增源文件放入 `src/` 对应模块目录（扩展名优先 `.cc`，兼容 `.cpp`）；GLOB 自动收集（CONFIGURE_DEPENDS），无需修改 CMakeLists。
- 头文件引用以 `src/` 为根，形如 `#include "log/xxx.h"`，保持模块前缀语义清晰。

## 可移植性约束（硬性）

代码后续可能在 CPU / GPU / NPU 上执行，架构覆盖 ARM / x86，现阶段：

- **禁止引入难以适配的第三方库**：新增依赖前必须确认 arm64 与 x86_64 均可编译，首选纯 C++ 实现的库。
- 优先使用 C++17 标准库等可移植能力，禁止直接调用平台相关接口。
- 禁止使用无编译开关保护的 SIMD / 内联汇编等架构相关特性；确需优化时通过预处理宏隔离，并提供通用回退实现。
- GPU / NPU 设备相关能力一律通过接口抽象隔离，实现可替换，业务代码不感知具体设备。
- 编译选项（编译器、FLAGS）保持架构无关，由目标平台原生工具链编译，不做交叉编译特殊处理。

以上规范同时以人类可读形式记录于 `README.md`（「编码规范」「可移植性约束」两节），两处修改必须保持同步。
