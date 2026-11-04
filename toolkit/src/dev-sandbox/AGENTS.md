# AGENTS.md

## 项目概述

C++17 项目，使用 CMake 构建系统，支持 CUDA、Ascend（华为 NPU）和 CPU 模拟三种运行模式。

## 构建命令

```bash
# 配置项目（默认 Release）
cmake -B build

# 配置 Debug 模式
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# 构建
cmake --build build -j
```

## 代码格式化

- 使用 `.clang-format`（基于 Google 风格的自定义配置）
- CI 会严格检查代码格式，格式错误会导致构建失败
- 建议在提交前运行：`clang-format -i <file>`

## 硬件条件编译

项目会根据环境自动选择编译目标：

**检测优先级**：
1. CACHE 变量：`-DCUDA_ROOT=...` 或 `-DASCEND_ROOT=...`
2. 环境变量：`CUDA_HOME`/`CUDA_PATH` 或 `ASCEND_HOME`/`ASCEND_TOOLKIT_HOME`
3. 默认路径：`/usr/local/cuda` 或 `/usr/local/Ascend/ascend-toolkit/latest`

**编译目标**：
- **CUDA**: 检测成功后编译 CUDA 内核（`*.cu`）
- **GDR**: CUDA 后端下检测到 `libibverbs` 时，额外编译 GDR 拷贝测试实现
- **Ascend**: 检测成功后编译 Ascend 实现
- **模拟模式**: 无 GPU 时编译纯 CPU 模拟版本

相关源文件位于 `module/*/cuda`、`module/*/ascend`、`module/*/simu` 子目录。
GDR 实现位于 `module/copy/cuda/gdr`，复用 `copy` 的公共基类体系，并注册到 `copy`
主程序中。

## 模块结构

- `cmake/`: CMake 模块（`DetectRuntime.cmake` 运行时检测）
- `module/aio`: 异步 I/O 性能测试工具
- `module/copy`: 设备间内存拷贝性能测试
- `module/copy/cuda/gdr`: CUDA GDR/RDMA 拷贝测试实现
- `module/logger`: 日志库
- `module/trans`: 数据传输性能测试
- `vendor/`: 第三方依赖源码

## 依赖管理

第三方依赖源码放在 `vendor/` 下：
- fmt (11.2.0): `vendor/fmt`

## CI 流程

每次推送触发：lint -> build (Debug) -> build (Release)

- Lint 使用 cpp-linter-action，只检查变更的文件
- 构建失败会立即停止

## 注意事项

- 编译器标志包含 `-Werror`，所有警告都会导致编译失败
- `compile_commands.json` 在构建时自动生成在 `build/` 目录
- 构建产物位于 `build/module/` 下各模块目录
