# CacheStore GDR Copy Engine 接入方案

## 背景

当前 `ucm/store/cache` 的 GPU 与 host 之间数据搬运，仍然是基于 `UC::Trans::Stream` 的 `cudaMemcpy/cudaMemcpyAsync`。

现有关键路径如下：

- `LoadQueue` 在后端数据回填后，执行 host -> device scatter：
  - `ucm/store/cache/cc/load_queue.cc`
  - 关键方法：`LoadQueue::HostToDeviceScatterAsync`
- `DumpQueue` 在写回后端前，执行 device -> host gather：
  - `ucm/store/cache/cc/dump_queue.cc`
  - 关键方法：`DumpQueue::DeviceToHostGatherAsync`
- 更底层的 CUDA 拷贝实现在：
  - `ucm/shared/trans/cuda/cuda_stream.cc`
  - `ucm/shared/trans/stream.h`

你现在希望新增一套基于 GDR 的 H2D / D2H 路径，并且已经有 `gdrcopy.c/.h`，同时可以参考 `D:\Project\gdr` 里的 `gdr_copy.h` / `gdr_copy.cpp` 实现。

这件事的核心不是“把一段 GDR 代码塞进某个 `.cc` 文件里”，而是要在 `cache store` 内部补上一层**可切换的 copy engine 抽象**，让 `cudaMemcpy` 和 `gdrdma` 成为两个后端实现。

---

## 目标

本方案的目标是：

1. 在 `ucm/store/cache` 中新增一个可配置的 copy engine。
2. 默认行为保持不变，仍然走 `cudaMemcpyAsync`。
3. 当配置为 `gdrdma` 时，`LoadQueue` / `DumpQueue` 的 H2D / D2H 改走 GDR 通道。
4. 接入方式尽量局部，避免污染 `ucm/shared/trans` 的通用 CUDA stream 抽象。
5. 在无 `ibverbs`、无 `nvidia-peermem`、无兼容 NIC 时，支持清晰的失败或回退策略。

非目标：

- 本次不把 GDR 抽象推广到 `pcstore`、`nfsstore`、`shared/trans` 全局共用。
- 本次不要求一开始就做多 channel 并发最优实现。
- 本次不修改外部 backend 的协议，只替换 store 内部 copy engine。

---

## 为什么不建议直接改 `ucm/shared/trans`

虽然当前 `cudaMemcpyAsync` 最底层在 `ucm/shared/trans`，但 GDR 并不适合直接并入现有 `Trans::Stream` 抽象，原因有三点：

1. `Trans::Stream` 本质上是“GPU stream 风格”的接口，带有 `WaitEvent`、`AppendCallback`、`Synchronized` 等 CUDA stream 语义。
2. GDR 的参考实现 `D:\Project\gdr\include\gdr_copy.h` 更像“RDMA copy channel”，核心是 `memcpy_async`、`poll_wc`、`sync`，不是 CUDA stream。
3. `cache store` 当前真正需要的是 host <-> device 的 gather/scatter copy engine，不是一个新的通用 runtime stream。

所以更稳妥的做法是：

- 保持 `ucm/shared/trans` 继续负责通用 `cudaMemcpy`。
- 在 `ucm/store/cache` 内部新增 `CopyEngine` 抽象。
- `cudaMemcpy` 和 `gdrdma` 都实现这个抽象。

这样改动范围最小，也最容易回退。

---

## 推荐接入位置

推荐把 GDR 代码加在 `ucm/store/cache/cc/` 下面，而不是直接加到 `ucm/shared/trans/`。

建议目录结构如下：

```text
ucm/store/cache/cc/
  copy_stream.h                 # 保留现有 CUDA stream 池
  copy_engine.h                 # 新增：copy engine 抽象
  copy_engine_factory.h         # 新增：工厂
  cuda_copy_engine.h            # 新增：基于现有 CopyStream 的实现
  cuda_copy_engine.cc
  gdr/
    gdr_copy_engine.h           # 新增：GDR 实现
    gdr_copy_engine.cc
    gdrcopy_adapter.h           # 新增：把 gdrcopy API 封成 UCM 风格
    gdrcopy_adapter.cc
    third_party/
      gdr_copy.h                # 参考 D:\\Project\\gdr\\include\\gdr_copy.h
      gdr_copy.cc               # 参考 D:\\Project\\gdr\\src\\gdr_copy.cpp
      mr_cache.h                # 若参考实现依赖
      pinned_pool.h             # 若参考实现依赖
```

如果你手里现在真的是 `gdrcopy.c/.h`，而不是 `.cc/.h`，我的建议是：

- **优先方案**：改成 `.cc/.h` 或者提供一个 `gdrcopy_adapter.cc` 做 C++ 封装。
- **不推荐方案**：为了这个文件把整个工程改成 `project(... LANGUAGES C CXX)`。

原因很简单：UCM 当前顶层 CMake 是 `project(... LANGUAGES CXX)`，整个 store 侧也是 C++17 风格。为了一个 GDR 文件引入 C 语言编译链，收益不高，维护成本更高。

---

## 建议新增的抽象

### 1. 新增 `CopyEngine`

建议在 `ucm/store/cache/cc/copy_engine.h` 定义一层 store 内部接口：

```cpp
class CopyEngine {
public:
    virtual ~CopyEngine() = default;

    virtual Status Setup() = 0;

    virtual Status WaitPrerequisite(void* event) = 0;

    virtual Status HostToDeviceScatterAsync(
        void* host, void** device, const std::vector<size_t>& tensorSizes) = 0;

    virtual Status DeviceToHostGatherAsync(
        void** device, void* host, const std::vector<size_t>& tensorSizes) = 0;

    virtual Status Synchronize() = 0;
};
```

这里故意不暴露更底层的单次 `memcpy`，而是直接暴露 `cache store` 真正需要的两个动作：

- `HostToDeviceScatterAsync`
- `DeviceToHostGatherAsync`

这样可以把 `LoadQueue` / `DumpQueue` 当前的 offset 循环一起收口。

### 2. `CudaCopyEngine`

`CudaCopyEngine` 直接复用现有 `CopyStream`：

- `WaitPrerequisite` -> `CopyStream::WaitEvent`
- `HostToDeviceScatterAsync` -> 现有 `LoadQueue::HostToDeviceScatterAsync` 逻辑
- `DeviceToHostGatherAsync` -> 现有 `DumpQueue::DeviceToHostGatherAsync` 逻辑
- `Synchronize` -> `CopyStream::Synchronize`

这一步先把现有逻辑搬家，不改行为。

### 3. `GdrCopyEngine`

`GdrCopyEngine` 封装 GDR 参考实现，接口保持和 `CudaCopyEngine` 一致。

内部做法建议如下：

- `HostToDeviceScatterAsync`
  - 对 `tensorSizes` 做 offset 遍历
  - 每个 tensor 调用一次 GDR H2D submit
- `DeviceToHostGatherAsync`
  - 对 `tensorSizes` 做 offset 遍历
  - 每个 tensor 调用一次 GDR D2H submit
- `Synchronize`
  - 负责 drain 当前 engine 内部所有未完成请求
- `WaitPrerequisite`
  - 对 CUDA event 直接执行 `cudaEventSynchronize`

这里有一个关键点：

现有 CUDA 路径的 `WaitEvent` 是“让 copy stream 等待 event”，而 GDR 路径没有 CUDA stream 语义，所以最直接可靠的做法是：

```cpp
cudaEventSynchronize(static_cast<cudaEvent_t>(event));
```

这会把等待从“GPU stream 内部依赖”变成“CPU 线程阻塞等待”，语义更保守，但正确性最好，适合作为第一版。

---

## 为什么建议先做 store 内部 `CopyEngine`，再接 GDR

因为这样可以分两步落地：

### Phase 1：只做抽象收口

先不接 GDR，只做下面两件事：

1. 把 `LoadQueue::HostToDeviceScatterAsync` 搬到 `CudaCopyEngine`
2. 把 `DumpQueue::DeviceToHostGatherAsync` 搬到 `CudaCopyEngine`

完成后，`LoadQueue` / `DumpQueue` 只依赖 `CopyEngine`。

### Phase 2：加 `GdrCopyEngine`

当抽象稳定后，再新增 `GdrCopyEngine` 和配置开关。

这样做的好处是：

- 每一步都能单独验证；
- 第一阶段不会引入额外依赖；
- 第二阶段出了问题，直接切回 `cudaMemcpy` 即可。

---

## 配置设计

建议在 `ucm/store/cache/cc/global_config.h` 的 `Config` 里新增：

```cpp
std::string copyEngine{"cudaMemcpy"};
std::string gdrNicName{};
bool gdrUseOdp{false};
bool gdrStrict{false};
size_t gdrChannelNumber{1};
```

并在 `ucm/store/cache/cc/cache_store.cc` 的 `ParseConfig` 中读取：

```cpp
config.Get("cache_copy_engine", param.copyEngine);
config.Get("gdr_nic_name", param.gdrNicName);
config.Get("gdr_use_odp", param.gdrUseOdp);
config.Get("gdr_strict", param.gdrStrict);
config.GetNumber("gdr_channel_number", param.gdrChannelNumber);
```

建议配置含义如下：

- `cache_copy_engine`
  - `cudaMemcpy`
  - `gdrdma`
- `gdr_nic_name`
  - 例如 `mlx5_0`
- `gdr_use_odp`
  - 直接透传给 GDR 实现
- `gdr_strict`
  - `true`：GDR 初始化失败则 `Setup()` 失败
  - `false`：GDR 初始化失败则回退到 `cudaMemcpy`
- `gdr_channel_number`
  - 第一版可以先默认为 `1`
  - 后续如果 GDR 要做多 channel 并发，再启用

示例配置：

```yaml
ucm_connector_config:
  cache_copy_engine: gdrdma
  gdr_nic_name: mlx5_0
  gdr_use_odp: false
  gdr_strict: false
  gdr_channel_number: 1
```

---

## 对 `D:\Project\gdr` 参考实现的接入建议

### 参考实现里哪些内容值得复用

`D:\Project\gdr` 里有几类内容：

1. 对外接口：`include/gdr_copy.h`
2. 核心实现：`src/gdr_copy.cpp`
3. 辅助类：`mr_cache.h`、`pinned_pool.h`
4. bench / demo / nixl 对照代码

对于 UCM 来说，真正有价值的是前 3 类，bench 和 demo 不建议直接并入主仓。

### 推荐复用方式

推荐把参考实现裁成“库代码 + UCM 适配层”两层：

#### 层 1：vendor 的 GDR 基础实现

这一层尽量保持接近 `D:\Project\gdr`，便于后续同步：

```text
ucm/store/cache/cc/gdr/third_party/
  gdr_copy.h
  gdr_copy.cc
  mr_cache.h
  pinned_pool.h
```

#### 层 2：UCM 适配层

这一层只做三件事：

1. 把 GDR 错误码转换成 `Status`
2. 把 GDR 的 async completion 模型转换成 `CopyEngine::Synchronize`
3. 处理 UCM 的配置、日志、回退逻辑

例如：

```text
ucm/store/cache/cc/gdr/
  gdrcopy_adapter.h
  gdrcopy_adapter.cc
  gdr_copy_engine.h
  gdr_copy_engine.cc
```

这样做比把 `gdr_copy.cpp` 直接塞进 `load_queue.cc` 或 `dump_queue.cc` 干净得多。

---

## GDR async 语义如何映射到 UCM

`D:\Project\gdr\include\gdr_copy.h` 的接口有几个关键点：

- `memcpy_async`
- `GdrMemcpyAsync`
- `PollCompletion`
- `sync`

而 UCM 当前 cache store 的使用方式是：

1. 先提交一批 H2D / D2H
2. 再调用一次 `Synchronize`

所以映射关系可以这样设计：

### UCM -> GDR

- `HostToDeviceScatterAsync`
  - 对每个 tensor 调用 `GdrMemcpyAsync(..., GDR_H2D, ...)`
  - 把返回的 `req_id` 记录到 engine 内部队列
- `DeviceToHostGatherAsync`
  - 对每个 tensor 调用 `GdrMemcpyAsync(..., GDR_D2H, ...)`
  - 同样记录 `req_id`
- `Synchronize`
  - 循环调用 `PollCompletion` / `sync`
  - 直到当前 engine 提交的请求全部完成

### 需要注意的地方

1. GDR 的 `memcpy()` 在参考实现里其实也是 submit-only 语义，不是阻塞 copy。
2. 因此 UCM 侧不能把它当同步接口使用。
3. `Synchronize()` 必须明确 drain 当前 batch 的请求，否则 `Wait()` 语义会被破坏。

我的建议是：**UCM 适配层统一只用 `GdrMemcpyAsync`，不要混用 `memcpy`。**

---

## 单 channel 还是多 channel

这里建议分阶段：

### 第一版：单 channel

`GdrCopyEngine` 内部只持有一个 `GDRCopyChannel`。

原因：

- `LoadQueue::TransferStage` 本身只有一个 transfer 线程；
- `DumpQueue::DispatchStage` 本身也只有一个 dispatch 线程；
- 单 channel 已经能利用 GDR 的异步请求队列；
- 改动最小，调试最容易。

### 第二版：多 channel

如果后续 benchmark 证明单 channel 不够，再做 channel 池：

```cpp
std::vector<std::shared_ptr<GDRCopyChannel>> channels_;
size_t channelIndex_{0};
```

但是这里要注意：`D:\Project\gdr` 当前 `GDRCopyLib::open(gpu_id, nic_name)` 带缓存语义，可能会把同一 `(gpu, nic)` 返回成同一个 channel。

所以如果你要做真正的多 channel，需要先改参考实现，至少满足下面之一：

1. 提供 `CreateChannel()`，每次返回独立实例；
2. 或者把缓存 key 从 `(gpu_id, nic_name)` 扩成 `(gpu_id, nic_name, channel_idx)`；
3. 或者允许显式关闭缓存。

因此对 UCM 第一版来说，**先做单 channel 更稳**。

---

## CMake 接入建议

当前 `ucm/store/cache/CMakeLists.txt` 很简单，只是把 `./cc/*.cc` 编进 `cachestore`。

建议增加一个显式选项：

```cmake
option(UCM_ENABLE_GDRDMA "Enable GDR copy engine for cache store" OFF)
```

然后在 `ucm/store/cache/CMakeLists.txt` 里按条件编译：

```cmake
if(RUNTIME_ENVIRONMENT STREQUAL "cuda" AND UCM_ENABLE_GDRDMA)
  find_package(CUDAToolkit REQUIRED)
  find_library(IBVERBS_LIB NAMES ibverbs)

  target_sources(cachestore PRIVATE
    cc/gdr/gdrcopy_adapter.cc
    cc/gdr/gdr_copy_engine.cc
    cc/gdr/third_party/gdr_copy.cc
  )

  target_include_directories(cachestore PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/gdr/third_party
  )

  target_link_libraries(cachestore PUBLIC
    ${IBVERBS_LIB}
    pthread
  )

  target_compile_definitions(cachestore PUBLIC UCM_ENABLE_GDRDMA=1)
endif()
```

### 这里的几个注意点

1. GDR 只应该在 `RUNTIME_ENVIRONMENT=cuda` 下启用。
2. GDR 只应该在 Linux 下启用。
3. 如果 `ibverbs` 或 CUDA toolkit 缺失，建议：
   - `UCM_ENABLE_GDRDMA=ON` 时直接 CMake fail fast；
   - `UCM_ENABLE_GDRDMA=OFF` 时完全不感知 GDR。
4. 不建议让 `cachestore` 在所有平台都无条件链接 `ibverbs`。

---

## `gdrcopy.c/.h` 的工程化处理建议

如果你当前手上的文件叫 `gdrcopy.c/.h`，我建议做一次整理，而不是原样扔进仓库：

### 推荐方式

1. 把外部接口整理成 `gdr_copy.h`
2. 把实现整理成 `gdr_copy.cc`
3. 保持接口类风格和 `D:\Project\gdr` 对齐
4. 在 `gdrcopy_adapter.cc` 里做 UCM 封装

### 不推荐方式

1. 在 `load_queue.cc` 里直接 `#include "gdrcopy.h"`
2. 在 `dump_queue.cc` 里直接写一堆 `#ifdef GDR`
3. 把 GDR 错误码直接向上传播到业务层

这些做法会让 copy 逻辑在两个 queue 里分叉，后面很难维护。

---

## 代码改动范围建议

按文件维度，建议改动如下。

### 新增

- `ucm/store/cache/cc/copy_engine.h`
- `ucm/store/cache/cc/copy_engine_factory.h`
- `ucm/store/cache/cc/cuda_copy_engine.h`
- `ucm/store/cache/cc/cuda_copy_engine.cc`
- `ucm/store/cache/cc/gdr/gdr_copy_engine.h`
- `ucm/store/cache/cc/gdr/gdr_copy_engine.cc`
- `ucm/store/cache/cc/gdr/gdrcopy_adapter.h`
- `ucm/store/cache/cc/gdr/gdrcopy_adapter.cc`
- `ucm/store/cache/cc/gdr/third_party/*`

### 修改

- `ucm/store/cache/cc/load_queue.h`
- `ucm/store/cache/cc/load_queue.cc`
- `ucm/store/cache/cc/dump_queue.h`
- `ucm/store/cache/cc/dump_queue.cc`
- `ucm/store/cache/cc/global_config.h`
- `ucm/store/cache/cc/cache_store.cc`
- `ucm/store/cache/CMakeLists.txt`

### 可选修改

- 顶层 `CMakeLists.txt`
  - 增加 `UCM_ENABLE_GDRDMA`

---

## 推荐的落地顺序

### Step 1

新增 `CopyEngine` 和 `CudaCopyEngine`，把当前逻辑搬过去，行为不变。

验收标准：

- `cache store` 现有测试行为不变；
- 不打开任何 GDR 编译选项时，生成物和现在一致。

### Step 2

新增配置项：

- `cache_copy_engine`
- `gdr_nic_name`
- `gdr_use_odp`
- `gdr_strict`

但仍然只支持 `cudaMemcpy`。

验收标准：

- 配置解析正确；
- 非法值能报清晰错误。

### Step 3

接入 `GdrCopyEngine`，打通 H2D / D2H。

验收标准：

- `cache_copy_engine=gdrdma` 时能正常 load / dump；
- 无 GDR 条件时，`strict=false` 能回退；
- `strict=true` 时能失败退出。

### Step 4

补 benchmark 和文档说明。

---

## 风险点

### 1. `WaitEvent` 语义差异

CUDA 路径是 stream wait，GDR 第一版建议用 `cudaEventSynchronize`。

影响：

- 正确性没问题；
- 但可能会把部分异步性变成 CPU 阻塞等待。

这是一个合理的第一版折中。

### 2. host memory 注册策略

参考实现里对 host pointer 也做了 MR 注册缓存。

要注意：

- UCM cache buffer 当前是否总是 pinned host memory；
- 如果不是，GDR 路径是否还能稳定工作；
- 出现不可注册的 host memory 时，是报错还是回退。

建议第一版只支持 UCM 自己分配的 buffer，不承诺对任意 host 指针都最优。

### 3. 依赖项复杂

GDR 需要：

- CUDA
- `libibverbs`
- 兼容 RDMA NIC
- `nvidia-peermem` / `nv_peer_mem`

因此必须有清晰的“编译期开关 + 运行时探测 + strict/fallback 策略”。

### 4. 单 channel 吞吐可能不足

第一版先追求稳定，不追求极限并发。

如果 benchmark 后发现瓶颈，再做多 channel 扩展。

---

## 测试建议

### 单元/构建级

1. `UCM_ENABLE_GDRDMA=OFF`
   - `cache store` 正常编译
2. `UCM_ENABLE_GDRDMA=ON` + `RUNTIME_ENVIRONMENT=cuda`
   - 正常编译并链接 `ibverbs`
3. `UCM_ENABLE_GDRDMA=ON` + 非 CUDA runtime
   - 配置阶段直接报错或跳过 GDR

### 功能级

1. `cache_copy_engine=cudaMemcpy`
   - load / dump 正常
2. `cache_copy_engine=gdrdma` + GDR 环境正常
   - load / dump 正常
3. `cache_copy_engine=gdrdma` + 无 `nvidia-peermem`
   - `gdr_strict=false` 回退
   - `gdr_strict=true` 失败

### 性能级

建议单独补一个 benchmark，而不是直接复用生产路径测试：

- 小包延迟：4KB / 16KB / 64KB
- 中包吞吐：256KB / 1MB / 4MB
- 对比：
  - `cudaMemcpyAsync`
  - `gdrdma`

---

## 最终建议

如果你现在就要开始改代码，我的建议非常明确：

1. **不要直接去改 `ucm/shared/trans/cuda/cuda_stream.cc`。**
2. **先在 `ucm/store/cache/cc/` 新增 `CopyEngine` 抽象。**
3. **先把现有 CUDA scatter/gather 逻辑搬到 `CudaCopyEngine`。**
4. **再把 `D:\Project\gdr` 的代码裁成 `third_party + adapter + gdr_copy_engine` 三层接进去。**
5. **第一版先做单 channel、`WaitEvent -> cudaEventSynchronize`、`strict/fallback` 明确化。**

这样改，风险最低，回退最容易，也最符合 UCM 当前代码结构。

---

## 后续扩展

当 `cache store` 上的 `gdrdma` 跑稳后，可以再评估两件事：

1. 是否把 `CopyEngine` 下沉为 `ucm/store/detail` 的公共组件，供 `pcstore` 复用。
2. 是否把 GDR 从 `cache store` 局部能力演进为更通用的 store copy backend。

在第一版之前，不建议一开始就做全局抽象。
