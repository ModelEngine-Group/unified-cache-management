# 扩展 Store

新增存储后端时，先决定需要接入整个 Store 接口，还是为现有 Pipeline 增加一个阶段。引擎集成负责模型布局、块标识与设备地址；Store 负责后端资源、数据搬运和任务完成。

## 选择扩展边界

| 扩展路径 | 要实现的部分 | 当前参考 |
| --- | --- | --- |
| Python V1 Store | 实现 `UcmKVStoreBaseV1`，通过 V1 工厂创建 | `ucm/store/pcstore/pcstore_connector_v1.py`、`ucm/store/pipeline/connector.py` |
| 原生 Pipeline 阶段 | 实现 `UC::StoreV1`，导出创建函数并注册阶段组合 | `ucm/store/empty/` 的接口形状、`ucm/store/posix/` 的实际 I/O |
| Python 包装已有原生库 | 同时满足 Python V1 接口和原生库的生命周期要求 | PcStore 与 Pipeline 的包装和绑定 |

扩展方式取决于已有库、数据路径和资源控制需求。Python 包装器可以调用原生 I/O；仅凭实现语言不能判断吞吐。`Empty` 只适合参考接口，不提供持久化行为。

## Python V1 接口

权威定义是 `ucm/store/ucmstore_v1.py`。当前抽象接口包括以下全部方法，具体类型标注以源码为准：

| 方法 | 合约 |
| --- | --- |
| `cc_store()` | 返回底层原生 Store 指针的整数表示；实际原生调用方要求有效的兼容对象，不能用占位数值替代 |
| `lookup(block_ids)` | 按输入顺序返回每个块是否存在 |
| `lookup_on_prefix(block_ids)` | 返回最后一个连续命中块的索引，首块未命中返回 `-1` |
| `lookup_on_reverse(block_ids)` | 从末尾向前找到存在的块并返回其索引，全部缺失返回 `-1` |
| `prefetch(block_ids)` | 发起预取 |
| `load(...)`、`dump(...)` | 使用张量描述提交传输，返回不透明 `Task` |
| `load_data(...)`、`dump_data(...)` | 使用设备地址描述提交传输；`dump_data` 包含 `prerequisite_handle=0` 同步参数 |
| `check(task)`、`wait(task)` | 分别查询是否完成、等待完成；错误必须按具体实现向调用者传播 |

不要将“实现了 lookup/load/dump”当作完整实现。`cc_store()`、反向查找和同步参数同样属于当前接口；若已有 Python 客户端不具备这些能力，需要先解决其与实际调用方的适配。

### 实现 Python 类

将下面的接口骨架保存为 `your_package/store.py`。它保留抽象方法，不能直接构造；需要逐个实现后端操作。当前接口还要求 `cc_store`、反向查找和 dump 的同步句柄，原先仅列出 lookup/load/dump 的骨架并不完整。

```python
from abc import abstractmethod
from typing import List

import numpy as np
import torch
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

class CustomStore(UcmKVStoreBaseV1):

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def cc_store(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        raise NotImplementedError()

    @abstractmethod
    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def lookup_on_reverse(self, block_ids: List[bytes]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def prefetch(self, block_ids: List[bytes]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, block_ids: List[bytes], shard_index: List[int], dst_tensor: List[List[torch.Tensor]]) -> Task:
        raise NotImplementedError()

    @abstractmethod
    def dump(self, block_ids: List[bytes], shard_index: List[int], src_tensor: List[List[torch.Tensor]]) -> Task:
        raise NotImplementedError()

    @abstractmethod
    def load_data(self, block_ids: List[bytes], shard_index: List[int], dst_addr: List[List[int]] | np.ndarray) -> Task:
        raise NotImplementedError()

    @abstractmethod
    def dump_data(self, block_ids: List[bytes], shard_index: List[int], src_addr: List[List[int]] | np.ndarray, prerequisite_handle: int=0) -> Task:
        raise NotImplementedError()

    @abstractmethod
    def wait(self, task: Task) -> None:
        raise NotImplementedError()

    @abstractmethod
    def check(self, task: Task) -> bool:
        raise NotImplementedError()
```

完整实现并可以导入后，在初始化路径注册。以下是注册形状，模块名和类名需替换为实际实现：

```python
from ucm.store.factory_v1 import UcmConnectorFactoryV1

UcmConnectorFactoryV1.register_connector(
    "CustomStore", "your_package.store", "CustomStore"
)
```

工厂要求类继承 `UcmKVStoreBaseV1`。注册必须发生在创建 Store 之前；跨进程部署中，各创建 Store 的进程都需要能够加载实现。

## 原生 Pipeline 阶段

`ucm/store/ucmstore_v1.h` 定义原生接口：`Setup`、`Readme`、`Lookup`、`LookupOnPrefix`、`LookupOnReverse`、`Prefetch`、`Load`、`Dump`、`Check` 和 `Wait`。`CheckHealth` 有默认成功实现，需要真实后端探测时覆盖它。接口要求公开方法支持并发调用。

一个名为 `Custom` 的阶段导出 `MakeCustomStore`，返回 `UC::StoreV1*`。Pipeline 的加载器按 `Make` + 阶段名 + `Store` 查找符号，并在构造后执行 `Setup`。构建时以相邻 Store 的 CMake 为依据，安装共享库及依赖。

### 实现并构建原生阶段

在 `ucm/store/custom/` 下创建 `cc/custom_store.h`，声明如下。在 `cc/custom_store.cc` 中实现实际后端 I/O 和任务完成语义；仅有声明还不是可用 Store。

```cpp
#pragma once
#include "ucmstore_v1.h"

namespace UC::CustomStore {

class CustomStore : public StoreV1 {
public:
    ~CustomStore() override;
    Status Setup(const Detail::Dictionary& config) override;
    std::string Readme() const override;

    // Required interfaces
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override;
    void Prefetch(const Detail::BlockId* blocks, size_t num) override;
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override;
    Expected<bool> Check(Detail::TaskHandle taskId) override;
    Status Wait(Detail::TaskHandle taskId) override;
};

}  // namespace UC::CustomStore
```

实现该类后，在实现文件中导出创建函数：

```cpp
extern "C" UC::StoreV1* MakeCustomStore() { return new UC::CustomStore::CustomStore(); }
```

参照已有 Empty Store target，创建 `ucm/store/custom/CMakeLists.txt`：

```cmake
file(GLOB_RECURSE UCM_CUSTOM_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(customstore SHARED ${UCM_CUSTOM_STORE_CC_SOURCE_FILES})
target_include_directories(customstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(customstore PUBLIC storeintf)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS customstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
```

在 `ucm/store/CMakeLists.txt` 添加 `add_subdirectory(custom)`，按[源码构建](build_from_source.md)在目标引擎环境重建。将下方 builder 的库路径指向安装后的 `libcustomstore.so`；在 Linux 上用 `nm -D` 核对 `MakeCustomStore` 导出符号。

下面仅展示注册逻辑；`Custom` 类和共享库必须先实现：

```python
from ucm.store.pipeline.connector import UcmPipelineStoreBuilder

def build_custom(config, pipeline):
    pipeline.Stack("Custom", "/opt/ucm/libcustomstore.so", config)

UcmPipelineStoreBuilder.register("Custom", build_custom)
```

随后通过 `UcmPipelineStore` 和 `store_pipeline: Custom` 选择它。组合 Cache 或其他阶段时，参考已有 builder 的阶段顺序和原生接口，不把多个名称随意拼成字符串。

在各创建进程导入并注册 builder 后使用以下 YAML：

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: Custom
```

### 用 Python 包装原生库

混合扩展需要让绑定与包装层保持一致。现有 `ucm/store/pcstore/cpy/pcstore.py.cc` 绑定 `PcStorePy`，通过 `SetupPy` 转换 Python 配置，并导出以下方法（模块定义中的片段）：

```cpp
auto store = py::class_<UC::PcStorePy>(module, "PcStore");
store.def("CCStoreImpl", &UC::PcStorePy::CCStoreImpl);
store.def("Setup", &UC::PcStorePy::SetupPy);
store.def("LookupBatch", &UC::PcStorePy::LookupBatch);
store.def("LoadToDevice", &UC::PcStorePy::LoadToDevice);
store.def("DumpFromDevice", &UC::PcStorePy::DumpFromDevice);
```

参照 `pcstore_connector_v1.py` 中的 `UcmPcStoreV1`：根据引擎提供的布局构造 `PcStore.Config`，调用 `Setup` 并检查状态，将张量／地址映射到原生操作，再将原生句柄包装为 `Task`。完成的 `UcmKVStoreBaseV1` 子类通过上面的 `UcmConnectorFactoryV1` 注册。保持错误语义和缓冲生命周期，不因为一次 lookup 未命中就增加自动重试包装。上面的绑定片段不是独立模块。

## 任务、缓冲区与错误

传输任务返回时，数据可能仍在搬运。实现必须在任务完成前保留所需资源，并遵守源缓冲的计算同步条件；`prerequisite_handle` 用于将这类设备依赖传入原生保存路径。调用方根据完成反馈决定何时可以复用缓冲。

原生 `Load` / `Dump` 返回任务句柄或错误，`Check` 返回完成状态或错误，`Wait` 返回最终状态。Python 绑定应保留错误含义。查询未命中是可预期的缓存结果；传输失败不应伪装成已加载的数据。健康包装器只限制新的存储操作，不接管引擎请求恢复。

## 验证扩展

1. 在目标环境确认原生库可加载、符号和配置正确，工厂创建的是预期实现。
2. 用已知内容的缓冲验证保存、加载、逐块内容和任务完成；覆盖未命中与一个实际错误路径。
3. 对照 Prefix/Reverse 的索引语义、并发调用和缓冲生命周期检查实现。
4. 通过[引擎快速开始](../user-guide/quick_start/index.md)接入，再执行[外部缓存验证](../user-guide/observability/verify-cache.md)。最后在相同负载下测量性能。

参考 `ucm/store/test/` 中与后端相关的用例；不要用只返回成功的接口示例作为持久化验证。需要观测新后端时，继续阅读[指标开发](add-metrics.md)。

## 现有后端入口 {#backend-entrypoints}

当前 V1 工厂将 `UcmNfsStore` 映射到 `UcmPcStoreV1`；旧的 `ucm.store.nfsstore` 是另一套接口。Pipeline 通过名称加载已注册的 builder 和原生库，具体入口如下。

### pipeline

- `ucm/store/pipeline/connector.py` 定义注册管线及加载的原生库。
- `ucm/store/cache/cc/cache_store.cc` 管理缓冲默认值和最小容量检查。
- `ucm/store/posix/cc/posix_store.cc` 管理文件系统配置与健康探针。
- `ucm/integration/vllm/ucm_connector.py` 提供布局、共享缓冲默认值和 GC 归属。

### nfs

- `ucm/store/factory_v1.py` 将公开 connector 名映射到 `UcmPcStoreV1`。
- `ucm/store/pcstore/pcstore_connector_v1.py` 定义接受的 key 映射和张量大小约束。
- `ucm/store/pcstore/cc/api/pcstore.h` 定义原生传输默认值。

### ds3fs

- `ucm/store/ds3fs/CMakeLists.txt` 定义可选依赖发现逻辑。
- `ucm/store/pipeline/connector.py` 定义 `Cache|Ds3fs` 及其传输几何。
- `ucm/store/ds3fs/cc/ds3fs_store.cc` 解析配置并分派 Store 操作。
- `ucm/store/ds3fs/cc/trans_queue.h` 和 `trans_queue.cc` 实现 3FS 客户端 I/O。

### mooncake

- `ucm/store/mooncakestore/CMakeLists.txt` 定义 Ascend/Mooncake 构建依赖。
- `ucm/store/pipeline/connector.py` 注册两种 Mooncake 流水线名称。
- `ucm/store/mooncakestore/cc/mooncake_store.cc` 解析大小、执行查找并探测健康状态。
- `ucm/store/mooncakestore/cc/dump_queue.cc` 和 `load_queue.cc` 实现层间传输。

### compress

- `ucm/store/pipeline/connector.py` 定义组合方式和存储大小计算。
- `ucm/store/compress/cc/compressor_action.cc` 接受 dtype/ratio 设置并运行编解码器。
- `ucm/store/compress/cc/compress_lib/tunstall_bf16.cc` 定义有损编码和回退行为。
- `ucm/store/compress/cc/global_config.h` 定义阶段默认值。
