# Extend Store

Choose whether the change implements the full Store interface or adds a stage to an existing Pipeline. Engine integration owns model layouts, block identifiers and device addresses. Store owns backend resources, movement and task completion.

## Choose the extension boundary

| Path | Implementation | Current references |
| --- | --- | --- |
| Python V1 Store | Implement `UcmKVStoreBaseV1` and construct it through the V1 factory | `ucm/store/pcstore/pcstore_connector_v1.py`, `ucm/store/pipeline/connector.py` |
| Native Pipeline stage | Implement `UC::StoreV1`, export its factory and register a stage combination | `ucm/store/empty/` for interface shape, `ucm/store/posix/` for real I/O |
| Python wrapper of a native library | Satisfy Python V1 and native resource-lifetime contracts | PcStore and Pipeline wrappers/bindings |

Choose based on existing libraries, data paths and resource control. Python wrappers can invoke native I/O; language alone does not establish throughput. `Empty` illustrates interfaces without persistence.

## Python V1 interface

`ucm/store/ucmstore_v1.py` is authoritative. The following table includes all current abstract methods; use the source for exact type annotations.

| Method | Contract |
| --- | --- |
| `cc_store()` | Return the underlying native Store pointer as an integer; native callers require a valid compatible object, not a placeholder value |
| `lookup(block_ids)` | Return per-block presence in input order |
| `lookup_on_prefix(block_ids)` | Return the last contiguous-hit index, or `-1` when the first block misses |
| `lookup_on_reverse(block_ids)` | Scan backward for an existing block and return its index, or `-1` if all are missing |
| `prefetch(block_ids)` | Initiate prefetch |
| `load(...)`, `dump(...)` | Submit tensor-described transfers and return an opaque `Task` |
| `load_data(...)`, `dump_data(...)` | Submit address-described transfers; `dump_data` includes synchronization parameter `prerequisite_handle=0` |
| `check(task)`, `wait(task)` | Poll completion or wait; preserve implementation errors for callers |

Implementing lookup/load/dump alone is insufficient. Native pointer access, reverse lookup and synchronization are also part of the current interface. A Python client missing these capabilities needs an explicit adaptation to the actual caller.

### Implement the Python class

Save the following interface scaffold in `your_package/store.py`. It deliberately remains abstract: replace every abstract method with the backend operation before constructing the Store. The current interface also requires `cc_store`, reverse lookup and the dump prerequisite handle; the old lookup/load/dump-only outline was incomplete.

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

Register a complete, importable implementation during initialization. This illustrates registration; replace the module and class with the actual implementation:

```python
from ucm.store.factory_v1 import UcmConnectorFactoryV1

UcmConnectorFactoryV1.register_connector(
    "CustomStore", "your_package.store", "CustomStore"
)
```

The factory requires a subclass of `UcmKVStoreBaseV1`. Registration precedes construction; every process constructing the Store must be able to load it.

## Native Pipeline stage

`ucm/store/ucmstore_v1.h` defines `Setup`, `Readme`, `Lookup`, `LookupOnPrefix`, `LookupOnReverse`, `Prefetch`, `Load`, `Dump`, `Check` and `Wait`. `CheckHealth` defaults to success; override it for a real backend probe. Public methods must support concurrent calls.

A stage named `Custom` exports `MakeCustomStore`, returning `UC::StoreV1*`. The loader resolves `Make` + stage name + `Store`, constructs the object and calls `Setup`. Follow adjacent Store CMake definitions to build and install the shared library and its dependencies.

### Implement and build the native stage

Under `ucm/store/custom/`, declare `cc/custom_store.h` as follows. Implement these methods in `cc/custom_store.cc` using the backend's real I/O and task completion; the declaration alone is not a functioning Store.

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

After defining the class, export the factory in its implementation file:

```cpp
extern "C" UC::StoreV1* MakeCustomStore() { return new UC::CustomStore::CustomStore(); }
```

Create `ucm/store/custom/CMakeLists.txt`, following the existing Empty Store target:

```cmake
file(GLOB_RECURSE UCM_CUSTOM_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(customstore SHARED ${UCM_CUSTOM_STORE_CC_SOURCE_FILES})
target_include_directories(customstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(customstore PUBLIC storeintf)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS customstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
```

Add `add_subdirectory(custom)` to `ucm/store/CMakeLists.txt`. Rebuild in the target engine environment using [Build from source](build_from_source.md). Point the builder below to the installed `libcustomstore.so`; use `nm -D` on Linux to confirm the exported `MakeCustomStore` symbol.

This shows registration only; implement the class and library first:

```python
from ucm.store.pipeline.connector import UcmPipelineStoreBuilder

def build_custom(config, pipeline):
    pipeline.Stack("Custom", "/opt/ucm/libcustomstore.so", config)

UcmPipelineStoreBuilder.register("Custom", build_custom)
```

Select it with `UcmPipelineStore` and `store_pipeline: Custom`. For combinations with Cache or other stages, follow an existing builder's ordering and native contracts rather than concatenating arbitrary names.

Use this YAML after the builder has been imported and registered in each creating process:

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: Custom
```

### Wrap a native library with Python

For the hybrid route, keep the binding and wrapper contracts together. The existing `ucm/store/pcstore/cpy/pcstore.py.cc` binds `PcStorePy`, converts Python configuration in `SetupPy`, and exposes these methods (excerpt from its module definition):

```cpp
auto store = py::class_<UC::PcStorePy>(module, "PcStore");
store.def("CCStoreImpl", &UC::PcStorePy::CCStoreImpl);
store.def("Setup", &UC::PcStorePy::SetupPy);
store.def("LookupBatch", &UC::PcStorePy::LookupBatch);
store.def("LoadToDevice", &UC::PcStorePy::LoadToDevice);
store.def("DumpFromDevice", &UC::PcStorePy::DumpFromDevice);
```

Follow `UcmPcStoreV1` in `pcstore_connector_v1.py`: construct `PcStore.Config` from the engine-provided geometry, call `Setup` and check its status, map tensor/address inputs to native operations, and wrap native handles as `Task` objects. Register the completed `UcmKVStoreBaseV1` subclass through `UcmConnectorFactoryV1` as above. Preserve errors and buffer lifetimes; a lookup miss does not justify adding an automatic retry wrapper. The binding excerpt is not a standalone module.

## Tasks, buffers and errors

A returned transfer task may still be moving data. Retain required resources until completion and honor source-buffer compute dependencies. `prerequisite_handle` passes such device dependencies to native saving. Callers use completion feedback to decide when buffers can be reused.

Native `Load`/`Dump` return a handle or error, `Check` returns completion or error, and `Wait` returns final status. Python bindings must preserve error meaning. A lookup miss is an expected cache result; a failed transfer must not masquerade as usable loaded data. Health wrappers restrict new storage operations without taking over engine request recovery.

## Verify the extension

1. Verify library loading, symbols, configuration and factory selection in the target environment.
2. Save and load known buffer contents, checking each block and task completion; cover a miss and one real error path.
3. Check Prefix/Reverse index semantics, concurrent calls and buffer lifetime.
4. Connect through an [engine quickstart](../user-guide/quick_start/index.md), [verify external reuse](../user-guide/observability/verify-cache.md), then measure performance under the same workload.

Use relevant cases in `ucm/store/test/`. A stub that always returns success does not verify persistence. Continue with [metrics development](add-metrics.md) to observe the backend.

## Existing backend entry points {#backend-entrypoints}

The V1 factory maps `UcmNfsStore` to `UcmPcStoreV1`; the older `ucm.store.nfsstore` is a separate interface. Pipeline loads registered builders and native libraries by name.

### pipeline

- `ucm/store/pipeline/connector.py` defines the registered compositions and loaded libraries.
- `ucm/store/cache/cc/cache_store.cc` owns buffer defaults and minimum-size checks.
- `ucm/store/posix/cc/posix_store.cc` owns filesystem configuration and health probes.
- `ucm/integration/vllm/ucm_connector.py` supplies layout, shared-buffer defaults, and GC ownership.

### nfs

- `ucm/store/factory_v1.py` maps the public connector name to `UcmPcStoreV1`.
- `ucm/store/pcstore/pcstore_connector_v1.py` defines the accepted key mapping and tensor-size constraint.
- `ucm/store/pcstore/cc/api/pcstore.h` defines native transfer defaults.

### ds3fs

- `ucm/store/ds3fs/CMakeLists.txt` defines the optional dependency discovery.
- `ucm/store/pipeline/connector.py` defines `Cache|Ds3fs` and its transfer geometry.
- `ucm/store/ds3fs/cc/ds3fs_store.cc` parses configuration and dispatches Store operations.
- `ucm/store/ds3fs/cc/trans_queue.h` and `trans_queue.cc` implement 3FS client I/O.

### mooncake

- `ucm/store/mooncakestore/CMakeLists.txt` owns the Ascend/Mooncake build requirements.
- `ucm/store/pipeline/connector.py` registers both Mooncake pipeline names.
- `ucm/store/mooncakestore/cc/mooncake_store.cc` parses sizes, performs lookup, and probes health.
- `ucm/store/mooncakestore/cc/dump_queue.cc` and `load_queue.cc` implement tier transfers.

### compress

- `ucm/store/pipeline/connector.py` defines composition and stored-size calculation.
- `ucm/store/compress/cc/compressor_action.cc` accepts dtype/ratio settings and runs the codec.
- `ucm/store/compress/cc/compress_lib/tunstall_bf16.cc` defines the lossy encoding and fallback.
- `ucm/store/compress/cc/global_config.h` defines the stage defaults.
