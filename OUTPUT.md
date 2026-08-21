# build.sh 行为

```bash
# 配置：统一 FetchContent 缓存、开启单测、Sparse 和 NUMA
cmake -S . -B build \
    -DFETCHCONTENT_BASE_DIR="$DEPS" \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_UCM_SPARSE=ON \
    -DBUILD_NUMA=ON

# 构建全部目标
cmake --build build --config Release -j8

# 默认只运行 BufferPool 相关测试
ctest --test-dir build \
    -R '^(BufferPoolTest|VariableBufferPoolTest|OffsetAllocatorTest)\.' \
    --output-on-failure

# 需要运行全部测试时取消注释
# ctest --test-dir build --output-on-failure
```

## shared pool 的 CMake 接入

```cmake
# ucm/shared/CMakeLists.txt
# trans 先选择当前 runtime 的具体后端并创建 trans target
add_subdirectory(trans)

# shared 只需要添加 pool
add_subdirectory(pool)

# ucm/shared/pool/CMakeLists.txt
# pool 创建自己的内部依赖
add_subdirectory(detail)
```

## Buffer 内存类型接口命名

```cpp
class Buffer {
public:
    virtual std::shared_ptr<void> MakeHostBuffer(std::size_t size) = 0;

    virtual bool SupportsHostMappedDeviceBuffer() const { return false; }
    // deviceAddress：返回设备侧用于访问该 host memory 的地址。
    virtual std::shared_ptr<void> MakeHostMappedDeviceBuffer(std::size_t size,
                                                             void** deviceAddress = nullptr) = 0;

    virtual std::shared_ptr<void> MakeDeviceBuffer(std::size_t size) = 0;

    virtual bool SupportsDeviceMappedHostBuffer() const { return false; }
    virtual std::shared_ptr<void> MakeDeviceMappedHostBuffer(std::size_t size) = 0;
};
```

## CUDA Memset 的同步边界

```cpp
// ptr: 待清零的 pool slot 地址，可能指向 host-pinned 或 device memory。
// size: 待清零的字节数。
// 返回语义: 成功返回时，slot 已经清零完成，可以重新分配。
Status Free(Slot slot)
{
    // CUDA device memory 上的 cudaMemset 对 host 异步，仅表示任务已提交。
    auto status = Memset(slot.ptr, slot.size, 0);
    if (status.Failure()) { return status; }

    // 等待 memset 所在的 default stream，防止 slot 重用时与清零并发。
    status = SynchronizeDefaultStream();
    if (status.Failure()) { return status; }

    // 只有清零完成后才将 slot 放回 allocator。
    allocator.Release(slot);
    return Status::OK();
}
```

## Ascend 与 CUDA Memset 的完成语义

```cpp
// Ascend: ptr 可指向 Host 或 Device 内存。
// 返回语义: aclrtMemset 完成当前内存初始化后返回。
aclrtMemset(ptr, size, value, size);
allocator.Release(slot);

// CUDA: ptr 指向 device memory 时，cudaMemset 对 host 异步。
// 返回语义: 同步 default stream 后才能保证清零完成。
cudaMemset(ptr, value, size);
cudaStreamSynchronize(nullptr);
allocator.Release(slot);
```

## CUDA mapped buffer 支持策略

```cpp
// type: 申请的 buffer memory type。
// CUDA capability: 两类 mapped memory 均不声明支持。
if (type == HostMappedDevice || type == DeviceMappedHost) {
    return Status::Unsupported();
}
```

## ASU 缓冲区接口名称

```cpp
// size: ASU HOST_PINNED 区域的字节数。
// deviceAddr: 返回该 host memory 的设备侧访问地址。
auto owner = ascendBuffer.MakeHostMappedDeviceBuffer(size, &deviceAddr);
if (!owner) { return Status::Error("failed to allocate host-pinned memory"); }

// localAddr 用于 CPU 访问，deviceAddr 用于 provider 的设备侧访问。
region = {owner, owner.get(), deviceAddr, TransProvider::MemType::MEM_DEVICE};
```

## Why DramStore needs Event synchronization

```cpp
// prerequisiteHandle: Native event recorded on the client's compute stream.
// Ownership: Remains with the framework; Event is only a typed, non-owning view.
Trans::Event prerequisite{task->desc.prerequisiteHandle};

// Empty handle: There is no prerequisite, so the dump can be submitted immediately.
if (!prerequisite.Valid()) { return SubmitDump(task); }

// DramStore has no local NPU copy stream; the remote DramPool directly RDMA-reads this memory.
// Synchronization boundary: The host must wait before sending the RDMA control request.
auto status = prerequisite.Synchronize();
if (status.Failure()) { return status; }
task->desc.prerequisiteHandle = 0;
return SendRdmaReadRequest(task);

// Other stores own a copy stream, so they can add a stream dependency without blocking the host.
copyStream.WaitEvent(prerequisite);
return copyStream.DeviceToHostAsync(device, host, size);
```
