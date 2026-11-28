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

#include "worker/device_allocator.h"
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include "kv_client.h"

// ---- 设备 SDK 头：仅宏编入时引入（可移植性约束——架构相关特性预处理宏隔离）----
#ifdef TESTTOOL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef TESTTOOL_ENABLE_ASCEND
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#endif

namespace {

#if defined(TESTTOOL_ENABLE_CUDA) || defined(TESTTOOL_ENABLE_ASCEND)
/** 值向上对齐到 alignment 的整数倍（alignment 须为正的 2 的幂） */
std::size_t AlignUp(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}
#endif

#ifdef TESTTOOL_ENABLE_CUDA

// ---- CUDA 显存分配器（宏隔离）：cudaMalloc/cudaMemcpy/cudaFree，进程内单 device ----
// cudaMalloc 只保证 256 字节对齐，协议要求 512（kv::kAlignmentBytes）——超额分配一截
// 后手动对齐，Deallocate 按记录的原始指针释放。
class CudaAllocator final : public DeviceAllocator {
public:
    /** @param deviceId CUDA device 编号（第 deviceId 块卡） */
    explicit CudaAllocator(std::int32_t deviceId) : deviceId_(deviceId) {}

    ~CudaAllocator() override
    {
        // 兜底释放（正常路径 Scheduler 已逐块 Deallocate）；释放前同样校验线程绑定
        std::string bindErr;
        (void)EnsureThreadDevice(bindErr);
        for (const auto &item : rawPtrs_) { ::cudaFree(item.second); }
    }

    bool Allocate(std::size_t size, std::uint64_t &addr, std::string &err) override
    {
        if (size == 0) {
            err = "cuda memory allocate size must be positive";
            return false;
        }
        if (!EnsureThreadDevice(err)) { return false; }
        // 超额分配：对齐粒度额外加一截，保证 [raw, raw+size) 内有 512 对齐的起点
        const std::size_t allocSize = AlignUp(size, kv::kAlignmentBytes) + kv::kAlignmentBytes;
        void *raw = nullptr;
        const cudaError_t malloc = ::cudaMalloc(&raw, allocSize);
        if (malloc != cudaSuccess) {
            err = std::string("cudaMalloc failed for ") + std::to_string(size) +
                  " bytes: " + ::cudaGetErrorString(malloc);
            return false;
        }
        const std::uint64_t aligned =
            AlignUp(reinterpret_cast<std::uint64_t>(raw), kv::kAlignmentBytes);
        rawPtrs_.emplace(aligned, raw);
        addr = aligned;
        return true;
    }

    void Deallocate(std::uint64_t addr) override
    {
        const auto it = rawPtrs_.find(addr);
        if (it == rawPtrs_.end()) { return; }
        std::string bindErr;  // 释放路径尽力绑定，失败交由 cudaFree 自行报错
        (void)EnsureThreadDevice(bindErr);
        ::cudaFree(it->second);
        rawPtrs_.erase(it);
    }

    bool Write(std::uint64_t addr, const void *hostData, std::size_t size,
               std::string &err) override
    {
        if (addr == 0) {
            err = "cuda memory write address is null";
            return false;
        }
        if (size == 0) { return true; }
        if (!EnsureThreadDevice(err)) { return false; }
        const cudaError_t copy =
            ::cudaMemcpy(reinterpret_cast<void *>(static_cast<std::uintptr_t>(addr)), hostData,
                         size, cudaMemcpyHostToDevice);
        if (copy != cudaSuccess) {
            err = std::string("cudaMemcpy H2D failed: ") + ::cudaGetErrorString(copy);
            return false;
        }
        return true;
    }

    bool Read(void *hostData, std::uint64_t addr, std::size_t size, std::string &err) override
    {
        if (addr == 0) {
            err = "cuda memory read address is null";
            return false;
        }
        if (size == 0) { return true; }
        if (!EnsureThreadDevice(err)) { return false; }
        const cudaError_t copy = ::cudaMemcpy(
            hostData, reinterpret_cast<const void *>(static_cast<std::uintptr_t>(addr)), size,
            cudaMemcpyDeviceToHost);
        if (copy != cudaSuccess) {
            err = std::string("cudaMemcpy D2H failed: ") + ::cudaGetErrorString(copy);
            return false;
        }
        return true;
    }

    bool IsDeviceMemory() const override { return true; }

    /**
     * @brief 确保当前线程绑定的 CUDA device 指向本分配器的设备
     * @param err 绑定失败时填充错误信息
     * @return 绑定成功返回 true；cudaSetDevice 失败返回 false
     */
    bool EnsureThreadDevice(std::string &err)
    {
        // cudaSetDevice 为线程级绑定：首次 Allocate 的绑定只作用于调用线程，worker 线程
        // 执行 H2D/D2H 前须自行绑定（未绑线程默认 device 0，deviceId 非 0 时会静默拷错
        // 卡）；同进程其他组件（kv client NPU 通路）可能切换过 current device，故每次
        // 调用前校验、不一致才重绑
        int current = -1;
        if (::cudaGetDevice(&current) == cudaSuccess && current == deviceId_) { return true; }
        const cudaError_t set = ::cudaSetDevice(deviceId_);
        if (set != cudaSuccess) {
            err = std::string("cudaSetDevice(") + std::to_string(deviceId_) +
                  ") failed: " + ::cudaGetErrorString(set);
            return false;
        }
        return true;
    }

private:
    /** CUDA device 编号（各线程使用前经 EnsureThreadDevice 校验绑定；本进程固定该卡） */
    const std::int32_t deviceId_;
    /** 对齐地址 → cudaMalloc 原始指针（Deallocate 需按原始指针 cudaFree） */
    std::unordered_map<std::uint64_t, void *> rawPtrs_;
};

#endif  // TESTTOOL_ENABLE_CUDA

#ifdef TESTTOOL_ENABLE_ASCEND

// ---- ACL NPU 内存分配器（宏隔离）：aclrtMalloc/aclrtMemcpy/aclrtFree ----
// aclrtMalloc 的对齐保证（大页 2MB / 普通页 64B）不保证协议要求的 512 字节，与 CUDA 同款
// 超额分配 + 手动对齐 + 原始指针记录。
class AclAllocator final : public DeviceAllocator {
public:
    /** @param deviceId ACL device 编号（第 deviceId 块卡） */
    explicit AclAllocator(std::int32_t deviceId) : deviceId_(deviceId) {}

    ~AclAllocator() override
    {
        // 兜底释放（正常路径 Scheduler 已逐块 Deallocate）；释放前同样校验线程绑定
        std::string bindErr;
        (void)EnsureThreadDevice(bindErr);
        for (const auto &item : rawPtrs_) { (void)::aclrtFree(item.second); }
    }

    bool Allocate(std::size_t size, std::uint64_t &addr, std::string &err) override
    {
        if (size == 0) {
            err = "acl npu memory allocate size must be positive";
            return false;
        }
        if (!EnsureThreadDevice(err)) { return false; }
        const std::size_t allocSize = AlignUp(size, kv::kAlignmentBytes) + kv::kAlignmentBytes;
        void *raw = nullptr;
        const aclError malloc = ::aclrtMalloc(&raw, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (malloc != ACL_SUCCESS) {
            err = "aclrtMalloc failed for " + std::to_string(size) +
                  " bytes: " + std::to_string(malloc);
            return false;
        }
        const std::uint64_t aligned =
            AlignUp(reinterpret_cast<std::uint64_t>(raw), kv::kAlignmentBytes);
        rawPtrs_.emplace(aligned, raw);
        addr = aligned;
        return true;
    }

    void Deallocate(std::uint64_t addr) override
    {
        const auto it = rawPtrs_.find(addr);
        if (it == rawPtrs_.end()) { return; }
        std::string bindErr;  // 释放路径尽力绑定，失败交由 aclrtFree 自行报错
        (void)EnsureThreadDevice(bindErr);
        (void)::aclrtFree(it->second);
        rawPtrs_.erase(it);
    }

    bool Write(std::uint64_t addr, const void *hostData, std::size_t size,
               std::string &err) override
    {
        if (addr == 0) {
            err = "acl npu memory write address is null";
            return false;
        }
        if (size == 0) { return true; }
        if (!EnsureThreadDevice(err)) { return false; }
        const aclError copy =
            ::aclrtMemcpy(reinterpret_cast<void *>(static_cast<std::uintptr_t>(addr)), size,
                          hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (copy != ACL_SUCCESS) {
            err = "aclrtMemcpy H2D failed: " + std::to_string(copy);
            return false;
        }
        return true;
    }

    bool Read(void *hostData, std::uint64_t addr, std::size_t size, std::string &err) override
    {
        if (addr == 0) {
            err = "acl npu memory read address is null";
            return false;
        }
        if (size == 0) { return true; }
        if (!EnsureThreadDevice(err)) { return false; }
        const aclError copy = ::aclrtMemcpy(
            hostData, size, reinterpret_cast<const void *>(static_cast<std::uintptr_t>(addr)), size,
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (copy != ACL_SUCCESS) {
            err = "aclrtMemcpy D2H failed: " + std::to_string(copy);
            return false;
        }
        return true;
    }

    bool IsDeviceMemory() const override { return true; }

    /**
     * @brief 确保当前线程绑定的 ACL 设备上下文指向本分配器的设备
     * @param err 绑定失败时填充错误信息
     * @return 绑定成功返回 true；aclrtSetDevice 失败返回 false
     */
    bool EnsureThreadDevice(std::string &err)
    {
        // ACL 设备上下文为线程级绑定：首次 Allocate 的 aclrtSetDevice 只作用于调用线程
        // （Scheduler::Init 的主线程），worker 线程执行 H2D/D2H 前须自行绑定，否则报
        // 107002（ACL_ERROR_RT_CONTEXT_NULL）；同进程 kv client（AICPU 通路）可能切换过
        // current device，故每次调用前校验、不一致才重绑
        std::int32_t current = -1;
        if (::aclrtGetDevice(&current) == ACL_SUCCESS && current == deviceId_) { return true; }
        const aclError set = ::aclrtSetDevice(deviceId_);
        if (set != ACL_SUCCESS) {
            err =
                "aclrtSetDevice(" + std::to_string(deviceId_) + ") failed: " + std::to_string(set);
            return false;
        }
        return true;
    }

private:
    /** ACL device 编号（各线程使用前经 EnsureThreadDevice 校验绑定；本进程固定该卡） */
    const std::int32_t deviceId_;
    /** 对齐地址 → aclrtMalloc 原始指针（Deallocate 需按原始指针 aclrtFree） */
    std::unordered_map<std::uint64_t, void *> rawPtrs_;
};

#endif  // TESTTOOL_ENABLE_ASCEND

}  // namespace

bool HostAllocator::Allocate(std::size_t size, std::uint64_t &addr, std::string &err)
{
    if (size == 0) {
        err = "host memory allocate size must be positive";
        return false;
    }
    void *ptr = std::malloc(size);
    if (ptr == nullptr) {
        err = "host memory allocate failed for " + std::to_string(size) + " bytes";
        return false;
    }
    addr = reinterpret_cast<std::uint64_t>(ptr);
    return true;
}

void HostAllocator::Deallocate(std::uint64_t addr)
{
    if (addr == 0) { return; }
    std::free(reinterpret_cast<void *>(static_cast<std::uintptr_t>(addr)));
}

bool HostAllocator::Write(std::uint64_t addr, const void *hostData, std::size_t size,
                          std::string &err)
{
    if (addr == 0) {
        err = "host memory write address is null";
        return false;
    }
    if (size != 0) {
        std::memcpy(reinterpret_cast<void *>(static_cast<std::uintptr_t>(addr)), hostData, size);
    }
    return true;
}

bool HostAllocator::Read(void *hostData, std::uint64_t addr, std::size_t size, std::string &err)
{
    if (addr == 0) {
        err = "host memory read address is null";
        return false;
    }
    if (size != 0) {
        std::memcpy(hostData, reinterpret_cast<const void *>(static_cast<std::uintptr_t>(addr)),
                    size);
    }
    return true;
}

bool HostAllocator::IsDeviceMemory() const { return false; }

std::unique_ptr<DeviceAllocator> CreateDeviceAllocator(const std::string &bufferType,
                                                       std::int32_t deviceId, std::string &err)
{
    if (bufferType == "cpu") { return std::make_unique<HostAllocator>(); }
    if (bufferType == "gpu") {
        if (deviceId < 0) {
            err = "deviceid must be >= 0 when buffertype=gpu";
            return nullptr;
        }
#ifdef TESTTOOL_ENABLE_CUDA
        return std::make_unique<CudaAllocator>(deviceId);
#else
        // 未编入 CUDA SDK：直接报错退出，不做主机内存模拟——模拟会让测试者误以为
        // GPU 链路已测通（注册类型仍为 DEVICE，实际无真实设备内存），假阳性结果
        err =
            "buffertype=gpu requires TESTTOOL_ENABLE_CUDA build "
            "(rebuild with -DTESTTOOL_ENABLE_CUDA=ON); host-memory simulation is removed, "
            "use buffertype=cpu for host-memory testing";
        return nullptr;
#endif
    }
    if (bufferType == "npu") {
        if (deviceId < 0) {
            err = "deviceid must be >= 0 when buffertype=npu";
            return nullptr;
        }
#ifdef TESTTOOL_ENABLE_ASCEND
        return std::make_unique<AclAllocator>(deviceId);
#else
        // 未编入 ACL SDK：直接报错退出，不做主机内存模拟——模拟会让测试者误以为
        // NPU 链路已测通（注册类型仍为 DEVICE，实际无真实设备内存），假阳性结果
        err =
            "buffertype=npu requires TESTTOOL_ENABLE_ASCEND build "
            "(rebuild with -DTESTTOOL_ENABLE_ASCEND=ON -DTESTTOOL_ASCEND_ROOT=<CANN dir>, "
            "or bash scripts/build-kv-link.sh --mode npu); host-memory simulation is removed, "
            "use buffertype=cpu for host-memory testing";
        return nullptr;
#endif
    }
    err = "invalid buffertype '" + bufferType + "', expect cpu, gpu or npu";
    return nullptr;
}
