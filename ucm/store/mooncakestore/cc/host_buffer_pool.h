/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_HOST_BUFFER_POOL_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_HOST_BUFFER_POOL_H

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include "logger/logger.h"
#include "status/status.h"
#include "thread/index_pool.h"
#include "trans/buffer.h"
#include "trans/device.h"

namespace UC::MooncakeStore {

class HostBufferPool {
public:
    using Handle = std::shared_ptr<void>;

    HostBufferPool() = default;
    ~HostBufferPool() = default;

    HostBufferPool(const HostBufferPool&) = delete;
    HostBufferPool& operator=(const HostBufferPool&) = delete;

    Status Setup(int32_t deviceId, uint32_t count, size_t unitSize, bool ioDirect)
    {
        if (count == 0 || unitSize == 0) { return Status::OK(); }
        size_t totalSize = static_cast<size_t>(count) * unitSize;
        if (totalSize / unitSize != count) {
            return Status::InvalidParam("host buffer pool size overflow");
        }

        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}) for host buffer pool.", s, deviceId);
            return s;
        }
        auto buffer = device.MakeBuffer();
        if (!buffer) [[unlikely]] {
            UC_ERROR("Failed to make buffer factory on device({}).", deviceId);
            return Status::Error();
        }
        auto holder = ioDirect ? buffer->MakeHostBuffer4DirectIo(totalSize)
                               : buffer->MakeHostBuffer(totalSize);
        if (!holder) [[unlikely]] {
            UC_ERROR("Failed to allocate {} host buffer pool, size={}B.",
                     ioDirect ? "DirectIO_HugePages" : "Pinned", totalSize);
            return Status::OutOfMemory();
        }

        unitSize_ = unitSize;
        count_ = count;
        pool_ = std::move(holder);
        index_.Setup(count);
        UC_INFO("HostBufferPool: {} x {}B = {}B, mode={}", count, unitSize, totalSize,
                ioDirect ? "DirectIO_HugePages" : "Pinned_aclrtMallocHost");
        return Status::OK();
    }

    Handle Acquire()
    {
        if (!pool_ || unitSize_ == 0) { return {}; }
        auto idx = index_.Acquire();
        if (idx == IndexPool::npos) { return {}; }
        return MakeHandle(idx);
    }

    Handle AcquireWithTimeout(std::chrono::milliseconds timeout)
    {
        if (!pool_ || unitSize_ == 0) { return {}; }
        auto idx = index_.Acquire();
        if (idx != IndexPool::npos) { return MakeHandle(idx); }
        std::unique_lock<std::mutex> lk(cvMtx_);
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
            cv_.wait_until(lk, deadline);
            idx = index_.Acquire();
            if (idx != IndexPool::npos) { return MakeHandle(idx); }
            if (std::chrono::steady_clock::now() >= deadline) { return {}; }
        }
    }

    size_t UnitSize() const { return unitSize_; }
    uint32_t Count() const { return count_; }

private:
    Handle MakeHandle(IndexPool::Index idx)
    {
        void* raw = static_cast<char*>(pool_.get()) + static_cast<size_t>(idx) * unitSize_;
        return Handle(raw, [this, idx](void*) noexcept { this->ReleaseByIndex(idx); });
    }

    void ReleaseByIndex(IndexPool::Index idx) noexcept
    {
        index_.Release(idx);
        cv_.notify_one();
    }

    std::shared_ptr<void> pool_;
    size_t unitSize_{0};
    uint32_t count_{0};
    IndexPool index_;
    std::mutex cvMtx_;
    std::condition_variable cv_;
};

}  // namespace UC::MooncakeStore

#endif
