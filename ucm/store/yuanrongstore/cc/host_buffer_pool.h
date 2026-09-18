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
 */
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_HOST_BUFFER_POOL_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_HOST_BUFFER_POOL_H

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include "status/status.h"
#include "thread/index_pool.h"
#include "trans/device.h"

namespace UC::YuanRongStore {

class HostBufferPool {
public:
    using Handle = std::shared_ptr<void>;

    Status Setup(int32_t deviceId, uint32_t count, size_t unitSize, bool ioDirect)
    {
        if (count == 0 || unitSize == 0) {
            return Status::InvalidParam("invalid YuanRong host buffer pool size");
        }
        const auto totalSize = static_cast<size_t>(count) * unitSize;
        if (totalSize / unitSize != count) { return Status::OutOfMemory(); }

        Trans::Device device;
        auto status = device.Setup(deviceId);
        if (status.Failure()) { return status; }
        auto buffer = device.MakeBuffer();
        if (!buffer) { return Status::Error("failed to create YuanRong host buffer factory"); }
        pool_ = ioDirect ? buffer->MakeHostBuffer4DirectIo(totalSize)
                         : buffer->MakeHostBuffer(totalSize);
        if (!pool_) { return Status::OutOfMemory(); }

        unitSize_ = unitSize;
        count_ = count;
        indexes_.Setup(count);
        return Status::OK();
    }

    Handle Acquire(std::chrono::milliseconds timeout)
    {
        auto index = indexes_.Acquire();
        if (index != IndexPool::npos) { return MakeHandle(index); }

        std::unique_lock<std::mutex> lock(mutex_);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            index = indexes_.Acquire();
            if (index != IndexPool::npos) { return MakeHandle(index); }
            if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                index = indexes_.Acquire();
                return index == IndexPool::npos ? Handle{} : MakeHandle(index);
            }
        }
    }

    size_t UnitSize() const { return unitSize_; }
    uint32_t Count() const { return count_; }

private:
    Handle MakeHandle(IndexPool::Index index)
    {
        auto* address = static_cast<uint8_t*>(pool_.get()) + static_cast<size_t>(index) * unitSize_;
        return Handle(address, [this, index](void*) {
            indexes_.Release(index);
            cv_.notify_one();
        });
    }

    std::shared_ptr<void> pool_;
    size_t unitSize_{0};
    uint32_t count_{0};
    IndexPool indexes_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace UC::YuanRongStore

#endif
