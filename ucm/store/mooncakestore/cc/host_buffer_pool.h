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
#include "thread/index_pool.h"

namespace UC::MooncakeStore {

class HostBufferPool {
public:
    HostBufferPool() = default;
    ~HostBufferPool() = default;

    void Setup(uint32_t count, size_t unitSize)
    {
        if (count == 0 || unitSize == 0) { return; }
        size_t totalSize = static_cast<size_t>(count) * unitSize;
        if (totalSize / unitSize != count) { return; }
        unitSize_ = unitSize;
        count_ = count;
        pool_ = std::make_unique<char[]>(totalSize);
        index_.Setup(count);
    }

    void* Acquire()
    {
        if (!pool_ || unitSize_ == 0) { return nullptr; }
        auto idx = index_.Acquire();
        if (idx == IndexPool::npos) { return nullptr; }
        return pool_.get() + static_cast<size_t>(idx) * unitSize_;
    }

    void* AcquireWithTimeout(std::chrono::milliseconds timeout)
    {
        if (!pool_ || unitSize_ == 0) { return nullptr; }
        auto idx = index_.Acquire();
        if (idx != IndexPool::npos) { return pool_.get() + static_cast<size_t>(idx) * unitSize_; }
        std::unique_lock<std::mutex> lk(cvMtx_);
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
            cv_.wait_until(lk, deadline);
            idx = index_.Acquire();
            if (idx != IndexPool::npos) {
                return pool_.get() + static_cast<size_t>(idx) * unitSize_;
            }
            if (std::chrono::steady_clock::now() >= deadline) { return nullptr; }
        }
    }

    void Release(void* buf)
    {
        if (!buf || !pool_ || unitSize_ == 0) { return; }
        auto offset = static_cast<char*>(buf) - pool_.get();
        if (offset < 0 || static_cast<size_t>(offset) >= static_cast<size_t>(count_) * unitSize_) {
            return;
        }
        if (static_cast<size_t>(offset) % unitSize_ != 0) { return; }
        auto idx = static_cast<IndexPool::Index>(static_cast<size_t>(offset) / unitSize_);
        index_.Release(idx);
        cv_.notify_one();
    }

    size_t UnitSize() const { return unitSize_; }
    uint32_t Count() const { return count_; }

private:
    std::unique_ptr<char[]> pool_;
    size_t unitSize_{0};
    uint32_t count_{0};
    IndexPool index_;
    std::mutex cvMtx_;
    std::condition_variable cv_;
};

}  // namespace UC::MooncakeStore

#endif
