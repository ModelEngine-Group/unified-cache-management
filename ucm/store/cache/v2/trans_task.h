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
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include "detail/dump_source_fence.h"
#include "status/status.h"
#include "template/hashset.h"
#include "thread/latch.h"
#include "type/types.h"

namespace UC::Cache2 {

class DumpSourceState {
    std::mutex mutex_;
    std::condition_variable ready_;
    bool published_{false};
    std::shared_ptr<Detail::DumpSourceFence> fence_;

public:
    void Publish(std::shared_ptr<Detail::DumpSourceFence> fence = nullptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (published_) { return; }
        fence_ = std::move(fence);
        published_ = true;
        ready_.notify_all();
    }
    std::shared_ptr<Detail::DumpSourceFence> Wait(size_t timeoutMs)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] { return published_; });
        return fence_;
    }
};

struct Task {
    enum class Type : uint8_t { LOAD, DUMP };

    Task(Type type, Detail::TaskDesc desc) : id{NextId()}, type{type}, desc{std::move(desc)} {}
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&& other) noexcept
        : id{other.id},
          type{other.type},
          desc{std::move(other.desc)},
          failureStatus{other.failureStatus.load()},
          dumpSource{std::move(other.dumpSource)}
    {
    }

    Detail::TaskHandle id{};
    Type type{Type::DUMP};
    Detail::TaskDesc desc;
    std::atomic<int32_t> failureStatus{Status::OK().Underlying()};
    std::shared_ptr<DumpSourceState> dumpSource;

    void Fail(const Status& status)
    {
        auto expected = Status::OK().Underlying();
        failureStatus.compare_exchange_strong(expected, status.Underlying());
    }
    Status FailureStatus() const
    {
        const auto status = failureStatus.load();
        return status == Status::OK().Underlying() ? Status::Error() : Status{status, {}};
    }

private:
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }
};

using TaskPtr = std::shared_ptr<Task>;
using WaiterPtr = std::shared_ptr<Latch>;
using TaskIdSet = HashSet<Detail::TaskHandle>;

}  // namespace UC::Cache2
