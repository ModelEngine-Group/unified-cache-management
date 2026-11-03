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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_TRANS_TASK_H
#define UNIFIEDCACHE_POSIX_STORE_CC_TRANS_TASK_H

#include <atomic>
#include <functional>
#include "status/status.h"
#include "type/types.h"

namespace UC::PosixStore {

class TransTask {
public:
    enum class Type : uint8_t { LOAD, DUMP };
    Detail::TaskHandle id{0};
    Type type{Type::DUMP};
    Detail::TaskDesc desc;
    size_t backendIndex{0};
    std::atomic<int32_t> failureStatus{Status::OK().Underlying()};
    // Set when a worker starts on the task. A timeout before that is queueing, not a backend
    // failure.
    std::atomic<bool> started{false};
    // When set, the engine reclaims the handle before invoking this callback.
    std::function<void(Status)> onComplete;

public:
    TransTask(Type type, Detail::TaskDesc desc) : id{NextId()}, type{type}, desc{std::move(desc)} {}
    TransTask(TransTask&& other) noexcept
        : id{other.id},
          type{other.type},
          desc{std::move(other.desc)},
          backendIndex{other.backendIndex},
          failureStatus{other.failureStatus.load()},
          started{other.started.load()},
          onComplete{std::move(other.onComplete)}
    {
    }
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
    static size_t NextId() noexcept
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::PosixStore

#endif
