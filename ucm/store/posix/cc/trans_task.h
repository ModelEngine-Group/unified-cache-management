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
#include <mutex>
#include "status/status.h"
#include "type/types.h"

namespace UC::PosixStore {

class TransTask {
public:
    enum class Type : uint8_t { LOAD, DUMP };
    Detail::TaskHandle id{0};
    Type type{Type::DUMP};
    Detail::TaskDesc desc;
    // When set, the engine reclaims the handle before invoking this callback.
    std::function<void(Status)> onComplete;

public:
    TransTask(Type type, Detail::TaskDesc desc) : id{NextId()}, type{type}, desc{std::move(desc)} {}
    TransTask(TransTask&& other)
        : id{other.id},
          type{other.type},
          desc{std::move(other.desc)},
          onComplete{std::move(other.onComplete)},
          result_{other.Result()}
    {
    }
    bool SetFirstFail(const Status& status)
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        if (result_.Failure()) { return false; }
        result_ = status;
        return true;
    }
    Status FailureStatus() const
    {
        const auto status = Result();
        return status.Success() ? Status::Error() : status;
    }
    Status Result() const
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        return result_;
    }

private:
    mutable std::mutex resultMutex_;
    Status result_{Status::OK()};
    static size_t NextId() noexcept
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::PosixStore

#endif
