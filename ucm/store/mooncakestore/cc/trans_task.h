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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_TASK_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_TASK_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>
#include "type/types.h"

namespace UC::MooncakeStore {

enum class TaskType { LOAD, DUMP };
enum class TaskStatus { PENDING, RUNNING, SUCCESS, FAILED };

struct TransShard {
    std::string key;
    Detail::BlockId owner;
    size_t index;
    std::vector<void*> addrs;
    std::vector<size_t> sizes;
};

struct TransTask {
    TaskType type;
    std::string brief;
    std::vector<TransShard> shards;
    uintptr_t prerequisiteHandle{0};
};

struct TaskState {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<TaskStatus> status{TaskStatus::PENDING};
    std::string errMsg;

    bool IsTerminal() const
    {
        auto s = status.load(std::memory_order_acquire);
        return s == TaskStatus::SUCCESS || s == TaskStatus::FAILED;
    }

    void Complete(TaskStatus s, std::string msg = {})
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            status.store(s, std::memory_order_release);
            errMsg = std::move(msg);
        }
        cv.notify_all();
    }
};

struct PendingItem {
    Detail::TaskHandle handle;
    TransTask task;
};

}  // namespace UC::MooncakeStore

#endif
