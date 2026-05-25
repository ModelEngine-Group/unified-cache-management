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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include "asu_transport/types.h"

namespace UC::ASU {

template <typename Context, typename State>
class TaskManagerBase {
public:
    TaskManagerBase(State initial_state, std::string task_name)
        : initial_state_(initial_state), task_name_(std::move(task_name))
    {
    }

    Status Submit(std::unique_ptr<Context> ctx, TaskId& task_id)
    {
        if (!ctx) {
            task_id = kInvalidTaskId;
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 task_name_ + " task context is null");
        }

        auto shared_ctx = std::shared_ptr<Context>(std::move(ctx));
        shared_ctx->state.store(initial_state_, std::memory_order_release);

        std::lock_guard<std::mutex> lock(mu_);
        do {
            task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
        } while (task_id == kInvalidTaskId || tasks_.find(task_id) != tasks_.end());

        shared_ctx->task_id = task_id;
        tasks_.emplace(task_id, std::move(shared_ctx));
        return Status::OK();
    }

    std::shared_ptr<Context> Get(TaskId task_id)
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto iter = tasks_.find(task_id);
        if (iter == tasks_.end()) { return nullptr; }
        return iter->second;
    }

    Status Remove(TaskId task_id)
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto erased = tasks_.erase(task_id);
        if (erased == 0) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, task_name_ + " task not found");
        }
        return Status::OK();
    }

private:
    State initial_state_;
    std::string task_name_;
    std::atomic<TaskId> next_task_id_{1};
    // TODO: consider using a lock-free structure !
    std::mutex mu_;
    std::unordered_map<TaskId, std::shared_ptr<Context>> tasks_;
};

}  // namespace UC::ASU
