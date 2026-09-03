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
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include "types.h"

namespace UC::ASU {

template <typename Context, typename State>
class TaskManagerBase {
public:
    TaskManagerBase(State initialState, std::string taskName)
        : initialState_(initialState), taskName_(std::move(taskName))
    {
    }

    Status Submit(std::unique_ptr<Context> ctx, TaskId& taskId)
    {
        if (!ctx) {
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::INVALID_ARGUMENT, taskName_ + " task context is null");
        }

        return Submit(std::shared_ptr<Context>(std::move(ctx)), taskId);
    }

    Status Submit(const std::shared_ptr<Context>& ctx, TaskId& taskId)
    {
        if (!ctx) {
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::INVALID_ARGUMENT, taskName_ + " task context is null");
        }

        auto sharedCtx = ctx;
        sharedCtx->state.store(initialState_, std::memory_order_release);

        std::lock_guard<std::shared_mutex> lock(mutex_);
        do {
            taskId = nextTaskId_.fetch_add(1, std::memory_order_relaxed);
        } while (taskId == kInvalidTaskId || tasks_.find(taskId) != tasks_.end());

        sharedCtx->taskId = taskId;
        tasks_.emplace(taskId, std::move(sharedCtx));
        return Status::OK();
    }

    std::shared_ptr<Context> Get(TaskId taskId)
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto iter = tasks_.find(taskId);
        if (iter == tasks_.end()) { return nullptr; }
        return iter->second;
    }

    std::vector<std::shared_ptr<Context>> GetAll()
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<std::shared_ptr<Context>> tasks;
        tasks.reserve(tasks_.size());
        for (const auto& item : tasks_) { tasks.emplace_back(item.second); }
        return tasks;
    }

    Status Remove(TaskId taskId)
    {
        std::lock_guard<std::shared_mutex> lock(mutex_);
        auto erased = tasks_.erase(taskId);
        if (erased == 0) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, taskName_ + " task not found");
        }
        return Status::OK();
    }

private:
    State initialState_;
    std::string taskName_;
    std::atomic<TaskId> nextTaskId_{1};
    // TODO: consider using a lock-free structure !
    std::shared_mutex mutex_;
    std::unordered_map<TaskId, std::shared_ptr<Context>> tasks_;
};

}  // namespace UC::ASU
