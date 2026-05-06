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
#include "task_manager.h"
#include "logger/logger.h"

namespace UC::MooncakeStore {

TaskManager::TaskManager() = default;

TaskManager::~TaskManager() { Close(); }

Status TaskManager::Setup(uint32_t loadWorkerNum, uint32_t dumpWorkerNum, HostBufferPool& bufPool,
                          LoadProcessFn loadFn, DumpProcessFn dumpFn)
{
    if (!loadFn) { return Status::InvalidParam("loadFn is null"); }
    if (!dumpFn) { return Status::InvalidParam("dumpFn is null"); }
    loadFn_ = std::move(loadFn);
    dumpFn_ = std::move(dumpFn);
    bufPool_ = &bufPool;

    loadPool_ = std::make_unique<ThreadPool<PendingItem>>();
    loadPool_->SetNWorker(loadWorkerNum)
        .SetWorkerFn(
            [this](PendingItem& item, auto&) { loadFn_(item.handle, item.task, *bufPool_); })
        .Run();

    dumpPool_ = std::make_unique<ThreadPool<PendingItem>>();
    dumpPool_->SetNWorker(dumpWorkerNum)
        .SetWorkerFn([this](PendingItem& item, auto&) { dumpFn_(item.handle, item.task); })
        .Run();

    UC_DEBUG("TaskManager setup ok, loadWorkers={}, dumpWorkers={}, bufPool={}x{}", loadWorkerNum,
             dumpWorkerNum, bufPool.Count(), bufPool.UnitSize());
    return Status::OK();
}

void TaskManager::Close()
{
    if (closed_.exchange(true, std::memory_order_acq_rel)) { return; }

    loadPool_.reset();
    dumpPool_.reset();

    std::lock_guard<std::mutex> lk(taskMtx_);
    for (auto& [h, state] : tasks_) {
        if (!state->IsTerminal()) { state->Complete(TaskStatus::FAILED, "shutting down"); }
    }
    tasks_.clear();
}

Expected<Detail::TaskHandle> TaskManager::Submit(TransTask task, bool isLoad)
{
    if (isLoad && !loadPool_) { return Status::Error("TaskManager load pool is not running"); }
    if (!isLoad && !dumpPool_) { return Status::Error("TaskManager dump pool is not running"); }

    auto handle = nextTaskId_.fetch_add(1, std::memory_order_relaxed);
    auto state = std::make_shared<TaskState>();
    {
        std::lock_guard<std::mutex> lk(taskMtx_);
        tasks_[handle] = state;
    }

    if (isLoad) {
        loadPool_->Push(PendingItem{handle, std::move(task)});
    } else {
        dumpPool_->Push(PendingItem{handle, std::move(task)});
    }
    return handle;
}

Expected<Detail::TaskHandle> TaskManager::SubmitLoad(TransTask task)
{
    task.type = TaskType::LOAD;
    return Submit(std::move(task), true);
}

Expected<Detail::TaskHandle> TaskManager::SubmitDump(TransTask task)
{
    task.type = TaskType::DUMP;
    return Submit(std::move(task), false);
}

std::shared_ptr<TaskState> TaskManager::GetState(Detail::TaskHandle handle)
{
    std::lock_guard<std::mutex> lk(taskMtx_);
    auto it = tasks_.find(handle);
    if (it == tasks_.end()) { return nullptr; }
    return it->second;
}

Expected<bool> TaskManager::Check(Detail::TaskHandle handle)
{
    auto state = GetState(handle);
    if (!state) { return Status::InvalidParam("unknown task id"); }
    bool done = state->IsTerminal();
    if (done) {
        std::lock_guard<std::mutex> lk(taskMtx_);
        tasks_.erase(handle);
    }
    return done;
}

Status TaskManager::Wait(Detail::TaskHandle handle)
{
    auto state = GetState(handle);
    if (!state) { return Status::InvalidParam("unknown task id"); }

    TaskStatus st;
    std::string msg;
    {
        std::unique_lock<std::mutex> lk(state->mtx);
        state->cv.wait(lk, [&] { return state->IsTerminal(); });
        st = state->status.load(std::memory_order_relaxed);
        msg = state->errMsg;
    }

    {
        std::lock_guard<std::mutex> lk(taskMtx_);
        tasks_.erase(handle);
    }

    if (st == TaskStatus::SUCCESS) { return Status::OK(); }
    return Status::Error("mooncake task failed: " + msg);
}

}  // namespace UC::MooncakeStore
