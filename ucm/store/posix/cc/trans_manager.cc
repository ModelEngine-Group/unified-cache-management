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
#include "trans_manager.h"

namespace UC::PosixStore {

TransManager::~TransManager()
{
    {
        // Drain callbacks before destroying any engine they could submit to.
        std::unique_lock<std::shared_mutex> lock(callbacks_->mutex);
        callbacks_->stopped = true;
    }
    aioEngine_.reset();
    psyncEngine_.reset();
}

Status TransManager::Setup(const Config& config, const SpaceLayout* layout,
                           const BackendManager* backendMgr)
{
    if (config.ioEngine != "aio" && config.ioEngine != "psync") {
        return Status::InvalidParam("invalid io engine({})", config.ioEngine);
    }
    backendMgr_ = backendMgr;
    timeoutMs_ = config.timeoutMs;
    auto backendConfig = config;
    backendConfig.timeoutMs = backendMgr->IoTimeoutMs();
    if (config.ioEngine == "aio") {
        aioEngine_ = std::make_unique<IoEngineAio>();
        engine_ = aioEngine_.get();
        return aioEngine_->Setup(backendConfig, layout, &backendMgr->Backends());
    } else {
        psyncEngine_ = std::make_unique<IoEnginePsync>();
        engine_ = psyncEngine_.get();
        return psyncEngine_->Setup(backendConfig, layout, &backendMgr->Backends());
    }
}

Expected<Detail::TaskHandle> TransManager::Submit(TransTask task)
{
    auto handle = task.id;
    std::shared_ptr<Request> request;
    try {
        request = std::make_shared<Request>(std::move(task));
        for (const auto& shard : request->task.desc) {
            request->shardTasks.push_back(ShardTask{shard});
        }
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (!requests_.emplace(handle, request).second) { return Status::DuplicateKey(); }
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }
    for (auto& shardTask : request->shardTasks) { TryNextBackend(request, shardTask); }
    return handle;
}

Expected<bool> TransManager::Check(Detail::TaskHandle handle)
{
    std::shared_ptr<Request> request;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        const auto found = requests_.find(handle);
        if (found == requests_.end()) { return Status::NotFound(); }
        request = found->second;
    }
    return request->waiter.Check();
}

Status TransManager::Wait(Detail::TaskHandle handle)
{
    std::shared_ptr<Request> request;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        const auto found = requests_.find(handle);
        if (found == requests_.end()) { return Status::NotFound(); }
        request = found->second;
    }
    request->waiter.Wait();
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        requests_.erase(handle);
    }
    for (const auto& shardTask : request->shardTasks) {
        if (shardTask.result.Failure()) { return shardTask.result; }
    }
    return Status::OK();
}

void TransManager::TryNextBackend(const std::shared_ptr<Request>& request, ShardTask& shardTask)
{
    if (shardTask.shard.addrs.empty()) {
        shardTask.result = Status::InvalidParam("empty shard addresses");
        request->waiter.Done();
        return;
    }
    if (timeoutMs_ > 0 && (NowTime::Now() - request->startTp) * 1000 >= timeoutMs_) {
        shardTask.result = Status::Timeout();
        request->waiter.Done();
        return;
    }
    // Dump only uses the primary backend, so all writers of a block go through the same path.
    auto attempt = request->task.type == TransTask::Type::DUMP
                       ? backendMgr_->PrimaryBackend(shardTask.shard.owner)
                       : backendMgr_->SelectNextBackend(shardTask.shard.owner, shardTask.attempted);
    if (attempt) {
        shardTask.attempt = attempt.Value();
    } else {
        // Once a backend has been tried, keep its failure instead of "no backend left".
        if (shardTask.attempted.empty()) { shardTask.result = attempt.Error(); }
        request->waiter.Done();
        return;
    }
    Detail::TaskDesc desc{shardTask.shard};
    desc.brief = request->task.desc.brief;
    TransTask task{request->task.type, std::move(desc)};
    task.backendIndex = shardTask.attempt.index;
    // check status in callback, if need retry then call TryNextBackend
    task.onComplete = [this, request, &shardTask, callbacks = callbacks_](Status status) {
        std::shared_lock<std::shared_mutex> lock(callbacks->mutex);
        if (callbacks->stopped) { return; }
        shardTask.result = status;
        const auto retry = backendMgr_->ShouldRetry(shardTask.attempt, status);
        if (retry && request->task.type == TransTask::Type::LOAD) {
            // A queued timeout is not the backend's fault, so the retry may pick it again.
            if (status == Status::TimeoutInQueue()) { shardTask.attempted.pop_back(); }
            TryNextBackend(request, shardTask);
        } else {
            request->waiter.Done();
        }
    };
    auto submitted = engine_->Submit(std::move(task));
    if (submitted) {
        return;
    } else {
        // Submit only fails on a duplicate task id or allocation; I/O errors arrive through
        // onComplete. Neither reflects backend health, so do not retry or record it.
        shardTask.result = submitted.Error();
        request->waiter.Done();
    }
}

}  // namespace UC::PosixStore
