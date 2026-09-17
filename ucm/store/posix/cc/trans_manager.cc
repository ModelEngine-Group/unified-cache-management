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
#include <algorithm>

namespace UC::PosixStore {

TransManager::~TransManager()
{
    {
        // Drain callbacks before destroying any engine they could submit to.
        std::unique_lock<std::shared_mutex> lock(callbacks_->mutex);
        callbacks_->stopped = true;
    }
    aioEngines_.clear();
    psyncEngines_.clear();
}

Status TransManager::Setup(const Config& config, const SpaceLayout* layout)
{
    if (config.ioEngine != "aio" && config.ioEngine != "psync") {
        return Status::InvalidParam("invalid io engine({})", config.ioEngine);
    }
    layout_ = layout;
    timeoutMs_ = config.timeoutMs;
    auto backendConfig = config;
    backendConfig.timeoutMs = layout->IoTimeoutMs();
    for (const auto& backend : layout->Backends()) {
        Status status = Status::OK();
        if (config.ioEngine == "aio") {
            auto engine = std::make_unique<IoEngineAio>();
            status = engine->Setup(backendConfig, layout, backend);
            engines_.push_back(engine.get());
            aioEngines_.push_back(std::move(engine));
        } else {
            auto engine = std::make_unique<IoEnginePsync>();
            status = engine->Setup(backendConfig, layout, backend);
            engines_.push_back(engine.get());
            psyncEngines_.push_back(std::move(engine));
        }
        if (status.Failure()) { return status; }
    }
    return Status::OK();
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
    for (;;) {
        if (timeoutMs_ > 0 && (NowTime::Now() - request->startTp) * 1000 >= timeoutMs_) {
            shardTask.result = Status::Timeout();
            break;
        }
        auto backend = layout_->StorageBackend(shardTask.shard.owner, shardTask.attempted);
        if (!backend) {
            if (shardTask.attempted.empty()) { shardTask.result = backend.Error(); }
            break;
        }
        const auto& backends = layout_->Backends();
        shardTask.backendIndex =
            std::find(backends.begin(), backends.end(), backend.Value()) - backends.begin();
        shardTask.attempted.push_back(backend.Value());
        Detail::TaskDesc desc{shardTask.shard};
        desc.brief = request->task.desc.brief;
        TransTask task{request->task.type, std::move(desc)};
        task.onComplete = [this, request, &shardTask, callbacks = callbacks_](Status status) {
            std::shared_lock<std::shared_mutex> lock(callbacks->mutex);
            if (callbacks->stopped) { return; }
            shardTask.result = status;
            layout_->RecordIoResult(layout_->Backends()[shardTask.backendIndex], status);
            if (status.Success() || status == Status::NotFound()) {
                request->waiter.Done();
            } else {
                TryNextBackend(request, shardTask);
            }
        };
        auto submitted = engines_[shardTask.backendIndex]->Submit(std::move(task));
        if (submitted) { return; }

        // error handling
        shardTask.result = submitted.Error();
        layout_->RecordIoResult(backend.Value(), shardTask.result);
        if (shardTask.result == Status::NotFound()) { break; }
    }
    request->waiter.Done();
}

}  // namespace UC::PosixStore
