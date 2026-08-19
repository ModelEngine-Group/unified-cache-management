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
 */
#include "backfill_queue.h"
#include <cstring>
#include <exception>
#include <fmt/format.h>
#include "logger/logger.h"
#ifdef __linux__
#include "thread/cpu_affinity.h"
#endif
#include "time/now_time.h"
#include "yuanrong_helper.h"

namespace UC::YuanRongStore {

BackfillQueue::~BackfillQueue() { Close(); }

Status BackfillQueue::Setup(const Config& config, std::shared_ptr<datasystem::KVClient> kvClient)
{
    config_ = config;
    kvClient_ = std::move(kvClient);
    queueDepth_ = config.backfillQueueDepth;
    try {
        workers_.reserve(config.backfillWorkerCount);
        for (size_t i = 0; i < config.backfillWorkerCount; ++i) {
            workers_.emplace_back(&BackfillQueue::WorkerStage, this);
        }
    } catch (const std::exception& e) {
        Close();
        return Status::Error(fmt::format("failed to start YuanRong backfill worker: {}", e.what()));
    }
    return Status::OK();
}

bool BackfillQueue::Submit(BackfillTask task)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_ || waiting_.size() >= queueDepth_) {
        UC_WARN("YuanRong backfill queue full, dropping ({} keys).", task.keys.size());
        return false;
    }
    waiting_.push_back(std::move(task));
    cv_.notify_one();
    return true;
}

void BackfillQueue::Close()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) { return; }
        stop_ = true;
        cv_.notify_all();
    }
    for (auto& worker : workers_) {
        if (worker.joinable()) { worker.join(); }
    }
    workers_.clear();
}

void BackfillQueue::WorkerStage()
{
#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto status = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (status.Failure()) { UC_WARN("Failed({}) to set YuanRong backfill affinity.", status); }
    }
#endif
    for (;;) {
        BackfillTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !waiting_.empty(); });
            if (waiting_.empty()) {
                if (stop_) { return; }
                continue;
            }
            task = std::move(waiting_.front());
            waiting_.pop_front();
        }
        RunOne(task);
    }
}

void BackfillQueue::RunOne(BackfillTask& task)
{
    auto start = NowTime::Now();
    if (task.keys.empty() || task.keys.size() != task.hostBuffers.size()) {
        UC_ERROR("Invalid YuanRong backfill task, keys={}, buffers={}.", task.keys.size(),
                 task.hostBuffers.size());
        return;
    }

    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    param.existence = datasystem::ExistenceOpt::NX;
    param.cacheType = datasystem::CacheType::MEMORY;
    const auto composedSize =
        YuanRongComposedObjectSize(config_.tensorSizes, config_.memoryAlignment);
    std::vector<uint64_t> objectSizes(task.keys.size(), composedSize);
    std::vector<std::shared_ptr<datasystem::Buffer>> buffers;
    auto createStart = NowTime::Now();
    auto createStatus = kvClient_->MCreate(task.keys, objectSizes, param, buffers);
    auto createEnd = NowTime::Now();
    if (createStatus.IsError() || buffers.size() != task.keys.size()) {
        UC_ERROR("YuanRong async backfill MCreate failed for ({} keys): {}.", task.keys.size(),
                 createStatus.ToString());
        return;
    }

    std::vector<std::shared_ptr<datasystem::Buffer>> created;
    created.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        if (!buffer) {
            UC_ERROR("YuanRong async backfill returned null buffer for key({}).", task.keys[i]);
            return;
        }
        if (buffer->GetSize() == 0) { continue; }
        void* payload = nullptr;
        auto status =
            InitYuanRongComposedBuffer(task.keys[i], buffer->MutableData(), buffer->GetSize(),
                                       config_.tensorSizes, config_.memoryAlignment, payload);
        if (status.Failure()) {
            UC_ERROR("YuanRong async backfill buffer init failed: {}.", status);
            return;
        }
        std::memcpy(payload, task.hostBuffers[i].get(), config_.objectSize);
        created.push_back(buffer);
    }

    if (created.empty()) { return; }
    auto setStart = NowTime::Now();
    auto publishStatus = kvClient_->MSet(created);
    auto setEnd = NowTime::Now();
    if (publishStatus.IsError()) {
        UC_ERROR("YuanRong async backfill MSet failed for ({} keys): {}.", created.size(),
                 publishStatus.ToString());
        return;
    }
    UC_DEBUG(
        "YuanRong async backfill({} keys) finished, mcreate={:.3f}ms, copy={:.3f}ms, "
        "mset={:.3f}ms, total={:.3f}ms.",
        task.keys.size(), (createEnd - createStart) * 1e3, (setStart - createEnd) * 1e3,
        (setEnd - setStart) * 1e3, (setEnd - start) * 1e3);
}

}  // namespace UC::YuanRongStore
