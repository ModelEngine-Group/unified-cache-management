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
#include "breaker_store.h"
#include "logger/logger.h"

namespace UC::PipelineStore {

BreakerStore::BreakerStore(StoreV1* store, std::string storeId, BreakerConfig config)
    : store_(store), storeId_(std::move(storeId)), config_(config)
{
}

BreakerStore::~BreakerStore() { Stop(); }

Status BreakerStore::Start()
{
    if (!store_) { return Status::InvalidParam("breaker store target is null"); }
    if (config_.healthWindowSize == 0 || config_.failureThreshold == 0 ||
        config_.failureThreshold > config_.healthWindowSize ||
        config_.healthCheckInterval.count() <= 0 || config_.healthCheckTimeout.count() <= 0) {
        return Status::InvalidParam("invalid breaker config");
    }
    std::lock_guard<std::mutex> lock(stopMutex_);
    if (probeThread_.joinable()) { return Status::OK(); }
    stop_ = false;
    try {
        probeThread_ = std::thread(&BreakerStore::ProbeLoop, this);
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed({}) to start breaker probe", e.what()));
    }
    UC_INFO("Started store health breaker({}).", storeId_);
    return Status::OK();
}

void BreakerStore::Stop()
{
    {
        std::lock_guard<std::mutex> lock(stopMutex_);
        stop_ = true;
    }
    stopCv_.notify_all();
    if (probeThread_.joinable()) {
        probeThread_.join();
        UC_INFO("Stopped store health breaker({}).", storeId_);
    }
}

size_t BreakerStore::FailureCount() const
{
    std::lock_guard<std::mutex> lock(healthMutex_);
    return failureCount_;
}

size_t BreakerStore::SampleCount() const
{
    std::lock_guard<std::mutex> lock(healthMutex_);
    return healthResults_.size();
}

Status BreakerStore::Setup(const Detail::Dictionary&) { return Status::OK(); }

std::string BreakerStore::Readme() const { return "Breaker(" + storeId_ + ")"; }

Expected<std::vector<uint8_t>> BreakerStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    if (!Enabled()) { return std::vector<uint8_t>(num, 0); }
    return store_->Lookup(blocks, num);
}

Expected<ssize_t> BreakerStore::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    if (!Enabled()) { return static_cast<ssize_t>(-1); }
    return store_->LookupOnPrefix(blocks, num);
}

void BreakerStore::Prefetch(const Detail::BlockId* blocks, size_t num)
{
    if (Enabled()) { store_->Prefetch(blocks, num); }
}

Status BreakerStore::CheckHealth()
{
    const auto start = std::chrono::steady_clock::now();
    auto status = store_->CheckHealth();
    if (std::chrono::steady_clock::now() - start > config_.healthCheckTimeout) {
        status = Status::Timeout();
    }
    RecordHealth(status.Success());
    return status;
}

Expected<Detail::TaskHandle> BreakerStore::Load(Detail::TaskDesc task)
{
    if (!Enabled()) { return Status::StoreUnhealthy(storeId_); }
    return store_->Load(std::move(task));
}

Expected<Detail::TaskHandle> BreakerStore::Dump(Detail::TaskDesc task)
{
    if (!Enabled()) { return Status::StoreUnhealthy(storeId_); }
    return store_->Dump(std::move(task));
}

Expected<bool> BreakerStore::Check(Detail::TaskHandle taskId) { return store_->Check(taskId); }

Status BreakerStore::Wait(Detail::TaskHandle taskId) { return store_->Wait(taskId); }

void BreakerStore::RecordHealth(bool healthy)
{
    bool oldEnabled = false;
    bool newEnabled = false;
    size_t failureCount = 0;
    size_t sampleCount = 0;
    {
        std::lock_guard<std::mutex> lock(healthMutex_);
        if (healthResults_.size() == config_.healthWindowSize) {
            if (!healthResults_.front()) { --failureCount_; }
            healthResults_.pop_front();
        }
        healthResults_.push_back(healthy);
        if (!healthy) { ++failureCount_; }

        oldEnabled = enabled_.load(std::memory_order_relaxed);
        newEnabled = oldEnabled;
        if (oldEnabled && failureCount_ >= config_.failureThreshold) {
            newEnabled = false;
        } else if (!oldEnabled && healthResults_.size() == config_.healthWindowSize &&
                   failureCount_ == 0) {
            newEnabled = true;
        }
        enabled_.store(newEnabled, std::memory_order_release);
        failureCount = failureCount_;
        sampleCount = healthResults_.size();
    }
    if (oldEnabled != newEnabled) {
        UC_WARN("Store health breaker({}) transitioned to {}, samples={}, failures={}, threshold={}.",
                storeId_, newEnabled ? "HEALTHY" : "UNHEALTHY", sampleCount, failureCount,
                config_.failureThreshold);
    }
}

void BreakerStore::ProbeLoop()
{
    std::unique_lock<std::mutex> lock(stopMutex_);
    while (!stopCv_.wait_for(lock, config_.healthCheckInterval, [this] { return stop_; })) {
        lock.unlock();
        CheckHealth();
        lock.lock();
    }
}

}  // namespace UC::PipelineStore
