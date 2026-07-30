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
#include "health_breaker_store.h"
#include <random>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::PipelineStore {

namespace {

std::chrono::milliseconds RandomProbeDelay(std::chrono::milliseconds interval)
{
    thread_local std::mt19937 generator{std::random_device{}()};
    std::uniform_int_distribution<std::chrono::milliseconds::rep> distribution{0, interval.count()};
    return std::chrono::milliseconds{distribution(generator)};
}

}  // namespace

HealthBreakerStore::HealthBreakerStore(StoreV1* store, std::string storeId,
                                       HealthBreakerConfig config)
    : store_(store),
      storeId_(std::move(storeId)),
      config_(config),
      healthCheck_(config.healthCheckTimeout)
{
}

HealthBreakerStore::~HealthBreakerStore() { Stop(); }

Status HealthBreakerStore::Start()
{
    if (!store_) { return Status::InvalidParam("health breaker store target is null"); }
    if (config_.healthWindowSize == 0 || config_.failureThreshold == 0 ||
        config_.failureThreshold > config_.healthWindowSize ||
        config_.healthCheckInterval.count() <= 0 || config_.healthCheckTimeout.count() <= 0 ||
        config_.healthCheckTimeout >= config_.healthCheckInterval) {
        return Status::InvalidParam("invalid health breaker config");
    }
    std::lock_guard<std::mutex> lock(stopMutex_);
    if (probeThread_.joinable()) { return Status::OK(); }
    stop_ = false;
    try {
        probeThread_ = std::thread(&HealthBreakerStore::ProbeLoop, this);
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed({}) to start health breaker probe", e.what()));
    }
    RecordEffectiveHealth();
    UC_INFO("Started store health breaker({}).", storeId_);
    return Status::OK();
}

void HealthBreakerStore::Stop()
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

size_t HealthBreakerStore::FailureCount() const
{
    std::lock_guard<std::mutex> lock(healthMutex_);
    return failureCount_;
}

size_t HealthBreakerStore::SampleCount() const
{
    std::lock_guard<std::mutex> lock(healthMutex_);
    return healthResults_.size();
}

Status HealthBreakerStore::Setup(const Detail::Dictionary&) { return Status::OK(); }

std::string HealthBreakerStore::Readme() const { return "HealthBreakerStore(" + storeId_ + ")"; }

Expected<std::vector<uint8_t>> HealthBreakerStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    if (!Enabled()) { return std::vector<uint8_t>(num, 0); }
    return store_->Lookup(blocks, num);
}

Expected<ssize_t> HealthBreakerStore::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    if (!Enabled()) { return static_cast<ssize_t>(-1); }
    return store_->LookupOnPrefix(blocks, num);
}

void HealthBreakerStore::Prefetch(const Detail::BlockId* blocks, size_t num)
{
    if (Enabled()) { store_->Prefetch(blocks, num); }
}

Status HealthBreakerStore::CheckHealth()
{
    auto status = healthCheck_.Run([this] { return store_->CheckHealth(); });
    if (status == Status::Timeout()) {
        UC_WARN_UNLIMITED("Store health check({}) timed out after {} ms.", storeId_,
                          config_.healthCheckTimeout.count());
    } else if (status.Failure()) {
        UC_WARN_UNLIMITED("Store health check({}) failed({}).", storeId_, status);
    }
    RecordHealth(status.Success());
    RecordProbeMetrics(status.Success());
    return status;
}

Expected<Detail::TaskHandle> HealthBreakerStore::Load(Detail::TaskDesc task)
{
    if (!Enabled()) { return Status::StoreUnhealthy(storeId_); }
    return store_->Load(std::move(task));
}

Expected<Detail::TaskHandle> HealthBreakerStore::Dump(Detail::TaskDesc task)
{
    if (!Enabled()) { return Status::StoreUnhealthy(storeId_); }
    return store_->Dump(std::move(task));
}

Expected<bool> HealthBreakerStore::Check(Detail::TaskHandle taskId)
{
    return store_->Check(taskId);
}

Status HealthBreakerStore::Wait(Detail::TaskHandle taskId) { return store_->Wait(taskId); }

void HealthBreakerStore::RecordHealth(bool healthy)
{
    bool oldEnabled = false;
    bool newEnabled = false;
    size_t failureCount = 0;
    size_t sampleCount = 0;
    std::string healthWindow;
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
        if (oldEnabled != newEnabled) {
            for (bool result : healthResults_) {
                if (!healthWindow.empty()) { healthWindow += ", "; }
                healthWindow += result ? "success" : "failure";
            }
        }
    }
    if (oldEnabled != newEnabled) {
        UC_WARN_UNLIMITED(
            "Store health breaker({}) transitioned to {}, window=[{}], samples={}, failures={}, "
            "threshold={}.",
            storeId_, newEnabled ? "HEALTHY" : "UNHEALTHY", healthWindow, sampleCount, failureCount,
            config_.failureThreshold);
    }
}

void HealthBreakerStore::RecordProbeMetrics(bool healthy)
{
    if (storeId_.find(":PosixStore") != std::string::npos) {
        UC::Metrics::UpdateStats(healthy ? NAME_TO_METRIC_ID("posix_healthy_count_total")
                                         : NAME_TO_METRIC_ID("posix_unhealthy_count_total"),
                                 1.0);
    } else if (storeId_.find(":MooncakeStore") != std::string::npos) {
        UC::Metrics::UpdateStats(healthy ? NAME_TO_METRIC_ID("mooncake_healthy_count_total")
                                         : NAME_TO_METRIC_ID("mooncake_unhealthy_count_total"),
                                 1.0);
    }
    RecordEffectiveHealth();
}

void HealthBreakerStore::RecordEffectiveHealth()
{
    if (storeId_.find(":PosixStore") != std::string::npos) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_store_health"), Enabled() ? 1.0 : 0.0);
    } else if (storeId_.find(":MooncakeStore") != std::string::npos) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_store_health"), Enabled() ? 1.0 : 0.0);
    }
}

void HealthBreakerStore::ProbeLoop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_health_mon");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM health monitor thread name.", nameStatus);
    }
    std::unique_lock<std::mutex> lock(stopMutex_);
    auto delay =
        config_.healthCheckInterval + RandomProbeDelay(config_.healthCheckInterval);
    while (!stopCv_.wait_for(lock, delay, [this] { return stop_; })) {
        lock.unlock();
        const auto start = std::chrono::steady_clock::now();
        CheckHealth();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        lock.lock();
        delay = elapsed < config_.healthCheckInterval ? config_.healthCheckInterval - elapsed
                                                      : std::chrono::milliseconds{0};
    }
}

}  // namespace UC::PipelineStore
