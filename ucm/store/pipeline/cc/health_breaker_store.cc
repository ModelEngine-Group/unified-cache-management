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

HealthBreakerStore::~HealthBreakerStore() { Stop(); }

Status HealthBreakerStore::Start()
{
    if (!store_ || !healthCheck_) {
        return Status::InvalidParam("health breaker store is not set up");
    }
    if (!config_.enabled) { return Status::InvalidParam("health breaker store is disabled"); }
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
    return healthState_ ? healthState_->FailureCount() : 0;
}

size_t HealthBreakerStore::SampleCount() const
{
    std::lock_guard<std::mutex> lock(healthMutex_);
    return healthState_ ? healthState_->SampleCount() : 0;
}

Status HealthBreakerStore::Setup(const Detail::Dictionary&)
{
    return Status::InvalidParam("health breaker store requires typed setup");
}

Status HealthBreakerStore::Setup(StoreV1* store, std::string storeId,
                                 const StoreHealthConfig& config)
{
    if (healthCheck_ || probeThread_.joinable()) {
        return Status::InvalidParam("health breaker store is already set up");
    }
    if (!store) { return Status::InvalidParam("health breaker store target is null"); }
    if (storeId.empty()) { return Status::InvalidParam("health breaker store id is empty"); }
    auto status = config.Validate();
    if (status.Failure()) { return status; }
    store_ = store;
    storeId_ = std::move(storeId);
    config_ = config;
    healthState_ = std::make_unique<StoreHealthState>(config_);
    healthCheck_ = std::make_unique<Detail::HealthCheckExecutor>(config_.healthCheckTimeout);
    return Status::OK();
}

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

Expected<ssize_t> HealthBreakerStore::LookupOnReverse(const Detail::BlockId* blocks, size_t num)
{
    if (!Enabled()) { return static_cast<ssize_t>(-1); }
    return store_->LookupOnReverse(blocks, num);
}

void HealthBreakerStore::Prefetch(const Detail::BlockId* blocks, size_t num)
{
    if (Enabled()) { store_->Prefetch(blocks, num); }
}

Status HealthBreakerStore::CheckHealth()
{
    if (!healthCheck_) { return Status::InvalidParam("health breaker store is not set up"); }
    const auto generation = healthState_->Generation();
    const auto started = StoreHealthState::Clock::now();
    auto status = healthCheck_->Run([this] { return store_->CheckHealth(); });
    if (status == Status::Timeout()) {
        UC_WARN("Store health check({}) timed out after {} ms.", storeId_,
                config_.healthCheckTimeout.count());
    } else if (status.Failure()) {
        UC_WARN("Store health check({}) failed({}).", storeId_, status);
    }
    std::lock_guard<std::mutex> lock(healthMutex_);
    const auto now = StoreHealthState::Clock::now();
    UpdateState(healthState_->RecordProbe(status.Success(), generation, started, now),
                "active_probe");
    if (generation == healthState_->Generation() && healthState_->FailureCount() == 0 &&
        healthState_->SampleCount() == config_.healthWindowSize) {
        const auto remaining = healthState_->CooldownRemaining(now);
        if (remaining.count() > 0) {
            UC_INFO_UNLIMITED(
                "Store health breaker({}) has a healthy probe window; waiting for "
                "cooldown, remaining_ms={}.",
                storeId_, remaining.count());
        }
    }
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

Status HealthBreakerStore::Wait(Detail::TaskHandle taskId)
{
    const auto generation = healthState_->Generation();
    auto status = store_->Wait(taskId);
    if (!config_.passiveEnabled || status == Status::NotFound() ||
        status == Status::StoreUnhealthy() || status == Status::InvalidParam() ||
        status == Status::DuplicateKey() || status == Status::Unsupported()) {
        return status;
    }
    if (status.Failure()) {
        if (storeId_.find(":PosixStore") != std::string::npos) {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_passive_failures_total"), 1.0);
        } else if (storeId_.find(":MooncakeStore") != std::string::npos) {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_passive_failures_total"), 1.0);
        }
    }
    healthState_->RecordIo(status.Success(), generation);
    if (healthState_->PassiveThresholdExceeded(generation)) {
        // only lock and set state when need to switch to unhealthy
        std::lock_guard<std::mutex> lock(healthMutex_);
        const auto changed =
            healthState_->UpdatePassiveHealth(generation, StoreHealthState::Clock::now());
        UpdateState(changed, "passive_io");
    }
    if (status.Failure()) {
        return Status::StoreUnhealthy(fmt::format("{}: {}", storeId_, status));
    }
    return status;
}

void HealthBreakerStore::UpdateState(bool changed, const char* source)
{
    if (!changed) { return; }
    RecordEffectiveHealth();
    UC_WARN_UNLIMITED(
        "Store health breaker({}) transitioned to {}, source={}, cooldown_ms={}, generation={}.",
        storeId_, healthState_->Enabled() ? "HEALTHY" : "UNHEALTHY", source,
        healthState_->Cooldown().count(), healthState_->Generation());
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
    auto delay = config_.healthCheckInterval + RandomProbeDelay(config_.healthCheckInterval);
    while (!stopCv_.wait_for(lock, delay, [this] { return stop_; })) {
        lock.unlock();
        const auto start = std::chrono::steady_clock::now();
        if (config_.passiveEnabled) {
            StoreHealthState::PassiveWindowStats stats;
            {
                std::lock_guard<std::mutex> healthLock(healthMutex_);
                healthState_->SamplePassiveWindow(start);
                stats = healthState_->GetPassiveWindowStats();
            }
            if (stats.failures > 0) {
                UC_INFO_UNLIMITED(
                    "Store passive health window({}): window_s={}, samples={}, failures={}, "
                    "failure_threshold={}.",
                    storeId_, config_.passiveWindow.count(), stats.total, stats.failures,
                    config_.passiveFailureThreshold);
            }
        }
        CheckHealth();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        lock.lock();
        delay = elapsed < config_.healthCheckInterval ? config_.healthCheckInterval - elapsed
                                                      : std::chrono::milliseconds{0};
    }
}

}  // namespace UC::PipelineStore
