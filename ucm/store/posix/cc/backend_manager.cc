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
#include "backend_manager.h"
#include <algorithm>
#include <array>
#include <future>
#include "detail/health_check_executor.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "posix_file.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

static Status CheckPathHealth(const std::string& path, bool ioDirect)
{
    constexpr size_t kHealthIoSize = 4096;
    alignas(kHealthIoSize) std::array<uint8_t, kHealthIoSize> expected{};
    alignas(kHealthIoSize) std::array<uint8_t, kHealthIoSize> actual{};
    expected.fill(0x5a);

    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::READ_WRITE;
    if (ioDirect) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto status = file.Open(flags);
    if (status.Failure()) { return status; }
    status = file.Write(expected.data(), expected.size(), 0);
    if (status.Success() && !ioDirect) { status = file.Sync(); }
    if (status.Success()) { status = file.Read(actual.data(), actual.size(), 0); }
    if (status.Success() && actual != expected) {
        status = Status::Error(fmt::format("verify('{}') failed: health data mismatch", path));
    }
    auto closed = file.Close();
    if (status.Success()) { status = closed; }
    auto cleanup = file.Remove();
    return status.Failure() ? status : cleanup;
}

static Status RunStartupCheck(std::function<Status()> check, size_t timeoutMs)
{
    auto result = std::make_shared<std::promise<Status>>();
    auto future = result->get_future();
    try {
        std::thread([check = std::move(check), result] { result->set_value(check()); }).detach();
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed to start backend startup check: {}", e.what()));
    }
    if (future.wait_for(std::chrono::milliseconds(timeoutMs)) == std::future_status::ready) {
        return future.get();
    } else {
        return Status{Status::Timeout().Underlying(),
                      fmt::format("no response in {} ms", timeoutMs)};
    }
}

BackendManager::~BackendManager()
{
    {
        std::lock_guard<std::mutex> lock(stopMutex_);
        stop_ = true;
    }
    stopCv_.notify_all();
    if (probeThread_.joinable()) { probeThread_.join(); }
}

Status BackendManager::Setup(const Config& config, const SpaceLayout* layout)
{
    if (!storageBackends_.empty()) {
        return Status::InvalidParam("backend manager already set up");
    }
    if (config.timeoutMs == 0) { return Status::InvalidParam("timeout_ms must be greater than 0"); }
    auto status = config.backendHealth.Validate();
    if (status.Failure()) { return status; }
    if (config.storageBackends.empty()) { return Status::InvalidParam("empty storage backends"); }
    layout_ = layout;
    ioDirect_ = config.ioDirect;
    passiveEnabled_ = config.backendHealth.enabled && config.backendHealth.passiveEnabled;
    for (const auto& path : config.storageBackends) {
        if (path.empty()) { return Status::InvalidParam("empty storage backend path"); }
        auto normalizedPath = path.back() == '/' ? path : path + '/';
        if (std::find(storageBackends_.begin(), storageBackends_.end(), normalizedPath) !=
            storageBackends_.end()) {
            continue;
        }
        const auto index = storageBackends_.size();
        storageBackends_.push_back(normalizedPath);
        const auto startupError = [&](const Status& error) {
            auto message = fmt::format("Storage backend '{}' failed startup I/O check: {}",
                                       normalizedPath, error);
            UC_ERROR("{}", message);
            return Status{error.Underlying(), std::move(message)};
        };
        status = RunStartupCheck(
            [layout = *layout_, backend = normalizedPath, create = index == 0,
             probe = layout_->ArchivedFilePath(normalizedPath, healthBlockId_) + ".health_probe",
             ioDirect = ioDirect_] {
                auto s = layout.InitBackend(backend, create);
                return s.Success() ? CheckPathHealth(probe, ioDirect) : s;
            },
            config.timeoutMs);
        if (status.Failure()) { return startupError(status); }
        backendHealth_.emplace_back(
            "PosixBackend", fmt::format("device({}) {}:{}", config.deviceId, index, normalizedPath),
            config.backendHealth);
    }
    // The thread pool checks for stuck workers once a second, so it may take up to 1s more to
    // notice a timed-out worker and finish its task. Leave that 1s out of each backend's share,
    // so a retry on the next backend still fits in timeout_ms. A share under 2s keeps at least
    // half of it. One backend has no retry, so it keeps timeout_ms as before.
    if (BackendCount() == 1) {
        ioTimeoutMs_ = config.timeoutMs;
    } else {
        const auto perBackendMs = config.timeoutMs / BackendCount();
        ioTimeoutMs_ = std::max<size_t>(1, perBackendMs - std::min<size_t>(1000, perBackendMs / 2));
    }
    if (config.posixGcEnable) {
        status = gcExecutor_.Setup(BackendCount(), ioTimeoutMs_);
        if (status.Failure()) { return status; }
    }
    try {
        probeThread_ = std::thread(&BackendManager::ProbeBackends, this, config.backendHealth);
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed to start backend recovery probes: {}", e.what()));
    }
    UpdateStoreHealthMetric();
    return Status::OK();
}

Expected<std::string> BackendManager::GetBackendForBlock(
    const Detail::BlockId& blockId, const std::vector<std::string>& excluded) const
{
    const auto count = storageBackends_.size();
    if (count == 0) { return Status::StoreUnhealthy(); }
    const auto primaryBackendIndex = Detail::BlockIdHasher{}(blockId) % count;
    // primary backend may not available, so traverse list to find next usable backend
    for (size_t offset = 0; offset < count; ++offset) {
        const auto index = (primaryBackendIndex + offset) % count;
        if (!backendHealth_[index].Healthy()) { continue; }
        const auto& backend = storageBackends_[index];
        if (std::find(excluded.begin(), excluded.end(), backend) == excluded.end()) {
            return std::string(backend);
        }
    }
    return Status::StoreUnhealthy();
}

Expected<BackendManager::BackendAttempt> BackendManager::PrimaryBackend(
    const Detail::BlockId& blockId) const
{
    const auto index = Detail::BlockIdHasher{}(blockId) % storageBackends_.size();
    if (backendHealth_[index].Healthy()) {
        return BackendAttempt{index, backendHealth_[index].Generation()};
    } else {
        return Status::StoreUnhealthy();
    }
}

Expected<BackendManager::BackendAttempt> BackendManager::SelectNextBackend(
    const Detail::BlockId& blockId, std::vector<std::string>& attempted) const
{
    auto backend = GetBackendForBlock(blockId, attempted);
    if (!backend) { return backend.Error(); }
    const size_t index =
        std::find(storageBackends_.begin(), storageBackends_.end(), backend.Value()) -
        storageBackends_.begin();
    attempted.push_back(std::move(backend.Value()));
    return BackendAttempt{index, backendHealth_[index].Generation()};
}

Status BackendManager::CheckHealth() const
{
    for (const auto& health : backendHealth_) {
        if (health.Healthy()) { return Status::OK(); }
    }
    return Status::StoreUnhealthy();
}

void BackendManager::ProbeBackends(const Detail::StoreHealthConfig& config)
{
    auto status = CpuAffinity::SetCurrentThreadName("ucm_health_be");
    if (status.Failure()) { UC_WARN("Failed to name backend health monitor: {}.", status); }
    Detail::HealthCheckExecutor executor(std::chrono::milliseconds{ioTimeoutMs_}, 8);
    std::unique_lock<std::mutex> stopLock(stopMutex_);
    auto delay = config.healthCheckInterval;
    while (!stopCv_.wait_for(stopLock, delay, [this] { return stop_; })) {
        stopLock.unlock();
        const auto start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < storageBackends_.size(); ++index) {
            {
                std::lock_guard<std::mutex> lock(stopMutex_);
                if (stop_) { return; }
            }
            auto& health = backendHealth_[index];
            if (passiveEnabled_) { health.SamplePassiveWindow(); }
            // only active probe when it's unhealthy
            if (health.Healthy()) { continue; }
            const auto healthProbeFilePath =
                layout_->ArchivedFilePath(storageBackends_[index], healthBlockId_) +
                ".health_probe";
            auto check = [healthProbeFilePath, ioDirect = ioDirect_] {
                return CheckPathHealth(healthProbeFilePath, ioDirect);
            };
            const auto generation = health.Generation();
            status = executor.Run(std::move(check));
            // Timed-out I/O may still access this file through any backend path.
            // So we use a new file after timeout.
            if (status == Status::Timeout()) { healthBlockId_ = Detail::RandomBlockId(); }
            UC::Metrics::UpdateStats(status.Success()
                                         ? NAME_TO_METRIC_ID("posix_healthy_count_total")
                                         : NAME_TO_METRIC_ID("posix_unhealthy_count_total"),
                                     1.0);
            if (health.RecordProbe(generation, status)) { UpdateStoreHealthMetric(); }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        delay = elapsed < config.healthCheckInterval ? config.healthCheckInterval - elapsed
                                                     : std::chrono::milliseconds{0};
        stopLock.lock();
    }
}

Status BackendManager::RunGcOp(std::function<Status()> op) const
{
    return gcExecutor_.Run(std::move(op));
}

// The store is usable while any backend is healthy.
void BackendManager::UpdateStoreHealthMetric() const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_store_health"),
                             CheckHealth().Success() ? 1.0 : 0.0);
}

void BackendManager::RecordResult(const BackendAttempt& attempt, const Status& status) const
{
    if (passiveEnabled_) {
        if (backendHealth_[attempt.index].RecordIo(attempt.generation, status)) {
            UpdateStoreHealthMetric();
        }
        if (status.Failure() && Detail::ShouldRecordResult(status)) {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_passive_failures_total"), 1.0);
        }
    }
}

bool BackendManager::ShouldRetry(const BackendAttempt& attempt, const Status& status) const
{
    if (status == Status::NotFound()) { return false; }
    RecordResult(attempt, status);
    return status.Failure();
}

}  // namespace UC::PosixStore
