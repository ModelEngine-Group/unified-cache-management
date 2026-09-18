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
#include "detail/health_check_executor.h"
#include "logger/logger.h"
#include "posix_file.h"
#include "thread/cpu_affinity.h"
#include "type/random_block_id.h"

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
    auto status = config.backendHealth.Validate();
    if (status.Failure()) { return status; }
    if (config.storageBackends.empty()) { return Status::InvalidParam("empty storage backends"); }
    layout_ = layout;
    ioDirect_ = config.ioDirect;
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
        status = layout_->InitBackend(normalizedPath, index == 0);
        if (status.Failure()) { return startupError(status); }
        status = CheckPathHealth(
            layout_->DataFilePath(normalizedPath, Detail::RandomBlockId(), true), ioDirect_);
        if (status.Failure()) { return startupError(status); }
        backendHealth_.emplace_back(config.backendHealth);
    }
    for (size_t i = 0; i < storageBackends_.size(); ++i) { availableBackends_.push_back(i); }
    ioTimeoutMs_ =
        config.timeoutMs == 0 ? 0 : std::max<size_t>(1, config.timeoutMs / BackendCount());
    try {
        probeThread_ = std::thread(&BackendManager::ProbeBackends, this, config.backendHealth);
    } catch (const std::exception& e) {
        availableBackends_.clear();
        return Status::Error(fmt::format("failed to start backend recovery probes: {}", e.what()));
    }
    return Status::OK();
}

Expected<std::string> BackendManager::StorageBackend(const Detail::BlockId& blockId,
                                                     const std::vector<std::string>& excluded) const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    if (availableBackends_.empty()) { return Status::StoreUnhealthy(); }
    const auto count = storageBackends_.size();
    const auto primary = Detail::BlockIdHasher{}(blockId) % count;
    for (size_t offset = 0; offset < count; ++offset) {
        const auto index = (primary + offset) % count;
        if (!backendHealth_[index].Healthy()) { continue; }
        const auto& backend = storageBackends_[index];
        if (std::find(excluded.begin(), excluded.end(), backend) == excluded.end()) {
            return std::string(backend);
        }
    }
    return Status::StoreUnhealthy();
}

Expected<size_t> BackendManager::SelectNextBackend(const Detail::BlockId& blockId,
                                                   std::vector<std::string>& attempted) const
{
    auto backend = StorageBackend(blockId, attempted);
    if (!backend) { return backend.Error(); }
    size_t backendIndex =
        std::find(storageBackends_.begin(), storageBackends_.end(), backend.Value()) -
        storageBackends_.begin();
    attempted.push_back(std::move(backend.Value()));
    return backendIndex;
}

Status BackendManager::RunOnAvailableBackend(const Detail::BlockId& blockId,
                                             const std::function<Status(size_t)>& operation) const
{
    std::vector<std::string> attempted;
    auto status = Status::StoreUnhealthy();
    while (auto backendIndex = SelectNextBackend(blockId, attempted)) {
        status = operation(backendIndex.Value());
        if (!ShouldRetry(backendIndex.Value(), status)) { break; }
    }
    return status;
}

void BackendManager::RecordIoResult(const std::string& backend, const Status& status) const
{
    if (status == Status::NotFound()) { return; }
    const auto found = std::find(storageBackends_.begin(), storageBackends_.end(), backend);
    if (found != storageBackends_.end()) {
        RecordHealth(std::distance(storageBackends_.begin(), found), status, false);
    }
}

void BackendManager::RecordHealth(size_t index, const Status& status, bool recovery) const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    auto& health = backendHealth_[index];
    const auto wasAvailable = health.Healthy();
    if (recovery && wasAvailable) { return; }
    // Late successful business I/O cannot restore an excluded backend.
    if (!recovery && !wasAvailable && status.Success()) { return; }
    health.Record(status.Success());
    if (wasAvailable == health.Healthy()) { return; }
    availableBackends_.clear();
    for (size_t i = 0; i < backendHealth_.size(); ++i) {
        if (backendHealth_[i].Healthy()) { availableBackends_.push_back(i); }
    }
    UC_WARN("Storage backend({}) is {}, samples={}, failures={}, status={}.",
            storageBackends_[index], health.Healthy() ? "HEALTHY" : "UNHEALTHY",
            health.SampleCount(), health.FailureCount(), status);
}

Status BackendManager::CheckHealth() const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    return availableBackends_.empty() ? Status::StoreUnhealthy() : Status::OK();
}

void BackendManager::ProbeBackends(const Detail::StoreHealthConfig& config)
{
    auto status = CpuAffinity::SetCurrentThreadName("ucm_health_pmon");
    if (status.Failure()) { UC_WARN("Failed to name backend health monitor: {}.", status); }
    Detail::HealthCheckExecutor executor(config.healthCheckTimeout, 0);
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
            {
                std::lock_guard<std::mutex> lock(backendMutex_);
                if (backendHealth_[index].Healthy()) { continue; }
            }
            // A timed-out probe may still run, so each probe owns a distinct file name.
            const auto path =
                layout_->DataFilePath(storageBackends_[index], Detail::RandomBlockId(), true);
            status = executor.Run(
                [path, ioDirect = ioDirect_] { return CheckPathHealth(path, ioDirect); });
            RecordHealth(index, status, true);
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        delay = elapsed < config.healthCheckInterval ? config.healthCheckInterval - elapsed
                                                     : std::chrono::milliseconds{0};
        stopLock.lock();
    }
}

bool BackendManager::ShouldRetry(size_t backendIndex, const Status& status) const
{
    RecordIoResult(storageBackends_[backendIndex], status);
    return status.Failure() && status != Status::NotFound();
}

}  // namespace UC::PosixStore
