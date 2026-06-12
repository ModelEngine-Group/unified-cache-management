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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_BACKEND_HEALTH_H
#define UNIFIEDCACHE_POSIX_STORE_CC_BACKEND_HEALTH_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <dirent.h>
#include <errno.h>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>
#include <fmt/format.h>
#include "global_config.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "space_layout.h"
#include "status/status.h"
#include "time/now_time.h"

namespace UC::PosixStore {

#ifdef UCM_ENABLE_TEST_HOOKS
namespace TestHooks {
using BackendProbeHook = std::function<Status(const std::vector<std::string>&)>;
inline std::mutex& BackendProbeHookMutex()
{
    static std::mutex mutex;
    return mutex;
}
inline BackendProbeHook& BackendProbeHookSlot()
{
    static BackendProbeHook hook;
    return hook;
}
inline void SetBackendProbeHook(BackendProbeHook hook)
{
    std::lock_guard<std::mutex> lock{BackendProbeHookMutex()};
    BackendProbeHookSlot() = std::move(hook);
}
inline void ClearBackendProbeHook()
{
    std::lock_guard<std::mutex> lock{BackendProbeHookMutex()};
    BackendProbeHookSlot() = nullptr;
}
inline BackendProbeHook GetBackendProbeHook()
{
    std::lock_guard<std::mutex> lock{BackendProbeHookMutex()};
    return BackendProbeHookSlot();
}
}  // namespace TestHooks
#endif

enum class BackendOperation { Lookup, Load, Dump };

class BackendHealth {
    static constexpr size_t kProbeIntervalMs = 1000;
    static constexpr size_t kMaxProbeWorkers = 2;
    static constexpr double kShortCircuitLogIntervalSec = 1.0;

    struct ProbeState {
        std::mutex mutex;
        std::condition_variable cv;
        bool done{false};
        Status status{Status::OK()};
    };

    struct ProbeControl {
        std::atomic_bool stop{false};
        std::atomic_size_t active{0};
    };

    struct SharedState {
        std::atomic_bool healthy{true};
        std::mutex mutex;
        std::condition_variable cv;
    };

public:
    BackendHealth() = default;
    BackendHealth(const BackendHealth&) = delete;
    BackendHealth& operator=(const BackendHealth&) = delete;
    ~BackendHealth() { Stop(); }

    Status Setup(const Config& config, const SpaceLayout* layout)
    {
        enabled_ = config.ioEngine == "aio" && config.timeoutMs > 0;
        timeoutMs_ = config.timeoutMs;
        if (!enabled_) { return Status::OK(); }
        storageBackends_ = layout->StorageBackends();
        state_ = GetSharedState(storageBackends_);
        probeControl_ = std::make_shared<ProbeControl>();
        try {
            monitor_ = std::thread([this] { MonitorLoop(); });
        } catch (const std::exception& e) {
            return Status::Error(e.what());
        }
        UpdateUnhealthyGauge(!state_->healthy.load(std::memory_order_acquire));
        return Status::OK();
    }

    bool IsHealthy() const noexcept
    {
        return !enabled_ || !state_ || state_->healthy.load(std::memory_order_acquire);
    }

    void MarkUnhealthy(const std::string& reason)
    {
        if (!enabled_ || !state_) { return; }
        bool expected = true;
        if (!state_->healthy.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
            return;
        }
        IncrementUnhealthyMetric();
        UpdateUnhealthyGauge(true);
        UC_ERROR("Posix AIO backend marked unhealthy: {}.", reason);
        state_->cv.notify_all();
    }

    void RecordShortCircuit(BackendOperation op)
    {
        IncrementShortCircuitMetric(op);
        auto now = NowTime::Now();
        std::lock_guard<std::mutex> lock{logMutex_};
        if (now < nextShortCircuitLogTp_) { return; }
        nextShortCircuitLogTp_ = now + kShortCircuitLogIntervalSec;
        UC_WARN("Posix AIO backend unhealthy; short-circuiting {}.", OperationName(op));
    }

private:
    void Stop()
    {
        if (probeControl_) { probeControl_->stop.store(true, std::memory_order_release); }
        stop_.store(true, std::memory_order_release);
        if (state_) { state_->cv.notify_all(); }
        if (monitor_.joinable()) { monitor_.join(); }
    }

    void MonitorLoop()
    {
        while (true) {
            {
                if (!state_) { return; }
                std::unique_lock<std::mutex> lock{state_->mutex};
                state_->cv.wait(lock, [this] {
                    return stop_.load(std::memory_order_acquire) || !IsHealthy();
                });
                if (stop_.load(std::memory_order_acquire)) { return; }
            }
            ProbeUntilRecovered();
        }
    }

    void ProbeUntilRecovered()
    {
        while (!ShouldStop() && !IsHealthy()) {
            auto control = probeControl_;
            if (!control || control->active.load(std::memory_order_acquire) >= kMaxProbeWorkers) {
                WaitProbeInterval();
                continue;
            }
            auto state = std::make_shared<ProbeState>();
            control->active.fetch_add(1, std::memory_order_acq_rel);
            std::thread worker;
            try {
                worker = std::thread([state, control, paths = storageBackends_] {
                    auto status = Status::OK();
                    try {
                        status = Probe(paths);
                    } catch (const std::exception& e) {
                        status = Status::Error(e.what());
                    }
                    {
                        std::lock_guard<std::mutex> lock{state->mutex};
                        state->status = status;
                        state->done = true;
                    }
                    state->cv.notify_all();
                    control->active.fetch_sub(1, std::memory_order_acq_rel);
                });
            } catch (const std::exception& e) {
                control->active.fetch_sub(1, std::memory_order_acq_rel);
                UC_ERROR("Failed({}) to start Posix AIO backend probe.", e.what());
                WaitProbeInterval();
                continue;
            }
            if (WaitProbe(state)) {
                if (worker.joinable()) { worker.join(); }
                if (state->status.Success()) {
                    MarkRecovered();
                    return;
                }
            } else {
                if (!ShouldStop()) { IncrementProbeTimeoutMetric(); }
                if (worker.joinable()) { worker.detach(); }
            }
            WaitProbeInterval();
        }
    }

    bool WaitProbe(const std::shared_ptr<ProbeState>& state)
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs_);
        std::unique_lock<std::mutex> lock{state->mutex};
        while (!state->done && !ShouldStop()) {
            auto now = std::chrono::steady_clock::now();
            if (now >= deadline) { break; }
            auto remain = deadline - now;
            auto slice = remain > std::chrono::milliseconds(50)
                             ? std::chrono::milliseconds(50)
                             : std::chrono::duration_cast<std::chrono::milliseconds>(remain);
            if (slice.count() <= 0) { break; }
            state->cv.wait_for(lock, slice);
        }
        return state->done;
    }

    void WaitProbeInterval()
    {
        if (!state_) { return; }
        std::unique_lock<std::mutex> lock{state_->mutex};
        state_->cv.wait_for(lock, std::chrono::milliseconds(kProbeIntervalMs),
                            [this] { return stop_.load(std::memory_order_acquire) || IsHealthy(); });
    }

    bool ShouldStop() const
    {
        return stop_.load(std::memory_order_acquire) ||
               (probeControl_ && probeControl_->stop.load(std::memory_order_acquire));
    }

    void MarkRecovered()
    {
        if (!state_) { return; }
        bool expected = false;
        if (!state_->healthy.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }
        IncrementRecoveredMetric();
        UpdateUnhealthyGauge(false);
        UC_INFO("Posix AIO backend recovered.");
        state_->cv.notify_all();
    }

    static std::shared_ptr<SharedState> GetSharedState(const std::vector<std::string>& paths)
    {
        static std::mutex mutex;
        static std::unordered_map<std::string, std::weak_ptr<SharedState>> states;
        auto key = BackendKey(paths);
        std::lock_guard<std::mutex> lock{mutex};
        if (auto state = states[key].lock()) { return state; }
        auto state = std::make_shared<SharedState>();
        states[key] = state;
        return state;
    }

    static std::string BackendKey(const std::vector<std::string>& paths)
    {
        std::string key;
        for (const auto& path : paths) {
            key += path;
            key += '\n';
        }
        return key;
    }

    static Status Probe(const std::vector<std::string>& paths)
    {
#ifdef UCM_ENABLE_TEST_HOOKS
        auto hook = TestHooks::GetBackendProbeHook();
        if (hook) { return hook(paths); }
#endif
        for (const auto& path : paths) {
            if (::access(path.c_str(), R_OK | W_OK) != 0) {
                return Status::OsApiError(fmt::format("failed to access {}", path));
            }
            std::string probeFile;
            if (FindProbeFile(path, probeFile)) {
                auto status = ProbeFile(probeFile);
                if (status.Failure()) { return status; }
            }
        }
        return Status::OK();
    }

    static bool FindProbeFile(const std::string& path, std::string& probeFile)
    {
        std::vector<std::string> dirs{path};
        for (size_t depth = 0; depth < 2 && !dirs.empty(); ++depth) {
            std::vector<std::string> nextDirs;
            for (const auto& dir : dirs) {
                if (FindProbeFileInDir(dir, probeFile, nextDirs)) { return true; }
            }
            dirs = std::move(nextDirs);
        }
        return false;
    }

    static bool FindProbeFileInDir(const std::string& dir, std::string& probeFile, std::vector<std::string>& nextDirs)
    {
        DIR* handle = ::opendir(dir.c_str());
        if (!handle) { return false; }
        while (auto* entry = ::readdir(handle)) {
            std::string name{entry->d_name};
            if (name == "." || name == "..") { continue; }
            auto child = JoinPath(dir, name);
            struct stat st {};
            if (::lstat(child.c_str(), &st) != 0) { continue; }
            if (S_ISREG(st.st_mode) && st.st_size > 0) {
                probeFile = std::move(child);
                ::closedir(handle);
                return true;
            }
            if (S_ISDIR(st.st_mode)) { nextDirs.push_back(std::move(child)); }
        }
        ::closedir(handle);
        return false;
    }

    static Status ProbeFile(const std::string& file)
    {
        int flags = O_RDONLY;
#ifdef O_DIRECT
        flags |= O_DIRECT;
#endif
        int fd = ::open(file.c_str(), flags);
#ifdef O_DIRECT
        if (fd < 0 && errno == EINVAL) { fd = ::open(file.c_str(), O_RDONLY); }
#endif
        if (fd < 0) { return Status::OsApiError(fmt::format("failed to open probe file {}", file)); }

        void* buffer = nullptr;
        if (::posix_memalign(&buffer, 4096, 4096) != 0) {
            ::close(fd);
            return Status::Error("failed to allocate probe buffer");
        }
        auto nread = ::pread(fd, buffer, 4096, 0);
        auto eno = errno;
        std::free(buffer);
        ::close(fd);
        if (nread < 0) {
            errno = eno;
            return Status::OsApiError(fmt::format("failed to read probe file {}", file));
        }
        return Status::OK();
    }

    static std::string JoinPath(const std::string& dir, const std::string& name)
    {
        if (!dir.empty() && dir.back() == '/') { return dir + name; }
        return dir + "/" + name;
    }

    static const char* OperationName(BackendOperation op)
    {
        switch (op) {
            case BackendOperation::Lookup: return "lookup";
            case BackendOperation::Load: return "load";
            case BackendOperation::Dump: return "dump";
            default: return "unknown";
        }
    }

    static void IncrementUnhealthyMetric()
    {
        static UC::Metrics::CachedMetric metric{"posix_backend_unhealthy_transitions_total"};
        UC::Metrics::UpdateStats(metric, 1.0);
    }

    static void IncrementRecoveredMetric()
    {
        static UC::Metrics::CachedMetric metric{"posix_backend_recovered_total"};
        UC::Metrics::UpdateStats(metric, 1.0);
    }

    static void IncrementProbeTimeoutMetric()
    {
        static UC::Metrics::CachedMetric metric{"posix_backend_probe_timeout_total"};
        UC::Metrics::UpdateStats(metric, 1.0);
    }

    static void IncrementShortCircuitMetric(BackendOperation op)
    {
        static UC::Metrics::CachedMetric lookup{"posix_backend_short_circuit_lookup_total"};
        static UC::Metrics::CachedMetric load{"posix_backend_short_circuit_load_total"};
        static UC::Metrics::CachedMetric dump{"posix_backend_short_circuit_dump_total"};
        auto& metric = op == BackendOperation::Lookup   ? lookup
                       : op == BackendOperation::Load   ? load
                                                        : dump;
        UC::Metrics::UpdateStats(metric, 1.0);
    }

    static void UpdateUnhealthyGauge(bool unhealthy)
    {
        static UC::Metrics::CachedMetric metric{"posix_backend_unhealthy"};
        UC::Metrics::UpdateStats(metric, unhealthy ? 1.0 : 0.0);
    }

private:
    bool enabled_{false};
    size_t timeoutMs_{0};
    std::atomic_bool stop_{false};
    std::vector<std::string> storageBackends_{};
    std::shared_ptr<SharedState> state_;
    std::shared_ptr<ProbeControl> probeControl_;
    std::thread monitor_;
    std::mutex logMutex_;
    double nextShortCircuitLogTp_{0};
};

}  // namespace UC::PosixStore

#endif
