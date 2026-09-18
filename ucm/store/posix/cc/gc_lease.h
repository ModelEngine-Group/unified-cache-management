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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_GC_LEASE_H
#define UNIFIEDCACHE_POSIX_STORE_CC_GC_LEASE_H

#include <atomic>
#include <condition_variable>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include "backend_manager.h"
#include "global_config.h"
#include "status/status.h"

namespace UC::PosixStore {

class GcLease {
public:
    enum class Acquisition {
        Acquired,
        HeldByPeer,
        Unavailable,
    };

    GcLease() = default;
    GcLease(const GcLease&) = delete;
    GcLease& operator=(const GcLease&) = delete;
    ~GcLease();

    void Setup(const Config& config, const BackendManager* backendMgr);

    Acquisition TryAcquire();
    void Release();
    bool HoldsLock() const;
    void RequestStop();

private:
    struct Paths {
        std::string backend;
        std::string lockDir;
        std::string checkTime;
        std::string heartbeat;
    };
    Expected<Paths> SelectPaths() const;
    Status Claim(const Paths& paths);
    bool EntryPresent(const Paths& paths) const;
    Status ProbeHolder(const Paths& paths, bool& stale);
    Status TakeOverStale(const Paths& paths);
    void SweepParked(const Paths& paths) const;
    void HeartbeatLoop();
    void StopHeartbeat();
    Status Touch(const std::string& path, time_t& stamp, bool create) const;

    const BackendManager* backendMgr_{nullptr};
    std::string identity_;
    size_t heartbeatIntervalSec_{5};
    size_t staleThresholdSec_{180};

    std::string suspectHeartbeat_;
    time_t suspectMtime_{0};
    bool haveSuspect_{false};

    std::atomic<bool> held_{false};
    std::thread heartbeatWorker_;
    std::mutex stopMtx_;
    std::condition_variable stopCv_;
    bool stopHeartbeat_{false};
};

}  // namespace UC::PosixStore

#endif
