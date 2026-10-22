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
#include <ctime>
#include <functional>
#include <memory>
#include <string>
#include "backend_manager.h"
#include "gc_liveness.h"
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
        BackendManager::BackendAttempt attempt;
        std::string backend;
        std::string lockDir;
        std::string checkTime;
    };
    struct Holder {
        bool locked{false};
        std::string name;
        time_t mtime{0};
        size_t lag{0};
    };
    static Status ProbeLock(const std::string& lockDir, const std::string& checkTime,
                            const std::shared_ptr<Holder>& holder);
    Expected<Paths> SelectPaths() const;
    Status Run(const Paths& paths, std::function<Status()> op) const;
    Status Claim(const Paths& paths);
    bool IsStale(const Paths& paths, const Holder& holder);
    bool EntryPresent(const Paths& paths) const;
    void StopHeartbeat();

    const BackendManager* backendMgr_{nullptr};
    std::string identity_;
    // Heartbeat file name of the current claim. Each claim gets a new name, so a late release
    // from an earlier claim cannot remove the current lock.
    std::string heartbeatName_;
    size_t heartbeatIntervalSec_{5};
    size_t staleThresholdSec_{180};

    std::string suspectHeartbeat_;
    time_t suspectMtime_{0};
    bool haveSuspect_{false};

    std::atomic<bool> held_{false};
    GcHeartbeat heartbeat_;
};

}  // namespace UC::PosixStore

#endif
