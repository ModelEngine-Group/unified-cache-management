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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_BACKEND_MANAGER_H
#define UNIFIEDCACHE_POSIX_STORE_CC_BACKEND_MANAGER_H

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include "detail/health_tracker.h"
#include "gc_executor.h"
#include "space_layout.h"
#include "type/random_block_id.h"

namespace UC::PosixStore {

class BackendManager {
public:
    struct BackendAttempt {
        size_t index{0};
        uint64_t generation{0};
    };

    ~BackendManager();
    Status Setup(const Config& config, const SpaceLayout* layout);
    Expected<std::string> GetBackendForBlock(const Detail::BlockId& blockId,
                                             const std::vector<std::string>& excluded = {}) const;
    // The block's primary backend, or StoreUnhealthy when that backend is unhealthy.
    Expected<BackendAttempt> PrimaryBackend(const Detail::BlockId& blockId) const;
    Expected<BackendAttempt> SelectNextBackend(const Detail::BlockId& blockId,
                                               std::vector<std::string>& attempted) const;
    size_t BackendCount() const { return storageBackends_.size(); }
    const std::vector<std::string>& Backends() const { return storageBackends_; }
    size_t IoTimeoutMs() const { return ioTimeoutMs_; }
    Status RunGcOp(std::function<Status()> op) const;
    void RecordResult(const BackendAttempt& attempt, const Status& status) const;
    bool ShouldRetry(const BackendAttempt& attempt, const Status& status) const;
    Status CheckHealth() const;

private:
    void ProbeBackends(const Detail::StoreHealthConfig& config);
    void UpdateStoreHealthMetric() const;
    const SpaceLayout* layout_{nullptr};
    bool ioDirect_{true};
    bool passiveEnabled_{true};
    Detail::BlockId healthBlockId_{Detail::RandomBlockId()};
    std::vector<std::string> storageBackends_;
    mutable std::deque<Detail::HealthTracker> backendHealth_;
    mutable GcExecutor gcExecutor_;
    std::mutex stopMutex_;
    std::condition_variable stopCv_;
    bool stop_{false};
    std::thread probeThread_;
    size_t ioTimeoutMs_{0};
};

}  // namespace UC::PosixStore

#endif
