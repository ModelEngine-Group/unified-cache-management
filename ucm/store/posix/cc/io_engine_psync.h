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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_IO_ENGINE_PSYNC_H
#define UNIFIEDCACHE_POSIX_STORE_CC_IO_ENGINE_PSYNC_H

#include <atomic>
#include <thread>
#include "logger/logger.h"
#include "metrics_api.h"
#include "template/spsc_ring_queue.h"
#include "template/task_wrapper.h"
#include "thread/cpu_affinity.h"
#include "thread/lock.h"
#include "trans_queue.h"

namespace UC::PosixStore {

class IoEnginePsync : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    static constexpr size_t kDispatchQueueDepth = 8192;

    TransQueue queue_;
    size_t shardSize_;
    SpscRingQueue<TaskPair> waiting_;
    alignas(64) std::atomic_bool stop_{false};
    SpinLock pushLock_;
    std::thread dispatcher_;

public:
    Status Setup(const Config& config, const SpaceLayout* layout)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        auto s = queue_.Setup(config, &failureSet_, layout);
        if (s.Failure()) [[unlikely]] { return s; }
        waiting_.Setup(kDispatchQueueDepth);
        dispatcher_ = std::thread(&IoEnginePsync::DispatchStage, this);
        return Status::OK();
    }
    ~IoEnginePsync() { Close(); }
    void Close()
    {
        if (stop_.exchange(true)) { return; }
        if (dispatcher_.joinable()) { dispatcher_.join(); }
        TaskPair pair;
        SpinLockGuard guard(pushLock_);
        while (waiting_.TryPop(pair)) {
            if (pair.first) { failureSet_.Insert(pair.first->id); }
            if (pair.second) { pair.second->Done(); }
        }
    }

protected:
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        if (t->type != TransTask::Type::LOAD) {
            DispatchOne({t, w});
            return;
        }
        w->Up();
        bool pushed = false;
        {
            SpinLockGuard guard(pushLock_);
            while (!stop_.load(std::memory_order_acquire)) {
                if (waiting_.TryPush({t, w})) {
                    pushed = true;
                    break;
                }
                std::this_thread::yield();
            }
        }
        if (!pushed) [[unlikely]] {
            UC_ERROR("Posix load task({}) dispatch aborted, engine stopped.", t->id);
            failureSet_.Insert(t->id);
            w->Done();
        }
    }
    void Cancel(TaskPtr t) override { queue_.Cancel(t); }

private:
    void DispatchStage()
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_psync_disp");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set psync dispatcher name.", nameStatus);
        }
        waiting_.ConsumerLoop(stop_, &IoEnginePsync::DispatchOne, this);
    }
    void DispatchOne(TaskPair&& pair)
    {
        auto& t = pair.first;
        auto& w = pair.second;
        if (failureSet_.Contains(t->id)) {
            w->Done();
            return;
        }
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        const auto isDump = (t->type == TransTask::Type::DUMP);
        UC_DEBUG("Posix task({},{},{},{}) dispatching.", id, brief, num, size);
        w->SetEpilog([id, brief = std::move(brief), num, size, tp, isDump] {
            auto cost = NowTime::Now() - tp;
            auto costMs = cost * 1e3;
            auto bwGbps = cost > 0 ? static_cast<double>(size) / cost / 1e9 : 0.0;
            UC_DEBUG("Posix task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     costMs);
            static UC::Metrics::CachedMetric loadDuration{"posix_load_task_duration_ms"};
            static UC::Metrics::CachedMetric dumpDuration{"posix_dump_task_duration_ms"};
            static UC::Metrics::CachedMetric loadBandwidth{"posix_s2h_bandwidth_gbps"};
            static UC::Metrics::CachedMetric dumpBandwidth{"posix_h2s_bandwidth_gbps"};
            static UC::Metrics::CachedMetric loadBytes{"posix_s2h_bytes_total"};
            static UC::Metrics::CachedMetric dumpBytes{"posix_h2s_bytes_total"};
            UC::Metrics::UpdateStats(isDump ? dumpDuration : loadDuration, costMs);
            UC::Metrics::UpdateStats(isDump ? dumpBandwidth : loadBandwidth, bwGbps);
            UC::Metrics::UpdateStats(isDump ? dumpBytes : loadBytes, static_cast<double>(size));
        });
        queue_.Push(t, w);
    }
};

}  // namespace UC::PosixStore

#endif
