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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_IO_ENGINE_AIO_H
#define UNIFIEDCACHE_POSIX_STORE_CC_IO_ENGINE_AIO_H

#include <algorithm>
#include <cerrno>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "aio_impl.h"
#include "block_operator.h"
#include "block_tracker.h"
#include "handle_cache.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::PosixStore {

class IoEngineAio : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    struct Inflight {
        TaskPtr t;
        WaiterPtr w;
        double deadlineTp{0};
        size_t shardCount{0};
        bool aborted{false};
    };
    size_t shardSize_;
    size_t nShardPerBlock_;
    BlockCommitTracker tracker_;
    std::deque<LoadHandleCache> loadHandles_;
    std::mutex regMutex_;
    std::unordered_map<Detail::TaskHandle, Inflight> registry_;
    const std::vector<std::string>* backends_{nullptr};
    BlockOperator blockOperator_;
    AioImpl aio_;

public:
    // aio_ is destroyed before blockOperator_, but open callbacks submit I/O to aio_ and
    // completion callbacks submit commits to blockOperator_; stop the open/commit workers first.
    ~IoEngineAio() { blockOperator_.Stop(); }
    Status Setup(const Config& config, const SpaceLayout* layout,
                 const std::vector<std::string>* backends)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        nShardPerBlock_ = config.blockSize / config.shardSize;
        backends_ = backends;
        tracker_.Setup(nShardPerBlock_);
        auto s = SetupLoadHandles(loadHandles_, backends_->size(), config.loadHandleCacheSize,
                                  config.ioDirect);
        if (s.Failure()) { return s; }
        // A stuck open is already past its task deadline. Sweep before the new worker starts,
        // so the stuck backend trips and queued tasks retry first.
        s = blockOperator_.Setup(
            layout, backends, config.openConcurrency, config.commitConcurrency, timeoutMs_,
            [this] { SweepDeadlines(); }, &loadHandles_);
        if (s.Failure()) { return s; }
        aio_.SetSweepFn([this] { SweepDeadlines(); });
        UC_INFO(
            "AIO engine setup: backends={}, timeoutMs={}, "
            "openConcurrency={}, commitConcurrency={}.",
            backends_->size(), timeoutMs_, config.openConcurrency, config.commitConcurrency);
        return aio_.Setup(timeoutMs_, config.aioQueueDepth);
    }

private:
    bool IsAio() const override { return true; }
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }
    template <bool dump>
    static void UpdateWaitMetrics(double wait)
    {
        static UC::Metrics::CachedMetric dumpQueueDuration{"posix_dump_queue_wait_duration_ms"};
        static UC::Metrics::CachedMetric loadQueueDuration{"posix_load_queue_wait_duration_ms"};
        UC::Metrics::UpdateStats(dump ? dumpQueueDuration : loadQueueDuration, wait * 1e3);
    }
    static void IncrementAioTimeoutMetric()
    {
        static UC::Metrics::CachedMetric timeoutTotal{"posix_aio_timeout_total"};
        UC::Metrics::UpdateStats(timeoutTotal, 1.0);
    }
    static void IncrementOpenErrorMetric()
    {
        static UC::Metrics::CachedMetric openErrors{"posix_open_errors_total"};
        UC::Metrics::UpdateStats(openErrors, 1.0);
    }
    static void IncrementIoErrorMetric()
    {
        static UC::Metrics::CachedMetric ioErrors{"posix_io_errors_total"};
        UC::Metrics::UpdateStats(ioErrors, 1.0);
    }
    std::string BackendTag(const TransTask& task) const
    {
        return fmt::format("{}:{}", task.backendIndex, (*backends_)[task.backendIndex]);
    }
    void CommitBlock(const TransTask& task, Detail::BlockId id, bool success, uint64_t ino = 0)
    {
        blockOperator_.Submit(
            BlockOperator::CommitTask{std::move(id), success, task.backendIndex, ino});
    }
    template <bool dump>
    void OnIoCallback(const TaskPtr& task, WaiterPtr w,
                      std::shared_ptr<LoadHandleCache::BorrowedFd> borrowed, int32_t fd,
                      size_t shardIndex, uint64_t ino, const Detail::BlockId& id,
                      const AioImpl::Result& result)
    {
        const auto tid = task->id;
        const auto shortIo = result.error == 0 && result.nBytes != static_cast<ssize_t>(shardSize_);
        if (result.error != 0 || shortIo) {
            UC_ERROR("Failed(error={}, bytes={}/{}) to do io on block({}) in backend({}).",
                     result.error, result.nBytes, shardSize_, id, BackendTag(*task));
            if (result.error != ECANCELED) { IncrementIoErrorMetric(); }
            task->Fail(!dump && shortIo ? Status::NotFound() : Status::Error());
            failureSet_.Insert(tid);
        }
        if constexpr (dump) {
            if (fd >= 0) { ::close(fd); }
            const auto ok = result.error == 0 && !shortIo && !failureSet_.Contains(tid);
            if (ok && tracker_.MarkDone(id, shardIndex, ino)) {
                CommitBlock(*task, id, true, ino);
            } else if (!ok && shardIndex + 1 == nShardPerBlock_) {
                CommitBlock(*task, id, false);
            }
        }
        w->Done();
    }
    template <bool dump>
    void OnOpenCallback(const TaskPtr& task, WaiterPtr w, const Detail::Shard& shard,
                        BlockOperator::OpenResult result)
    {
        const auto tid = task->id;
        const auto shardIndex = shard.index;
        const auto last = shardIndex + 1 == nShardPerBlock_;
        const auto ino = dump ? InoOf(result.fd) : 0;
        const auto& id = shard.owner;
        auto handleFailure = [this, task, tid, w, last, id](int32_t fd, Status status) {
            task->Fail(status);
            failureSet_.Insert(tid);
            if constexpr (dump) {
                if (fd >= 0) { ::close(fd); }
                if (last) { CommitBlock(*task, id, false); }
            }
            w->Done();
        };
        if (result.error != 0) {
            UC_ERROR("Failed({}) to do open on block({}) in backend({}).", result.error,
                     shard.owner, BackendTag(*task));
            if (result.error != ECANCELED) { IncrementOpenErrorMetric(); }
            auto status = !dump && result.error == ENOENT ? Status::NotFound() : Status::Error();
            handleFailure(result.fd, status);
            return;
        }
        if (failureSet_.Contains(tid)) {
            if constexpr (dump) {
                if (result.fd >= 0) { ::close(result.fd); }
                if (last) { CommitBlock(*task, id, false); }
            }
            w->Done();
            return;
        }
        auto fd = result.fd;
        auto borrowed = std::make_shared<LoadHandleCache::BorrowedFd>(std::move(result.borrowed));
        AioImpl::Io io;
        io.fd = fd;
        io.offset = shard.index * shardSize_;
        io.length = shardSize_;
        io.buffer = shard.addrs.front();
        io.tag = tid;
        io.callback = [this, task, w, borrowed, fd, shardIndex, ino, id](AioImpl::Result ioResult) {
            OnIoCallback<dump>(task, w, borrowed, fd, shardIndex, ino, id, ioResult);
        };
        auto status = dump ? aio_.WriteAsync(std::move(io)) : aio_.ReadAsync(std::move(io));
        if (status.Failure()) {
            UC_ERROR("Failed({}) to submit io on block({}) in backend({}).", status, id,
                     BackendTag(*task));
            if (status == Status::Timeout()) {
                IncrementAioTimeoutMetric();
            } else {
                IncrementIoErrorMetric();
            }
            task->Fail(status);
            failureSet_.Insert(tid);
            if constexpr (dump) {
                if (fd >= 0) { ::close(fd); }
                if (last) { CommitBlock(*task, id, false); }
            }
            w->Done();
        }
    }
    template <bool dump>
    void Dispatch(TaskPtr t, WaiterPtr w, double deadline)
    {
        const auto flags = O_DIRECT | (dump ? (O_CREAT | O_WRONLY) : O_RDONLY);
        const auto number = t->desc.size();
        w->Set(number);
        std::list<BlockOperator::OpenTask> tasks;
        for (size_t i = 0; i < number; ++i) {
            BlockOperator::OpenTask task;
            const auto& shard = t->desc[i];
            task.id = shard.owner;
            task.activated = dump;
            task.flags = flags;
            task.tag = t->id;
            task.useHandleCache = !dump;
            task.backendIndex = t->backendIndex;
            task.callback = [this, t, w, i](BlockOperator::OpenResult result) {
                OnOpenCallback<dump>(t, w, t->desc[i], std::move(result));
            };
            // A task taken after its deadline gives up without opening the file.
            task.onStart = [t, deadline] {
                if (NowTime::Now() < deadline) {
                    t->started.store(true, std::memory_order_relaxed);
                    return true;
                } else {
                    t->Fail(TimeoutStatus(t->started.load(std::memory_order_relaxed)));
                    return false;
                }
            };
            tasks.push_back(std::move(task));
        }
        blockOperator_.Submit(std::move(tasks));
    }
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        const auto isDump = (t->type == TransTask::Type::DUMP);
        UC_DEBUG("Posix task({},{},{},{}) dispatching.", id, brief, num, size);
        const auto wait = NowTime::Now() - tp;
        w->SetEpilog([this, t, id, brief = std::move(brief), num, size, tp, isDump] {
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
            {
                std::lock_guard<std::mutex> lk(regMutex_);
                registry_.erase(id);
            }
            if (t->onComplete) {
                auto onComplete = std::move(t->onComplete);
                // The latch is complete, so Wait only reclaims the task and its status.
                onComplete(Wait(id));
            }
        });
        const auto deadline = timeoutMs_ > 0 ? NowTime::Now() + timeoutMs_ / 1000.0
                                             : std::numeric_limits<double>::max();
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            registry_[id] = Inflight{t, w, deadline, num, false};
        }
        if (timeoutMs_ > 0) {
            UC_DEBUG("AIO task({}) registered: type={}, shardCount={}, deadlineMs={}.", id,
                     isDump ? "dump" : "load", num, timeoutMs_);
        } else {
            UC_DEBUG("AIO task({}) registered: type={}, shardCount={}, deadline=disabled.", id,
                     isDump ? "dump" : "load", num);
        }
        if (isDump) {
            UpdateWaitMetrics<true>(wait);
            Dispatch<true>(t, w, deadline);
        } else {
            UpdateWaitMetrics<false>(wait);
            Dispatch<false>(t, w, deadline);
        }
    }
    static Status TimeoutStatus(bool started)
    {
        if (started) {
            return Status::Timeout();
        } else {
            return Status::TimeoutInQueue();
        }
    }
    void SweepDeadlines()
    {
        auto now = NowTime::Now();
        std::vector<std::tuple<TaskPtr, WaiterPtr, size_t, bool>> due;
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            for (auto& [id, inf] : registry_) {
                if (!inf.aborted && now >= inf.deadlineTp) {
                    inf.aborted = true;
                    const auto started = inf.t->started.load(std::memory_order_relaxed);
                    inf.t->Fail(TimeoutStatus(started));
                    due.emplace_back(inf.t, inf.w, inf.shardCount, started);
                }
            }
        }
        // partition by if task has started, process started task first, they will be
        // recorded in passive health window to make them become unhealthy,
        // avoid later tasks to use the unhealthy backend again.
        std::stable_partition(due.begin(), due.end(),
                              [](const auto& item) { return std::get<3>(item); });
        for (auto& [t, w, shardCount, started] : due) {
            auto stuckMs = (now - w->startTp) * 1e3;
            auto pending = w->Pending();
            IncrementAioTimeoutMetric();
            UC_ERROR(
                "AIO task({}) in backend({}) timed out while {} after {:.1f}ms; force-draining "
                "latch "
                "({} of {} shard(s) outstanding).",
                t->id, BackendTag(*t), started ? "running" : "queued", stuckMs, pending,
                shardCount);
            ForceComplete(t, w);
        }
    }
    void Cancel(TaskPtr t) override
    {
        WaiterPtr w;
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            auto it = registry_.find(t->id);
            if (it == registry_.end()) { return; }
            if (it->second.aborted) { return; }
            it->second.aborted = true;
            w = it->second.w;
        }
        UC_ERROR("AIO task({}) in backend({}) cancelled on wait timeout; force-draining latch.",
                 t->id, BackendTag(*t));
        IncrementAioTimeoutMetric();
        t->Fail(TimeoutStatus(t->started.load(std::memory_order_relaxed)));
        ForceComplete(t, w);
    }
    void ForceComplete(const TaskPtr& t, const WaiterPtr& w)
    {
        const auto id = t->id;
        auto pending = w ? w->Pending() : 0;
        UC_WARN(
            "AIO task({}) in backend({}) force-completing; pending latch count before "
            "abort={}.",
            id, BackendTag(*t), pending);
        failureSet_.Insert(id);
        aio_.CancelTask(id);
        blockOperator_.CancelQueued(id);
        if (w) { w->Abort(); }
    }
};

}  // namespace UC::PosixStore

#endif
