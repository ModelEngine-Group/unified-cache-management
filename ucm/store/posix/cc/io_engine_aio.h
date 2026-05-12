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

#include "aio_impl.h"
#include "block_operator.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::PosixStore {

class IoEngineAio : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    size_t shardSize_;
    size_t nShardPerBlock_;
    const SpaceLayout* layout_;
    BlockOperator blockOperator_;
    AioImpl aio_;

public:
    Status Setup(const Config& config, const SpaceLayout* layout)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        nShardPerBlock_ = config.blockSize / config.shardSize;
        layout_ = layout;
        blockOperator_.Setup(layout, config.openConcurrency, config.commitConcurrency);
        return aio_.Setup();
    }

private:
    template <bool dump>
    static void UpdateWaitMetrics(double wait)
    {
        UC::Metrics::UpdateStats(dump ? "posix_dump_queue_wait_duration_ms"
                                      : "posix_load_queue_wait_duration_ms",
                                 wait * 1e3);
    }
    template <bool dump>
    static void UpdateIoMetrics(double tp, size_t ioBytes)
    {
        auto cost = NowTime::Now() - tp;
        auto bwGbps = cost > 0 ? static_cast<double>(ioBytes) / cost / 1e9 : 0.0;
        UC::Metrics::UpdateStats(dump ? "posix_h2s_duration_ms"
                                      : "posix_s2h_duration_ms",
                                 cost * 1e3);
        UC::Metrics::UpdateStats(dump ? "posix_h2s_bandwidth_gbps"
                                      : "posix_s2h_bandwidth_gbps",
                                 bwGbps);
    }
    template <bool dump>
    static void UpdateFailureMetrics()
    {
        UC::Metrics::UpdateStats(dump ? "posix_dump_failures_total"
                                      : "posix_load_failures_total",
                                 1.0);
    }
    void CommitBlock(Detail::BlockId id, bool success)
    {
        blockOperator_.Submit(BlockOperator::CommitTask{std::move(id), success});
    }
    template <bool dump>
    void OnIoCallback(const Detail::TaskHandle& tid, WaiterPtr w, int32_t fd, bool last,
                      const Detail::BlockId& id, double tp, const AioImpl::Result& result)
    {
        UpdateIoMetrics<dump>(tp, shardSize_);
        if (result.error != 0) {
            UC_ERROR("Failed({}) to do io on block({}).", result.error, id);
            failureSet_.Insert(tid);
            UpdateFailureMetrics<dump>();
        } else {
            UC::Metrics::UpdateStats(dump ? "posix_h2s_bytes_total" : "posix_s2h_bytes_total",
                                     static_cast<double>(shardSize_));
        }
        ::close(fd);
        if constexpr (dump) {
            if (last) { CommitBlock(id, !failureSet_.Contains(tid)); }
        }
        w->Done();
    }
    template <bool dump>
    void OnOpenCallback(const Detail::TaskHandle& tid, WaiterPtr w, const Detail::Shard& shard,
                        double tp, const BlockOperator::OpenResult& result)
    {
        const auto last = shard.index + 1 == nShardPerBlock_;
        const auto& id = shard.owner;
        auto handleFailure = [&](int32_t fd, bool recordMetrics) {
            if (recordMetrics) {
                failureSet_.Insert(tid);
                UpdateIoMetrics<dump>(tp, shardSize_);
                UpdateFailureMetrics<dump>();
            }
            if (fd >= 0) { ::close(fd); }
            if constexpr (dump) {
                if (last) { CommitBlock(id, false); }
            }
            w->Done();
        };
        if (result.error != 0) {
            UC_ERROR("Failed({}) to do open on block({}).", result.error, shard.owner);
            handleFailure(result.fd, true);
            return;
        }
        if (failureSet_.Contains(tid)) { handleFailure(result.fd, false); return; }
        AioImpl::Io io;
        io.fd = result.fd;
        io.offset = shard.index * shardSize_;
        io.length = shardSize_;
        io.buffer = shard.addrs.front();
        io.callback = [this, tid, w, fd = result.fd, last, id, tp](AioImpl::Result ioResult) {
            OnIoCallback<dump>(tid, w, fd, last, id, tp, ioResult);
        };
        auto status = dump ? aio_.WriteAsync(std::move(io)) : aio_.ReadAsync(std::move(io));
        if (status.Failure()) { handleFailure(result.fd, true); }
    }
    template <bool dump>
    void Dispatch(TaskPtr t, WaiterPtr w)
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
            const auto tp = NowTime::Now();
            task.callback = [this, tid = t->id, w,
                             shard = std::ref(t->desc[i]),
                             tp](BlockOperator::OpenResult result) {
                OnOpenCallback<dump>(tid, w, shard, tp, result);
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
        w->SetEpilog([id, brief = std::move(brief), num, size, tp, isDump] {
            auto cost = NowTime::Now() - tp;
            UC_DEBUG("Posix task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     cost * 1e3);
            UC::Metrics::UpdateStats(
                isDump ? "posix_dump_task_duration_ms" : "posix_load_task_duration_ms",
                cost * 1e3);
        });
        if (isDump) {
            UpdateWaitMetrics<true>(wait);
            Dispatch<true>(t, w);
        } else {
            UpdateWaitMetrics<false>(wait);
            Dispatch<false>(t, w);
        }
    }
};

}  // namespace UC::PosixStore

#endif
