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

#include "aio_engine.h"
#include "block_operator.h"
#include "logger/logger.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::PosixStore {

class IoEngineAio : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    size_t shardSize_;
    size_t nShardPerBlock_;
    const SpaceLayout* layout_;
    BlockOperator blockOperator_;
    AioEngine aio_;

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
    void CommitBlock(Detail::BlockId id, bool success)
    {
        blockOperator_.Submit(BlockOperator::CommitTask{std::move(id), success});
    }
    template <bool dump>
    void OnIoCallback(const Detail::TaskHandle& tid, WaiterPtr w, int32_t fd, bool last,
                      const Detail::BlockId& id, const AioEngine::Result& result)
    {
        if (result.error != 0) {
            UC_ERROR("Failed({}) to do io on block({}).", result.error, id);
            failureSet_.Insert(tid);
        }
        ::close(fd);
        if constexpr (dump) {
            if (last) { CommitBlock(id, !failureSet_.Contains(tid)); }
        }
        w->Done();
    }
    template <bool dump>
    void OnOpenCallback(const Detail::TaskHandle& tid, WaiterPtr w, const Detail::Shard& shard,
                        const BlockOperator::OpenResult& result)
    {
        const auto last = shard.index + 1 == nShardPerBlock_;
        const auto& id = shard.owner;
        auto handleFailure = [&](int32_t error, int32_t fd) {
            if (error != 0) { failureSet_.Insert(tid); }
            if (fd >= 0) { ::close(fd); }
            if constexpr (dump) {
                if (last) { CommitBlock(id, false); }
            }
            w->Done();
        };
        if (result.error != 0) {
            UC_ERROR("Failed({}) to do open on block({}).", result.error, shard.owner);
            failureSet_.Insert(tid);
        }
        if (failureSet_.Contains(tid)) {
            handleFailure(0, result.fd);
            return;
        }
        AioEngine::Io io;
        io.fd = result.fd;
        io.offset = shard.index * shardSize_;
        io.length = shardSize_;
        io.buffer = shard.addrs.front();
        io.callback = [this, tid, w, fd = result.fd, last, id](AioEngine::Result ioResult) {
            OnIoCallback<dump>(tid, w, fd, last, id, ioResult);
        };
        auto status = dump ? aio_.WriteAsync(std::move(io)) : aio_.ReadAsync(std::move(io));
        if (status.Failure()) { handleFailure(-1, result.fd); }
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
            task.callback = [this, tid = t->id, w,
                             shard = std::ref(t->desc[i])](BlockOperator::OpenResult result) {
                OnOpenCallback<dump>(tid, w, shard, result);
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
        UC_DEBUG("Posix task({},{},{},{}) dispatching.", id, brief, num, size);
        w->SetEpilog([id, brief = std::move(brief), num, size, tp] {
            auto cost = NowTime::Now() - tp;
            UC_DEBUG("Posix task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     cost * 1e3);
        });
        if (t->type == TransTask::Type::DUMP) {
            Dispatch<true>(t, w);
        } else {
            Dispatch<false>(t, w);
        }
    }
};

}  // namespace UC::PosixStore

#endif
