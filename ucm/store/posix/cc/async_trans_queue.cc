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
#include "async_trans_queue.h"
#include "logger/logger.h"
#include "posix_file.h"
#include <cerrno>

namespace UC::PosixStore {
namespace {
constexpr int32_t kRingEntries = 4096;
constexpr size_t kSubmitBatchSize = 512;
constexpr unsigned kMaxBatchCqe = 256;
}

AsyncTransQueue::~AsyncTransQueue()
{
    stop_.store(true);
    if (loadReaperThread_.joinable()) { loadReaperThread_.join(); }
    if (dumpReaperThread_.joinable()) { dumpReaperThread_.join(); }
}

Status AsyncTransQueue::Setup(const Config& config, TaskIdSet* failureSet,
                              const SpaceLayout* layout)
{
    failureSet_ = failureSet;
    layout_ = layout;
    ioSize_ = config.tensorSize;
    shardSize_ = config.shardSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    waitCqeTimeoutMs_ = config.timeoutMs;
    ioDirect_ = config.ioDirect;

    auto s = loadRing_.Init(kRingEntries);
    if (s.Failure()) [[unlikely]] { return s; }
    s = dumpRing_.Init(kRingEntries);
    if (s.Failure()) [[unlikely]] { return s; }

    auto success = loadSubmitterPool_.SetNWorker(config.dataTransConcurrency)
                       .SetWorkerInitFn([this](auto& args) {
                           args = &loadRing_;
                           return true;
                       })
                       .SetWorkerFn([this](auto& unit, auto& ring) { LoadSubmitter(unit, ring); })
                       .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("load submitter workers start failed"));
    }
    success = dumpSubmitterPool_.SetNWorker(config.dataTransConcurrency)
                  .SetWorkerInitFn([this](auto& args) {
                      args = &dumpRing_;
                      return true;
                  })
                  .SetWorkerFn([this](auto& unit, auto& ring) { DumpSubmitter(unit, ring); })
                  .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("dump submitter workers start failed"));
    }

    loadReaperThread_ = std::thread([this] { LoadReaperLoop(); });
    dumpReaperThread_ = std::thread([this] { DumpReaperLoop(); });
    return Status::OK();
}

void AsyncTransQueue::Push(TaskPtr task, WaiterPtr waiter)
{
    const size_t nShard = task->desc.size();
    waiter->Set(nShard);
    if (nShard == 0) [[unlikely]] { return; }
    IoTask ioTask{task->id, {}, waiter, true};
    ioTask.shards.reserve(nShard);
    for (auto&& shard : task->desc) {
        ioTask.shards.emplace_back(std::move(shard));
    }
    if (task->type == TransTask::Type::DUMP) {
        dumpSubmitterPool_.Push(std::move(ioTask));
    } else {
        loadSubmitterPool_.Push(std::move(ioTask));
    }
}

void AsyncTransQueue::SubmitTask(IoTask& task, IoUringContext* ring, bool isDump)
{
    if (task.firstIo) {
        auto wait = NowTime::Now() - task.waiter->startTp;
        UC_DEBUG("IoUring {} task({}) start running, wait {:.3f}ms.", isDump ? "dump" : "load",
                 task.owner, wait * 1e3);
    }
    if (failureSet_->Contains(task.owner)) {
        for (size_t i = 0; i < task.shards.size(); ++i) {
            task.waiter->Done();
        }
        return;
    }

    std::vector<IoUnitCtx*> preparedCtxs;
    preparedCtxs.reserve(task.shards.size());
    auto doneRemaining = [&](size_t startIdx) {
        for (size_t j = startIdx; j < task.shards.size(); ++j) {
            task.waiter->Done();
        }
    };
    auto failAndCleanup = [&](size_t begin, size_t end) {
        failureSet_->Insert(task.owner);
        for (size_t i = begin; i < end; ++i) {
            preparedCtxs[i]->unit.waiter->Done();
            delete preparedCtxs[i];
        }
    };

    for (size_t i = 0; i < task.shards.size(); ++i) {
        if (failureSet_->Contains(task.owner)) {
            failAndCleanup(0, preparedCtxs.size());
            doneRemaining(i);
            return;
        }
        auto* ctx = new IoUnitCtx{
            IoUnit{task.owner, std::move(task.shards[i]), task.waiter}, failureSet_, layout_, 0,
            false, {}, nullptr};
        ctx->isLastShardOfBlock = ctx->unit.shard.index + 1 == nShardPerBlock_;
        const auto& path = layout_->DataFilePath(ctx->unit.shard.owner, isDump);
        ctx->file = std::make_unique<PosixFile>(path);
        auto flags = isDump ? PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY
                            : PosixFile::OpenFlag::READ_ONLY;
        if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
        auto s = ctx->file->Open(flags);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
            failureSet_->Insert(ctx->unit.owner);
            ctx->unit.waiter->Done();
            delete ctx;
            failAndCleanup(0, preparedCtxs.size());
            doneRemaining(i + 1);
            return;
        }
        ctx->iov.resize(ctx->unit.shard.addrs.size());
        for (size_t j = 0; j < ctx->unit.shard.addrs.size(); ++j) {
            ctx->iov[j] = {ctx->unit.shard.addrs[j], ioSize_};
        }
        ctx->expectedBytes = ioSize_ * ctx->unit.shard.addrs.size();
        preparedCtxs.push_back(ctx);
    }

    if (preparedCtxs.empty()) { return; }

    auto& sqMtx = isDump ? dumpSqMtx_ : loadSqMtx_;
    std::unique_lock<std::mutex> lk(sqMtx);
    auto flushRange = [&](size_t begin, size_t end) -> bool {
        if (begin >= end) { return true; }
        size_t submitted = 0;
        const size_t nPending = end - begin;
        while (submitted < nPending) {
            int ret = ring->Submit();
            if (ret == -EINTR) { continue; }
            if (ret <= 0) {
                failAndCleanup(begin + submitted, end);
                return false;
            }
            submitted += static_cast<size_t>(ret);
        }
        return true;
    };

    size_t rangeBegin = 0;
    for (size_t i = 0; i < preparedCtxs.size(); ++i) {
        auto* ctx = preparedCtxs[i];
        auto* sqe = ring->GetSqe();
        if (!sqe) [[unlikely]] {
            if (!flushRange(rangeBegin, i)) {
                failAndCleanup(i, preparedCtxs.size());
                return;
            }
            rangeBegin = i;
            sqe = ring->GetSqe();
            if (!sqe) [[unlikely]] {
                failAndCleanup(i, preparedCtxs.size());
                return;
            }
        }
        if (isDump) {
            io_uring_prep_writev(sqe, ctx->file->Handle(), ctx->iov.data(),
                                 static_cast<size_t>(ctx->iov.size()),
                                 static_cast<off64_t>(shardSize_ * ctx->unit.shard.index));
        } else {
            io_uring_prep_readv(sqe, ctx->file->Handle(), ctx->iov.data(),
                                static_cast<size_t>(ctx->iov.size()),
                                static_cast<off64_t>(shardSize_ * ctx->unit.shard.index));
        }
        io_uring_sqe_set_data(sqe, ctx);

        if (i + 1 - rangeBegin >= kSubmitBatchSize) {
            if (!flushRange(rangeBegin, i + 1)) {
                failAndCleanup(i + 1, preparedCtxs.size());
                return;
            }
            rangeBegin = i + 1;
        }
    }
    (void)flushRange(rangeBegin, preparedCtxs.size());
}

void AsyncTransQueue::LoadSubmitter(IoTask& task, IoUringContext* ring)
{
    SubmitTask(task, ring, false);
}

void AsyncTransQueue::DumpSubmitter(IoTask& task, IoUringContext* ring)
{
    SubmitTask(task, ring, true);
}

void AsyncTransQueue::LoadReaperLoop()
{
    io_uring_cqe* cqes[kMaxBatchCqe];
    while (!stop_.load(std::memory_order_relaxed)) {
        struct io_uring_cqe* cqe = nullptr;
        int ret = loadRing_.WaitCqe(&cqe, waitCqeTimeoutMs_);
        if (ret == -ETIME || ret == -EINTR) { continue; }
        if (ret < 0 || cqe == nullptr) {
            UC_ERROR("io_uring wait cqe failed, ret={}", ret);
            continue;
        }

        unsigned nCqe = loadRing_.PeekBatchCqe(cqes, kMaxBatchCqe);
        if (nCqe == 0) {
            cqes[0] = cqe;
            nCqe = 1;
        }
        for (unsigned i = 0; i < nCqe; ++i) {
            auto* ctx = static_cast<IoUnitCtx*>(io_uring_cqe_get_data(cqes[i]));
            if (ctx) {
                if (cqes[i]->res < 0 || static_cast<size_t>(cqes[i]->res) != ctx->expectedBytes) {
                    ctx->failureSet->Insert(ctx->unit.owner);
                }
                ctx->unit.waiter->Done();
                delete ctx;
            }
        }
        loadRing_.CqAdvance(nCqe);
    }
}

void AsyncTransQueue::DumpReaperLoop()
{
    io_uring_cqe* cqes[kMaxBatchCqe];
    while (!stop_.load(std::memory_order_relaxed)) {
        struct io_uring_cqe* cqe = nullptr;
        int ret = dumpRing_.WaitCqe(&cqe, waitCqeTimeoutMs_);
        if (ret == -ETIME || ret == -EINTR) { continue; }
        if (ret < 0 || cqe == nullptr) {
            UC_ERROR("io_uring wait cqe failed, ret={}", ret);
            continue;
        }

        unsigned nCqe = dumpRing_.PeekBatchCqe(cqes, kMaxBatchCqe);
        if (nCqe == 0) {
            cqes[0] = cqe;
            nCqe = 1;
        }
        for (unsigned i = 0; i < nCqe; ++i) {
            auto* ctx = static_cast<IoUnitCtx*>(io_uring_cqe_get_data(cqes[i]));
            if (ctx) {
                const bool success =
                    cqes[i]->res >= 0 && static_cast<size_t>(cqes[i]->res) == ctx->expectedBytes;
                if (!success) { ctx->failureSet->Insert(ctx->unit.owner); }
                if (ctx->isLastShardOfBlock) {
                    layout_->CommitFile(ctx->unit.shard.owner, success);
                }
                ctx->unit.waiter->Done();
                delete ctx;
            }
        }
        dumpRing_.CqAdvance(nCqe);
    }
}

}  // namespace UC::PosixStore
