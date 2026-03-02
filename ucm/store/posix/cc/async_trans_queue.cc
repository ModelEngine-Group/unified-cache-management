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

    auto s = loadRing_.Init();
    if (s.Failure()) [[unlikely]] { return s; }
    s = dumpRing_.Init();
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
    std::list<IoUnit> units;
    for (auto&& shard : task->desc) {
        units.emplace_back(IoUnit{task->id, std::move(shard), waiter});
    }
    units.front().firstIo = true;
    if (task->type == TransTask::Type::DUMP) {
        dumpSubmitterPool_.Push(units);
    } else {
        loadSubmitterPool_.Push(units);
    }
}

void AsyncTransQueue::LoadSubmitter(IoUnit& unit, IoUringContext* ring)
{
    if (unit.firstIo) {
        auto wait = NowTime::Now() - unit.waiter->startTp;
        UC_DEBUG("IoUring load task({}) start running, wait {:.3f}ms.", unit.owner, wait * 1e3);
    }
    if (failureSet_->Contains(unit.owner)) {
        unit.waiter->Done();
        return;
    }

    const auto& path = layout_->DataFilePath(unit.shard.owner, false);
    bool isLast = unit.shard.index + 1 == nShardPerBlock_;
    auto* ctx = new IoUnitCtx{std::move(unit), failureSet_, layout_, 0, isLast, {}, nullptr};
    ctx->file = std::make_unique<PosixFile>(path);
    auto flags = PosixFile::OpenFlag::READ_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = ctx->file->Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
        failureSet_->Insert(ctx->unit.owner);
        ctx->unit.waiter->Done();
        delete ctx;
        return;
    }

    ctx->iov.resize(ctx->unit.shard.addrs.size());
    for (size_t i = 0; i < ctx->unit.shard.addrs.size(); ++i) {
        ctx->iov[i] = {ctx->unit.shard.addrs[i], ioSize_};
    }
    ctx->expectedBytes = ioSize_ * ctx->unit.shard.addrs.size();

    {
        std::unique_lock<std::mutex> lk(loadSqMtx_);
        struct io_uring_sqe* sqe = ring->GetSqe();
        if (!sqe) [[unlikely]] {
            failureSet_->Insert(ctx->unit.owner);
            ctx->unit.waiter->Done();
            delete ctx;
            return;
        }
        io_uring_prep_readv(sqe, ctx->file->Handle(), ctx->iov.data(),
                            static_cast<size_t>(ctx->iov.size()),
                            static_cast<off64_t>(shardSize_ * ctx->unit.shard.index));
        io_uring_sqe_set_data(sqe, ctx);
        int ret = ring->Submit();
        if (ret < 1) [[unlikely]] {
            failureSet_->Insert(ctx->unit.owner);
            ctx->unit.waiter->Done();
            delete ctx;
            return;
        }
    }
}

void AsyncTransQueue::DumpSubmitter(IoUnit& unit, IoUringContext* ring)
{
    if (unit.firstIo) {
        auto wait = NowTime::Now() - unit.waiter->startTp;
        UC_DEBUG("IoUring dump task({}) start running, wait {:.3f}ms.", unit.owner, wait * 1e3);
    }
    if (failureSet_->Contains(unit.owner)) {
        unit.waiter->Done();
        return;
    }

    const auto& path = layout_->DataFilePath(unit.shard.owner, true);
    bool isLast = unit.shard.index + 1 == nShardPerBlock_;
    auto* ctx = new IoUnitCtx{std::move(unit), failureSet_, layout_, 0, isLast, {}, nullptr};
    ctx->file = std::make_unique<PosixFile>(path);
    auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = ctx->file->Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
        failureSet_->Insert(ctx->unit.owner);
        ctx->unit.waiter->Done();
        delete ctx;
        return;
    }

    ctx->iov.resize(ctx->unit.shard.addrs.size());
    for (size_t i = 0; i < ctx->unit.shard.addrs.size(); ++i) {
        ctx->iov[i] = {ctx->unit.shard.addrs[i], ioSize_};
    }
    ctx->expectedBytes = ioSize_ * ctx->unit.shard.addrs.size();

    {
        std::unique_lock<std::mutex> lk(dumpSqMtx_);
        struct io_uring_sqe* sqe = ring->GetSqe();
        if (!sqe) [[unlikely]] {
            failureSet_->Insert(ctx->unit.owner);
            ctx->unit.waiter->Done();
            delete ctx;
            return;
        }
        io_uring_prep_writev(sqe, ctx->file->Handle(), ctx->iov.data(),
                             static_cast<size_t>(ctx->iov.size()),
                             static_cast<off64_t>(shardSize_ * ctx->unit.shard.index));
        io_uring_sqe_set_data(sqe, ctx);
        int ret = ring->Submit();
        if (ret < 1) [[unlikely]] {
            failureSet_->Insert(ctx->unit.owner);
            ctx->unit.waiter->Done();
            delete ctx;
            return;
        }
    }
}

void AsyncTransQueue::LoadReaperLoop()
{
    while (!stop_.load(std::memory_order_relaxed)) {
        struct io_uring_cqe* cqe = nullptr;
        int ret = loadRing_.WaitCqe(&cqe, waitCqeTimeoutMs_);
        if (ret == -ETIME || ret == -EINTR) { continue; }
        if (ret < 0 || cqe == nullptr) {
            UC_ERROR("io_uring wait cqe failed, ret={}", ret);
            continue;
        }
        auto* ctx = static_cast<IoUnitCtx*>(io_uring_cqe_get_data(cqe));
        if (ctx) {
            if (cqe->res < 0 || static_cast<size_t>(cqe->res) != ctx->expectedBytes) {
                ctx->failureSet->Insert(ctx->unit.owner);
            }
            ctx->unit.waiter->Done();
            delete ctx;
        }
        loadRing_.CqeSeen(cqe);
    }
}

void AsyncTransQueue::DumpReaperLoop()
{
    while (!stop_.load(std::memory_order_relaxed)) {
        struct io_uring_cqe* cqe = nullptr;
        int ret = dumpRing_.WaitCqe(&cqe, waitCqeTimeoutMs_);
        if (ret == -ETIME || ret == -EINTR) { continue; }
        if (ret < 0 || cqe == nullptr) {
            UC_ERROR("io_uring wait cqe failed, ret={}", ret);
            continue;
        }
        auto* ctx = static_cast<IoUnitCtx*>(io_uring_cqe_get_data(cqe));
        if (ctx) {
            const bool success =
                cqe->res >= 0 && static_cast<size_t>(cqe->res) == ctx->expectedBytes;
            if (!success) { ctx->failureSet->Insert(ctx->unit.owner); }
            if (ctx->isLastShardOfBlock) { layout_->CommitFile(ctx->unit.shard.owner, success); }
            ctx->unit.waiter->Done();
            delete ctx;
        }
        dumpRing_.CqeSeen(cqe);
    }
}

}  // namespace UC::PosixStore
