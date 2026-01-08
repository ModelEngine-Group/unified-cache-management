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
#include "trans_queue.h"
#include <cstring>
#include <hf3fs_usrbio.h>
#include "ds3fs_file.h"
#include "logger/logger.h"

namespace UC::Ds3fsStore {

Status TransQueue::Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout)
{
    failureSet_ = failureSet;
    layout_ = layout;
    ioSize_ = config.tensorSize;
    shardSize_ = config.shardSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    ioDirect_ = config.ioDirect;
    mountPoint_ = config.storageBackends[0];
    iorEntries_ = config.iorEntries;
    iorDepth_ = config.iorDepth;
    numaId_ = config.numaId;

    auto success = pool_.SetNWorker(config.streamNumber)
                       .SetWorkerInitFn([this](auto& ctx) { return InitWorkerContext(ctx); })
                       .SetWorkerFn([this](auto& ios, auto& ctx) { Worker(ios, ctx); })
                       .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.streamNumber));
    }
    return Status::OK();
}

bool TransQueue::InitWorkerContext(std::unique_ptr<WorkerContext>& ctx)
{
    ctx = std::make_unique<WorkerContext>();

    auto s = ctx->Init(mountPoint_, ioSize_, iorEntries_, iorDepth_, numaId_);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed to initialize worker context: {}", s);
        return false;
    }

    return true;
}

void TransQueue::Push(TaskPtr task, WaiterPtr waiter)
{
    waiter->Set(task->desc.size());
    std::list<IoUnit> ios;
    for (auto&& shard : task->desc) {
        ios.emplace_back<IoUnit>({task->id, task->type, std::move(shard), waiter});
    }
    ios.front().firstIo = true;
    pool_.Push(ios);
}

void TransQueue::Worker(IoUnit& ios, const std::unique_ptr<WorkerContext>& ctx)
{
    if (ios.firstIo) {
        auto wait = NowTime::Now() - ios.waiter->startTp;
        UC_DEBUG("Ds3fs task({}) start running, wait {:.3f}ms.", ios.owner, wait * 1e3);
    }
    if (failureSet_->Contains(ios.owner)) {
        ios.waiter->Done();
        return;
    }
    auto s = Status::OK();
    if (ios.type == TransTask::Type::DUMP) {
        s = H2S(ios, ctx);
        if (ios.shard.index + 1 == nShardPerBlock_) {
            layout_->CommitFile(ios.shard.owner, s.Success());
        }
    } else {
        s = S2H(ios, ctx);
    }
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(ios.owner); }
    ios.waiter->Done();
}

Status TransQueue::H2S(IoUnit& ios, const std::unique_ptr<WorkerContext>& ctx)
{
    const auto& path = layout_->DataFilePath(ios.shard.owner, true);
    auto flags = Ds3fsFile::OpenFlag::CREATE | Ds3fsFile::OpenFlag::WRITE_ONLY;
    if (ioDirect_) { flags |= Ds3fsFile::OpenFlag::DIRECT; }

    Ds3fsFile file{path};
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed to open file({}): {}", path, s);
        return s;
    }

    int fd = file.ReleaseHandle();

    FdGuard fdGuard;
    s = fdGuard.Register(fd);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed to register fd({}) for file({}): {}", fd, path, s);
        return s;
    }

    auto offset = shardSize_ * ios.shard.index;
    return DoIoWrite(ctx, fd, offset, ios);
}

Status TransQueue::S2H(IoUnit& ios, const std::unique_ptr<WorkerContext>& ctx)
{
    const auto& path = layout_->DataFilePath(ios.shard.owner, false);
    auto flags = Ds3fsFile::OpenFlag::READ_ONLY;
    if (ioDirect_) { flags |= Ds3fsFile::OpenFlag::DIRECT; }

    Ds3fsFile file{path};
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed to open file({}): {}", path, s);
        return s;
    }

    int fd = file.ReleaseHandle();

    FdGuard fdGuard;
    s = fdGuard.Register(fd);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed to register fd({}) for file({}): {}", fd, path, s);
        return s;
    }

    auto offset = shardSize_ * ios.shard.index;
    return DoIoRead(ctx, fd, offset, ios);
}

Status TransQueue::DoIoRead(const std::unique_ptr<WorkerContext>& ctx, int fd, size_t offset,
                            IoUnit& ios)
{
    int prepRes = hf3fs_prep_io(ctx->iorRead.Get(), ctx->iov.Get(), true, ctx->iov.Base(), fd,
                                offset, ioSize_, nullptr);
    if (prepRes < 0) [[unlikely]] {
        UC_ERROR("Failed to prep read io: result={}", prepRes);
        return Status::OsApiError(fmt::format("Failed to prep read io: {}", prepRes));
    }

    int submitRes = hf3fs_submit_ios(ctx->iorRead.Get());
    if (submitRes < 0) [[unlikely]] {
        UC_ERROR("Failed to submit read io: result={}", submitRes);
        return Status::OsApiError(fmt::format("Failed to submit read ios: {}", submitRes));
    }

    struct hf3fs_cqe cqe;
    int waitRes = hf3fs_wait_for_ios(ctx->iorRead.Get(), &cqe, 1, 1, nullptr);
    if (waitRes <= 0) [[unlikely]] {
        UC_ERROR("Failed to wait for read io: result={}", waitRes);
        return Status::OsApiError(fmt::format("Failed to wait for read ios: {}", waitRes));
    }

    auto s = CheckIoResult(cqe, layout_->DataFilePath(ios.shard.owner, false), offset, true);
    if (s.Failure()) { return s; }

    std::memcpy(reinterpret_cast<void*>(ios.shard.addrs[0]), ctx->iov.Base(), ioSize_);
    return Status::OK();
}

Status TransQueue::DoIoWrite(const std::unique_ptr<WorkerContext>& ctx, int fd, size_t offset,
                             IoUnit& ios)
{
    std::memcpy(ctx->iov.Base(), reinterpret_cast<const void*>(ios.shard.addrs[0]), ioSize_);

    int prepRes = hf3fs_prep_io(ctx->iorWrite.Get(), ctx->iov.Get(), false, ctx->iov.Base(), fd,
                                offset, ioSize_, nullptr);
    if (prepRes < 0) [[unlikely]] {
        UC_ERROR("Failed to prep write io: result={}", prepRes);
        return Status::OsApiError(fmt::format("Failed to prep write io: {}", prepRes));
    }

    int submitRes = hf3fs_submit_ios(ctx->iorWrite.Get());
    if (submitRes < 0) [[unlikely]] {
        UC_ERROR("Failed to submit write io: result={}", submitRes);
        return Status::OsApiError(fmt::format("Failed to submit write ios: {}", submitRes));
    }

    struct hf3fs_cqe cqe;
    int waitRes = hf3fs_wait_for_ios(ctx->iorWrite.Get(), &cqe, 1, 1, nullptr);
    if (waitRes <= 0) [[unlikely]] {
        UC_ERROR("Failed to wait for write io: result={}", waitRes);
        return Status::OsApiError(fmt::format("Failed to wait for write ios: {}", waitRes));
    }

    auto s = CheckIoResult(cqe, layout_->DataFilePath(ios.shard.owner, true), offset, false);
    if (s.Failure()) { return s; }

    return Status::OK();
}

Status TransQueue::CheckIoResult(const hf3fs_cqe& cqe, const std::string& path, size_t offset,
                                 bool isRead)
{
    if (cqe.result < 0) [[unlikely]] {
        const char* op = isRead ? "Read" : "Write";
        UC_ERROR("{} operation failed: result={}, offset={}, size={}, path={}", op, cqe.result,
                 offset, ioSize_, path);
        return Status::OsApiError(fmt::format("{} operation failed: {}", op, cqe.result));
    }
    return Status::OK();
}

}  // namespace UC::Ds3fsStore