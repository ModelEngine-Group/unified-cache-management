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
#include <memory>
#include <vector>

namespace UC::PosixStore {

Status AsyncTransQueue::Setup(const Config& config, TaskIdSet* failureSet,
                              const SpaceLayout* layout)
{
    failureSet_ = failureSet;
    layout_ = layout;
    ioSize_ = config.tensorSize;
    shardSize_ = config.shardSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    ioDirect_ = config.ioDirect;
    auto success = loadIoUringPool_.SetNWorker(config.dataTransConcurrency)
                       .SetWorkerInitFn([](auto& ctx) {
                           ctx = std::make_unique<IoUringContext>();
                           return ctx->Init().Success();
                       })
                       .SetWorkerExitFn([](auto& ctx) { ctx.reset(); })
                       .SetWorkerFn([this](auto& ios, auto& ctx) { LoadWorkerAsync(ios, ctx); })
                       .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.dataTransConcurrency));
    }
    success = dumpIoUringPool_.SetNWorker(config.dataTransConcurrency)
                  .SetWorkerInitFn([](auto& ctx) {
                      ctx = std::make_unique<IoUringContext>();
                      return ctx->Init().Success();
                  })
                  .SetWorkerExitFn([](auto& ctx) { ctx.reset(); })
                  .SetWorkerFn([this](auto& ios, auto& ctx) { DumpWorkerAsync(ios, ctx); })
                  .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.dataTransConcurrency));
    }
    return Status::OK();
}

void AsyncTransQueue::Push(TaskPtr task, WaiterPtr waiter)
{
    waiter->Set(1);
    IoUnit ios{task->id, std::move(task->desc), waiter};
    if (task->type == TransTask::Type::DUMP) {
        dumpIoUringPool_.Push(std::move(ios));
    } else {
        loadIoUringPool_.Push(std::move(ios));
    }
}

void AsyncTransQueue::LoadWorkerAsync(IoUnit& ios, const IoUringContextPtr& ctx)
{
    auto wait = NowTime::Now() - ios.waiter->startTp;
    UC_DEBUG("IoUring load task({}) start running, wait {:.3f}ms.", ios.owner, wait * 1e3);
    if (failureSet_->Contains(ios.owner)) {
        ios.waiter->Done();
        return;
    }
    auto s = S2HAsync(ios, ctx);
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(ios.owner); }
    ios.waiter->Done();
}

void AsyncTransQueue::DumpWorkerAsync(IoUnit& ios, const IoUringContextPtr& ctx)
{

    auto wait = NowTime::Now() - ios.waiter->startTp;
    UC_DEBUG("IoUring dump task({}) start running, wait {:.3f}ms.", ios.owner, wait * 1e3);
    if (failureSet_->Contains(ios.owner)) {
        ios.waiter->Done();
        return;
    }
    auto s = H2SAsync(ios, ctx);
    for (const auto& shard : ios.shards) {
        if (shard.index + 1 == nShardPerBlock_) {
            layout_->CommitFile(shard.owner, s.Success());
        }
    }
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(ios.owner); }
    ios.waiter->Done();
}

Status AsyncTransQueue::S2HAsync(IoUnit& ios, const IoUringContextPtr& ctx)
{
    std::vector<PosixFile> files;
    files.reserve(ios.shards.size());
    std::vector<IoUringTask> tasks;
    for (const auto& shard : ios.shards) {
        const auto& path = layout_->DataFilePath(shard.owner, false);
        files.emplace_back(path);
        auto flags = PosixFile::OpenFlag::READ_ONLY;
        if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
        auto s = files.back().Open(flags);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
            return s;
        }
        auto offset = shardSize_ * shard.index;
        for (const auto& addr : shard.addrs) {
            tasks.push_back({files.back().Handle(), addr, ioSize_, static_cast<off64_t>(offset)});
            offset += ioSize_;
        }
    }
    return ctx->S2HBatch(tasks);
}

Status AsyncTransQueue::H2SAsync(IoUnit& ios, const IoUringContextPtr& ctx)
{
    std::vector<PosixFile> files;
    files.reserve(ios.shards.size());
    std::vector<IoUringTask> tasks;
    for (const auto& shard : ios.shards) {
        const auto& path = layout_->DataFilePath(shard.owner, true);
        files.emplace_back(path);
        auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY;
        if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
        auto s = files.back().Open(flags);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
            return s;
        }
        auto offset = shardSize_ * shard.index;
        for (const auto& addr : shard.addrs) {
            tasks.push_back({files.back().Handle(), addr, ioSize_, static_cast<off64_t>(offset)});
            offset += ioSize_;
        }
    }
    return ctx->H2SBatch(tasks);
}
}  // namespace UC::PosixStore
