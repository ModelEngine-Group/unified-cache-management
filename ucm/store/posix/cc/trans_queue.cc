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
#include "logger/logger.h"
#include "posix_file.h"

namespace UC::PosixStore {

Status TransQueue::Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout)
{
    failureSet_ = failureSet;
    layout_ = layout;
    ioSize_ = config.ioSize;
    shardSize_ = config.shardSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    ioDirect_ = config.transferIoDirect;
    auto success = pool_.SetNWorker(config.transferStreamNumber)
                       .SetWorkerFn([this](auto& ios, auto&) { Worker(ios); })
                       .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.transferStreamNumber));
    }
    return Status::OK();
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

void TransQueue::Worker(IoUnit& ios)
{
    if (ios.firstIo) {
        auto wait = NowTime::Now() - ios.waiter->startTp;
        UC_DEBUG("Posix task({}) start running, wait {:.3f}ms.", ios.owner, wait * 1e3);
    }
    if (failureSet_->Contains(ios.owner)) {
        ios.waiter->Done();
        return;
    }
    auto s = Status::OK();
    if (ios.type == TransTask::Type::DUMP) {
        s = H2S(ios);
        if (ios.shard.index + 1 == nShardPerBlock_) {
            layout_->CommitFile(ios.shard.owner, s.Success());
        }
    } else {
        s = S2H(ios);
    }
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(ios.owner); }
    ios.waiter->Done();
}

Status TransQueue::H2S(IoUnit& ios)
{
    const auto& path = layout_->DataFilePath(ios.shard.owner, true);
    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
        return s;
    }
    auto offset = shardSize_ * ios.shard.index;
    for (const auto& addr : ios.shard.addrs) {
        s = file.Write(addr, ioSize_, offset);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to write file({}:{}).", s, path, offset);
            return s;
        }
        offset += ioSize_;
    }
    return Status::OK();
}

Status TransQueue::S2H(IoUnit& ios)
{
    const auto& path = layout_->DataFilePath(ios.shard.owner, false);
    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::READ_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
        return s;
    }
    auto offset = shardSize_ * ios.shard.index;
    for (const auto& addr : ios.shard.addrs) {
        s = file.Read(addr, ioSize_, offset);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to read file({}:{}).", s, path, offset);
            return s;
        }
        offset += ioSize_;
    }
    return Status::OK();
}

}  // namespace UC::PosixStore
