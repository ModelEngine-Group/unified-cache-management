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
#include <cerrno>
#include <unistd.h>
#include "logger/logger.h"
#include "metrics_api.h"
#include "posix_file.h"

namespace UC::PosixStore {

Status TransQueue::Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout,
                         const std::vector<std::string>* backends)
{
    failureSet_ = failureSet;
    layout_ = layout;
    backends_ = backends;
    ioSize_ = config.tensorSize;
    shardSize_ = config.shardSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    ioDirect_ = config.ioDirect;
    timeoutMs_ = config.timeoutMs;
    tracker_.Setup(nShardPerBlock_);
    auto cacheStatus = SetupLoadHandles(loadHandles_, backends_->size(), config.loadHandleCacheSize,
                                        config.ioDirect);
    if (cacheStatus.Failure()) [[unlikely]] { return cacheStatus; }
    auto success =
        loadPool_.SetNWorker(config.dataTransConcurrency)
            .SetWorkerFn([this](auto& ios, auto&) { LoadWorker(ios); })
            .SetWorkerTimeoutFn([this](IoUnit& ios, ssize_t tid) { OnIoUnitTimeout(ios); },
                                config.timeoutMs)
            .SetCpuAffinity(config.cpuAffinityCores)
            .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.dataTransConcurrency));
    }
    success = dumpPool_.SetNWorker(config.dataTransConcurrency)
                  .SetWorkerFn([this](auto& ios, auto&) { DumpWorker(ios); })
                  .SetWorkerTimeoutFn([this](IoUnit& ios, ssize_t tid) { OnIoUnitTimeout(ios); },
                                      config.timeoutMs)
                  .SetCpuAffinity(config.cpuAffinityCores)
                  .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.dataTransConcurrency));
    }
    return Status::OK();
}

void TransQueue::OnIoUnitTimeout(IoUnit& ios)
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_timeout_total"), 1.0);
    const auto backendIndex = ios.task->backendIndex;
    UC_ERROR("Posix io on block({}) in backend({}:{}) timed out after {} ms.", ios.shard.owner,
             backendIndex, (*backends_)[backendIndex], timeoutMs_);
    ios.task->Fail(Status::Timeout());
    if (!failureSet_->Contains(ios.task->id)) { failureSet_->Insert(ios.task->id); }
    ios.waiter->Done();
}

void TransQueue::OnIoUnitQueueTimeout(IoUnit& ios)
{
    const auto backendIndex = ios.task->backendIndex;
    UC_WARN("Posix io on block({}) in backend({}:{}) waited in queue over {} ms; giving up.",
            ios.shard.owner, backendIndex, (*backends_)[backendIndex], timeoutMs_);
    ios.task->Fail(Status::TimeoutInQueue());
    if (!failureSet_->Contains(ios.task->id)) { failureSet_->Insert(ios.task->id); }
    ios.waiter->Done();
}

void TransQueue::Push(TaskPtr task, WaiterPtr waiter)
{
    waiter->Set(task->desc.size());
    std::list<IoUnit> ios;
    for (auto&& shard : task->desc) { ios.emplace_back<IoUnit>({task, std::move(shard), waiter}); }
    ios.front().firstIo = true;
    if (task->type == TransTask::Type::DUMP) {
        dumpPool_.Push(ios);
    } else {
        loadPool_.Push(ios);
    }
}

void TransQueue::Cancel(TaskPtr task)
{
    auto& pool = task->type == TransTask::Type::DUMP ? dumpPool_ : loadPool_;
    const auto tid = task->id;
    pool.TraverseWaitQueue(
        [this, tid](IoUnit& ios) {
            return ios.task->id == tid || ios.waiter->IsTimeout(timeoutMs_);
        },
        [this](IoUnit& ios) { OnIoUnitTimeout(ios); },
        [this, tid](IoUnit& ios) {
            return ios.task->id > tid && !ios.waiter->IsTimeout(timeoutMs_);
        });
}

void TransQueue::LoadWorker(IoUnit& ios)
{
    if (ios.firstIo) {
        auto wait = NowTime::Now() - ios.waiter->startTp;
        UC_DEBUG("Posix load task({}) start running, wait {:.3f}ms.", ios.task->id, wait * 1e3);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_load_queue_wait_duration_ms"),
                                 wait * 1e3);
    }
    if (failureSet_->Contains(ios.task->id)) {
        ios.waiter->Done();
        return;
    }
    if (timeoutMs_ > 0 && ios.waiter->IsTimeout(timeoutMs_)) {
        OnIoUnitQueueTimeout(ios);
        return;
    }
    auto s = S2H(ios);
    if (s.Failure()) [[unlikely]] {
        ios.task->Fail(s);
        failureSet_->Insert(ios.task->id);
    }
    ios.waiter->Done();
}

void TransQueue::DumpWorker(IoUnit& ios)
{
    if (ios.firstIo) {
        auto wait = NowTime::Now() - ios.waiter->startTp;
        UC_DEBUG("Posix dump task({}) start running, wait {:.3f}ms.", ios.task->id, wait * 1e3);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_dump_queue_wait_duration_ms"),
                                 wait * 1e3);
    }
    if (failureSet_->Contains(ios.task->id)) {
        ios.waiter->Done();
        return;
    }
    if (timeoutMs_ > 0 && ios.waiter->IsTimeout(timeoutMs_)) {
        OnIoUnitQueueTimeout(ios);
        return;
    }
    auto s = H2S(ios);
    const auto& backend = (*backends_)[ios.task->backendIndex];
    if (s.Success()) {
        if (tracker_.MarkDone(ios.shard.owner, ios.shard.index, ios.ino)) {
            const auto path = layout_->ActivatedFilePath(backend, ios.shard.owner);
            if (ios.ino != 0 && InoOf(path) == ios.ino) {
                layout_->CommitFile(backend, ios.shard.owner, true);
            }
        }
    } else if (ios.shard.index + 1 == nShardPerBlock_) {
        layout_->CommitFile(backend, ios.shard.owner, false);
    }
    if (s.Failure()) [[unlikely]] {
        ios.task->Fail(s);
        failureSet_->Insert(ios.task->id);
    }
    ios.waiter->Done();
}

Status TransQueue::H2S(IoUnit& ios)
{
    const auto backendIndex = ios.task->backendIndex;
    const auto& backend = (*backends_)[backendIndex];
    const auto& path = layout_->ActivatedFilePath(backend, ios.shard.owner);
    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}) in backend({}:{}).", s, path, flags,
                 backendIndex, backend);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_open_errors_total"), 1.0);
        return s;
    }
    ios.ino = InoOf(file.Handle());
    auto offset = shardSize_ * ios.shard.index;
    for (const auto& addr : ios.shard.addrs) {
        s = file.Write(addr, ioSize_, offset);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to write file({}:{}) in backend({}:{}).", s, path, offset,
                     backendIndex, backend);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_errors_total"), 1.0);
            return s;
        }
        offset += ioSize_;
    }
    return Status::OK();
}

Status TransQueue::S2H(IoUnit& ios)
{
    const auto backendIndex = ios.task->backendIndex;
    const auto& backend = (*backends_)[backendIndex];
    const auto& path = layout_->ArchivedFilePath(backend, ios.shard.owner);
    auto borrowed = loadHandles_[backendIndex].GetOrOpen(ios.shard.owner, path);
    if (!borrowed.Valid()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) for load in backend({}:{}).", borrowed.Error(), path,
                 backendIndex, backend);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_open_errors_total"), 1.0);
        return borrowed.Error() == ENOENT ? Status::NotFound()
                                          : Status::OsApiError(std::to_string(borrowed.Error()));
    }
    auto offset = shardSize_ * ios.shard.index;
    for (const auto& addr : ios.shard.addrs) {
        auto nBytes = ::pread(borrowed.Fd(), addr, ioSize_, offset);
        auto eno = errno;
        if (nBytes < 0) [[unlikely]] {
            UC_ERROR("Failed({}) to read file({}:{}) in backend({}:{}).", eno, path, offset,
                     backendIndex, backend);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_errors_total"), 1.0);
            return Status::OsApiError(std::to_string(eno));
        }
        if (nBytes != static_cast<ssize_t>(ioSize_)) [[unlikely]] {
            UC_ERROR("Failed({}) to read file({}:{}) in backend({}:{}).", nBytes, path, offset,
                     backendIndex, backend);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_errors_total"), 1.0);
            return Status::NotFound();
        }
        offset += ioSize_;
    }
    return Status::OK();
}

}  // namespace UC::PosixStore
