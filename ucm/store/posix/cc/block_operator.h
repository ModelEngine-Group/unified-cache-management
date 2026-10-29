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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_BLOCK_OPERATOR_H
#define UNIFIEDCACHE_POSIX_STORE_CC_BLOCK_OPERATOR_H

#include <cerrno>
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "block_tracker.h"
#include "handle_cache.h"
#include "logger/logger.h"
#include "space_layout.h"
#include "thread/cpu_affinity.h"
#include "thread/thread_pool.h"
#include "type/types.h"

namespace UC::PosixStore {

class BlockOperator {
public:
    struct OpenResult {
        int32_t fd;
        int32_t error;
        LoadHandleCache::BorrowedFd borrowed;
    };
    using OpenCallback = std::function<void(OpenResult)>;
    struct OpenTask {
        Detail::BlockId id;
        bool activated;
        int32_t flags;
        OpenCallback callback;
        uint64_t tag{0};
        bool useHandleCache{false};
        size_t backendIndex{0};
        // Called when a worker takes the task; returning false skips the open.
        std::function<bool()> onStart;
    };
    struct CommitTask {
        Detail::BlockId id;
        bool success;
        size_t backendIndex{0};
        uint64_t ino{0};
    };

    ~BlockOperator() { Stop(); }
    void Stop()
    {
        std::unique_ptr<ThreadPool<OpenTask>> openSrv;
        std::unique_ptr<ThreadPool<CommitTask>> commitSrv;
        {
            std::unique_lock<std::shared_mutex> lock{srvMutex_};
            openSrv = std::move(openSrv_);
            commitSrv = std::move(commitSrv_);
        }
        openSrv.reset();
        commitSrv.reset();
    }
    // loadHandles holds one handle cache per backend.
    Status Setup(const SpaceLayout* layout, const std::vector<std::string>* backends,
                 size_t nOpenWorker, size_t nCommitWorker, size_t timeoutMs,
                 std::function<void()> onOpenStuck,
                 std::deque<LoadHandleCache>* loadHandles = nullptr)
    {
        layout_ = layout;
        backends_ = backends;
        loadHandles_ = loadHandles;
        openSrv_ = std::make_unique<ThreadPool<OpenTask>>();
        auto success =
            openSrv_->SetWorkerInitFn([](auto&) { return SetThreadName("ucm_posix_open"); })
                .SetWorkerFn([this](OpenTask& task, auto&) { Open(task); })
                .SetWorkerTimeoutFn(
                    [this, timeoutMs, onOpenStuck](OpenTask& task, auto) {
                        UC_WARN(
                            "Open of block({}) in backend({}:{}) is stuck over {} ms; a new "
                            "worker takes over.",
                            task.id, task.backendIndex, (*backends_)[task.backendIndex], timeoutMs);
                        if (onOpenStuck) { onOpenStuck(); }
                    },
                    timeoutMs)
                .SetNWorker(nOpenWorker)
                .Run();
        if (!success) { return Status::Error("failed to run posix open thread pool"); }
        commitSrv_ = std::make_unique<ThreadPool<CommitTask>>();
        success =
            commitSrv_->SetWorkerInitFn([](auto&) { return SetThreadName("ucm_posix_cmit"); })
                .SetWorkerFn([this](CommitTask& task, auto&) { Commit(task); })
                .SetWorkerTimeoutFn(
                    [this, timeoutMs](CommitTask& task, auto) {
                        UC_WARN(
                            "Commit of block({}) in backend({}:{}) is stuck over {} ms; a new "
                            "worker takes over.",
                            task.id, task.backendIndex, (*backends_)[task.backendIndex], timeoutMs);
                    },
                    timeoutMs)
                .SetNWorker(nCommitWorker)
                .Run();
        if (!success) { return Status::Error("failed to run posix commit thread pool"); }
        return Status::OK();
    }
    void Submit(std::list<OpenTask>&& tasks)
    {
        std::shared_lock<std::shared_mutex> lock{srvMutex_};
        if (openSrv_) { openSrv_->Push(tasks); }
    }
    void Submit(CommitTask&& task)
    {
        std::shared_lock<std::shared_mutex> lock{srvMutex_};
        if (commitSrv_) { commitSrv_->Push(std::move(task)); }
    }
    void CancelQueued(uint64_t tag)
    {
        std::list<OpenTask> purged;
        {
            std::shared_lock<std::shared_mutex> lock{srvMutex_};
            if (openSrv_) {
                openSrv_->TraverseWaitQueue(
                    [tag](OpenTask& task) { return task.tag == tag; },
                    [&purged](OpenTask& task) { purged.push_back(std::move(task)); }, nullptr);
            }
        }
        if (!purged.empty()) {
            UC_WARN("AIO task({}) cancelled {} queued open task(s).", tag, purged.size());
        }
        // Run callbacks outside the lock, since they may submit commits.
        for (auto& task : purged) {
            if (task.callback) { task.callback(OpenResult{-1, ECANCELED, {}}); }
        }
    }

private:
    static bool SetThreadName(const char* name)
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName(name);
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set UCM posix worker name({}).", nameStatus, name);
        }
        return true;
    }
    void Open(OpenTask& task)
    {
        constexpr const auto mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
        const auto& backend = (*backends_)[task.backendIndex];
        const auto path = task.activated ? layout_->ActivatedFilePath(backend, task.id)
                                         : layout_->ArchivedFilePath(backend, task.id);
        const auto shouldStart = !task.onStart || task.onStart();
        if (shouldStart) {
            OpenResult result{-1, 0, {}};
            if (task.useHandleCache && loadHandles_) {
                result.borrowed = (*loadHandles_)[task.backendIndex].GetOrOpen(task.id, path);
                if (result.borrowed.Valid()) {
                    result.fd = result.borrowed.Fd();
                    result.error = 0;
                } else {
                    result.fd = -1;
                    result.error = result.borrowed.Error();
                }
            } else {
#ifdef UCM_ENABLE_TEST_HOOKS
                auto hook = TestHooks::GetOpenHook();
                result.fd =
                    hook ? hook(path, task.flags, mode) : ::open(path.c_str(), task.flags, mode);
#else
                result.fd = ::open(path.c_str(), task.flags, mode);
#endif
                result.error = (result.fd < 0) ? errno : 0;
            }
            if (task.callback) { task.callback(std::move(result)); }
        } else {
            if (task.callback) { task.callback(OpenResult{-1, ECANCELED, {}}); }
        }
    }
    // The inode check stats the file, so it runs here rather than on the AIO completion thread,
    // where a hung backend would block completions of every backend.
    void Commit(const CommitTask& task)
    {
        const auto& backend = (*backends_)[task.backendIndex];
        if (!task.success ||
            (task.ino != 0 && InoOf(layout_->ActivatedFilePath(backend, task.id)) == task.ino)) {
            layout_->CommitFile(backend, task.id, task.success);
        }
    }

    const SpaceLayout* layout_{nullptr};
    const std::vector<std::string>* backends_{nullptr};
    std::deque<LoadHandleCache>* loadHandles_{nullptr};
    std::shared_mutex srvMutex_;
    std::unique_ptr<ThreadPool<OpenTask>> openSrv_;
    std::unique_ptr<ThreadPool<CommitTask>> commitSrv_;
};

}  // namespace UC::PosixStore

#endif
