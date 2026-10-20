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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_LOOKUP_MANAGER_H
#define UNIFIEDCACHE_POSIX_STORE_CC_LOOKUP_MANAGER_H

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include "global_config.h"
#include "logger/logger.h"
#include "posix_file.h"
#include "space_layout.h"
#include "thread/thread_pool.h"

namespace UC::PosixStore {

class LookupManager {
    struct Task {
        std::string path;
        size_t backendIndex{0};
        std::promise<Status> result;
        std::atomic<bool> completed{false};

        void Complete(Status status)
        {
            // Timeout and a late access result can race.
            if (!completed.exchange(true)) { result.set_value(std::move(status)); }
        }
    };

public:
    Status Setup(const Config& config, const SpaceLayout* layout,
                 const std::vector<std::string>* backends)
    {
        layout_ = layout;
        backends_ = backends;
        const auto timeoutMs = config.timeoutMs;
        const auto success =
            lookupSrv_
                .SetWorkerFn([this](std::shared_ptr<Task>& task, auto&) {
                    constexpr auto mode = PosixFile::AccessMode::EXIST |
                                          PosixFile::AccessMode::READ |
                                          PosixFile::AccessMode::WRITE;
                    auto status = PosixFile{task->path}.Access(mode);
                    // NotFound is a cache miss, not a backend error.
                    if (status.Failure() && status != Status::NotFound()) {
                        UC_ERROR("Failed({}) to look up file({}) in backend({}:{}).", status,
                                 task->path, task->backendIndex, (*backends_)[task->backendIndex]);
                    }
                    task->Complete(std::move(status));
                })
                .SetWorkerTimeoutFn(
                    [this, timeoutMs](std::shared_ptr<Task>& task, auto) {
                        UC_ERROR("Lookup of file({}) in backend({}:{}) timed out after {} ms.",
                                 task->path, task->backendIndex, (*backends_)[task->backendIndex],
                                 timeoutMs);
                        task->Complete(Status::Timeout());
                    },
                    config.timeoutMs, 100)
                .SetNWorker(config.lookupConcurrency)
                .SetCpuAffinity(config.cpuAffinityCores)
                .Run();
        if (!success) { return Status::Error("failed to run lookup service thread pool"); }
        return Status::OK();
    }

    Status Lookup(const Detail::BlockId& block, size_t backendIndex)
    {
        auto task = std::make_shared<Task>();
        task->path = layout_->ArchivedFilePath((*backends_)[backendIndex], block);
        task->backendIndex = backendIndex;
        auto result = task->result.get_future();
        lookupSrv_.Push(std::move(task));
        return result.get();
    }

private:
    const SpaceLayout* layout_{nullptr};
    const std::vector<std::string>* backends_{nullptr};
    ThreadPool<std::shared_ptr<Task>> lookupSrv_;
};

}  // namespace UC::PosixStore

#endif
