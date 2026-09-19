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
#include "global_config.h"
#include "posix_file.h"
#include "space_layout.h"
#include "thread/thread_pool.h"

namespace UC::PosixStore {

class LookupManager {
    struct Task {
        std::string path;
        std::promise<Status> result;
        std::atomic<bool> completed{false};

        void Complete(Status status)
        {
            // Timeout and a late access result can race.
            if (!completed.exchange(true)) { result.set_value(std::move(status)); }
        }
    };

public:
    Status Setup(const Config& config, const SpaceLayout* layout, const std::string& backend)
    {
        layout_ = layout;
        backend_ = backend;
        const auto success =
            lookupSrv_
                .SetWorkerFn([](std::shared_ptr<Task>& task, auto&) {
                    constexpr auto mode = PosixFile::AccessMode::EXIST |
                                          PosixFile::AccessMode::READ |
                                          PosixFile::AccessMode::WRITE;
                    task->Complete(PosixFile{task->path}.Access(mode));
                })
                .SetWorkerTimeoutFn(
                    [](std::shared_ptr<Task>& task, auto) { task->Complete(Status::Timeout()); },
                    config.timeoutMs, 100)
                .SetNWorker(config.lookupConcurrency)
                .SetCpuAffinity(config.cpuAffinityCores)
                .Run();
        if (!success) { return Status::Error("failed to run lookup service thread pool"); }
        return Status::OK();
    }

    Status Lookup(const Detail::BlockId& block)
    {
        auto task = std::make_shared<Task>();
        task->path = layout_->DataFilePath(backend_, block, false);
        auto result = task->result.get_future();
        lookupSrv_.Push(std::move(task));
        return result.get();
    }

private:
    const SpaceLayout* layout_{nullptr};
    std::string backend_;
    ThreadPool<std::shared_ptr<Task>> lookupSrv_;
};

}  // namespace UC::PosixStore

#endif
