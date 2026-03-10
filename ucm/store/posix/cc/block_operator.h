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

#include <atomic>
#include <condition_variable>
#include <fcntl.h>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include "space_layout.h"
#include "type/types.h"

namespace UC::PosixStore {

class BlockOperator {
public:
    struct OpenResult {
        int32_t fd;
        int32_t error;
    };
    using OpenCallback = std::function<void(OpenResult)>;
    struct OpenTask {
        Detail::BlockId id;
        bool activated;
        int32_t flags;
        OpenCallback callback;
    };
    struct CommitTask {
        Detail::BlockId id;
        bool success;
    };

    ~BlockOperator()
    {
        stop_ = true;
        {
            std::lock_guard<std::mutex> lock{openQueue_.mutex};
            openQueue_.cv.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock{commitQueue_.mutex};
            commitQueue_.cv.notify_all();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) { worker.join(); }
        }
    }
    void Setup(const SpaceLayout* layout, const size_t nOpenWorker, const size_t nCommitWorker)
    {
        layout_ = layout;
        for (size_t i = 0; i < nOpenWorker; ++i) {
            workers_.push_back(std::thread{[this] { OpenWorkerLoop(); }});
        }
        for (size_t i = 0; i < nCommitWorker; ++i) {
            workers_.push_back(std::thread{[this] { CommitWorkerLoop(); }});
        }
    }
    void Submit(std::list<OpenTask>&& tasks)
    {
        auto& q = openQueue_;
        std::lock_guard<std::mutex> lock{q.mutex};
        q.queue.splice(q.queue.end(), tasks);
        q.cv.notify_all();
    }
    void Submit(CommitTask&& task)
    {
        auto& q = commitQueue_;
        std::lock_guard<std::mutex> lock{q.mutex};
        q.queue.push_back(std::move(task));
        q.cv.notify_one();
    }

private:
    void OpenWorkerLoop()
    {
        constexpr const auto mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
        for (;;) {
            OpenTask task;
            {
                std::unique_lock<std::mutex> lock{openQueue_.mutex};
                openQueue_.cv.wait(lock, [this] { return stop_ || !openQueue_.queue.empty(); });
                if (stop_) { break; }
                if (openQueue_.queue.empty()) { continue; }
                task = std::move(openQueue_.queue.front());
                openQueue_.queue.pop_front();
            }
            const auto path = layout_->DataFilePath(task.id, task.activated);
            auto fd = ::open(path.c_str(), task.flags, mode);
            auto err = (fd < 0) ? errno : 0;
            if (task.callback) { task.callback(OpenResult{fd, err}); }
        }
    }
    void CommitWorkerLoop()
    {
        for (;;) {
            CommitTask task;
            {
                std::unique_lock<std::mutex> lock{commitQueue_.mutex};
                commitQueue_.cv.wait(lock, [this] { return stop_ || !commitQueue_.queue.empty(); });
                if (stop_) { break; }
                if (commitQueue_.queue.empty()) { continue; }
                task = std::move(commitQueue_.queue.front());
                commitQueue_.queue.pop_front();
            }
            layout_->CommitFile(task.id, task.success);
        }
    }

    template <class T>
    struct TaskQueue {
        std::list<T> queue;
        std::mutex mutex;
        std::condition_variable cv;
    };

    std::atomic_bool stop_{false};
    const SpaceLayout* layout_;
    std::list<std::thread> workers_;
    TaskQueue<OpenTask> openQueue_;
    TaskQueue<CommitTask> commitQueue_;
};

}  // namespace UC::PosixStore

#endif
