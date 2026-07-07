/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef AIO_BLOCK_OPENER_H
#define AIO_BLOCK_OPENER_H

#include <atomic>
#include <condition_variable>
#include <fcntl.h>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include "space_layout.h"

namespace aio {

class BlockOpener {
public:
    struct OpenResult {
        int32_t fd;
        int32_t error;
    };
    using OpenCallback = std::function<void(OpenResult)>;
    struct OpenTask {
        BlockId id;
        int32_t flags;
        OpenCallback callback;
    };

    BlockOpener(const SpaceLayout* layout, const size_t nOpenWorker)
    {
        layout_ = layout;
        for (size_t i = 0; i < nOpenWorker; ++i) {
            workers_.push_back(std::thread{[this] { OpenWorkerLoop(); }});
        }
    }
    ~BlockOpener()
    {
        stop_ = true;
        {
            std::lock_guard<std::mutex> lock{openQueue_.mutex};
            openQueue_.cv.notify_all();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) { worker.join(); }
        }
    }
    void Submit(std::list<OpenTask>&& tasks)
    {
        auto& q = openQueue_;
        std::lock_guard<std::mutex> lock{q.mutex};
        q.queue.splice(q.queue.end(), tasks);
        q.cv.notify_all();
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
            const auto path = layout_->BlockPath(task.id);
            auto fd = ::open(path.c_str(), task.flags, mode);
            auto err = (fd < 0) ? errno : 0;
            if (task.callback) { task.callback(OpenResult{fd, err}); }
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
};

}  // namespace aio

#endif
