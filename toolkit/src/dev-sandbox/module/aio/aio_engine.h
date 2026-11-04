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
#ifndef AIO_AIO_ENGINE_H
#define AIO_AIO_ENGINE_H

#include <memory>
#include <unistd.h>
#include "aio_impl.h"
#include "block_opener.h"

namespace aio {

class AioEngine {
public:
    struct IoShard {
        BlockId id;
        void* addr;
        size_t length;
    };
    using IoTask = std::vector<IoShard>;
    class IoWaiter {
    public:
        IoWaiter(size_t expected) noexcept : counter_(expected) {}
        void SetEpilog(std::function<void(void)> finish) noexcept { finish_ = std::move(finish); }
        void Done() noexcept
        {
            auto counter = this->counter_.load(std::memory_order_acquire);
            while (counter > 0) {
                auto desired = counter - 1;
                if (this->counter_.compare_exchange_weak(counter, desired,
                                                         std::memory_order_acq_rel)) {
                    if (desired == 0) {
                        if (finish_) { finish_(); }
                        std::lock_guard<std::mutex> lg(this->mutex_);
                        this->cv_.notify_all();
                    }
                    return;
                }
            }
        }
        void Wait() noexcept
        {
            std::unique_lock<std::mutex> lk(this->mutex_);
            if (this->counter_ == 0) { return; }
            this->cv_.wait(lk, [this] { return this->counter_ == 0; });
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        std::atomic<size_t> counter_{0};
        std::function<void(void)> finish_{nullptr};
    };

    AioEngine(const SpaceLayout* layout, const size_t nOpenWorker)
        : aio_(std::make_unique<AioImpl>()),
          opener_(std::make_unique<BlockOpener>(layout, nOpenWorker))
    {
    }
    std::shared_ptr<IoWaiter> SubmitWrite(const IoTask& ioTask) { return Submit<true>(ioTask); }
    std::shared_ptr<IoWaiter> SubmitRead(const IoTask& ioTask) { return Submit<false>(ioTask); }

private:
    void OnIoCallback(std::shared_ptr<IoWaiter> w, int32_t fd, const BlockId& id,
                      const AioImpl::Result& result)
    {
        if (result.error != 0) {
            fmt::println("Failed({}) to do io on block({}).", result.error, id);
        }
        ::close(fd);
        w->Done();
    }
    template <bool dump>
    void OnOpenCallback(std::shared_ptr<IoWaiter> w, const IoShard& shard,
                        const BlockOpener::OpenResult& result)
    {
        if (result.error != 0) {
            fmt::println("Failed({}) to do open on block({}).", result.error, shard.id);
            if (result.fd >= 0) { ::close(result.fd); }
            w->Done();
            return;
        }
        AioImpl::Io io;
        io.fd = result.fd;
        io.offset = 0;
        io.length = shard.length;
        io.buffer = shard.addr;
        io.callback = [this, w, fd = result.fd, id = shard.id](AioImpl::Result ioResult) {
            OnIoCallback(w, fd, id, ioResult);
        };
        if constexpr (dump) {
            aio_->WriteAsync(std::move(io));
        } else {
            aio_->ReadAsync(std::move(io));
        }
    }
    template <bool dump>
    std::shared_ptr<IoWaiter> Submit(const IoTask& ioTask)
    {
        const auto number = ioTask.size();
        auto waiter = std::make_shared<IoWaiter>(number);
        const auto flags = O_DIRECT | (dump ? (O_CREAT | O_WRONLY) : O_RDONLY);
        std::list<BlockOpener::OpenTask> tasks;
        for (size_t i = 0; i < number; ++i) {
            BlockOpener::OpenTask task;
            const auto& shard = ioTask[i];
            task.id = shard.id;
            task.flags = flags;
            task.callback = [this, waiter, shard = ioTask[i]](BlockOpener::OpenResult result) {
                OnOpenCallback<dump>(waiter, shard, result);
            };
            tasks.push_back(std::move(task));
        }
        opener_->Submit(std::move(tasks));
        return waiter;
    }

    std::unique_ptr<AioImpl> aio_;
    std::unique_ptr<BlockOpener> opener_;
};

}  // namespace aio

#endif
