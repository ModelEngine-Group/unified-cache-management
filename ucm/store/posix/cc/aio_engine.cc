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
#include "aio_engine.h"
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include "logger/logger.h"

namespace UC::PosixStore {

AioEngine::~AioEngine()
{
    stop_ = true;
    if (eventThread_.joinable()) {
        uint64_t val = 1;
        auto ret = write(eventFd_, &val, sizeof(val));
        if (ret < 0) { UC_WARN("Failed to call write."); }
        eventThread_.join();
    }
    if (epollFd_ >= 0) { close(epollFd_); }
    if (eventFd_ >= 0) { close(eventFd_); }
    if (ctx_) { io_destroy(ctx_); }
}

Status AioEngine::Setup()
{
    auto ret = io_setup(queueDepth_, &ctx_);
    if (ret != 0) {
        UC_ERROR("Failed({}) to call io_setup.", ret);
        return Status::Error(std::to_string(ret));
    }
    eventFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    auto eno = errno;
    if (eventFd_ < 0) {
        UC_ERROR("Failed({}) to call eventfd.", eno);
        return Status::Error(std::string(strerror(eno)));
    }
    epollFd_ = epoll_create1(EPOLL_CLOEXEC);
    eno = errno;
    if (epollFd_ < 0) {
        UC_ERROR("Failed({}) to call epoll_create1.", eno);
        return Status::Error(std::string(strerror(eno)));
    }
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.ptr = nullptr;
    ret = epoll_ctl(epollFd_, EPOLL_CTL_ADD, eventFd_, &ev);
    eno = errno;
    if (ret < 0) {
        UC_ERROR("Failed({}) to call epoll_ctl.", eno);
        return Status::Error(std::string(strerror(eno)));
    }
    eventThread_ = std::thread([this] { CompletionLoop(); });
    return Status::OK();
}

Status AioEngine::ReadAsync(Io&& io)
{
    auto cb = std::make_unique<struct iocb>();
    auto data = std::make_unique<Callback>(std::move(io.callback));
    io_prep_pread(cb.get(), io.fd, io.buffer, io.length, io.offset);
    cb->data = static_cast<void*>(data.get());
    auto status = SubmitIo(cb.get());
    if (status.Failure()) {
        UC_ERROR("Failed({}) to submit read io.", status);
        return status;
    }
    data.release();
    return Status::OK();
}

Status AioEngine::WriteAsync(Io&& io)
{
    auto cb = std::make_unique<struct iocb>();
    auto data = std::make_unique<Callback>(std::move(io.callback));
    io_prep_pwrite(cb.get(), io.fd, io.buffer, io.length, io.offset);
    cb->data = static_cast<void*>(data.get());
    auto status = SubmitIo(cb.get());
    if (status.Failure()) {
        UC_ERROR("Failed({}) to submit write io.", status);
        return status;
    }
    data.release();
    cb.release();
    return Status::OK();
}

void AioEngine::CompletionLoop()
{
    std::vector<epoll_event> epollEvents(128);
    std::vector<io_event> aioEvents(batchCompleteSize);
    while (!stop_) {
        auto nfds = epoll_wait(epollFd_, epollEvents.data(), epollEvents.size(), epollTimeoutMs);
        for (auto i = 0; i < nfds; i++) {
            if (epollEvents[i].data.ptr == nullptr) {
                uint64_t count;
                auto ret = read(eventFd_, &count, sizeof(count));
                if (ret < 0) { UC_WARN("Failed to call read."); }
                HarvestCompletions(aioEvents);
            }
        }
    }
}

void AioEngine::HarvestCompletions(std::vector<io_event>& events)
{
    auto batchSize = static_cast<int>(events.size());
    while (!stop_) {
        auto num = io_getevents(ctx_, 1, batchSize, events.data(), nullptr);
        for (auto i = 0; i < num; i++) {
            auto cb = static_cast<Callback*>(events[i].data);
            if (!cb) { continue; }
            Result res;
            if (events[i].res >= 0) {
                res.nBytes = events[i].res;
                res.error = 0;
            } else {
                res.nBytes = -1;
                res.error = -static_cast<int>(events[i].res);
            }
            (*cb)(res);
            delete cb;
        }
        if (num < batchSize) { break; }
    }
}

Status AioEngine::SubmitIo(iocb* cb)
{
    io_set_eventfd(cb, eventFd_);
    auto ret = 0;
    for (;;) {
        ret = io_submit(ctx_, 1, &cb);
        if (ret == 1) { return Status::OK(); }
        if (ret == -EAGAIN) {
            std::this_thread::yield();
            continue;
        }
        return Status::Error(std::string(strerror(-ret)));
    }
}

}  // namespace UC::PosixStore
