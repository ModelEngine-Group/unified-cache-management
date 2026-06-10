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
#include "aio_impl.h"
#include <cstring>
#include <fmt/format.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "error_handle.h"

namespace aio {

static inline int32_t AioSetup(int32_t nEvents, aio_context_t* pCtx)
{
    return syscall(SYS_io_setup, nEvents, pCtx);
}
static inline int32_t AioGetEvents(aio_context_t ctx, int64_t minNr, int64_t maxNr,
                                   io_event* events, timespec* timeout)
{
    return syscall(SYS_io_getevents, ctx, minNr, maxNr, events, timeout);
}
static inline int32_t AioSubmit(aio_context_t ctx, int64_t nr, iocb** ios)
{
    return syscall(SYS_io_submit, ctx, nr, ios);
}
static inline int32_t AioDestroy(aio_context_t ctx) { return syscall(SYS_io_destroy, ctx); }
static inline void AioPrepareRead(struct iocb* iocb, int32_t fd, void* buf, size_t count,
                                  size_t offset)
{
    std::memset(iocb, 0, sizeof(*iocb));
    iocb->aio_fildes = fd;
    iocb->aio_lio_opcode = 0; /* IO_CMD_PREAD */
    iocb->aio_reqprio = 0;
    iocb->aio_buf = reinterpret_cast<uintptr_t>(buf);
    iocb->aio_nbytes = count;
    iocb->aio_offset = offset;
}
static inline void AioPrepareWrite(struct iocb* iocb, int32_t fd, void* buf, size_t count,
                                   size_t offset)
{
    std::memset(iocb, 0, sizeof(*iocb));
    iocb->aio_fildes = fd;
    iocb->aio_lio_opcode = 1; /* IO_CMD_PWRITE */
    iocb->aio_reqprio = 0;
    iocb->aio_buf = reinterpret_cast<uintptr_t>(buf);
    iocb->aio_nbytes = count;
    iocb->aio_offset = offset;
}
static inline void AioSetEventFd(struct iocb* iocb, int32_t eventfd)
{
    iocb->aio_flags |= (1 << 0) /* IOCB_FLAG_RESFD */;
    iocb->aio_resfd = eventfd;
}

AioImpl::AioImpl()
{
    AIO_ASSERT(AioSetup(queueDepth_, &ctx_) == 0);
    eventFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    AIO_ASSERT(eventFd_ >= 0);
    epollFd_ = epoll_create1(EPOLL_CLOEXEC);
    AIO_ASSERT(epollFd_ >= 0);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.ptr = nullptr;
    AIO_ASSERT(epoll_ctl(epollFd_, EPOLL_CTL_ADD, eventFd_, &ev) == 0);
    eventThread_ = std::thread([this] { CompletionLoop(); });
}

AioImpl::~AioImpl()
{
    stop_ = true;
    if (eventThread_.joinable()) {
        uint64_t val = 1;
        auto ret = write(eventFd_, &val, sizeof(val));
        if (ret < 0) { fmt::println("Failed to call write."); }
        eventThread_.join();
    }
    if (epollFd_ >= 0) { close(epollFd_); }
    if (eventFd_ >= 0) { close(eventFd_); }
    if (ctx_) { AioDestroy(ctx_); }
}

void AioImpl::ReadAsync(Io&& io)
{
    struct iocb cb;
    auto data = std::make_unique<Callback>(std::move(io.callback));
    AioPrepareRead(&cb, io.fd, io.buffer, io.length, io.offset);
    cb.aio_data = (uintptr_t)(void*)data.get();
    SubmitIo(&cb);
    data.release();
}

void AioImpl::WriteAsync(Io&& io)
{
    struct iocb cb;
    auto data = std::make_unique<Callback>(std::move(io.callback));
    AioPrepareWrite(&cb, io.fd, io.buffer, io.length, io.offset);
    cb.aio_data = (uintptr_t)(void*)data.get();
    SubmitIo(&cb);
    data.release();
}

void AioImpl::CompletionLoop()
{
    std::vector<epoll_event> epollEvents(128);
    std::vector<io_event> aioEvents(batchCompleteSize);
    while (!stop_) {
        auto nfds = epoll_wait(epollFd_, epollEvents.data(), epollEvents.size(), epollTimeoutMs);
        for (auto i = 0; i < nfds; i++) {
            if (epollEvents[i].data.ptr == nullptr) {
                uint64_t count;
                auto ret = read(eventFd_, &count, sizeof(count));
                if (ret < 0) { fmt::println("Failed to call read."); }
                HarvestCompletions(aioEvents);
            }
        }
    }
}

void AioImpl::HarvestCompletions(std::vector<io_event>& events)
{
    auto batchSize = static_cast<int>(events.size());
    while (!stop_) {
        auto num = AioGetEvents(ctx_, 1, batchSize, events.data(), nullptr);
        for (auto i = 0; i < num; i++) {
            auto cb = (Callback*)(void*)events[i].data;
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

void AioImpl::SubmitIo(iocb* cb)
{
    AioSetEventFd(cb, eventFd_);
    auto ret = 0;
    for (;;) {
        ret = AioSubmit(ctx_, 1, &cb);
        auto eno = errno;
        if (ret == 1) { return; }
        if (eno == EAGAIN) {
            std::this_thread::yield();
            continue;
        }
        AIO_ASSERT(eno == 0);
    }
}

}  // namespace aio
