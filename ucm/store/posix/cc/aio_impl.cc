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
#include "aio_impl.h"
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include "logger/logger.h"
#include "time/now_time.h"

namespace UC::PosixStore {

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
static inline int32_t AioCancel(aio_context_t ctx, struct iocb* cb, io_event* result)
{
    return syscall(SYS_io_cancel, ctx, cb, result);
}
static inline int32_t AioDestroy(aio_context_t ctx) { return syscall(SYS_io_destroy, ctx); }
static inline void AioPrepareRead(struct iocb* iocb, int32_t fd, void* buf, size_t count,
                                  size_t offset)
{
    memset(iocb, 0, sizeof(*iocb));
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
    memset(iocb, 0, sizeof(*iocb));
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

AioImpl::~AioImpl()
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
    if (ctx_) { AioDestroy(ctx_); }
    // The completion thread is joined and the kernel context destroyed; any iocb still tracked
    // here belongs to an IO that never completed (a truly hung device). Its heap iocb/Callback is
    // intentionally not reclaimed (doing so could race a kernel still touching the buffer); the
    // process is exiting, so the OS reclaims the memory. Just surface the count.
    std::lock_guard<std::mutex> lk(tableMutex_);
    if (!iocbToTag_.empty()) {
        UC_WARN("AIO teardown: {} in-flight IO(s) abandoned (never completed; not reclaimed).",
                iocbToTag_.size());
    }
}

Status AioImpl::Setup(size_t epollTimeoutMs, size_t sweepIntervalMs, size_t submitTimeoutMs)
{
    epollTimeoutMs_ = epollTimeoutMs > 0 ? epollTimeoutMs : 10;
    sweepIntervalMs_ = sweepIntervalMs;
    submitTimeoutMs_ = submitTimeoutMs;
    auto ret = AioSetup(queueDepth_, &ctx_);
    if (ret != 0) {
        UC_ERROR("Failed({}) to call AioSetup.", ret);
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

Status AioImpl::ReadAsync(Io&& io)
{
    // Heap-allocate the iocb (stable address) so it can later be passed to io_cancel and so the
    // completion thread can reclaim it. Track before submit: once submitted, a completion may be
    // harvested concurrently, and harvest looks the iocb up in the side table.
    auto cb = std::make_unique<struct iocb>();
    auto data = std::make_unique<Callback>(std::move(io.callback));
    AioPrepareRead(cb.get(), io.fd, io.buffer, io.length, io.offset);
    cb->aio_data = (uintptr_t)(void*)data.get();
    Track(io.tag, cb.get());
    auto status = SubmitIo(cb.get());
    if (status.Failure()) {
        Untrack(cb.get());
        UC_ERROR("Failed({}) to submit read io.", status);
        return status;  // cb and data reclaimed by unique_ptr
    }
    data.release();
    cb.release();  // ownership transferred to "kernel + completion thread"
    return Status::OK();
}

Status AioImpl::WriteAsync(Io&& io)
{
    auto cb = std::make_unique<struct iocb>();
    auto data = std::make_unique<Callback>(std::move(io.callback));
    AioPrepareWrite(cb.get(), io.fd, io.buffer, io.length, io.offset);
    cb->aio_data = (uintptr_t)(void*)data.get();
    Track(io.tag, cb.get());
    auto status = SubmitIo(cb.get());
    if (status.Failure()) {
        Untrack(cb.get());
        UC_ERROR("Failed({}) to submit write io.", status);
        return status;  // cb and data reclaimed by unique_ptr
    }
    data.release();
    cb.release();  // ownership transferred to "kernel + completion thread"
    return Status::OK();
}

void AioImpl::CompletionLoop()
{
    std::vector<epoll_event> epollEvents(128);
    std::vector<io_event> aioEvents(batchCompleteSize);
    while (!stop_) {
        auto nfds = epoll_wait(epollFd_, epollEvents.data(), epollEvents.size(), epollTimeoutMs_);
        for (auto i = 0; i < nfds; i++) {
            if (epollEvents[i].data.ptr == nullptr) {
                uint64_t count;
                auto ret = read(eventFd_, &count, sizeof(count));
                if (ret < 0) { UC_WARN("Failed to call read."); }
                HarvestCompletions(aioEvents);
            }
        }
        // epoll_wait wakes at least every epollTimeoutMs_ even with no completions, so the
        // deadline watchdog runs on this same thread without an extra timer thread.
        MaybeSweep();
    }
}

void AioImpl::MaybeSweep()
{
    if (!sweepFn_) { return; }
    auto now = NowTime::Now();
    if (sweepIntervalMs_ == 0 ||
        (now - lastSweepTp_) * 1e3 >= static_cast<double>(sweepIntervalMs_)) {
        lastSweepTp_ = now;
        sweepFn_();
    }
}

void AioImpl::HarvestCompletions(std::vector<io_event>& events)
{
    auto batchSize = static_cast<int>(events.size());
    while (!stop_) {
        auto num = AioGetEvents(ctx_, 1, batchSize, events.data(), nullptr);
        for (auto i = 0; i < num; i++) {
            auto* iocbPtr = reinterpret_cast<struct iocb*>(events[i].obj);
            auto cb = (Callback*)(void*)events[i].data;
            Result res;
            if (events[i].res >= 0) {
                res.nBytes = events[i].res;
                res.error = 0;
            } else {
                res.nBytes = -1;
                res.error = -static_cast<int>(events[i].res);
            }
            if (cb) {
                (*cb)(res);
                delete cb;
            }
            // The completion thread is the sole reclaimer of the heap iocb (and its side-table
            // entry) for IOs that the kernel reports. Successfully io_cancel'd IOs are NOT
            // reported here and are reclaimed by CancelTask instead.
            if (iocbPtr) {
                Untrack(iocbPtr);
                delete iocbPtr;
            }
        }
        if (num < batchSize) { break; }
    }
}

void AioImpl::Track(uint64_t tag, struct iocb* cb)
{
    if (tag == 0) { return; }  // untracked: no cancellation support requested
    std::lock_guard<std::mutex> lk(tableMutex_);
    iocbTable_[tag].push_back(cb);
    iocbToTag_[cb] = tag;
}

void AioImpl::Untrack(struct iocb* cb)
{
    std::lock_guard<std::mutex> lk(tableMutex_);
    auto it = iocbToTag_.find(cb);
    if (it == iocbToTag_.end()) { return; }
    auto tag = it->second;
    iocbToTag_.erase(it);
    auto vit = iocbTable_.find(tag);
    if (vit == iocbTable_.end()) { return; }
    auto& vec = vit->second;
    vec.erase(std::remove(vec.begin(), vec.end(), cb), vec.end());
    if (vec.empty()) { iocbTable_.erase(vit); }
}

void AioImpl::CancelTask(uint64_t tag)
{
    // Collect successfully-cancelled iocbs under the lock (mutually exclusive with harvest's
    // Untrack and with submit's Track), then invoke their callbacks and reclaim them outside the
    // lock to avoid holding tableMutex_ across user callbacks. A successful io_cancel means the
    // kernel will NOT report the IO via io_getevents, so CancelTask becomes its sole reclaimer.
    std::vector<struct iocb*> cancelled;
    size_t uncancellable = 0;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        auto it = iocbTable_.find(tag);
        if (it == iocbTable_.end()) { return; }
        auto& vec = it->second;
        for (auto vi = vec.begin(); vi != vec.end();) {
            io_event ev{};
            auto r = AioCancel(ctx_, *vi, &ev);
            if (r == 0) {
                iocbToTag_.erase(*vi);
                cancelled.push_back(*vi);
                vi = vec.erase(vi);
            } else {
                ++vi;
                ++uncancellable;
            }
        }
        if (vec.empty()) { iocbTable_.erase(it); }
    }
    for (auto* cbp : cancelled) {
        auto* cb = reinterpret_cast<Callback*>(cbp->aio_data);
        if (cb) {
            (*cb)(Result{-1, ECANCELED});
            delete cb;
        }
        delete cbp;
    }
    if (!cancelled.empty() || uncancellable > 0) {
        UC_WARN("AIO cancel task({}): {} cancelled, {} in-flight/uncancellable.", tag,
                cancelled.size(), uncancellable);
    }
}

Status AioImpl::SubmitIo(iocb* cb)
{
    AioSetEventFd(cb, eventFd_);
    // The in-flight queue (queueDepth_) can stay full when earlier IOs are stuck on a dead disk.
    // Cap the EAGAIN retry so the submitting open-worker thread is not lost spinning forever; on
    // timeout the shard fails fast and the worker returns to serve new requests after recovery.
    const auto deadline = NowTime::Now() + static_cast<double>(submitTimeoutMs_) / 1000.0;
    for (;;) {
        auto ret = AioSubmit(ctx_, 1, &cb);
        auto eno = errno;
        if (ret == 1) { return Status::OK(); }
        if (eno == EAGAIN) {
            if (submitTimeoutMs_ > 0 && NowTime::Now() >= deadline) {
                UC_ERROR("io_submit EAGAIN for {}ms (in-flight queue saturated); giving up.",
                         submitTimeoutMs_);
                return Status::Timeout();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }
        return Status::Error(std::string(strerror(eno)));
    }
}

}  // namespace UC::PosixStore
