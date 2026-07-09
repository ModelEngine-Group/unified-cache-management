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
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "logger/logger.h"
#include "thread/cpu_affinity.h"
#include "time/now_time.h"

namespace UC::PosixStore {

#ifdef UCM_ENABLE_TEST_HOOKS
namespace {
std::mutex& AioHookMutex()
{
    static std::mutex mutex;
    return mutex;
}
TestHooks::AioSubmitHook& AioSubmitHookSlot()
{
    static TestHooks::AioSubmitHook hook;
    return hook;
}
TestHooks::AioCancelHook& AioCancelHookSlot()
{
    static TestHooks::AioCancelHook hook;
    return hook;
}
}  // namespace

namespace TestHooks {
void SetAioSubmitHook(AioSubmitHook hook)
{
    std::lock_guard<std::mutex> lock{AioHookMutex()};
    AioSubmitHookSlot() = std::move(hook);
}
void SetAioCancelHook(AioCancelHook hook)
{
    std::lock_guard<std::mutex> lock{AioHookMutex()};
    AioCancelHookSlot() = std::move(hook);
}
void ClearAioHooks()
{
    std::lock_guard<std::mutex> lock{AioHookMutex()};
    AioSubmitHookSlot() = nullptr;
    AioCancelHookSlot() = nullptr;
}
}  // namespace TestHooks

static TestHooks::AioSubmitHook GetAioSubmitHook()
{
    std::lock_guard<std::mutex> lock{AioHookMutex()};
    return AioSubmitHookSlot();
}
static TestHooks::AioCancelHook GetAioCancelHook()
{
    std::lock_guard<std::mutex> lock{AioHookMutex()};
    return AioCancelHookSlot();
}
#endif

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
#ifdef UCM_ENABLE_TEST_HOOKS
    auto hook = GetAioSubmitHook();
    if (hook) { return hook(ctx, nr, ios); }
#endif
    return syscall(SYS_io_submit, ctx, nr, ios);
}
static inline int32_t AioCancel(aio_context_t ctx, struct iocb* cb, io_event* result)
{
#ifdef UCM_ENABLE_TEST_HOOKS
    auto hook = GetAioCancelHook();
    if (hook) { return hook(ctx, cb, result); }
#endif
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
static inline void ReleaseToAioCompletion(std::unique_ptr<struct iocb>& cb,
                                          std::unique_ptr<AioImpl::Callback>& data)
{
    data.release();
    cb.release();
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
    std::lock_guard<std::mutex> lk(tableMutex_);
    if (!iocbToTag_.empty()) {
        UC_WARN("AIO teardown: {} in-flight IO(s) abandoned (never completed; not reclaimed).",
                iocbToTag_.size());
    }
}

Status AioImpl::Setup(size_t timeoutMs)
{
    constexpr size_t defaultSweepIntervalMs = 100;
    constexpr size_t defaultEpollTimeoutMs = 10;
    sweepIntervalMs_ = defaultSweepIntervalMs;
    epollTimeoutMs_ = defaultEpollTimeoutMs;
    submitTimeoutMs_ = timeoutMs;
    UC_INFO(
        "AIO setup: queueDepth={}, epollTimeoutMs={}, sweepIntervalMs={}, "
        "submitTimeoutMs={}.",
        queueDepth_, epollTimeoutMs_, sweepIntervalMs_, submitTimeoutMs_);
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
    auto cb = std::make_unique<struct iocb>();
    auto data = std::make_unique<Callback>(std::move(io.callback));
    AioPrepareRead(cb.get(), io.fd, io.buffer, io.length, io.offset);
    cb->aio_data = (uintptr_t)(void*)data.get();
    Track(io.tag, cb.get());
    auto status = SubmitIo(cb.get());
    if (status.Failure()) {
        Untrack(cb.get());
        UC_ERROR("Failed({}) to submit read io.", status);
        return status;
    }
    ReleaseToAioCompletion(cb, data);
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
        return status;
    }
    ReleaseToAioCompletion(cb, data);
    return Status::OK();
}

void AioImpl::CompletionLoop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_posix_aio");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM posix AIO thread name.", nameStatus);
    }
    std::vector<epoll_event> epollEvents(128);
    std::vector<io_event> aioEvents(batchCompleteSize);
    while (!stop_) {
        auto nfds = epoll_wait(epollFd_, epollEvents.data(), epollEvents.size(), epollTimeoutMs_);
        for (auto i = 0; i < nfds; i++) {
            if (epollEvents[i].data.ptr == nullptr) {
                uint64_t count;
                auto ret = read(eventFd_, &count, sizeof(count));
                auto eno = errno;
                if (ret < 0 && eno != EAGAIN) { UC_WARN("Failed({}) to read aio eventfd.", eno); }
                HarvestCompletions(aioEvents);
            }
        }
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
    if (tag == 0) { return; }
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
    std::vector<struct iocb*> cancelled;
    size_t uncancellable = 0;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        auto it = iocbTable_.find(tag);
        if (it == iocbTable_.end()) {
            UC_DEBUG("AIO cancel task({}): no tracked in-flight IO.", tag);
            return;
        }
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
