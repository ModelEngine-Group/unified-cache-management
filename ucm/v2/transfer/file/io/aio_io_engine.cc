/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <linux/aio_abi.h>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <system_error>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "fmt/format.h"
#include "io_engine.h"
#include "io_engine_registry.h"

namespace UC::Transfer {
namespace {

inline int32_t AioSetup(int32_t nEvents, aio_context_t* pCtx)
{
    return syscall(SYS_io_setup, nEvents, pCtx);
}

inline int32_t AioGetEvents(aio_context_t ctx, int64_t minNr, int64_t maxNr, io_event* events,
                            timespec* timeout)
{
    return syscall(SYS_io_getevents, ctx, minNr, maxNr, events, timeout);
}

inline int32_t AioSubmit(aio_context_t ctx, int64_t nr, iocb** ios)
{
    return syscall(SYS_io_submit, ctx, nr, ios);
}

inline int32_t AioDestroy(aio_context_t ctx) { return syscall(SYS_io_destroy, ctx); }

inline void AioPrepareRead(struct iocb* iocb, int32_t fd, void* buf, size_t count, size_t offset)
{
    memset(iocb, 0, sizeof(*iocb));
    iocb->aio_fildes = fd;
    iocb->aio_lio_opcode = 0; /* IO_CMD_PREAD */
    iocb->aio_reqprio = 0;
    iocb->aio_buf = reinterpret_cast<uintptr_t>(buf);
    iocb->aio_nbytes = count;
    iocb->aio_offset = offset;
}

inline void AioPrepareWrite(struct iocb* iocb, int32_t fd, void* buf, size_t count, size_t offset)
{
    memset(iocb, 0, sizeof(*iocb));
    iocb->aio_fildes = fd;
    iocb->aio_lio_opcode = 1; /* IO_CMD_PWRITE */
    iocb->aio_reqprio = 0;
    iocb->aio_buf = reinterpret_cast<uintptr_t>(buf);
    iocb->aio_nbytes = count;
    iocb->aio_offset = offset;
}

inline void AioSetEventFd(struct iocb* iocb, int32_t eventfd)
{
    iocb->aio_flags |= (1 << 0) /* IOCB_FLAG_RESFD */;
    iocb->aio_resfd = eventfd;
}

/**
 * @brief Kicks the completion loop by signaling the eventfd.
 *
 * An interrupted kick must be retried: losing it while nothing is in flight
 * would leave the completion loop blocked in epoll forever (no completion
 * and no submit would ever signal again).
 */
inline void AioKickEventFd(int32_t eventfd)
{
    uint64_t wake = 1;
    while (::write(eventfd, &wake, sizeof(wake)) < 0 && errno == EINTR) {}
}

/** Batch bound of the completion thread's reaping and topping-up syscalls. */
constexpr std::size_t kBatchSize = 64;

/**
 * @brief IoEngine over one Linux asynchronous I/O context (io_setup /
 *        io_submit / io_getevents): a submitted job is split into
 *        per-segment pread/pwrite requests that the kernel executes
 *        concurrently, bounded by the context's ring capacity
 *        (InitParam::ioDepth, the maximum number of in-flight kernel
 *        requests). Excess segments wait in the engine's pending queue and
 *        enter the ring as completions free it, so Submit neither blocks
 *        nor fails on capacity.
 *
 * A single completion thread owns the context: every request carries
 * IOCB_FLAG_RESFD, so the kernel signals the engine's eventfd on each
 * completion; the thread blocks in epoll on that eventfd (submissions kick
 * the same eventfd), reaps everything completed with io_getevents, and tops
 * the ring up from the pending queue.
 *
 * Segment errors (bad descriptor, I/O failure, short transfer) are not
 * diagnosed at submission: Submit* validates only the job shape and the
 * jobId lifecycle, and execution failures surface through the job's
 * completion event (GetFinished's success flag, Wait's error status; the
 * first failure detail is retained). The first failing segment cancels the
 * job's queued segments -- they never enter the kernel -- while its
 * in-flight requests cannot be cancelled and settle the job once they
 * complete.
 *
 * Job lifecycle: a successful submit enters jobId into the in-flight table,
 * and its completion is reported exactly once, via GetFinished or Wait;
 * consumption frees the id for reuse and removes the settled entry on the
 * spot, so no completion state outlives its report -- Wait-only callers
 * cannot accumulate settled entries. Completions are reported
 * first-settled-first.
 *
 * Thread contract: callers serialize the control plane (Setup / Submit* /
 * GetFinished / Wait) and keep job buffers alive until completion; Setup is
 * called at most once. The engine must be drained before destruction; the
 * destructor is a drain-and-join safety net regardless.
 */
class AioIoEngine : public IoEngine {
public:
    ~AioIoEngine() override;

    Status Setup(const InitParam& param) override;
    Status SubmitReadV(std::size_t jobId, const IoJob& job) override;
    Status SubmitWriteV(std::size_t jobId, const IoJob& job) override;
    Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() override;
    Status Wait(std::size_t jobId) override;

private:
    /**
     * One queued kernel request; the iocb's aio_data points back here, so a
     * completion maps to its job. Segments are immutable once the job is
     * submitted (their addresses must stay stable while the request flies).
     */
    struct JobEntry;

    struct Segment {
        struct iocb iocb;
        JobEntry* job;
    };

    struct JobEntry {
        std::size_t jobId = 0;
        bool write = false;
        std::vector<Segment> segments; /* immutable once the job is submitted */
        std::size_t remaining = 0;     /* segments not finished yet; 0 == settled */
        std::string error; /* first segment failure detail; empty == success once settled */
    };

    struct Task {
        std::shared_ptr<JobEntry> job;
        std::size_t segmentIndex;
    };

    /* control-plane helpers */
    Status Submit(std::size_t jobId, const IoJob& job, bool write);
    /** @return Ok, or the shape violation; a rejected jobId stays free. */
    static Status ValidateJobShape(std::size_t jobId, const IoJob& job);
    /** @brief Builds the job's table entry with its per-segment kernel requests. */
    std::shared_ptr<JobEntry> PrepareJobEntry(std::size_t jobId, const IoJob& job, bool write);
    /** @return Ok with both fds, or an error having rolled both back. */
    Status CreateCompletionChannel(int32_t* eventfd, int32_t* epollFd);

    /* completion-thread helpers */
    void CompletionLoop();
    void ReapCompletions(io_event* events, timespec* timeout);
    /** @return Empty on a full transfer; the failure detail otherwise. */
    static std::string DescribeResult(const io_event& event);
    void CompleteBatch(io_event* events, std::size_t count);
    /** @return True once stopping and fully drained: the completion loop exits. */
    bool TopUpRing(iocb** requests);
    void FailPendingSegments(const std::string& error);
    /** @brief Records entry's first failure and cancels its queued segments. Caller holds mutex_.
     */
    void FailJobLocked(JobEntry* entry, const std::string& error);
    /** @brief Counts one finished segment down, reporting the settled job. Caller holds mutex_. */
    bool SettleSegmentLocked(JobEntry* entry);

    void Stop();

    std::mutex mutex_;               /* guards every field below except worker_ */
    std::condition_variable cvDone_; /* wakes Wait() on settled jobs */
    std::deque<Task> pending_;       /* queued (unstarted) segments in submission order */
    /**
     * Settled-unconsumed {jobId, succeeded} completions in settle order; one
     * element per jobs_ entry, so a jobId appears at most once here -- both
     * consumption paths remove their element.
     */
    std::deque<std::pair<std::size_t, bool>> doneOrder_;
    std::unordered_map<std::size_t, std::shared_ptr<JobEntry>> jobs_; /* in flight + settled */
    std::thread worker_; /* the completion thread; not touched by anyone but Setup/Stop */
    aio_context_t ctx_ = 0;
    int32_t eventfd_ = -1;
    int32_t epollFd_ = -1;
    std::size_t ringCapacity_ = 0; /* == InitParam::ioDepth; bounds inFlight_ */
    std::size_t inFlight_ = 0;     /* submitted and not yet reaped == ring occupancy */
    bool stopping_ = false;
};

AioIoEngine::~AioIoEngine() { Stop(); }

Status AioIoEngine::Setup(const InitParam& param)
{
    if (param.ioDepth == 0) { return Status::InvalidParam("io depth must be greater than 0"); }
    if (param.ioDepth > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        return Status::InvalidParam("io depth must not exceed {}",
                                    std::numeric_limits<int32_t>::max());
    }
    aio_context_t ctx = 0;
    if (AioSetup(static_cast<int32_t>(param.ioDepth), &ctx) < 0) {
        return Status::OsApiError("io_setup(depth={}) failed: {}", param.ioDepth,
                                  std::strerror(errno));
    }
    int32_t eventfd = -1;
    int32_t epollFd = -1;
    const Status channel = CreateCompletionChannel(&eventfd, &epollFd);
    if (!channel.Success()) {
        AioDestroy(ctx);
        return channel;
    }
    /* members are published before the completion thread starts reading them */
    ctx_ = ctx;
    eventfd_ = eventfd;
    epollFd_ = epollFd;
    ringCapacity_ = param.ioDepth;
    try {
        worker_ = std::thread([this] { CompletionLoop(); });
    } catch (const std::system_error& error) {
        ctx_ = 0;
        eventfd_ = -1;
        epollFd_ = -1;
        ringCapacity_ = 0;
        AioDestroy(ctx);
        ::close(eventfd);
        ::close(epollFd);
        return Status::OsApiError("spawning io completion worker failed: {}", error.what());
    }
    return Status::Ok();
}

Status AioIoEngine::CreateCompletionChannel(int32_t* eventfd, int32_t* epollFd)
{
    *eventfd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (*eventfd < 0) { return Status::OsApiError("eventfd() failed: {}", std::strerror(errno)); }
    *epollFd = ::epoll_create1(EPOLL_CLOEXEC);
    if (*epollFd < 0) {
        const int32_t reason = errno;
        ::close(*eventfd);
        return Status::OsApiError("epoll_create1() failed: {}", std::strerror(reason));
    }
    epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = *eventfd;
    if (::epoll_ctl(*epollFd, EPOLL_CTL_ADD, *eventfd, &event) < 0) {
        const int32_t reason = errno;
        ::close(*epollFd);
        ::close(*eventfd);
        return Status::OsApiError("epoll_ctl(add) failed: {}", std::strerror(reason));
    }
    return Status::Ok();
}

Status AioIoEngine::SubmitReadV(std::size_t jobId, const IoJob& job)
{
    return Submit(jobId, job, false);
}

Status AioIoEngine::SubmitWriteV(std::size_t jobId, const IoJob& job)
{
    return Submit(jobId, job, true);
}

Expected<std::vector<std::pair<std::size_t, bool>>> AioIoEngine::GetFinished()
{
    std::vector<std::pair<std::size_t, bool>> finished;
    const std::lock_guard<std::mutex> lock(mutex_);
    finished.reserve(doneOrder_.size());
    while (!doneOrder_.empty()) {
        finished.push_back(std::move(doneOrder_.front()));
        doneOrder_.pop_front();
        jobs_.erase(finished.back().first);
    }
    return finished;
}

Status AioIoEngine::Wait(std::size_t jobId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = jobs_.find(jobId);
    if (it == jobs_.end()) { return Status::InvalidParam("job {} is not in flight", jobId); }
    const std::shared_ptr<JobEntry> entry = it->second;
    cvDone_.wait(lock, [&entry] { return entry->remaining == 0; });
    jobs_.erase(jobId);
    const auto settled = std::find_if(doneOrder_.begin(), doneOrder_.end(),
                                      [jobId](const std::pair<std::size_t, bool>& candidate) {
                                          return candidate.first == jobId;
                                      });
    if (settled != doneOrder_.end()) { doneOrder_.erase(settled); }
    return entry->error.empty() ? Status::Ok() : Status::OsApiError("{}", entry->error);
}

Status AioIoEngine::Submit(std::size_t jobId, const IoJob& job, bool write)
{
    const Status valid = ValidateJobShape(jobId, job);
    if (!valid.Success()) { return valid; }

    const std::shared_ptr<JobEntry> entry = PrepareJobEntry(jobId, job, write);

    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (jobs_.find(jobId) != jobs_.end()) {
            return Status::DuplicateKey("job {} is already in flight", jobId);
        }
        jobs_.emplace(jobId, entry);
        std::size_t queued = 0;
        try {
            for (; queued < entry->segments.size(); ++queued) {
                pending_.push_back(Task{entry, queued});
            }
        } catch (const std::bad_alloc&) {
            /**
             * A segment task left unqueued would keep the job unsettled forever
             * (and its id permanently taken), so undo the partial registration.
             * */
            for (std::size_t i = 0; i < queued; ++i) { pending_.pop_back(); }
            jobs_.erase(jobId);
            return Status::OutOfMemory("enqueueing segment tasks of job {} failed", jobId);
        }
    }
    AioKickEventFd(eventfd_); /* kick the completion loop */
    return Status::Ok();
}

Status AioIoEngine::ValidateJobShape(std::size_t jobId, const IoJob& job)
{
    const std::size_t segmentCount = job.fileHandles.size();
    if (segmentCount == 0 || job.offsets.size() != segmentCount ||
        job.lengths.size() != segmentCount || job.addresses.size() != segmentCount) {
        return Status::InvalidParam(
            "job {} column vectors must be non-empty and of equal length ({} segments)", jobId,
            segmentCount);
    }
    for (std::size_t i = 0; i < segmentCount; ++i) {
        if (job.addresses[i] == nullptr) {
            return Status::InvalidParam("job {} segment {} has a null address", jobId, i);
        }
        if (job.lengths[i] == 0) {
            return Status::InvalidParam("job {} segment {} has zero length", jobId, i);
        }
    }
    return Status::Ok();
}

std::shared_ptr<AioIoEngine::JobEntry> AioIoEngine::PrepareJobEntry(std::size_t jobId,
                                                                    const IoJob& job, bool write)
{
    auto entry = std::make_shared<JobEntry>();
    entry->jobId = jobId;
    entry->write = write;
    entry->remaining = job.fileHandles.size();
    entry->segments.resize(job.fileHandles.size());
    for (std::size_t i = 0; i < entry->segments.size(); ++i) {
        Segment& segment = entry->segments[i];
        segment.job = entry.get();
        if (write) {
            AioPrepareWrite(&segment.iocb, job.fileHandles[i], job.addresses[i], job.lengths[i],
                            job.offsets[i]);
        } else {
            AioPrepareRead(&segment.iocb, job.fileHandles[i], job.addresses[i], job.lengths[i],
                           job.offsets[i]);
        }
        AioSetEventFd(&segment.iocb, eventfd_);
        segment.iocb.aio_data = reinterpret_cast<uintptr_t>(&segment);
    }
    return entry;
}

void AioIoEngine::CompletionLoop()
{
    struct io_event events[kBatchSize];
    struct iocb* requests[kBatchSize];
    timespec zeroTimeout{0, 0};

    for (;;) {
        /* block until a completion, a submit kick, or the stop wakes the eventfd */
        epoll_event wake{};
        int32_t woken = 0;
        do {
            woken = ::epoll_wait(epollFd_, &wake, 1, -1);
        } while (woken < 0 && errno == EINTR);
        if (woken <= 0) { return; } /* the epoll fd is gone: destruction in progress */
        uint64_t signals = 0;
        /* clears the counter; EINTR must be retried or the wake is lost */
        while (::read(eventfd_, &signals, sizeof(signals)) < 0 && errno == EINTR) {}

        ReapCompletions(events, &zeroTimeout);
        if (TopUpRing(requests)) { return; } /* stopping and fully drained */
    }
}

void AioIoEngine::ReapCompletions(io_event* events, timespec* timeout)
{
    for (;;) {
        const int32_t reaped =
            AioGetEvents(ctx_, 0, static_cast<int64_t>(kBatchSize), events, timeout);
        if (reaped < 0) {
            if (errno == EINTR) { continue; }
            break; /* reap errors are impossible with a valid context */
        }
        if (reaped == 0) { break; }
        CompleteBatch(events, static_cast<std::size_t>(reaped));
    }
}

std::string AioIoEngine::DescribeResult(const io_event& event)
{
    Segment* const segment = reinterpret_cast<Segment*>(event.data);
    const long moved = static_cast<long>(event.res);
    const long wanted = static_cast<long>(segment->iocb.aio_nbytes);
    if (moved == wanted) { return {}; }
    if (moved < 0) {
        return fmt::format("{}(fd={}, offset={}, length={}) failed: {}",
                           segment->job->write ? "pwrite" : "pread",
                           static_cast<int32_t>(segment->iocb.aio_fildes), segment->iocb.aio_offset,
                           segment->iocb.aio_nbytes, std::strerror(-moved));
    }
    return fmt::format("{}(fd={}, offset={}, length={}) stopped after {} bytes at end of file",
                       segment->job->write ? "pwrite" : "pread",
                       static_cast<int32_t>(segment->iocb.aio_fildes), segment->iocb.aio_offset,
                       segment->iocb.aio_nbytes, moved);
}

void AioIoEngine::CompleteBatch(io_event* events, std::size_t count)
{
    std::array<std::string, kBatchSize> failures;
    for (std::size_t i = 0; i < count; ++i) { failures[i] = DescribeResult(events[i]); }

    bool settled = false;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < count; ++i) {
            Segment* const segment = reinterpret_cast<Segment*>(events[i].data);
            --inFlight_;
            if (!failures[i].empty()) { FailJobLocked(segment->job, failures[i]); }
            settled = SettleSegmentLocked(segment->job) || settled;
        }
    }
    if (settled) { cvDone_.notify_all(); }
}

bool AioIoEngine::TopUpRing(iocb** requests)
{
    for (;;) {
        std::size_t batch = 0;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ && pending_.empty()) { return true; } /* stopping and fully drained */
            batch = std::min(std::min(pending_.size(), ringCapacity_ - inFlight_), kBatchSize);
            if (batch == 0) { break; } /* ring full: the next completion re-kicks */
            for (std::size_t i = 0; i < batch; ++i) {
                requests[i] = &pending_[i].job->segments[pending_[i].segmentIndex].iocb;
            }
        }
        const int32_t submitted = AioSubmit(ctx_, static_cast<int64_t>(batch), requests);
        if (submitted >= 0) {
            const std::lock_guard<std::mutex> lock(mutex_);
            inFlight_ += static_cast<std::size_t>(submitted);
            pending_.erase(pending_.begin(),
                           pending_.begin() + static_cast<std::ptrdiff_t>(submitted));
            if (static_cast<std::size_t>(submitted) < batch) { break; } /* ring now full */
            continue;
        }
        const int32_t reason = errno;
        /**
         * Interrupted before submitting anything: retry in place instead of
         * waiting for a wake -- with nothing in flight no completion ever
         * signals, so the loop (and Wait) would block forever.
         * */
        if (reason == EINTR) { continue; }
        if (reason == EAGAIN) { break; } /* ring full: the next completion re-kicks */
        /**
         * A hard submit error (impossible with a valid context) fails every
         * queued segment right here; the in-flight requests still complete
         * and settle their jobs, so the accounting stays exact.
         * */
        FailPendingSegments(fmt::format("io_submit failed: {}", std::strerror(reason)));
        break;
    }
    return false;
}

void AioIoEngine::FailPendingSegments(const std::string& error)
{
    bool settled = false;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty()) {
            JobEntry* const entry = pending_.front().job.get();
            pending_.pop_front();
            FailJobLocked(entry, error);
            settled = SettleSegmentLocked(entry) || settled;
        }
    }
    if (settled) { cvDone_.notify_all(); }
}

void AioIoEngine::FailJobLocked(JobEntry* entry, const std::string& error)
{
    if (!entry->error.empty()) { return; }
    entry->error = error;
    /**
     * First failure cancels the job's queued (unstarted) segments: each
     * cancelled segment settles one remaining unit here, so the countdown
     * reaches zero exactly when the last in-flight request (kernel I/O
     * cannot be interrupted) finishes. Popped tasks are never cancelled.
     * */
    const auto newEnd =
        std::remove_if(pending_.begin(), pending_.end(),
                       [entry](const Task& candidate) { return candidate.job.get() == entry; });
    entry->remaining -= static_cast<std::size_t>(std::distance(newEnd, pending_.end()));
    pending_.erase(newEnd, pending_.end());
}

bool AioIoEngine::SettleSegmentLocked(JobEntry* entry)
{
    if (--entry->remaining != 0) { return false; }
    doneOrder_.emplace_back(entry->jobId, entry->error.empty());
    return true;
}

void AioIoEngine::Stop()
{
    if (worker_.joinable()) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        AioKickEventFd(eventfd_);
        worker_.join();
    }
    if (ctx_ != 0) { AioDestroy(ctx_); } /* cancels whatever is still in flight */
    if (eventfd_ >= 0) { ::close(eventfd_); }
    if (epollFd_ >= 0) { ::close(epollFd_); }
}

UC_TRANSFER_REGISTER_IO_ENGINE(AioIoEngine, IoEngineKind::Aio);

}  // namespace
}  // namespace UC::Transfer
