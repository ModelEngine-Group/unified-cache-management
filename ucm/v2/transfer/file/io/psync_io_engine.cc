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
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "fmt/format.h"
#include "io_engine.h"
#include "io_engine_registry.h"

namespace UC::Transfer {
namespace {

/**
 * @brief IoEngine over a fixed thread pool of positional reads and writes:
 *        a submitted job is split into per-segment pread/pwrite tasks, so
 *        both multiple jobs and the segments of a single job execute
 *        concurrently, bounded by the pool size (InitParam::ioDepth).
 *
 * Segment errors (bad descriptor, I/O failure, short transfer) are not
 * diagnosed at submission: Submit* validates only the job shape and the
 * jobId lifecycle, and execution failures surface through the job's
 * completion event (GetFinished's success flag, Wait's error status;
 * the first failure detail is retained). The first failing segment
 * cancels the job's queued segments -- they never execute -- and the
 * job settles once its in-flight segments finish.
 *
 * Job lifecycle: a successful submit enters jobId into the in-flight
 * table, and its completion is reported exactly once, via GetFinished
 * or Wait; consumption frees the id for reuse and removes the settled
 * entry on the spot, so no completion state outlives its report --
 * Wait-only callers cannot accumulate settled entries. Completions
 * are reported first-settled-first.
 *
 * Thread contract: callers serialize the control plane (Setup / Submit* /
 * GetFinished / Wait) and keep job buffers alive until completion. The
 * engine must be drained before destruction; the destructor is a
 * drain-and-join safety net regardless.
 */
class PsyncIoEngine : public IoEngine {
public:
    ~PsyncIoEngine() override;

    Status Setup(const InitParam& param) override;
    Status SubmitReadV(std::size_t jobId, const IoJob& job) override;
    Status SubmitWriteV(std::size_t jobId, const IoJob& job) override;
    Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() override;
    Status Wait(std::size_t jobId) override;

private:
    struct Segment {
        int32_t fd;
        std::size_t offset;
        std::size_t length;
        void* address;
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

    Status Submit(std::size_t jobId, const IoJob& job, bool write);
    void WorkerLoop();
    static std::string ExecuteSegment(bool write, const Segment& segment);
    void Stop(std::vector<std::thread>& workers);

    std::mutex mutex_;               /* guards every field below except workers_ */
    std::condition_variable cvWork_; /* wakes idle workers on new tasks or stop */
    std::condition_variable cvDone_; /* wakes Wait() on settled jobs */
    std::deque<Task> pending_;
    /**
     * Settled-unconsumed {jobId, succeeded} completions in settle order; one
     * element per jobs_ entry, so a jobId appears at most once here -- both
     * consumption paths remove their element.
     */
    std::deque<std::pair<std::size_t, bool>> doneOrder_;
    std::unordered_map<std::size_t, std::shared_ptr<JobEntry>> jobs_; /* in flight + settled */
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

PsyncIoEngine::~PsyncIoEngine() { Stop(workers_); }

Status PsyncIoEngine::Setup(const InitParam& param)
{
    if (param.ioDepth == 0) { return Status::InvalidParam("io depth must be greater than 0"); }
    std::vector<std::thread> workers;
    workers.reserve(param.ioDepth);
    for (std::size_t i = 0; i < param.ioDepth; ++i) {
        try {
            workers.emplace_back([this] { WorkerLoop(); });
        } catch (const std::system_error& error) {
            Stop(workers);
            return Status::OsApiError("spawning io worker {} failed: {}", i, error.what());
        }
    }
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        workers_ = std::move(workers);
    }
    return Status::Ok();
}

Status PsyncIoEngine::SubmitReadV(std::size_t jobId, const IoJob& job)
{
    return Submit(jobId, job, false);
}

Status PsyncIoEngine::SubmitWriteV(std::size_t jobId, const IoJob& job)
{
    return Submit(jobId, job, true);
}

Expected<std::vector<std::pair<std::size_t, bool>>> PsyncIoEngine::GetFinished()
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

Status PsyncIoEngine::Wait(std::size_t jobId)
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

Status PsyncIoEngine::Submit(std::size_t jobId, const IoJob& job, bool write)
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

    auto entry = std::make_shared<JobEntry>();
    entry->jobId = jobId;
    entry->write = write;
    entry->remaining = segmentCount;
    entry->segments.reserve(segmentCount);
    for (std::size_t i = 0; i < segmentCount; ++i) {
        entry->segments.push_back(
            Segment{job.fileHandles[i], job.offsets[i], job.lengths[i], job.addresses[i]});
    }

    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (jobs_.find(jobId) != jobs_.end()) {
            return Status::DuplicateKey("job {} is already in flight", jobId);
        }
        jobs_.emplace(jobId, entry);
        std::size_t queued = 0;
        try {
            for (; queued < segmentCount; ++queued) { pending_.push_back(Task{entry, queued}); }
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
    cvWork_.notify_all();
    return Status::Ok();
}

void PsyncIoEngine::WorkerLoop()
{
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cvWork_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (pending_.empty()) { return; } /* stopping_ and fully drained */
            task = std::move(pending_.front());
            pending_.pop_front();
        }

        const std::string error =
            ExecuteSegment(task.job->write, task.job->segments[task.segmentIndex]);

        bool settled = false;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!error.empty() && task.job->error.empty()) {
                task.job->error = error;
                /**
                 * First failure cancels the job's queued (unstarted) segments: each
                 * cancelled segment settles one remaining unit here, so the countdown
                 * reaches zero exactly when the last in-flight segment (pread/pwrite
                 * cannot be interrupted) finishes. Popped tasks are never cancelled.
                 * */
                const auto newEnd = std::remove_if(
                    pending_.begin(), pending_.end(),
                    [&job = task.job](const Task& candidate) { return candidate.job == job; });
                task.job->remaining -=
                    static_cast<std::size_t>(std::distance(newEnd, pending_.end()));
                pending_.erase(newEnd, pending_.end());
            }
            settled = (--task.job->remaining == 0);
            if (settled) { doneOrder_.emplace_back(task.job->jobId, task.job->error.empty()); }
        }
        if (settled) { cvDone_.notify_all(); }
    }
}

std::string PsyncIoEngine::ExecuteSegment(bool write, const Segment& segment)
{
    char* const base = static_cast<char*>(segment.address);
    std::size_t done = 0;
    while (done < segment.length) {
        const ssize_t moved = write ? ::pwrite(segment.fd, base + done, segment.length - done,
                                               static_cast<off_t>(segment.offset + done))
                                    : ::pread(segment.fd, base + done, segment.length - done,
                                              static_cast<off_t>(segment.offset + done));
        if (moved < 0) {
            if (errno == EINTR) { continue; }
            return fmt::format("{}(fd={}, offset={}, length={}) failed: {}",
                               write ? "pwrite" : "pread", segment.fd, segment.offset + done,
                               segment.length - done, std::strerror(errno));
        }
        if (moved == 0) {
            return fmt::format(
                "{}(fd={}, offset={}, length={}) stopped after {} bytes at end of file",
                write ? "pwrite" : "pread", segment.fd, segment.offset, segment.length, done);
        }
        done += static_cast<std::size_t>(moved);
    }
    return {};
}

void PsyncIoEngine::Stop(std::vector<std::thread>& workers)
{
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cvWork_.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) { worker.join(); }
    }
}

UC_TRANSFER_REGISTER_IO_ENGINE(PsyncIoEngine, IoEngineKind::Psync);

}  // namespace
}  // namespace UC::Transfer
