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
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dma_engine.h"
#include "dma_engine_registry.h"

namespace UC::Transfer {
namespace {

/**
 * @brief DmaEngine of the simu runtime: device pointers are ordinary host
 *        pointers, so a submitted batch degrades into std::memcpy calls,
 *        executed asynchronously by a single FIFO worker thread.
 *
 * The single worker settles jobs one at a time, in submission order --
 * like a single device stream, which in the simulated environment also
 * subsumes DmaJob::prerequisite ordering (the field is read and
 * ignored). Submission therefore returns before the copy lands, giving
 * callers a genuine in-flight window to poll, wait and synchronize
 * against. std::memcpy cannot fail, so every job settles successfully
 * and the engine carries no failure or cancellation path.
 *
 * Job lifecycle: a successful submit enters jobId into the in-flight
 * table, and its completion is reported exactly once, via GetFinished
 * or Wait; consumption frees the id for reuse and removes the settled
 * entry on the spot, so no completion state outlives its report --
 * Wait-only callers cannot accumulate settled entries. Completions
 * are reported first-settled-first.
 *
 * Lifecycle conventions: the upper layer calls Setup exactly once and
 * touches the other entry points only after it returns Ok. This
 * simulation validates both conventions solely to expose violations
 * while supporting tests -- they fail loudly as Unsupported instead
 * of exercising undefined engine behavior; real engine implementations
 * treat the conventions as invariants that always hold and may omit
 * these redundant checks.
 *
 * Thread contract: callers serialize the control plane (Setup / Submit* /
 * GetFinished / Wait / Synchronize) and keep job buffers alive until
 * completion. The engine must be drained before destruction; the
 * destructor is a drain-and-join safety net regardless.
 */
class SimuDmaEngine : public DmaEngine {
public:
    ~SimuDmaEngine() override;

    Status Setup(const InitParam& param) override;
    Status SubmitH2DBatch(std::size_t jobId, const DmaJob& job) override;
    Status SubmitD2HBatch(std::size_t jobId, const DmaJob& job) override;
    Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() override;
    Status Wait(std::size_t jobId) override;
    Status Synchronize() override;

private:
    struct Segment {
        const void* source;
        void* destination;
        std::size_t size;
    };

    struct JobEntry {
        std::size_t jobId = 0;
        std::vector<Segment> segments; /* immutable once the job is submitted */
        bool settled = false;          /* every segment copied; awaiting report */
    };

    Status Submit(std::size_t jobId, const DmaJob& job);
    void WorkerLoop();
    void Stop();

    std::mutex mutex_;               /* guards every field below except worker_ */
    std::condition_variable cvWork_; /* wakes the worker on new jobs or stop */
    std::condition_variable cvDone_; /* wakes Wait/Synchronize on settlements */
    std::deque<std::shared_ptr<JobEntry>> pending_;
    /**
     * Settled-unconsumed {jobId, succeeded} completions in settle order; one
     * element per jobs_ entry, so a jobId appears at most once here -- both
     * consumption paths remove their element.
     */
    std::deque<std::pair<std::size_t, bool>> doneOrder_;
    std::unordered_map<std::size_t, std::shared_ptr<JobEntry>> jobs_; /* in flight + settled */
    std::size_t unsettled_ = 0; /* submitted and not settled yet -- Synchronize's wait condition */
    std::thread worker_;
    bool stopping_ = false;
    std::size_t maxLength_ = 0;
    bool setup_ = false;
};

SimuDmaEngine::~SimuDmaEngine() { Stop(); }

Status SimuDmaEngine::Setup(const InitParam& param)
{
    /* simulation-only convention check (see class doc): Setup never re-enters */
    if (setup_) {
        return Status::Unsupported("setup re-entry is not supported: dma engine is already set up");
    }
    if (param.maxLength == 0) { return Status::InvalidParam("max length must be greater than 0"); }
    try {
        worker_ = std::thread([this] { WorkerLoop(); });
    } catch (const std::system_error& error) {
        return Status::OsApiError("spawning dma worker failed: {}", error.what());
    }
    maxLength_ = param.maxLength;
    setup_ = true;
    return Status::Ok();
}

Status SimuDmaEngine::SubmitH2DBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job);
}

Status SimuDmaEngine::SubmitD2HBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job);
}

Expected<std::vector<std::pair<std::size_t, bool>>> SimuDmaEngine::GetFinished()
{
    /* simulation-only convention check (see class doc): nothing runs before a successful Setup */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
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

Status SimuDmaEngine::Wait(std::size_t jobId)
{
    /* simulation-only convention check (see class doc): nothing runs before a successful Setup */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = jobs_.find(jobId);
    if (it == jobs_.end()) { return Status::InvalidParam("job {} is not in flight", jobId); }
    const std::shared_ptr<JobEntry> entry = it->second;
    cvDone_.wait(lock, [&entry] { return entry->settled; });
    jobs_.erase(jobId);
    const auto settled = std::find_if(doneOrder_.begin(), doneOrder_.end(),
                                      [jobId](const std::pair<std::size_t, bool>& candidate) {
                                          return candidate.first == jobId;
                                      });
    if (settled != doneOrder_.end()) { doneOrder_.erase(settled); }
    return Status::Ok(); /* memcpy cannot fail, so a settled job always succeeded */
}

Status SimuDmaEngine::Synchronize()
{
    /* simulation-only convention check (see class doc): nothing runs before a successful Setup */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
    std::unique_lock<std::mutex> lock(mutex_);
    cvDone_.wait(lock, [this] { return unsettled_ == 0; });
    return Status::Ok();
}

Status SimuDmaEngine::Submit(std::size_t jobId, const DmaJob& job)
{
    /* simulation-only convention check (see class doc): nothing runs before a successful Setup */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
    const std::size_t segmentCount = job.sizes.size();
    if (segmentCount == 0 || job.sources.size() != segmentCount ||
        job.destinations.size() != segmentCount) {
        return Status::InvalidParam(
            "job {} column vectors must be non-empty and of equal length ({} segments)", jobId,
            segmentCount);
    }
    /* the optional grouping must describe the same batch */
    std::size_t grouped = 0;
    for (std::size_t i = 0; i < job.nSegmentPerGroup.size(); ++i) {
        if (job.nSegmentPerGroup[i] == 0) {
            return Status::InvalidParam("job {} group {} has zero segments", jobId, i);
        }
        if (job.nSegmentPerGroup[i] > segmentCount - grouped) {
            return Status::InvalidParam("job {} groups describe more than the {} segments", jobId,
                                        segmentCount);
        }
        grouped += job.nSegmentPerGroup[i];
    }
    if (!job.nSegmentPerGroup.empty() && grouped != segmentCount) {
        return Status::InvalidParam("job {} groups describe only {} of the {} segments", jobId,
                                    grouped, segmentCount);
    }
    for (std::size_t i = 0; i < segmentCount; ++i) {
        if (job.sources[i] == nullptr || job.destinations[i] == nullptr) {
            return Status::InvalidParam("job {} segment {} has a null address", jobId, i);
        }
        if (job.sizes[i] == 0) {
            return Status::InvalidParam("job {} segment {} has zero length", jobId, i);
        }
        if (job.sizes[i] > maxLength_) {
            return Status::InvalidParam("job {} segment {} exceeds the max length {}", jobId, i,
                                        maxLength_);
        }
    }

    auto entry = std::make_shared<JobEntry>();
    entry->jobId = jobId;
    entry->segments.reserve(segmentCount);
    for (std::size_t i = 0; i < segmentCount; ++i) {
        entry->segments.push_back(Segment{job.sources[i], job.destinations[i], job.sizes[i]});
    }

    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (jobs_.find(jobId) != jobs_.end()) {
            return Status::DuplicateKey("job {} is already in flight", jobId);
        }
        bool registered = false;
        try {
            jobs_.emplace(jobId, entry);
            registered = true;
            pending_.push_back(entry);
        } catch (const std::bad_alloc&) {
            /**
             * A job left half-registered would keep its id permanently taken
             * (and its segments forever unsettled), so undo the partial state.
             * */
            if (registered) { jobs_.erase(jobId); }
            return Status::OutOfMemory("enqueueing job {} failed", jobId);
        }
        ++unsettled_;
    }
    cvWork_.notify_one();
    return Status::Ok();
}

void SimuDmaEngine::WorkerLoop()
{
    for (;;) {
        std::shared_ptr<JobEntry> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cvWork_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (pending_.empty()) { return; } /* stopping_ and fully drained */
            job = std::move(pending_.front());
            pending_.pop_front();
        }

        for (const Segment& segment : job->segments) {
            std::memcpy(segment.destination, segment.source, segment.size);
        }

        {
            const std::lock_guard<std::mutex> lock(mutex_);
            job->settled = true;
            doneOrder_.emplace_back(job->jobId, true);
            --unsettled_;
        }
        cvDone_.notify_all();
    }
}

void SimuDmaEngine::Stop()
{
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cvWork_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

UC_TRANSFER_REGISTER_DMA_ENGINE(SimuDmaEngine, DmaEngineKind::Naive);

}  // namespace
}  // namespace UC::Transfer
