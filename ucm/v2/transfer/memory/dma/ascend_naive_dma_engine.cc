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
#include <acl/acl_rt.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dma_engine.h"
#include "dma_engine_registry.h"

namespace UC::Transfer {
namespace {

/** @brief Event flavor of the engine: a pure stream-progress marker. */
constexpr uint32_t kEventFlags = ACL_EVENT_CAPTURE_STREAM_PROGRESS;

/**
 * @brief Naive DmaEngine of the Ascend runtime: a round-robin pool of
 *        private streams per direction -- H2D and D2H each on their
 *        own -- one recorded event per (submission, stream), no
 *        staging, no engine threads.
 *
 * Setup creates InitParam::nStream streams per direction -- 0 (unset)
 * keeps each direction single-stream. A submission maps its batch onto
 * its direction's pool: each DmaJob::nSegmentPerGroup group (the whole
 * batch when the field is empty) takes the next stream of that
 * direction's group-granular round-robin cursor, preceded by an
 * optional aclrtStreamWaitEvent for DmaJob::prerequisite (the handle
 * is an aclrtEvent) and followed, once the whole batch is enqueued, by
 * one closing aclrtRecordEvent per touched stream. The two directions
 * execute concurrently, with no order between them, and GetFinished
 * walks their settled prefixes back to back, the h2d one first; within
 * one direction a stream executes its queue in order, so completions
 * are reported in submission order. Concurrent groups -- of one
 * submission or of several -- must not alias.
 *
 * Host memory: a fixed host region (InitParam::hostBase/hostLength)
 * is page-locked once at Setup, best effort -- success makes both
 * directions asynchronous DMA over the region. Without a region, or
 * when the registration is refused, the engine runs with pageable
 * semantics: H2D submissions return once the runtime has staged the
 * source, D2H submissions once the copy has landed. The pinning is an
 * optimization, never a correctness condition: no registration
 * outcome fails Setup, and what this engine did not register it
 * never unregisters.
 *
 * Failure semantics: an ACL call failing mid-submission aborts the
 * submission -- every stream the submission touched is drained first,
 * because earlier segments may already be in flight and the caller may
 * release its buffers right after the error returns -- and the
 * completion is never registered; the other direction's streams are
 * left untouched, their in-flight jobs keep the buffer-alive contract
 * and settle through their own completions. A job failing
 * asynchronously surfaces through its completion: {jobId, false} from
 * GetFinished, an error status from Wait.
 *
 * Job lifecycle: a successful submit enters jobId into the in-flight
 * table, and its completion is reported exactly once, via GetFinished
 * or Wait; consumption frees the id for reuse and returns the events to
 * the pool.
 *
 * The runtime lifecycle is the process's: initialization, finalization
 * and device resets live with the caller, and the engine's teardown
 * releases only what it created -- the streams, the events, the
 * pinning.
 *
 * The upper layer calls Setup once and touches the other entry points
 * only after it returns Ok; the engine must be drained before
 * destruction, and the destructor is a drain-and-release safety net
 * regardless.
 *
 * Callers serialize the control plane (Setup / Submit* / GetFinished /
 * Wait / Synchronize) and keep job buffers alive until completion;
 * that single thread is the engine's only one, so the bookkeeping
 * carries no locks.
 */
class NaiveDmaEngine : public DmaEngine {
public:
    ~NaiveDmaEngine() override;

    Status Setup(const InitParam& param) override;
    Status SubmitH2DBatch(std::size_t jobId, const DmaJob& job) override;
    Status SubmitD2HBatch(std::size_t jobId, const DmaJob& job) override;
    Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() override;
    Status Wait(std::size_t jobId) override;
    Status Synchronize() override;

private:
    using JobCompletion = std::pair<aclrtStream, aclrtEvent>;
    /** One in-flight job: its id and, per stream its groups were placed
     *  on, the closing event. */
    struct Job {
        std::size_t jobId;
        std::vector<JobCompletion> completions;
    };

    /** @brief Transfer direction: each direction owns one private stream
     *         pool, one round-robin cursor and one in-flight queue, and
     *         the two run concurrently. */
    enum Direction : std::size_t {
        kH2D = 0, /* ACL_MEMCPY_HOST_TO_DEVICE lands here */
        kD2H = 1, /* ACL_MEMCPY_DEVICE_TO_HOST lands here */
        kDirectionCount
    };

    Status Submit(std::size_t jobId, const DmaJob& job, aclrtMemcpyKind kind);
    /** @brief Rejects malformed jobs before the submission touches a
     *         resource: empty or unequal column vectors, a grouping that
     *         does not describe the segments, null addresses, zero
     *         lengths, lengths beyond the bound. */
    Status ValidateJob(std::size_t jobId, const DmaJob& job) const;
    /** @brief One completion event: reused from the LIFO pool when it
     *         holds one, freshly created when it does not. */
    Expected<aclrtEvent> AcquireEvent(std::size_t jobId);
    /** @brief Enters the id and its completion pairs into the in-flight
     *         bookkeeping, on its direction's queue; a memory failure
     *         rejects the job with nothing started and rolls a partially
     *         registered entry back on the spot. */
    Status RegisterJob(std::size_t jobId, const std::vector<JobCompletion>& completions,
                       Direction lane);
    /** @brief Puts one group's copy calls on the given stream and reports
     *         the code of the calls that took effect: one aclrtMemcpyAsync
     *         per segment, destMax carrying the segment's own length. */
    aclError EnqueueCopies(const DmaJob& job, aclrtMemcpyKind kind, aclrtStream stream,
                           std::size_t start, std::size_t count);
    /**
     * @brief Rolls one submission back after an ACL call failed mid-batch.
     *
     * The failed call enqueued nothing, but earlier groups of the batch
     * may already be on their streams, and the caller may release its
     * buffers right after this error returns -- so every stream the
     * submission touched is drained first (each drain's own outcome is
     * irrelevant: the submit error below is the precise one) before the
     * bookkeeping lets the id go. The touched streams all sit on the
     * submission's direction; the other direction's streams are left
     * untouched, and its in-flight jobs settle through their own
     * completions.
     */
    Status AbortSubmit(std::size_t jobId, const std::vector<JobCompletion>& completions,
                       aclError code, Direction lane);
    /** @brief Returns one settled event to the pool; a pool that cannot
     *         grow destroys the event instead of leaking it. */
    void ReleaseEvent(aclrtEvent event);

    /* one FIFO per direction: submission order, and each direction's
     * streams settle their own queue in that order */
    std::array<std::deque<Job>, kDirectionCount> inFlight_;
    std::unordered_map<std::size_t, std::vector<JobCompletion>>
        jobs_;                          /* in-flight ids: Wait's O(1) lookup */
    std::vector<aclrtEvent> eventPool_; /* LIFO pool: settled events, reusable */
    /* one round-robin pool per direction; both empty or both full --
     * Setup's rollback keeps the pair whole, and the destructor's guard
     * reads the first */
    std::array<std::vector<aclrtStream>, kDirectionCount> streams_ = {};
    /* one group-granular round-robin cursor per direction */
    std::array<std::size_t, kDirectionCount> nextStream_ = {};
    void* registeredBase_ = nullptr; /* page-locked region this engine owns, for unregister */
    std::size_t maxLength_ = 0;
    bool registered_ = false;
};

NaiveDmaEngine::~NaiveDmaEngine()
{
    if (streams_[kH2D].empty()) { return; } /* Setup never completed: nothing to release */
    /* drain every pool: segment buffers must outlive the engine, either direction */
    for (const Direction lane : {kH2D, kD2H}) {
        for (aclrtStream stream : streams_[lane]) { aclrtSynchronizeStream(stream); }
    }
    if (registered_) { aclrtHostUnregister(registeredBase_); }
    for (const std::deque<Job>& queue : inFlight_) {
        for (const Job& job : queue) {
            for (const JobCompletion& completion : job.completions) {
                aclrtDestroyEvent(completion.second);
            }
        }
    }
    for (const aclrtEvent event : eventPool_) { aclrtDestroyEvent(event); }
    for (const Direction lane : {kH2D, kD2H}) {
        for (aclrtStream stream : streams_[lane]) { aclrtDestroyStream(stream); }
    }
}

Status NaiveDmaEngine::Setup(const InitParam& param)
{
    if (param.maxLength == 0) { return Status::InvalidParam("max length must be greater than 0"); }
    aclError code = aclrtSetDevice(param.deviceId);
    if (code != ACL_SUCCESS) {
        return Status::OsApiError("binding the dma engine to device {} failed: acl error {}",
                                  param.deviceId, code);
    }
    /* 0 (unset) keeps each direction single-stream */
    const std::size_t streamCount = param.nStream != 0 ? param.nStream : std::size_t{1};
    if (streamCount > streams_[kH2D].max_size()) {
        return Status::InvalidParam("stream number {} exceeds the addressable range", streamCount);
    }
    try {
        streams_[kH2D].reserve(streamCount);
        streams_[kD2H].reserve(streamCount);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory("allocating the dma stream table failed");
    }
    /* each direction fills its own pool, the h2d one first; a create
     * failure undoes every stream both pools hold, so the pair stays
     * all-or-nothing for the destructor's guard */
    for (const Direction lane : {kH2D, kD2H}) {
        for (std::size_t i = 0; i < streamCount; ++i) {
            aclrtStream stream = nullptr;
            code = aclrtCreateStream(&stream);
            if (code != ACL_SUCCESS) {
                for (const Direction made : {kH2D, kD2H}) {
                    for (const aclrtStream created : streams_[made]) {
                        aclrtDestroyStream(created);
                    }
                    streams_[made].clear();
                }
                return Status::OsApiError(
                    "creating the {} dma stream {} of {} failed: acl error {}",
                    lane == kH2D ? "h2d" : "d2h", i, streamCount, code);
            }
            streams_[lane].push_back(stream); /* nothrow: the table is reserved */
        }
    }
    maxLength_ = param.maxLength;

    /* best effort: a failure costs performance only */
    if (param.hostBase != nullptr && param.hostLength > 0) {
        code = aclrtHostRegisterV2(param.hostBase, param.hostLength, ACL_HOST_REG_PINNED);
        if (code == ACL_SUCCESS) {
            registered_ = true;
            registeredBase_ = param.hostBase;
        }
    }
    return Status::Ok();
}

Status NaiveDmaEngine::SubmitH2DBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job, ACL_MEMCPY_HOST_TO_DEVICE);
}

Status NaiveDmaEngine::SubmitD2HBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job, ACL_MEMCPY_DEVICE_TO_HOST);
}

Expected<std::vector<std::pair<std::size_t, bool>>> NaiveDmaEngine::GetFinished()
{
    std::vector<std::pair<std::size_t, bool>> finished;
    try {
        /* settles stay within capacity: no allocation can throw mid-loop */
        finished.reserve(inFlight_[kH2D].size() + inFlight_[kD2H].size());
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory("collecting dma completions failed");
    }
    /* each direction drains its own FIFO, the h2d one first: within one,
     * a not-ready head holds the queue behind it. The poll carries no
     * task outcome -- its return code is the call's own status, the
     * recorded status only distinguishes not-ready from complete -- so a
     * head whose every event is recorded resolves through
     * aclrtSynchronizeEvent: each event is already settled, the calls
     * return immediately, and the first failing code is the job's
     * outcome, the per-job attribution. Across the two directions
     * nothing is implied: each report is the settled prefix of its own
     * direction's streams. */
    for (const Direction lane : {kH2D, kD2H}) {
        std::deque<Job>& queue = inFlight_[lane];
        while (!queue.empty()) {
            Job& head = queue.front();
            bool pending = false;
            for (const JobCompletion& completion : head.completions) {
                aclrtEventRecordedStatus state = ACL_EVENT_RECORDED_STATUS_NOT_READY;
                const aclError query = aclrtQueryEventStatus(completion.second, &state);
                if (query == ACL_SUCCESS && state == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
                    pending = true;
                    break;
                }
            }
            if (pending) { break; }
            aclError outcome = ACL_SUCCESS;
            for (const JobCompletion& completion : head.completions) {
                const aclError code = aclrtSynchronizeEvent(completion.second);
                if (outcome == ACL_SUCCESS) { outcome = code; }
            }
            const std::size_t jobId = head.jobId;
            const std::vector<JobCompletion> completions = std::move(head.completions);
            finished.emplace_back(jobId, outcome == ACL_SUCCESS);
            queue.pop_front();
            jobs_.erase(jobId);
            for (const JobCompletion& completion : completions) { ReleaseEvent(completion.second); }
        }
    }
    return finished;
}

Status NaiveDmaEngine::Wait(std::size_t jobId)
{
    const auto it = jobs_.find(jobId);
    if (it == jobs_.end()) { return Status::InvalidParam("job {} is not in flight", jobId); }
    const std::vector<JobCompletion> completions = std::move(it->second);
    jobs_.erase(it);
    /* every event is drained -- an early-out would recycle an event that
     * may still be pending on its stream */
    aclError code = ACL_SUCCESS;
    for (const JobCompletion& completion : completions) {
        const aclError outcome = aclrtSynchronizeEvent(completion.second);
        if (code == ACL_SUCCESS) { code = outcome; }
    }
    /* the job sits on exactly one direction's queue: the first hit is it */
    for (std::deque<Job>& queue : inFlight_) {
        const auto settled =
            std::find_if(queue.begin(), queue.end(),
                         [jobId](const Job& candidate) { return candidate.jobId == jobId; });
        if (settled != queue.end()) {
            queue.erase(settled);
            break;
        }
    }
    for (const JobCompletion& completion : completions) { ReleaseEvent(completion.second); }
    if (code != ACL_SUCCESS) {
        return Status::OsApiError("job {} failed: acl error {}", jobId, code);
    }
    return Status::Ok();
}

Status NaiveDmaEngine::Synchronize()
{
    /* every stream of both pools drains, whichever failed first is the
     * report -- the rest are still waited on, so no direction is left
     * un-waited */
    aclError code = ACL_SUCCESS;
    for (const Direction lane : {kH2D, kD2H}) {
        for (aclrtStream stream : streams_[lane]) {
            const aclError outcome = aclrtSynchronizeStream(stream);
            if (code == ACL_SUCCESS) { code = outcome; }
        }
    }
    if (code != ACL_SUCCESS) {
        return Status::OsApiError("synchronizing the dma streams failed: acl error {}", code);
    }
    return Status::Ok();
}

Status NaiveDmaEngine::Submit(std::size_t jobId, const DmaJob& job, aclrtMemcpyKind kind)
{
    /* the kind picks the lane: its pool carries the batch below */
    const Direction lane = kind == ACL_MEMCPY_HOST_TO_DEVICE ? kH2D : kD2H;
    if (streams_[lane].empty()) { return Status::Unsupported("dma engine is not set up"); }
    const Status rejected = ValidateJob(jobId, job);
    if (rejected.Failure()) { return rejected; }
    if (jobs_.find(jobId) != jobs_.end()) {
        return Status::DuplicateKey("job {} is already in flight", jobId);
    }

    /* Placement: group g of the batch runs on stream (base + g) % n of
     * the lane's pool, so the groups of one submission fan out over
     * consecutive streams of that pool while the lane's cursor --
     * advanced only by placed groups -- spreads the next submission on
     * the same lane the same way. A batch without grouping is one group
     * and takes a single stream. */
    const std::size_t n = streams_[lane].size();
    const std::size_t base = nextStream_[lane];
    const std::size_t groupCount =
        job.nSegmentPerGroup.empty() ? std::size_t{1} : job.nSegmentPerGroup.size();
    const std::size_t used = std::min(groupCount, n);

    std::vector<JobCompletion> completions;
    try {
        completions.reserve(used);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory("placing job {} on the stream pool failed", jobId);
    }
    for (std::size_t i = 0; i < used; ++i) {
        auto acquired = AcquireEvent(jobId);
        if (!acquired.HasValue()) {
            for (const JobCompletion& completion : completions) { ReleaseEvent(completion.second); }
            return acquired.Error();
        }
        completions.emplace_back(streams_[lane][(base + i) % n],
                                 acquired.Value()); /* nothrow: reserved */
    }

    const Status registered = RegisterJob(jobId, completions, lane);
    if (registered.Failure()) { return registered; }

    /* The walk maps the batch onto the lane's streams: every group's
     * segments follow the optional prerequisite wait on the group's
     * stream (a stream touched by several groups re-waits, a no-op once
     * satisfied); the closing records are enqueued after the walk, one
     * per touched stream, behind that stream's last group. The first
     * failure aborts. */
    aclError code = ACL_SUCCESS;
    std::size_t start = 0;
    for (std::size_t g = 0; g < groupCount && code == ACL_SUCCESS; ++g) {
        aclrtStream stream = streams_[lane][(base + g) % n];
        if (job.prerequisite != 0) {
            code = aclrtStreamWaitEvent(stream, reinterpret_cast<aclrtEvent>(job.prerequisite));
        }
        if (code == ACL_SUCCESS) {
            const std::size_t count =
                job.nSegmentPerGroup.empty() ? job.sizes.size() : job.nSegmentPerGroup[g];
            code = EnqueueCopies(job, kind, stream, start, count);
            start += count;
        }
    }
    for (std::size_t i = 0; i < used && code == ACL_SUCCESS; ++i) {
        code = aclrtRecordEvent(completions[i].second, streams_[lane][(base + i) % n]);
    }
    if (code != ACL_SUCCESS) { return AbortSubmit(jobId, completions, code, lane); }
    nextStream_[lane] = (base + groupCount) % n;
    return Status::Ok();
}

Status NaiveDmaEngine::ValidateJob(std::size_t jobId, const DmaJob& job) const
{
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
    return Status::Ok();
}

Expected<aclrtEvent> NaiveDmaEngine::AcquireEvent(std::size_t jobId)
{
    if (!eventPool_.empty()) {
        const aclrtEvent event = eventPool_.back();
        eventPool_.pop_back();
        return event;
    }
    aclrtEvent event = nullptr;
    const aclError code = aclrtCreateEventExWithFlag(&event, kEventFlags);
    if (code != ACL_SUCCESS) {
        return Status::OsApiError("creating the completion event of job {} failed: acl error {}",
                                  jobId, code);
    }
    return event;
}

Status NaiveDmaEngine::RegisterJob(std::size_t jobId, const std::vector<JobCompletion>& completions,
                                   Direction lane)
{
    try {
        /* both containers take a copy, not a move -- whichever allocation
         * fails, completions still holds every event for the catch to
         * release; a move would strand them in the destroyed temporary */
        jobs_.emplace(jobId, completions);
        inFlight_[lane].push_back(Job{jobId, completions});
    } catch (const std::bad_alloc&) {
        jobs_.erase(jobId); /* a no-op when the map emplace itself failed */
        for (const JobCompletion& completion : completions) { ReleaseEvent(completion.second); }
        return Status::OutOfMemory("registering job {} failed", jobId);
    }
    return Status::Ok();
}

aclError NaiveDmaEngine::EnqueueCopies(const DmaJob& job, aclrtMemcpyKind kind, aclrtStream stream,
                                       std::size_t start, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) {
        const aclError code =
            aclrtMemcpyAsync(job.destinations[start + i], job.sizes[start + i],
                             job.sources[start + i], job.sizes[start + i], kind, stream);
        if (code != ACL_SUCCESS) { return code; }
    }
    return ACL_SUCCESS;
}

Status NaiveDmaEngine::AbortSubmit(std::size_t jobId, const std::vector<JobCompletion>& completions,
                                   aclError code, Direction lane)
{
    for (const JobCompletion& completion : completions) {
        aclrtSynchronizeStream(completion.first);
    }
    for (const JobCompletion& completion : completions) { ReleaseEvent(completion.second); }
    jobs_.erase(jobId);
    inFlight_[lane].pop_back();
    return Status::OsApiError("submitting job {} failed: acl error {}", jobId, code);
}

void NaiveDmaEngine::ReleaseEvent(aclrtEvent event)
{
    try {
        eventPool_.push_back(event);
    } catch (const std::bad_alloc&) {
        aclrtDestroyEvent(event);
    }
}

UC_TRANSFER_REGISTER_DMA_ENGINE(NaiveDmaEngine, DmaEngineKind::Naive);

}  // namespace
}  // namespace UC::Transfer
