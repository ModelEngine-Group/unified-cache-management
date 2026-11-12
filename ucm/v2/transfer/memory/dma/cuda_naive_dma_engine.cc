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
#include <cstddef>
#include <cuda_runtime_api.h>
#include <deque>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dma_engine.h"
#include "dma_engine_registry.h"

namespace UC::Transfer {
namespace {

/** @brief Event flavor of the engine: no timing cost, blocking waits. */
constexpr unsigned int kEventFlags = cudaEventDisableTiming | cudaEventBlockingSync;

/**
 * @brief Naive DmaEngine of the CUDA runtime: one private non-blocking
 *        stream per direction -- H2D and D2H each on their own -- one
 *        recorded event per batch, no staging, no engine threads.
 *
 * A submission maps its batch onto its direction's stream: an optional
 * cudaStreamWaitEvent for DmaJob::prerequisite (the handle is a
 * cudaEvent_t), the copy calls, a closing cudaEventRecord. Each stream
 * executes its submissions in order, so within one direction the
 * completions are reported first-settled-first; the two directions
 * execute concurrently, with no order between them, and GetFinished
 * walks their settled prefixes back to back, the h2d one first. The
 * segments within one submission have no guaranteed order and must
 * not alias.
 *
 * Host memory: a fixed host region (InitParam::hostBase/hostLength) is
 * page-locked once at Setup, best effort -- success makes both
 * directions asynchronous DMA over the region. Without a region, or
 * when the registration is refused, the engine runs with pageable
 * semantics: H2D submissions return once the driver has staged the
 * source, D2H submissions once the copy has landed. The pinning is an
 * optimization, never a correctness condition: no registration outcome
 * fails Setup, and what this engine did not register it never
 * unregisters.
 *
 * Failure semantics: a CUDA call failing mid-submission aborts the
 * submission -- its direction's stream is drained first, because
 * earlier segments may already be in flight and the caller may
 * release its buffers right after the error returns -- and the
 * completion is never registered; a failing batch copy call enqueues
 * nothing and only switches the submission to the per-segment shape.
 * A job failing asynchronously surfaces through its completion:
 * {jobId, false} from GetFinished, an error status from Wait. No
 * recovery from a poisoned context is attempted; the jobs behind a
 * failure settle as failed too, reported as-is.
 *
 * Job lifecycle: a successful submit enters jobId into the in-flight
 * table, and its completion is reported exactly once, via GetFinished
 * or Wait; consumption frees the id for reuse and returns the event to
 * the pool.
 *
 * The upper layer calls Setup once and touches the other entry points
 * only after it returns Ok, so the redundant re-entry and pre-Setup
 * guards are omitted; the engine must be drained before destruction,
 * and the destructor is a drain-and-release safety net regardless.
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
    /** One in-flight job: its id and the event closing its batch on the stream. */
    struct Job {
        std::size_t jobId;
        cudaEvent_t event;
    };

    /** @brief Transfer direction: each direction owns one private stream
     *         and one in-flight queue, and the two run concurrently. */
    enum Direction : std::size_t {
        kH2D = 0, /* cudaMemcpyHostToDevice lands here */
        kD2H = 1, /* cudaMemcpyDeviceToHost lands here */
        kDirectionCount
    };

    Status Submit(std::size_t jobId, const DmaJob& job, cudaMemcpyKind kind);
    /** @brief Rejects malformed jobs before the submission touches a
     *         resource: empty or unequal column vectors, null addresses,
     *         zero lengths, lengths beyond the bound. */
    Status ValidateJob(std::size_t jobId, const DmaJob& job) const;
    /** @brief One completion event: reused from the LIFO pool when it
     *         holds one, freshly created when it does not. */
    Expected<cudaEvent_t> AcquireEvent(std::size_t jobId);
    /** @brief Enters the id and its event into the in-flight bookkeeping,
     *         on its direction's queue; a memory failure rejects the job
     *         with nothing started and rolls a partially registered pair
     *         back on the spot. */
    Status RegisterJob(std::size_t jobId, cudaEvent_t event, Direction lane);
    /**
     * @brief Puts the job's copy calls on the given stream and reports
     *        the code of the calls that took effect: one
     *        cudaMemcpyBatchAsync carrying every segment where the
     *        toolkit provides it, one cudaMemcpyAsync per segment
     *        otherwise. A failing batch call enqueues nothing, so the
     *        submission falls back to the per-segment shape, and the
     *        engine stops batching for good.
     */
    cudaError_t EnqueueCopies(const DmaJob& job, cudaMemcpyKind kind, cudaStream_t stream);
    /**
     * @brief Rolls one submission back after a CUDA call failed mid-batch.
     *
     * The failed call enqueued nothing, but earlier segments of the batch
     * may already be on the lane's stream, and the caller may release its
     * buffers right after this error returns -- so that stream is drained
     * first (its own outcome is irrelevant: the submit error below is the
     * precise one) before the bookkeeping lets the id go. The other
     * direction's stream is left untouched: its in-flight jobs keep the
     * buffer-alive contract and settle through their own completions.
     */
    Status AbortSubmit(std::size_t jobId, cudaEvent_t event, cudaError_t code, Direction lane);
    /** @brief Returns one settled event to the pool; a pool that cannot
     *         grow destroys the event instead of leaking it. */
    void ReleaseEvent(cudaEvent_t event);

    /* one FIFO per direction: submission order, and each stream settles
     * its own queue in that order */
    std::array<std::deque<Job>, kDirectionCount> inFlight_;
    std::unordered_map<std::size_t, cudaEvent_t> jobs_; /* in-flight ids: Wait's O(1) lookup */
    std::vector<cudaEvent_t> eventPool_;                /* LIFO pool: settled events, reusable */
    /* one stream per direction; all null or all valid -- Setup's rollback
     * keeps the pair whole, and the destructor's guard reads the first */
    std::array<cudaStream_t, kDirectionCount> streams_ = {};
    void* registeredBase_ = nullptr; /* page-locked region this engine owns, for unregister */
    std::size_t maxLength_ = 0;
    bool registered_ = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    bool batchUsable_ = true;
#endif
};

NaiveDmaEngine::~NaiveDmaEngine()
{
    if (streams_[kH2D] == nullptr) { return; } /* Setup never completed: nothing to release */
    /* drain both: segment buffers must outlive the engine, either direction */
    cudaStreamSynchronize(streams_[kH2D]);
    cudaStreamSynchronize(streams_[kD2H]);
    if (registered_) { cudaHostUnregister(registeredBase_); }
    for (const std::deque<Job>& queue : inFlight_) {
        for (const Job& job : queue) { cudaEventDestroy(job.event); }
    }
    for (const cudaEvent_t event : eventPool_) { cudaEventDestroy(event); }
    cudaStreamDestroy(streams_[kH2D]);
    cudaStreamDestroy(streams_[kD2H]);
}

Status NaiveDmaEngine::Setup(const InitParam& param)
{
    if (param.maxLength == 0) { return Status::InvalidParam("max length must be greater than 0"); }
    cudaError_t code = cudaSetDevice(param.deviceId);
    if (code != cudaSuccess) {
        return Status::OsApiError("binding the dma engine to device {} failed: {}", param.deviceId,
                                  cudaGetErrorString(code));
    }
    code = cudaStreamCreateWithFlags(&streams_[kH2D], cudaStreamNonBlocking);
    if (code != cudaSuccess) {
        streams_[kH2D] = nullptr; /* a failed create may leave the handle indeterminate */
        return Status::OsApiError("creating the h2d dma stream failed: {}",
                                  cudaGetErrorString(code));
    }
    code = cudaStreamCreateWithFlags(&streams_[kD2H], cudaStreamNonBlocking);
    if (code != cudaSuccess) {
        /* the h2d stream is already made: undo it, so the pair stays
         * all-or-nothing for the destructor's guard */
        cudaStreamDestroy(streams_[kH2D]);
        streams_[kH2D] = nullptr;
        streams_[kD2H] = nullptr;
        return Status::OsApiError("creating the d2h dma stream failed: {}",
                                  cudaGetErrorString(code));
    }
    maxLength_ = param.maxLength;

    /* best effort: a failure costs performance only */
    if (param.hostBase != nullptr && param.hostLength > 0) {
        code = cudaHostRegister(param.hostBase, param.hostLength, cudaHostRegisterDefault);
        if (code == cudaSuccess) {
            registered_ = true;
            registeredBase_ = param.hostBase;
        }
    }
    return Status::Ok();
}

Status NaiveDmaEngine::SubmitH2DBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job, cudaMemcpyHostToDevice);
}

Status NaiveDmaEngine::SubmitD2HBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job, cudaMemcpyDeviceToHost);
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
    /* each direction drains its own FIFO, the h2d one first: a settled
     * queue head proves nothing about its successors, but a
     * cudaErrorNotReady head proves nothing behind it can be done either --
     * and an event query other than cudaErrorNotReady closes its batch, the
     * failure code included, which is the per-job attribution. Across the
     * two directions nothing is implied: each report is the settled prefix
     * of its own stream. */
    for (const Direction lane : {kH2D, kD2H}) {
        std::deque<Job>& queue = inFlight_[lane];
        while (!queue.empty()) {
            const Job head = queue.front();
            const cudaError_t state = cudaEventQuery(head.event);
            if (state == cudaErrorNotReady) { break; }
            finished.emplace_back(head.jobId, state == cudaSuccess);
            queue.pop_front();
            jobs_.erase(head.jobId);
            ReleaseEvent(head.event);
        }
    }
    return finished;
}

Status NaiveDmaEngine::Wait(std::size_t jobId)
{
    const auto it = jobs_.find(jobId);
    if (it == jobs_.end()) { return Status::InvalidParam("job {} is not in flight", jobId); }
    const cudaEvent_t event = it->second;
    const cudaError_t code = cudaEventSynchronize(event);
    jobs_.erase(it);
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
    ReleaseEvent(event);
    if (code != cudaSuccess) {
        return Status::OsApiError("job {} failed: {}", jobId, cudaGetErrorString(code));
    }
    return Status::Ok();
}

Status NaiveDmaEngine::Synchronize()
{
    /* both streams drain, whichever failed first is the report -- the
     * second one is still waited on, so no direction is left un-waited */
    cudaError_t code = cudaStreamSynchronize(streams_[kH2D]);
    const cudaError_t d2hCode = cudaStreamSynchronize(streams_[kD2H]);
    if (code == cudaSuccess) { code = d2hCode; }
    if (code != cudaSuccess) {
        return Status::OsApiError("synchronizing the dma streams failed: {}",
                                  cudaGetErrorString(code));
    }
    return Status::Ok();
}

Status NaiveDmaEngine::Submit(std::size_t jobId, const DmaJob& job, cudaMemcpyKind kind)
{
    const Status rejected = ValidateJob(jobId, job);
    if (rejected.Failure()) { return rejected; }
    if (jobs_.find(jobId) != jobs_.end()) {
        return Status::DuplicateKey("job {} is already in flight", jobId);
    }

    /* the kind picks the lane: its stream carries the batch below */
    const Direction lane = kind == cudaMemcpyHostToDevice ? kH2D : kD2H;
    cudaStream_t& stream = streams_[lane];

    auto acquired = AcquireEvent(jobId);
    if (!acquired.HasValue()) { return acquired.Error(); }
    const cudaEvent_t event = acquired.Value();
    const Status registered = RegisterJob(jobId, event, lane);
    if (registered.Failure()) { return registered; }

    /* the lane's stream maps the batch: an optional prerequisite wait, the
     * copy calls, the closing record; the first failure aborts */
    cudaError_t code = cudaSuccess;
    if (job.prerequisite != 0) {
        code = cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(job.prerequisite), 0);
    }
    if (code == cudaSuccess) { code = EnqueueCopies(job, kind, stream); }
    if (code == cudaSuccess) { code = cudaEventRecord(event, stream); }
    if (code != cudaSuccess) { return AbortSubmit(jobId, event, code, lane); }
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

Expected<cudaEvent_t> NaiveDmaEngine::AcquireEvent(std::size_t jobId)
{
    if (!eventPool_.empty()) {
        const cudaEvent_t event = eventPool_.back();
        eventPool_.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    const cudaError_t code = cudaEventCreateWithFlags(&event, kEventFlags);
    if (code != cudaSuccess) {
        return Status::OsApiError("creating the completion event of job {} failed: {}", jobId,
                                  cudaGetErrorString(code));
    }
    return event;
}

Status NaiveDmaEngine::RegisterJob(std::size_t jobId, cudaEvent_t event, Direction lane)
{
    try {
        jobs_.emplace(jobId, event);
        inFlight_[lane].push_back(Job{jobId, event});
    } catch (const std::bad_alloc&) {
        jobs_.erase(jobId); /* a no-op when the map emplace itself failed */
        ReleaseEvent(event);
        return Status::OutOfMemory("registering job {} failed", jobId);
    }
    return Status::Ok();
}

cudaError_t NaiveDmaEngine::EnqueueCopies(const DmaJob& job, cudaMemcpyKind kind,
                                          cudaStream_t stream)
{
    const std::size_t segmentCount = job.sizes.size();
    bool batched = false;
    /* the batch prototype settled in CUDA 13.0 */
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    if (batchUsable_) {
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream; /* sources read in stream order */
        std::size_t attrIndex = 0; /* the one attribute set covers the whole batch */
        /* the call spells both arrays as const void** while a const
         * DmaJob exposes them element-const, and the two const positions
         * admit no single-cast bridge: each array is widened to
         * const void* const* and const_cast off the element const --
         * the driver reads these caller-owned arrays, never rewrites
         * them */
        const void* const* destinations = job.destinations.data();
        const void* const* sources = job.sources.data();
        const cudaError_t batchCode = cudaMemcpyBatchAsync(
            const_cast<const void**>(destinations), const_cast<const void**>(sources),
            job.sizes.data(), segmentCount, &attr, &attrIndex, 1, stream);
        batched = (batchCode == cudaSuccess);
        if (!batched) { batchUsable_ = false; }
    }
#endif
    if (!batched) {
        for (std::size_t i = 0; i < segmentCount; ++i) {
            const cudaError_t code =
                cudaMemcpyAsync(job.destinations[i], job.sources[i], job.sizes[i], kind, stream);
            if (code != cudaSuccess) { return code; }
        }
    }
    return cudaSuccess;
}

Status NaiveDmaEngine::AbortSubmit(std::size_t jobId, cudaEvent_t event, cudaError_t code,
                                   Direction lane)
{
    cudaStreamSynchronize(streams_[lane]);
    jobs_.erase(jobId);
    inFlight_[lane].pop_back();
    ReleaseEvent(event);
    return Status::OsApiError("submitting job {} failed: {}", jobId, cudaGetErrorString(code));
}

void NaiveDmaEngine::ReleaseEvent(cudaEvent_t event)
{
    try {
        eventPool_.push_back(event);
    } catch (const std::bad_alloc&) {
        cudaEventDestroy(event);
    }
}

UC_TRANSFER_REGISTER_DMA_ENGINE(NaiveDmaEngine, DmaEngineKind::Naive);

}  // namespace
}  // namespace UC::Transfer
