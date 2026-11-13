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
#include "fault_dma_engine.h"
#include <algorithm>
#include <new>
#include "memory/dma/dma_engine_registry.h"

namespace UC::Transfer {
namespace {

constexpr const char* kFailureDetail = "fault dma engine fails every job by design";

std::vector<CapturedDmaBatch>& CaptureSink()
{
    static std::vector<CapturedDmaBatch> sink;
    return sink;
}

}  // namespace

const std::vector<CapturedDmaBatch>& CapturedDmaBatches() { return CaptureSink(); }

void ResetCapturedDmaBatches() { CaptureSink().clear(); }

Status FaultDmaEngine::Setup(const InitParam& param)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    /* Test-only convention check: Setup never re-enters. */
    if (setup_) {
        return Status::Unsupported("setup re-entry is not supported: dma engine is already set up");
    }
    if (param.maxLength == 0) { return Status::InvalidParam("max length must be greater than 0"); }
    maxLength_ = param.maxLength;
    setup_ = true;
    return Status::Ok();
}

Status FaultDmaEngine::SubmitH2DBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job, true);
}

Status FaultDmaEngine::SubmitD2HBatch(std::size_t jobId, const DmaJob& job)
{
    return Submit(jobId, job, false);
}

Expected<std::vector<std::pair<std::size_t, bool>>> FaultDmaEngine::GetFinished()
{
    /* Test-only convention check: nothing runs before a successful Setup. */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
    std::vector<std::pair<std::size_t, bool>> finished;
    const std::lock_guard<std::mutex> lock(mutex_);
    finished.reserve(doneOrder_.size());
    while (!doneOrder_.empty()) {
        finished.push_back(doneOrder_.front());
        doneOrder_.pop_front();
        jobs_.erase(finished.back().first);
    }
    return finished;
}

Status FaultDmaEngine::Wait(std::size_t jobId)
{
    /* Test-only convention check: nothing runs before a successful Setup. */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (jobs_.count(jobId) == 0) { return Status::InvalidParam("job {} is not in flight", jobId); }
    jobs_.erase(jobId);
    const auto settled = std::find_if(doneOrder_.begin(), doneOrder_.end(),
                                      [jobId](const std::pair<std::size_t, bool>& candidate) {
                                          return candidate.first == jobId;
                                      });
    if (settled != doneOrder_.end()) { doneOrder_.erase(settled); }
    return Status::General("job {}: {}", jobId, kFailureDetail);
}

Status FaultDmaEngine::Synchronize()
{
    /* Test-only convention check: nothing runs before a successful Setup. */
    if (!setup_) { return Status::Unsupported("dma engine is not set up"); }
    return Status::Ok(); /* every admitted job settled at submit time */
}

Status FaultDmaEngine::Submit(std::size_t jobId, const DmaJob& job, bool hostToDevice)
{
    /* Test-only convention check: nothing runs before a successful Setup. */
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

    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (jobs_.count(jobId) != 0) {
            return Status::DuplicateKey("job {} is already in flight", jobId);
        }
        bool registered = false;
        try {
            jobs_.insert(jobId);
            registered = true;
            doneOrder_.emplace_back(jobId, false);
        } catch (const std::bad_alloc&) {
            /* A job left half-registered would keep its id permanently
             * taken, so undo the partial state. */
            if (registered) { jobs_.erase(jobId); }
            return Status::OutOfMemory("enqueueing job {} failed", jobId);
        }
    }
    /* The capture records what the data plane built; a recording that
     * cannot allocate is skipped, never failed. */
    try {
        CaptureSink().push_back(CapturedDmaBatch{jobId, hostToDevice, job});
    } catch (const std::bad_alloc&) {
    }
    return Status::Ok();
}

/* The registration lands in the Gdr slot: only an explicit "gdr" engine
 * name selects this engine, never the default selection. */
UC_TRANSFER_REGISTER_DMA_ENGINE(FaultDmaEngine, DmaEngineKind::Gdr);

}  // namespace UC::Transfer
