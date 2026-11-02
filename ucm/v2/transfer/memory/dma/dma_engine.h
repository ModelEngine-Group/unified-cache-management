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
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "status/status.h"

namespace UC::Transfer {

/**
 * @brief Asynchronous host<->device batch DMA engine interface with
 *        job-granular completion tracking.
 *
 * Implementations validate submissions and enforce the jobId lifecycle:
 * a successful submit enters the in-flight set and its completion is
 * reported exactly once, via GetFinished or Wait.
 *
 * Thread safety: instances are not thread-safe; callers must serialize
 * access to a single engine.
 */
class DmaEngine {
public:
    /**
     * @brief Engine initialization and registration data.
     *
     * @note Append-only convention: new fields may only be appended at the
     *       tail with value-neutral defaults, so aggregate initialization
     *       by older callers stays valid; implementations read the fields
     *       they understand and ignore the rest.
     */
    struct InitParam {
        /** Device ordinal the engine operates on. */
        int32_t deviceId;
    };

    /**
     * @brief Transfer content of one asynchronous batch job,
     *        column-oriented.
     *
     * @note sources, destinations and sizes must be non-empty and of
     *       equal length; sources[i], destinations[i] and sizes[i]
     *       describe one segment. H2D submissions read sources as host
     *       addresses and destinations as device addresses; D2H the
     *       reverse.
     *
     * @note Append-only convention: new fields may only be appended at the
     *       tail with value-neutral defaults, so aggregate initialization
     *       by older callers stays valid; implementations read the fields
     *       they understand and ignore the rest.
     */
    struct DmaJob {
        std::vector<const void*> sources;
        std::vector<void*> destinations;
        std::vector<std::size_t> sizes;
    };

public:
    virtual ~DmaEngine() = default;

    /** @brief Initializes the engine on the device given by param.
     *         Must be called (and return Ok) before any submit. */
    virtual Status Setup(const InitParam& param) = 0;

    /**
     * @brief Submits an asynchronous host-to-device batch under jobId.
     *
     * @note job must be well-formed (else InvalidParam) and jobId must
     *       not be in flight (else DuplicateKey). On Ok the job is in
     *       flight and its completion is reported exactly once via
     *       GetFinished or Wait.
     */
    virtual Status SubmitH2DBatch(std::size_t jobId, const DmaJob& job) = 0;
    /**
     * @brief Submits an asynchronous device-to-host batch under jobId,
     *        with the same job and jobId contract as SubmitH2DBatch.
     */
    virtual Status SubmitD2HBatch(std::size_t jobId, const DmaJob& job) = 0;

    /**
     * @brief Non-blocking poll for jobs completed since the last call.
     *
     * @return one {jobId, succeeded} entry per completed in-flight job,
     *         each reported exactly once; an empty vector means nothing
     *         has finished yet.
     */
    virtual Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() = 0;
    /**
     * @brief Blocks until the given job has completed, then consumes and
     *        reports its completion.
     *
     * @param jobId Job to wait for; must currently be in flight.
     *
     * @return Status::Ok() when the job's batch completed successfully;
     *         an error status carrying the failure detail otherwise.
     *
     * @note Waiting on an id that is not in flight (unknown or already
     *       consumed) yields InvalidParam.
     */
    virtual Status Wait(std::size_t jobId) = 0;
    /**
     * @brief Blocks until all in-flight jobs have completed.
     *
     * @note Aggregate status only -- no per-job attribution; use
     *       GetFinished or Wait beforehand when attribution is needed.
     *       Returns Ok immediately when nothing is in flight.
     */
    virtual Status Synchronize() = 0;
};

}  // namespace UC::Transfer
