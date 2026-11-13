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
#include <memory>
#include <utility>
#include <vector>
#include "cache_config.h"
#include "memory/dma/dma_engine.h"
#include "shm_layout.h"
#include "store.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The cache domain's data plane: one DmaEngine moving bytes between
 *        the shared pool and the caller's xPU memory.
 *
 * A job consumes the token's slot addresses over the pool: every payload
 * segment flattens into one columnar engine batch, pairing the caller's
 * pointer with its slot address plus offset. Dump moves caller bytes into
 * the pool (D2H); Load moves pool bytes out to the caller (H2D). The
 * prerequisite handle rides along as the batch's prerequisite.
 *
 * Job bookkeeping is the engine's, not ours: jobIds pass through unchanged,
 * GetFinished()/Wait() forward directly, and each completion is consumed
 * exactly once. Submit-time errors (shape violations, a duplicate in-flight
 * id, resource exhaustion) return from Dump/Load with nothing started;
 * asynchronous failures arrive as {jobId, false} completions and error
 * statuses from Wait.
 *
 * Engine selection: an empty config.dmaEngine takes the build's default
 * (the registry's highest-priority registered kind); a name selects the
 * registered kind it denotes -- an unknown name or a kind unregistered in
 * this build fails here, as does the engine's Setup, both before any
 * transfer and both surfacing as the host's Setup error.
 *
 * Geometry: a legal segment fits its slot (offset + size <= blockSize);
 * a segment that starts past or runs past its slot bounds is rejected at
 * submit time with nothing started, and the engine's maxLength bound
 * (= blockSize) is the size-side guard behind it. Slot validity is not
 * verified here; content is never inspected.
 *
 * Threading: callers are serialized by the Store contract, and that single
 * calling thread is the engine's only thread.
 *
 * Destruction ordering: the engine's destructor drains in-flight jobs
 * whose segment addresses point into the pool, so the host must destruct
 * this component before releasing the pool mapping (declare it after the
 * mapping holders).
 */
class DataPlane {
public:
    /**
     * @brief Brings the data plane up over the given pool.
     *
     * @param config  The domain configuration (engine name, device id,
     *                block geometry).
     * @param pool    The shared pool mapping transfers address; must stay
     *                mapped for the component's whole lifetime.
     *
     * @return Expected the data plane on success; the engine-selection or
     *         engine-Setup error otherwise, with nothing left behind.
     */
    [[nodiscard]] static Expected<std::unique_ptr<DataPlane>> Create(const CacheConfig& config,
                                                                     const ShmMapping& pool);

    /**
     * @brief Submits one offload job: caller bytes into the pool, as an
     *         asynchronous engine batch under jobId.
     *
     * @return Status Ok with the job in flight; InvalidParam when the job
     *         and its token disagree in shape (an empty job included) or a
     *         segment runs past its slot bounds, with nothing started.
     */
    Status Dump(std::size_t jobId, const JobDescriptor& job, const AddressedToken& token);

    /**
     * @brief Submits one reload job: pool bytes out to the caller, with
     *         the same contract as Dump.
     */
    Status Load(std::size_t jobId, const JobDescriptor& job, const AddressedToken& token);

    /** @brief The engine's completions, drained and consumed exactly once. */
    std::vector<std::pair<std::size_t, bool>> GetFinished();

    /** @brief The engine's wait: blocks, consumes, and reports one job. */
    Status Wait(std::size_t jobId);

private:
    DataPlane(std::size_t blockSize, void* base,
              std::unique_ptr<Transfer::DmaEngine>&& engine) noexcept;

    /**
     * @brief Shape-checks and submits one flattened batch.
     *
     * @param intoPool  true moves caller bytes into the slots (Dump, D2H);
     *                  false moves slot bytes out to the caller (Load, H2D).
     */
    Status Submit(std::size_t jobId, const JobDescriptor& job, const AddressedToken& token,
                  bool intoPool);

    std::size_t blockSize_;
    void* base_;
    std::unique_ptr<Transfer::DmaEngine> engine_;
};

}  // namespace UC::Store
