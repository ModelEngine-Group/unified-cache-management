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
#include <deque>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>
#include "memory/dma/dma_engine.h"

namespace UC::Transfer {

/** One submission the fault engine admitted: its id, direction, and the
 *  flattened batch as the data plane built it. */
struct CapturedDmaBatch {
    std::size_t jobId;
    bool hostToDevice;
    DmaEngine::DmaJob batch;
};

/** Every batch admitted since the last reset, oldest first: the registry
 *  -- not the test -- creates the engine instance, so this process-wide
 *  sink is the only channel back to the test. Callers serialize with the
 *  engine, so the sink carries no lock of its own. */
const std::vector<CapturedDmaBatch>& CapturedDmaBatches();

/** Clears the capture sink; call between tests. */
void ResetCapturedDmaBatches();

/**
 * @brief Fault-injection DmaEngine for tests: every well-formed submission
 *        is admitted in flight, settles immediately as a failure, moves no
 *        bytes, and is recorded in the capture sink for shape inspection.
 *
 * Job lifecycle: a successful submit enters jobId into the in-flight table;
 * its completion -- always failed -- is reported exactly once via
 * GetFinished or Wait; consumption frees the id for reuse. GetFinished
 * reports first-settled-first; Wait answers an error Status carrying the
 * failure detail.
 *
 * Registration borrows the Gdr slot: the simu build -- the only build that
 * runs unit tests -- registers no engine there, and Gdr ranks below the
 * simu default (Naive), so the borrowed registration can never win the
 * default selection. Consumers opt in explicitly through the "gdr" engine
 * name; nothing that asks for the default ever sees this engine.
 *
 * Thread contract: callers serialize the calls; the internal mutex only
 * keeps the shape defensive.
 */
class FaultDmaEngine : public DmaEngine {
public:
    Status Setup(const InitParam& param) override;
    Status SubmitH2DBatch(std::size_t jobId, const DmaJob& job) override;
    Status SubmitD2HBatch(std::size_t jobId, const DmaJob& job) override;
    Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() override;
    Status Wait(std::size_t jobId) override;
    Status Synchronize() override;

private:
    Status Submit(std::size_t jobId, const DmaJob& job, bool hostToDevice);

    std::mutex mutex_;                                   /* guards every field below */
    std::deque<std::pair<std::size_t, bool>> doneOrder_; /* settled-unconsumed, in settle order */
    std::unordered_set<std::size_t> jobs_;               /* in flight + settled-unconsumed */
    std::size_t maxLength_ = 0;
    bool setup_ = false;
};

}  // namespace UC::Transfer
