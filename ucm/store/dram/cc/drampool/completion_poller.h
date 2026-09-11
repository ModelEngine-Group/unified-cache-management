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

#include <atomic>
#include <cstdint>
#include <deque>
#include "drampool_types.h"

namespace UC::DramPool {

class CompletionPoller final {
public:
    explicit CompletionPoller(DramPoolRuntime& runtime) : runtime_(runtime) {}

    void Run(const std::atomic_bool& stop);

private:
    void FillPendingWindow();
    void PollPendingCompletions();
    // Returns true after the data transfer has been settled and the record is ready
    // to advance to response submission. Waiting transfers remain pending.
    bool PollDataTransfer(CompletionRecord& record);
    bool SubmitResponse(CompletionRecord& record);
    bool PollResponseTransfer(CompletionRecord& record);
    void SettleDataTransfer(CompletionRecord& record, transport::TransferStatus terminalStatus);
    bool OperationTimedOut(const CompletionRecord& record, std::uint64_t nowMs) const;

    DramPoolRuntime& runtime_;
    std::deque<CompletionRecord> pending_;
};

}  // namespace UC::DramPool
