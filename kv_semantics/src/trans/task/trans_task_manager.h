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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "buffer/buffer_manager.h"
#include "conn/connection_manager.h"
#include "kv_transport.h"
#include "proto/kv_protocol.h"
#include "task_context.h"
#include "task_manager_base.h"

namespace kv {

// Pending retry category of a sub-batch. NONE doubles as the "no pending
// retry" flag (no separate retryPending boolean); the derived attempt states
// are READY (no in-flight slots, NONE), SENDING (inside the Send call),
// WAITING_RESPONSE (slots allocated, NONE, awaiting adjudication), and
// RETRY_BACKOFF (!= NONE).
enum class RetryCategory : std::uint8_t {
    NONE = 0,
    BUSY = 1,
    LINK_FAILURE = 2,
};

struct TransportSubBatchContext {
    std::uint16_t cid{0};
    AsuOpType opType{AsuOpType::QUERY};
    TransportSubBatchState state{TransportSubBatchState::PENDING};
    Status status{Status::OK()};
    std::shared_ptr<ConnectionChannel> channel;
    bool useSeekControl{false};
    ScatterGatherEntry sendSge;
    ScatterGatherEntry flagBuffer;
    std::vector<Status> entryStatus;

    // ---- Retry routing state ----
    // Per-attempt deadline: now + min(attemptTimeoutMs, remaining task time).
    std::chrono::steady_clock::time_point attemptDeadline{};
    // Consecutive same-channel busy retries (<= busyRetryCount M).
    std::uint32_t busyRetries{0};
    // Pending retry category; LINK_FAILURE also books ENTRY_RETRY_ADVISED
    // failovers (shared FAILOVER mechanics, no independent count).
    RetryCategory pendingRetryCategory{RetryCategory::NONE};
    // Channels already attempted; object identity, used for failover
    // exclusion and the failover budget (failoverAttempts = size()).
    std::vector<ConnectionChannel*> attemptedChannels;
    // Earliest time a new attempt may start (BUSY backoff; failover is now).
    std::chrono::steady_clock::time_point retryAt{};
    // Retained request object so PrepareRetryRequest can re-pack the SQE with
    // a fresh CID and response buffer/MR.
    std::shared_ptr<SqeRequest> request;

    // Derived attempt state helpers.
    bool InFlight() const
    {
        return state != TransportSubBatchState::COMPLETED &&
               pendingRetryCategory == RetryCategory::NONE &&
               (sendSge.slot_index != UINT32_MAX || flagBuffer.slot_index != UINT32_MAX);
    }
};

void FillEntryStatusFromCqeResult(const KvResponse& response,
                                  TransportSubBatchContext& subBatchContext);

class TransportTaskManager : public TaskManagerBase<TransportTask, TransportTaskState> {
public:
    TransportTaskManager() : TaskManagerBase(TransportTaskState::PENDING, "transport") {}

    void NotifyCompletion(const TransportTaskPtr& task);
    static void BuildResult(const TransportTask& task, TaskResult& result);
};

}  // namespace kv
