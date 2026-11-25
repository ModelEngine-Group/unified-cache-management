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

#include <cstdint>
#include "kv_types.h"
#include "task/trans_task_manager.h"

namespace kv {

// Error categories ordered by classification priority:
// LINK_FAILURE > BUSY > RETRY_ADVISORY > APPLICATION. The classifier only
// handles failed results; normal results (including query miss) are excluded
// up front by IsNormalResult. The fallback bucket for unknown/unmapped codes
// is LINK_FAILURE with the fallback flag set; it is not a separate category.
enum class ErrorCategory : std::uint8_t {
    LINK_FAILURE = 0,
    BUSY = 1,
    RETRY_ADVISORY = 2,
    APPLICATION = 3,
};

enum class RetryAction : std::uint8_t {
    RETURN_RESULT = 0,
    RETRY_SAME_CHANNEL = 1,
    FAILOVER = 2,
};

enum class ChannelHealthAction : std::uint8_t {
    NO_CHANGE = 0,
    RECORD_FAILURE = 1,
};

enum class FlowControlAction : std::uint8_t {
    NO_CHANGE = 0,
    ON_BUSY = 1,
};

// Single point of error routing for one failed sub-batch attempt. The three
// action fields are derived from the category inside the classifier.
struct ErrorDecision {
    ErrorCategory category{ErrorCategory::APPLICATION};
    // True when the entry-level expression carries more than one distinct
    // failure code. MIXED is a bookkeeping flag, not a category value.
    bool hasMultipleErrorCodes{false};
    // True when the decision came from the unknown-code fallback bucket
    // (drives the FALLBACK label in logs/metrics, distinct from EXPLICIT).
    bool fallback{false};
    RetryAction retryAction{RetryAction::RETURN_RESULT};
    ChannelHealthAction healthAction{ChannelHealthAction::NO_CHANGE};
    FlowControlAction flowAction{FlowControlAction::NO_CHANGE};
};

// Classifies failed sub-batch results into retry routing decisions.
// The classifier never mutates connection state, never creates threads, and
// never drives I/O; it is a pure mapping from (opType, status, entryStatus)
// to an ErrorDecision. Probes, aggregate values (PARTIAL_FAILED), and the
// statically excluded status list never enter the mapping tables.
class ErrorClassifier {
public:
    // True when the sub-batch result is a normal business outcome. Query
    // exist/miss results are normal outcomes driven by opType+status, so this
    // is intentionally stronger than status.ok().
    static bool IsNormalResult(const TransportSubBatchContext& subBatch);

    // Classifies one failed sub-batch. rawStatus is the status captured at the
    // parse point (Poll after Unpack); it is preserved for logging so an
    // unknown raw value is never folded into IO_ERROR by the classifier.
    static ErrorDecision ClassifySubBatch(const TransportSubBatchContext& subBatch,
                                          const Status& rawStatus);

    // Static whitelist of statuses that never enter the classification tables:
    // API-layer rejections, local execution errors, local RESOURCE_BUSY, and
    // cancellation. Connection-build errors, unsent timeouts, and Wait
    // timeouts do not reach the classifier by construction.
    static bool IsExcludedStatus(const Status& status);

    // Maps a single status code to its category without entry analysis.
    // Unknown/unmapped codes return LINK_FAILURE with fallback=true.
    static ErrorDecision ClassifyCode(StatusCode code, AsuOpType opType);
};

}  // namespace kv
