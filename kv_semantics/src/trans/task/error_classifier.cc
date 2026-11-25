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
#include "error_classifier.h"
#include <algorithm>
#include <string>
#include <vector>
#include "logger.h"

namespace kv {
namespace {

// Category for one explicitly mapped status code, or nullopt-equivalent marker
// via the fallback flag. Ordering follows the core mapping table:
//   - LINK_FAILURE: transport/link level codes reported by Send or by the CQE.
//   - BUSY: remote busy expressed through a legal CQE.
//   - RETRY_ADVISORY: entry-level retry hint (RA only exists in per-entry
//     expression scenarios).
//   - APPLICATION: business/protocol-validation codes, never retried.
enum class CodeClass { LINK_FAILURE, BUSY, RETRY_ADVISORY, APPLICATION, FALLBACK };

CodeClass ClassOfExplicitCode(StatusCode code, AsuOpType opType)
{
    switch (code) {
        // ---- Link failures (explicit): FAILOVER, record failure on channel.
        case StatusCode::CONNECTION_ERROR:
        case StatusCode::CQE_INTERNAL_ERROR:
        case StatusCode::CQE_WRITE_FAULT:
        case StatusCode::CQE_UNRECOVERED_READ_ERROR:
        case StatusCode::CQE_IO_TIMEOUT:
        // TIMEOUT is only classified here after an attempt was actually sent;
        // unsent total-deadline timeouts are terminal and never classified.
        case StatusCode::TIMEOUT:
            return CodeClass::LINK_FAILURE;

        // ---- Remote busy expressed through a legal CQE.
        case StatusCode::CQE_RESOURCE_BUSY:
            return CodeClass::BUSY;

        // ---- Retry advisory: switch channel immediately, no backoff, no
        // failure recording, no load reduction.
        case StatusCode::ENTRY_RETRY_ADVISED:
            return CodeClass::RETRY_ADVISORY;

        // ---- Application / protocol-validation codes: return to caller.
        case StatusCode::CQE_KEY_ALREADY_EXISTED:
        case StatusCode::CQE_OUT_OF_CREATE_SIZE:
        case StatusCode::CQE_INVALID_COMMAND_OPCODE:
        case StatusCode::CQE_INVALID_FIELD_IN_COMMAND:
        case StatusCode::ENTRY_NO_RETRY_ADVISED:
        case StatusCode::ENTRY_KEY_NOT_FOUND:
        case StatusCode::ENTRY_DATA_NOT_EXIST:
        case StatusCode::ENTRY_DELETE_FAILED:
        // CQE_KEY_NOT_EXIST depends on opType: QUERY is a normal result (query
        // miss) that IsNormalResult excludes up front; LOAD/STORE is an
        // application result returned to the caller.
        case StatusCode::CQE_KEY_NOT_EXIST:
            return CodeClass::APPLICATION;

        default:
            // Unknown/protocol-mismatch/count-mismatch/IO_ERROR without clear
            // semantics: fallback link failure (correctness preserved; the
            // caller logs/metrics it with the FALLBACK label).
            return CodeClass::FALLBACK;
    }
}

ErrorDecision DecisionForClass(CodeClass cls)
{
    ErrorDecision decision;
    switch (cls) {
        case CodeClass::LINK_FAILURE:
            decision.category = ErrorCategory::LINK_FAILURE;
            decision.retryAction = RetryAction::FAILOVER;
            decision.healthAction = ChannelHealthAction::RECORD_FAILURE;
            break;
        case CodeClass::BUSY:
            decision.category = ErrorCategory::BUSY;
            decision.retryAction = RetryAction::RETRY_SAME_CHANNEL;
            decision.flowAction = FlowControlAction::ON_BUSY;
            break;
        case CodeClass::RETRY_ADVISORY:
            decision.category = ErrorCategory::RETRY_ADVISORY;
            decision.retryAction = RetryAction::FAILOVER;
            break;
        case CodeClass::APPLICATION:
            decision.category = ErrorCategory::APPLICATION;
            decision.retryAction = RetryAction::RETURN_RESULT;
            break;
        case CodeClass::FALLBACK:
            decision.category = ErrorCategory::LINK_FAILURE;
            decision.fallback = true;
            decision.retryAction = RetryAction::FAILOVER;
            decision.healthAction = ChannelHealthAction::RECORD_FAILURE;
            break;
    }
    return decision;
}

// Priority order: LINK_FAILURE > BUSY > RETRY_ADVISORY > APPLICATION.
int ClassPriority(CodeClass cls)
{
    switch (cls) {
        case CodeClass::LINK_FAILURE: return 3;
        case CodeClass::BUSY: return 2;
        case CodeClass::RETRY_ADVISORY: return 1;
        case CodeClass::APPLICATION: return 0;
        case CodeClass::FALLBACK: return 3;
    }
    return 0;
}

}  // namespace

bool ErrorClassifier::IsExcludedStatus(const Status& status)
{
    switch (status.code) {
        // 1) API-layer/lifecycle errors rejected synchronously.
        case StatusCode::INVALID_ARGUMENT:
        case StatusCode::NOT_INITIALIZED:
        case StatusCode::TASK_NOT_FOUND:
        case StatusCode::UNSUPPORTED:
        case StatusCode::BUFFER_NOT_REGISTERED:
        case StatusCode::BUFFER_NOT_SUPPORTED:
        case StatusCode::NOT_FOUND:
        // 4) Local execution errors are terminated by the producer.
        case StatusCode::INTERNAL_ERROR:
        // 5) Local queue/slot busy is reported before acceptance; remote busy
        // is only expressed through CQE_RESOURCE_BUSY.
        case StatusCode::RESOURCE_BUSY:
        // Cancellation and aggregates never enter the mapping tables.
        case StatusCode::CANCELED:
        case StatusCode::PARTIAL_FAILED:
        case StatusCode::IN_PROGRESS:
            return true;
        default:
            return false;
    }
}

bool ErrorClassifier::IsNormalResult(const TransportSubBatchContext& subBatch)
{
    if (subBatch.status.ok()) { return true; }

    switch (subBatch.status.code) {
        // Query result buffer: the flag buffer carries per-key exist/miss
        // results, this is the normal query expression, not an error. Every
        // decoded entry must stay within the query expression codes.
        case StatusCode::CQE_CHECK_RESULT_BUFFER: {
            if (subBatch.opType != AsuOpType::QUERY) { return false; }
            for (const auto& entry : subBatch.entryStatus) {
                if (entry.ok()) { continue; }
                if (entry.code != StatusCode::ENTRY_KEY_EXIST &&
                    entry.code != StatusCode::ENTRY_KEY_NOT_EXIST) {
                    return false;
                }
            }
            return true;
        }
        // Query miss for the whole sub-batch (opType+status driven).
        case StatusCode::CQE_KEY_NOT_EXIST:
        case StatusCode::ENTRY_KEY_NOT_EXIST:
            return subBatch.opType == AsuOpType::QUERY;
        // Key-exists is a normal business result in any expression.
        case StatusCode::ENTRY_KEY_EXIST:
            return true;
        default:
            return false;
    }
}

ErrorDecision ErrorClassifier::ClassifyCode(StatusCode code, AsuOpType opType)
{
    return DecisionForClass(ClassOfExplicitCode(code, opType));
}

ErrorDecision ErrorClassifier::ClassifySubBatch(const TransportSubBatchContext& subBatch,
                                               const Status& rawStatus)
{
    // Excluded statuses are statically rejected before any mapping: they are
    // terminal producer/API outcomes, not retry routing inputs. They must not
    // fall into the fallback bucket.
    if (IsExcludedStatus(subBatch.status)) {
        ErrorDecision decision;
        decision.category = ErrorCategory::APPLICATION;
        decision.retryAction = RetryAction::RETURN_RESULT;
        return decision;
    }

    // CQE_CHECK_RESULT_BUFFER is not a final classification: the per-entry
    // statuses have already been decoded by the parse step. Classify the
    // entry-level mixture. Failure codes only mix at entry level; busy, link
    // faults, timeouts, and unknown codes are always sub-batch level single
    // codes (one CQE per attempt).
    if (subBatch.status.code == StatusCode::CQE_CHECK_RESULT_BUFFER) {
        CodeClass worst = CodeClass::APPLICATION;
        bool hasFailure = false;
        bool hasFallback = false;
        std::vector<StatusCode> distinctCodes;
        for (const auto& entry : subBatch.entryStatus) {
            if (entry.ok()) { continue; }
            hasFailure = true;
            if (std::find(distinctCodes.begin(), distinctCodes.end(), entry.code) ==
                distinctCodes.end()) {
                distinctCodes.push_back(entry.code);
            }
            auto entryClass = ClassOfExplicitCode(entry.code, subBatch.opType);
            if (entryClass == CodeClass::FALLBACK) {
                hasFallback = true;
                entryClass = CodeClass::LINK_FAILURE;
            }
            if (ClassPriority(entryClass) > ClassPriority(worst)) { worst = entryClass; }
        }
        auto decision = DecisionForClass(hasFailure ? worst : CodeClass::APPLICATION);
        decision.hasMultipleErrorCodes = distinctCodes.size() > 1;
        decision.fallback = decision.fallback || hasFallback;
        return decision;
    }

    // Single-code path: sub-batch status drives the mapping. The raw status is
    // preserved by the caller for the first-parse record so unknown raw values
    // are never folded into IO_ERROR here.
    auto decision = ClassifyCode(subBatch.status.code, subBatch.opType);
    if (decision.fallback) {
        KV_DEBUG(
            "ErrorClassifier fallback classification op_type={} raw_code={} message={} "
            "category=LINK_FAILURE fallback=true",
            static_cast<int>(subBatch.opType), static_cast<int>(rawStatus.code),
            rawStatus.message);
    }
    return decision;
}

}  // namespace kv
