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

// ErrorClassifier DT.
//
// 测试因子分解：
//   F1 opType        ∈ {QUERY, LOAD, STORE, BATCH_LOAD, BATCH_STORE, DELETE}
//   F2 subBatch.status ∈ 核心映射四类 + 排除清单 + 未知兜底 + 正常结果
//   F3 entryStatus   ∈ {全OK, 纯应用码混合, 含RA, 含未知码, 空}
// 断言面：category / retryAction / healthAction / fallback /
//         hasMultipleErrorCodes / IsNormalResult / IsExcludedStatus。
// 期望值全部来自《连接可靠性设计》第5节映射表，不来自实现。

#include "task/error_classifier.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>
#include "task/trans_task_manager.h"

namespace kv::test {
namespace {

Status MakeStatus(StatusCode code)
{
    return Status::Error(code, "dt");
}

TransportSubBatchContext MakeSubBatch(AsuOpType opType, StatusCode code,
                                      std::vector<Status> entryStatus = {})
{
    TransportSubBatchContext subBatch;
    subBatch.opType = opType;
    subBatch.status = MakeStatus(code);
    subBatch.entryStatus = std::move(entryStatus);
    return subBatch;
}

TransportSubBatchContext MakeOkSubBatch(AsuOpType opType, std::size_t entryCount)
{
    return MakeSubBatch(opType, StatusCode::OK, std::vector<Status>(entryCount, Status::OK()));
}

void ExpectDecision(const ErrorDecision& decision, ErrorCategory category, RetryAction retry,
                     ChannelHealthAction health)
{
    EXPECT_EQ(decision.category, category);
    EXPECT_EQ(decision.retryAction, retry);
    EXPECT_EQ(decision.healthAction, health);
}

}  // namespace

// ─── EC1: 链路类单码 → LINK_FAILURE / FAILOVER / RECORD_FAILURE ───

TEST(ErrorClassifierTest, LinkFailureCodesMapToFailover)
{
    const StatusCode linkCodes[] = {
        StatusCode::CONNECTION_ERROR,       StatusCode::CQE_INTERNAL_ERROR,
        StatusCode::CQE_WRITE_FAULT,        StatusCode::CQE_UNRECOVERED_READ_ERROR,
        StatusCode::CQE_IO_TIMEOUT,         StatusCode::TIMEOUT,
    };
    for (const auto code : linkCodes) {
        const auto subBatch = MakeSubBatch(AsuOpType::BATCH_LOAD, code);
        const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
        ExpectDecision(decision, ErrorCategory::LINK_FAILURE, RetryAction::FAILOVER,
                       ChannelHealthAction::RECORD_FAILURE);
        EXPECT_FALSE(decision.fallback) << "code=" << static_cast<int>(code);
        EXPECT_FALSE(decision.hasMultipleErrorCodes);
    }
}

// 已实际 Send 后的 TIMEOUT 按链路故障换链（调用方保证非未发超时）。
TEST(ErrorClassifierTest, TimeoutAfterSendIsLinkFailure)
{
    const auto subBatch = MakeSubBatch(AsuOpType::LOAD, StatusCode::TIMEOUT);
    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    EXPECT_EQ(decision.category, ErrorCategory::LINK_FAILURE);
    EXPECT_EQ(decision.retryAction, RetryAction::FAILOVER);
}

// ─── EC2: 远端忙单码 → BUSY / RETRY_SAME_CHANNEL / ON_BUSY ───

TEST(ErrorClassifierTest, RemoteBusyMapsToSameChannelRetryWithLoadReduction)
{
    const auto subBatch = MakeSubBatch(AsuOpType::BATCH_STORE, StatusCode::CQE_RESOURCE_BUSY);
    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::BUSY, RetryAction::RETRY_SAME_CHANNEL,
                   ChannelHealthAction::NO_CHANGE);
    EXPECT_FALSE(decision.fallback);
}

// ─── EC3: ENTRY_RETRY_ADVISED → RETRY_ADVISORY / FAILOVER / 无退避无降载 ───

TEST(ErrorClassifierTest, RetryAdvisedMapsToFailoverWithoutSideEffects)
{
    const auto subBatch = MakeSubBatch(AsuOpType::BATCH_LOAD, StatusCode::ENTRY_RETRY_ADVISED);
    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::RETRY_ADVISORY, RetryAction::FAILOVER,
                   ChannelHealthAction::NO_CHANGE);
    EXPECT_FALSE(decision.fallback);
}

// ─── EC4: 应用类 → APPLICATION / RETURN_RESULT / 无副作用 ───

TEST(ErrorClassifierTest, ApplicationCodesReturnResultWithoutSideEffects)
{
    const StatusCode appCodes[] = {
        StatusCode::CQE_KEY_ALREADY_EXISTED,      StatusCode::CQE_OUT_OF_CREATE_SIZE,
        StatusCode::ENTRY_NO_RETRY_ADVISED,       StatusCode::ENTRY_KEY_NOT_FOUND,
        StatusCode::ENTRY_DATA_NOT_EXIST,         StatusCode::ENTRY_DELETE_FAILED,
        StatusCode::CQE_INVALID_COMMAND_OPCODE,   StatusCode::CQE_INVALID_FIELD_IN_COMMAND,
    };
    for (const auto code : appCodes) {
        const auto subBatch = MakeSubBatch(AsuOpType::STORE, code);
        const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
        ExpectDecision(decision, ErrorCategory::APPLICATION, RetryAction::RETURN_RESULT,
                       ChannelHealthAction::NO_CHANGE);
        EXPECT_FALSE(decision.fallback) << "code=" << static_cast<int>(code);
    }
}

// CQE_KEY_NOT_EXIST 按 OpType 区分：LOAD/STORE 是应用结果。
TEST(ErrorClassifierTest, KeyNotExistOnLoadIsApplication)
{
    const auto subBatch = MakeSubBatch(AsuOpType::LOAD, StatusCode::CQE_KEY_NOT_EXIST);
    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::APPLICATION, RetryAction::RETURN_RESULT,
                   ChannelHealthAction::NO_CHANGE);
}

// ─── EC5: 未知/无明确语义 → 兜底 LINK_FAILURE + fallback 标记 ───

TEST(ErrorClassifierTest, UnknownCodesFallBackToLinkFailure)
{
    const StatusCode unknownCodes[] = {
        StatusCode::IO_ERROR,
        static_cast<StatusCode>(0x10000 | 0x999),  // 未定义的 CQE 值
    };
    for (const auto code : unknownCodes) {
        const auto subBatch = MakeSubBatch(AsuOpType::BATCH_LOAD, code);
        const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
        ExpectDecision(decision, ErrorCategory::LINK_FAILURE, RetryAction::FAILOVER,
                       ChannelHealthAction::RECORD_FAILURE);
        EXPECT_TRUE(decision.fallback) << "code=" << static_cast<int>(code);
    }
}

// ─── EC6: 排除清单 → 终态返回，不换链不记失败不降载 ───

TEST(ErrorClassifierTest, ExcludedStatusesTerminateWithoutRetry)
{
    const StatusCode excludedCodes[] = {
        StatusCode::INVALID_ARGUMENT,   StatusCode::NOT_INITIALIZED,
        StatusCode::TASK_NOT_FOUND,      StatusCode::UNSUPPORTED,
        StatusCode::BUFFER_NOT_REGISTERED, StatusCode::BUFFER_NOT_SUPPORTED,
        StatusCode::NOT_FOUND,           StatusCode::INTERNAL_ERROR,
        StatusCode::RESOURCE_BUSY,       StatusCode::CANCELED,
        StatusCode::PARTIAL_FAILED,     StatusCode::IN_PROGRESS,
    };
    for (const auto code : excludedCodes) {
        const auto subBatch = MakeSubBatch(AsuOpType::BATCH_LOAD, code);
        EXPECT_TRUE(ErrorClassifier::IsExcludedStatus(subBatch.status))
            << "code=" << static_cast<int>(code);
        const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
        ExpectDecision(decision, ErrorCategory::APPLICATION, RetryAction::RETURN_RESULT,
                       ChannelHealthAction::NO_CHANGE);
        EXPECT_FALSE(decision.fallback) << "code=" << static_cast<int>(code);
    }
}

TEST(ErrorClassifierTest, CoreCodesAreNotExcluded)
{
    const StatusCode coreCodes[] = {
        StatusCode::CONNECTION_ERROR, StatusCode::CQE_RESOURCE_BUSY,
        StatusCode::ENTRY_RETRY_ADVISED, StatusCode::CQE_IO_TIMEOUT,
    };
    for (const auto code : coreCodes) {
        EXPECT_FALSE(ErrorClassifier::IsExcludedStatus(MakeStatus(code)));
    }
}

// ─── EC7: CQE_CHECK_RESULT_BUFFER 逐 Entry 混合裁决 ───

// 纯应用码混合：不重试直接返回，hasMultipleErrorCodes=true。
TEST(ErrorClassifierTest, PureApplicationMixtureReturnsResult)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::BATCH_STORE, 3);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);
    subBatch.entryStatus[1] = MakeStatus(StatusCode::ENTRY_KEY_NOT_FOUND);
    subBatch.entryStatus[2] = MakeStatus(StatusCode::ENTRY_DATA_NOT_EXIST);

    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::APPLICATION, RetryAction::RETURN_RESULT,
                   ChannelHealthAction::NO_CHANGE);
    EXPECT_TRUE(decision.hasMultipleErrorCodes);
}

// 存在 ENTRY_RETRY_ADVISED：完整原始 subBatch 换链，无退避。
TEST(ErrorClassifierTest, RetryAdvisedEntryForcesFailover)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::BATCH_LOAD, 3);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);
    subBatch.entryStatus[0] = MakeStatus(StatusCode::ENTRY_KEY_NOT_FOUND);
    subBatch.entryStatus[2] = MakeStatus(StatusCode::ENTRY_RETRY_ADVISED);

    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::RETRY_ADVISORY, RetryAction::FAILOVER,
                   ChannelHealthAction::NO_CHANGE);
    EXPECT_TRUE(decision.hasMultipleErrorCodes);
}

// 应用错误与成功 Entry 随整体处理：全 OK entry 无失败码 → 无重试。
TEST(ErrorClassifierTest, AllEntriesOkUnderResultBufferIsApplication)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::BATCH_LOAD, 2);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);

    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::APPLICATION, RetryAction::RETURN_RESULT,
                   ChannelHealthAction::NO_CHANGE);
    EXPECT_FALSE(decision.hasMultipleErrorCodes);
}

// 未知 Entry 码（IO_ERROR 无明确语义）→ 兜底换链 + fallback 标记。
TEST(ErrorClassifierTest, UnknownEntryCodeFallsBackToLinkFailure)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::QUERY, 2);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);
    subBatch.entryStatus[0] = MakeStatus(StatusCode::IO_ERROR);
    subBatch.entryStatus[1] = MakeStatus(StatusCode::ENTRY_RETRY_ADVISED);

    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    ExpectDecision(decision, ErrorCategory::LINK_FAILURE, RetryAction::FAILOVER,
                   ChannelHealthAction::RECORD_FAILURE);
    EXPECT_TRUE(decision.fallback);
    EXPECT_TRUE(decision.hasMultipleErrorCodes);
}

// 空 entryStatus 边界：无失败码可判，按应用终态处理。
TEST(ErrorClassifierTest, EmptyEntryStatusUnderResultBufferIsApplication)
{
    auto subBatch = MakeSubBatch(AsuOpType::BATCH_LOAD, StatusCode::CQE_CHECK_RESULT_BUFFER);
    const auto decision = ErrorClassifier::ClassifySubBatch(subBatch, subBatch.status);
    EXPECT_EQ(decision.category, ErrorCategory::APPLICATION);
    EXPECT_FALSE(decision.hasMultipleErrorCodes);
}

// ─── IsNormalResult：正常业务结果前置排除（不得只调 status.ok()） ───

TEST(ErrorClassifierTest, IsNormalResultAcceptsOkStatus)
{
    const auto subBatch = MakeOkSubBatch(AsuOpType::BATCH_LOAD, 2);
    EXPECT_TRUE(ErrorClassifier::IsNormalResult(subBatch));
}

// Query miss：opType+status 驱动的正常查询结果。
TEST(ErrorClassifierTest, IsNormalResultTreatsQueryMissAsNormal)
{
    EXPECT_TRUE(ErrorClassifier::IsNormalResult(
        MakeSubBatch(AsuOpType::QUERY, StatusCode::CQE_KEY_NOT_EXIST)));
    EXPECT_TRUE(ErrorClassifier::IsNormalResult(
        MakeSubBatch(AsuOpType::QUERY, StatusCode::ENTRY_KEY_NOT_EXIST)));
}

// Query 结果缓冲携带逐 key exists/miss：正常。
TEST(ErrorClassifierTest, IsNormalResultTreatsQueryResultBufferAsNormal)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::QUERY, 3);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);
    subBatch.entryStatus[0] = MakeStatus(StatusCode::ENTRY_KEY_EXIST);
    subBatch.entryStatus[2] = MakeStatus(StatusCode::ENTRY_KEY_NOT_EXIST);
    EXPECT_TRUE(ErrorClassifier::IsNormalResult(subBatch));
}

// Query 结果缓冲内含非查询表达码（IO_ERROR）：不是正常结果。
TEST(ErrorClassifierTest, IsNormalResultRejectsQueryResultBufferWithUnknownEntry)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::QUERY, 2);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);
    subBatch.entryStatus[0] = MakeStatus(StatusCode::IO_ERROR);
    EXPECT_FALSE(ErrorClassifier::IsNormalResult(subBatch));
}

// 同一码不同 opType 等价类边界：CQE_KEY_NOT_EXIST 在 LOAD 下不是正常结果。
TEST(ErrorClassifierTest, IsNormalResultRejectsKeyMissOnLoadOps)
{
    EXPECT_FALSE(ErrorClassifier::IsNormalResult(
        MakeSubBatch(AsuOpType::LOAD, StatusCode::CQE_KEY_NOT_EXIST)));
    EXPECT_FALSE(ErrorClassifier::IsNormalResult(
        MakeSubBatch(AsuOpType::QUERY, StatusCode::CQE_IO_TIMEOUT)));
}

// 非 Query 的结果缓冲（Batch 表达逐 Entry 错误）不是正常结果。
TEST(ErrorClassifierTest, IsNormalResultRejectsNonQueryResultBuffer)
{
    auto subBatch = MakeOkSubBatch(AsuOpType::BATCH_LOAD, 2);
    subBatch.status = MakeStatus(StatusCode::CQE_CHECK_RESULT_BUFFER);
    subBatch.entryStatus[0] = MakeStatus(StatusCode::ENTRY_KEY_NOT_FOUND);
    EXPECT_FALSE(ErrorClassifier::IsNormalResult(subBatch));
}

// ENTRY_KEY_EXIST 是正常业务结果。
TEST(ErrorClassifierTest, IsNormalResultAcceptsKeyExist)
{
    EXPECT_TRUE(ErrorClassifier::IsNormalResult(
        MakeSubBatch(AsuOpType::QUERY, StatusCode::ENTRY_KEY_EXIST)));
    EXPECT_TRUE(ErrorClassifier::IsNormalResult(
        MakeSubBatch(AsuOpType::DELETE, StatusCode::ENTRY_KEY_EXIST)));
}

// ─── ClassifyCode 直查（单码映射，无 entry 分析） ───

TEST(ErrorClassifierTest, ClassifyCodeMapsSingleCodes)
{
    EXPECT_EQ(ErrorClassifier::ClassifyCode(StatusCode::CQE_IO_TIMEOUT, AsuOpType::LOAD).category,
              ErrorCategory::LINK_FAILURE);
    EXPECT_EQ(ErrorClassifier::ClassifyCode(StatusCode::CQE_RESOURCE_BUSY, AsuOpType::LOAD).category,
              ErrorCategory::BUSY);
    EXPECT_EQ(
        ErrorClassifier::ClassifyCode(StatusCode::ENTRY_RETRY_ADVISED, AsuOpType::LOAD).category,
        ErrorCategory::RETRY_ADVISORY);
    EXPECT_EQ(
        ErrorClassifier::ClassifyCode(StatusCode::CQE_KEY_ALREADY_EXISTED, AsuOpType::LOAD).category,
        ErrorCategory::APPLICATION);
    const auto fallback = ErrorClassifier::ClassifyCode(StatusCode::IO_ERROR, AsuOpType::LOAD);
    EXPECT_EQ(fallback.category, ErrorCategory::LINK_FAILURE);
    EXPECT_TRUE(fallback.fallback);
}

}  // namespace kv::test
