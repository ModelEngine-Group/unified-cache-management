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
#include "utils/trans_task_utils.h"
#include "task/trans_task_manager.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace kv::test {
namespace {

TEST(TransTaskUtilsTest, MapsCqeRawStatusToSubBatchStatus)
{
    EXPECT_TRUE(KvResponseStatusToSubBatchStatus(0x00).ok());
    EXPECT_EQ(KvResponseStatusToSubBatchStatus(0x731).code, StatusCode::ASU_CQE_RESOURCE_BUSY);
    EXPECT_EQ(KvResponseStatusToSubBatchStatus(0xFFFF).code, StatusCode::IO_ERROR);
}

TEST(TransTaskUtilsTest, SuccessfulSubBatchFillsAllEntriesOk)
{
    KvResponse response;
    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::BATCH_STORE;
    subBatchContext.status = Status::OK();
    subBatchContext.entryStatus.assign(3, Status::Error(StatusCode::IO_ERROR, "old"));

    FillEntryStatusFromCqeResult(response, subBatchContext);

    for (const auto& status : subBatchContext.entryStatus) { EXPECT_TRUE(status.ok()); }
}

TEST(TransTaskUtilsTest, CheckResultBufferFillsPerEntryBatchStatus)
{
    KvResponse response;
    response.result_buffer = {0x00, 0x03, 0x04};

    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::BATCH_LOAD;
    subBatchContext.status =
        Status::Error(StatusCode::ASU_CQE_CHECK_RESULT_BUFFER, "check result buffer");
    subBatchContext.entryStatus.assign(3, Status::OK());

    FillEntryStatusFromCqeResult(response, subBatchContext);

    EXPECT_TRUE(subBatchContext.entryStatus[0].ok());
    EXPECT_EQ(subBatchContext.entryStatus[1].code, StatusCode::ASU_ENTRY_KEY_NOT_FOUND);
    EXPECT_EQ(subBatchContext.entryStatus[2].code, StatusCode::ASU_ENTRY_DATA_NOT_EXIST);
}

TEST(TransTaskUtilsTest, QueryOkWithoutResultBufferMarksAllKeysExist)
{
    KvResponse response;
    response.existing_key_number = 1;

    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::QUERY;
    subBatchContext.status = Status::OK();
    subBatchContext.entryStatus.assign(3, Status::OK());

    FillEntryStatusFromCqeResult(response, subBatchContext);
    const auto queryResult = BuildQueryResultFromEntryStatus(subBatchContext.entryStatus);

    ASSERT_EQ(queryResult.exists.size(), std::size_t{3});
    EXPECT_EQ(queryResult.exists[0], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[1], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[2], std::uint8_t{1});
    EXPECT_EQ(queryResult.prefixHitKeys, std::uint32_t{3});
}

TEST(TransTaskUtilsTest, QueryOkIgnoresExistingKeyNumber)
{
    KvResponse response;
    response.existing_key_number = 2;

    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::QUERY;
    subBatchContext.useSeekControl = false;
    subBatchContext.status = Status::OK();
    subBatchContext.entryStatus.assign(4, Status::OK());

    FillEntryStatusFromCqeResult(response, subBatchContext);
    const auto queryResult = BuildQueryResultFromEntryStatus(subBatchContext.entryStatus);

    ASSERT_EQ(queryResult.exists.size(), std::size_t{4});
    EXPECT_EQ(queryResult.exists[0], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[1], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[2], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[3], std::uint8_t{1});
    EXPECT_EQ(queryResult.prefixHitKeys, std::uint32_t{4});
}

TEST(TransTaskUtilsTest, QueryCheckResultBufferWithoutSeekControlUsesExistingKeyNumber)
{
    KvResponse response;
    response.existing_key_number = 2;

    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::QUERY;
    subBatchContext.useSeekControl = false;
    subBatchContext.status =
        Status::Error(StatusCode::ASU_CQE_CHECK_RESULT_BUFFER, "check result buffer");
    subBatchContext.entryStatus.assign(4, Status::OK());

    FillEntryStatusFromCqeResult(response, subBatchContext);
    const auto queryResult = BuildQueryResultFromEntryStatus(subBatchContext.entryStatus);

    ASSERT_EQ(queryResult.exists.size(), std::size_t{4});
    EXPECT_EQ(queryResult.exists[0], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[1], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[2], std::uint8_t{0});
    EXPECT_EQ(queryResult.exists[3], std::uint8_t{0});
    EXPECT_EQ(queryResult.prefixHitKeys, std::uint32_t{2});
}

TEST(TransTaskUtilsTest, QueryCheckResultBufferUsesExistEntryStatuses)
{
    KvResponse response;
    response.result_buffer = {0x01, 0x00, 0x01};

    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::QUERY;
    subBatchContext.useSeekControl = true;
    subBatchContext.status =
        Status::Error(StatusCode::ASU_CQE_CHECK_RESULT_BUFFER, "check result buffer");
    subBatchContext.entryStatus.assign(3, Status::OK());

    FillEntryStatusFromCqeResult(response, subBatchContext);
    const auto queryResult = BuildQueryResultFromEntryStatus(subBatchContext.entryStatus);

    ASSERT_EQ(queryResult.exists.size(), std::size_t{3});
    EXPECT_EQ(queryResult.exists[0], std::uint8_t{1});
    EXPECT_EQ(queryResult.exists[1], std::uint8_t{0});
    EXPECT_EQ(queryResult.exists[2], std::uint8_t{1});
    EXPECT_EQ(queryResult.prefixHitKeys, std::uint32_t{1});
}

TEST(TransTaskUtilsTest, MissingResultBufferPropagatesSubBatchError)
{
    KvResponse response;
    TransportSubBatchContext subBatchContext;
    subBatchContext.opType = AsuOpType::DELETE;
    subBatchContext.status = Status::Error(StatusCode::ASU_CQE_RESOURCE_BUSY, "busy");
    subBatchContext.entryStatus.assign(2, Status::OK());

    FillEntryStatusFromCqeResult(response, subBatchContext);

    EXPECT_EQ(subBatchContext.entryStatus[0].code, StatusCode::ASU_CQE_RESOURCE_BUSY);
    EXPECT_EQ(subBatchContext.entryStatus[1].code, StatusCode::ASU_CQE_RESOURCE_BUSY);
}

}  // namespace
}  // namespace kv::test
