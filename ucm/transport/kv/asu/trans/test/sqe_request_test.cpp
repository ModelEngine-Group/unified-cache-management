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
#include "sqe_request.h"
#include <acl/acl.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "buffer_manager.h"
#include "kv_protocol.h"

namespace UC::ASU {

namespace {

constexpr std::size_t kFlagBufferHeaderSize = 16;
constexpr std::size_t kTestSendBufferSlotSize = 4096;
constexpr std::size_t kTestSendBufferSlotNum = 1;
constexpr std::size_t kFlagBufferSlotSize = 128;
constexpr std::size_t kFlagBufferSlotNum = 16;

std::unordered_map<std::string, std::string> DefaultAttrs()
{
    return {
        {"kv_ns_id",     "3"   },
        {"dtype",        "2"   },
        {"dspec",        "10"  },
        {"lr",           "true"},
        {"sc",           "true"},
        {"kernel_count", "1"   },
        {"quiet_count",  "5"   },
    };
}

std::vector<KVBuffer> MakeEntries(std::size_t count)
{
    std::vector<KVBuffer> entries(count);
    for (std::size_t index = 0; index < count; ++index) {
        entries[index].key = "key_" + std::to_string(index);
        entries[index].buffer.region.addr = 0x100000 + index * 0x1000;
        entries[index].buffer.region.size = 4096;
        entries[index].buffer.handle = 0x20 + index;
    }
    return entries;
}

}  // namespace

class SqeRequestTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
    }

    static void TearDownTestSuite()
    {
        aclrtResetDevice(0);
        aclFinalize();
    }

    void SetUp() override
    {
        auto status = flagBufferManager_.Init("test flag buffer", MemoryType::HOST,
                                              kFlagBufferSlotSize, kFlagBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = sendBufferManager_.Init("test send buffer", MemoryType::HOST,
                                         kTestSendBufferSlotSize, kTestSendBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        protocolManager_ = std::make_unique<ProtocolManager>();
    }

    BufferManager sendBufferManager_;
    BufferManager flagBufferManager_;
    std::unique_ptr<ProtocolManager> protocolManager_;
};

TEST_F(SqeRequestTest, ValidateSqeRequestAttrsRejectsMalformedValues)
{
    EXPECT_TRUE(ValidateSqeRequestAttrs(DefaultAttrs()).ok());

    auto attrs = DefaultAttrs();
    attrs["dtype"] = "256";
    EXPECT_EQ(ValidateSqeRequestAttrs(attrs).code, StatusCode::INVALID_ARGUMENT);

    attrs = DefaultAttrs();
    attrs["lr"] = "maybe";
    EXPECT_EQ(ValidateSqeRequestAttrs(attrs).code, StatusCode::INVALID_ARGUMENT);

    attrs = DefaultAttrs();
    attrs.erase("kernel_count");
    EXPECT_EQ(ValidateSqeRequestAttrs(attrs).code, StatusCode::INVALID_ARGUMENT);

    attrs = DefaultAttrs();
    attrs["quiet_count"] = "0";
    EXPECT_EQ(ValidateSqeRequestAttrs(attrs).code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(SqeRequestTest, SubmitBatchStoreAllocatesFlagBufferAndBuildsRequest)
{
    auto entries = MakeEntries(3);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    std::uint16_t nextCid = 41;

    const auto status = SubmitEntrySubBatchRequest(
        TransportOpType::BATCH_STORE, subBatch, DefaultAttrs(), [&nextCid] { return nextCid++; },
        sendBufferManager_, flagBufferManager_, *protocolManager_, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + (entries.size() + 1) / 2);
    EXPECT_EQ(subBatchContext.cid, std::uint32_t{41});
    EXPECT_EQ(subBatchContext.opType, TransportOpType::BATCH_STORE);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_NE(subBatchContext.sendSge.addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), entries.size());
    for (const auto& entryStatus : subBatchContext.entryStatus) { EXPECT_TRUE(entryStatus.ok()); }
}

TEST_F(SqeRequestTest, SubmitBatchRetrieveUsesRetrieveOpcodeAndRequest)
{
    auto entries = MakeEntries(2);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = SubmitEntrySubBatchRequest(
        TransportOpType::BATCH_LOAD, subBatch, DefaultAttrs(), [] { return std::uint16_t{9}; },
        sendBufferManager_, flagBufferManager_, *protocolManager_, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, TransportOpType::BATCH_LOAD);
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{9});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_NE(subBatchContext.sendSge.addr, std::uint64_t{0});
}

TEST_F(SqeRequestTest, SubmitDeleteCopiesKeysAndBuildsFlagBackedRequest)
{
    std::vector<CacheKey> keys = {"k0", "k1"};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = SubmitKeySubBatchRequest(
        TransportOpType::DELETE, subBatch, DefaultAttrs(), [] { return std::uint16_t{55}; },
        sendBufferManager_, flagBufferManager_, *protocolManager_, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, TransportOpType::DELETE);
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{55});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_NE(subBatchContext.sendSge.addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), keys.size());
    for (const auto& entryStatus : subBatchContext.entryStatus) { EXPECT_TRUE(entryStatus.ok()); }
}

TEST_F(SqeRequestTest, SubmitExistReadsScAttribute)
{
    std::vector<CacheKey> keys = {"k0"};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = SubmitKeySubBatchRequest(
        TransportOpType::QUERY, subBatch, DefaultAttrs(), [] { return std::uint16_t{13}; },
        sendBufferManager_, flagBufferManager_, *protocolManager_, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, TransportOpType::QUERY);
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{13});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_TRUE(subBatchContext.useSeekControl);
}

TEST_F(SqeRequestTest, SubmitExistDisablesSeekControlWhenScDisabled)
{
    auto attrs = DefaultAttrs();
    attrs["sc"] = "false";

    std::vector<CacheKey> keys = {"k0"};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = SubmitKeySubBatchRequest(
        TransportOpType::QUERY, subBatch, attrs, [] { return std::uint16_t{13}; },
        sendBufferManager_, flagBufferManager_, *protocolManager_, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_FALSE(subBatchContext.useSeekControl);
}

TEST_F(SqeRequestTest, AllocationFailureMarksWholeSubBatchFailed)
{
    auto entries = MakeEntries(2);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    BufferManager uninitializedFlagBufferManager;

    const auto status = SubmitEntrySubBatchRequest(
        TransportOpType::BATCH_STORE, subBatch, DefaultAttrs(), [] { return std::uint16_t{3}; },
        sendBufferManager_, uninitializedFlagBufferManager, *protocolManager_, subBatchContext);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::FAILED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(subBatchContext.flagBuffer.addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.sendSge.addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), entries.size());
    for (const auto& entryStatus : subBatchContext.entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::NOT_INITIALIZED);
    }
}

TEST_F(SqeRequestTest, SubmitKeepAliveBuildsFlagBackedRequest)
{
    TransportSubBatchContext subBatchContext;

    const auto status =
        SubmitKeepAliveRequest([] { return std::uint16_t{77}; }, sendBufferManager_,
                               flagBufferManager_, *protocolManager_, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{77});
    EXPECT_EQ(subBatchContext.opType, TransportOpType::KEEP_ALIVE);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + 1);
    EXPECT_NE(subBatchContext.sendSge.addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), 1);
    EXPECT_TRUE(subBatchContext.entryStatus[0].ok());
}

}  // namespace UC::ASU
