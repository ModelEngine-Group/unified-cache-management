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
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "core/transport_manager.h"
#include "dram/cc/drampool/buffer_manager.h"
#include "dram/cc/drampool/drampool_config.h"
#include "dram/cc/drampool/drampool_types.h"
#include "dram/cc/drampool/metadata.h"
#include "dram/cc/kv_protocol.h"
#include "dram/dram_test_common.h"
#include "pool/buffer_pool.h"
#include "status/status.h"
#include "trans/device.h"

// Match the white-box style used by the other DramPool unit tests. This keeps test access in
// this translation unit and does not add test-only seams to production code.
#define private public
#include "dram/cc/drampool/completion_poller.h"
#undef private

namespace UC::DramPool {
namespace {

constexpr std::size_t kQueueCapacity = 16;
constexpr std::size_t kValueLength = 16;
constexpr std::size_t kFlagSlotSize = 64;
constexpr std::uint64_t kRequestId = 42;
constexpr char kUnavailablePeer[] = "127.0.0.1:29000";

using UC::Test::Dram::Clock;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeBufferManager;
using UC::Test::Dram::MakeEntry;

std::uint8_t ResultCode(DumpLoadResult result) { return static_cast<std::uint8_t>(result); }

bool MarkTimeoutReportedForTest(CompletionPoller& poller, CompletionRecord& record,
                                std::uint64_t nowMs)
{
    if (record.timeout_reported || !poller.OperationTimedOut(record, nowMs)) { return false; }
    record.timeout_reported = true;
    return true;
}

class CompletionPollerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        auto status = device_.Init();
        deviceRuntimeOwned_ = status.Success();
        ASSERT_TRUE(deviceRuntimeOwned_ || status == Status::DuplicateKey()) << status.ToString();
        status = device_.Setup(0);
        ASSERT_TRUE(status.Success()) << status.ToString();
    }

    static void TearDownTestSuite()
    {
        if (!deviceRuntimeOwned_) { return; }
        EXPECT_TRUE(device_.Reset(0).Success());
        EXPECT_TRUE(device_.Finalize().Success());
        deviceRuntimeOwned_ = false;
    }

    inline static UC::Trans::Device device_;
    inline static bool deviceRuntimeOwned_{false};

    void SetUp() override
    {
        savedConfig_ = g_config;
        g_config.pollerPendingDepth = 2;
        g_config.opTimeoutMs = 100;

        requestQueue_.Setup(kQueueCapacity);
        completionQueue_.Setup(kQueueCapacity);
        bufferManager_ = MakeBufferManager({
            {kValueLength, kQueueCapacity}
        });
        const MetadataConfig metadataConfig{EvictionPolicyType::TTL, EvictionPolicyType::POSITION,
                                            std::chrono::milliseconds(100), 0.0};
        metadata_ = std::make_unique<MetadataManager>(metadataConfig, *bufferManager_);
        runtime_ = std::make_unique<DramPoolRuntime>(*metadata_, flagBufferPool_, manager_,
                                                     protocols_, requestQueue_, completionQueue_);
        poller_ = std::make_unique<CompletionPoller>(*runtime_);
    }

    void TearDown() override
    {
        poller_.reset();
        runtime_.reset();
        metadata_.reset();
        flagBufferPool_.Reset();
        bufferManager_.reset();
        g_config = std::move(savedConfig_);
    }

    void InitFlagBuffer(std::size_t slotSize = kFlagSlotSize, std::size_t slotCount = 1)
    {
        ASSERT_TRUE(flagBufferPool_
                        .Init("completion_poller_test_flags", BufferPool::MemoryType::Host,
                              slotSize, slotCount)
                        .Success());
    }

    EntryPtr ReserveEntry(const BlockId& key)
    {
        auto entry = MakeEntry(key, 0, Clock::now() + std::chrono::hours(1),
                               EntryStatus::INITIALIZED, 0, {}, kValueLength);
        EXPECT_TRUE(metadata_->StoreBegin(key, entry).Success());
        return entry;
    }

    EntryPtr PublishEntry(const BlockId& key)
    {
        auto entry = ReserveEntry(key);
        EXPECT_TRUE(metadata_->StoreEnd(key).Success());
        return entry;
    }

    CompletionRecord MakeResponseRecord(KvOpcode opcode = KvOpcode::Lookup)
    {
        CompletionRecord record;
        record.stage = CompletionStage::SubmitResponse;
        record.request_id = kRequestId;
        record.opcode = opcode;
        record.remote_resp_addr = 0x9000;
        record.peer_one_sided_id = kUnavailablePeer;
        record.results = {1, 0, 1};
        return record;
    }

    RequestQueue requestQueue_;
    CompletionQueue completionQueue_;
    std::unique_ptr<BufferManager> bufferManager_;
    std::unique_ptr<MetadataManager> metadata_;
    BufferPool flagBufferPool_;
    ProtocolManager protocols_;
    transport::TransportManager manager_{"127.0.0.1:28000"};
    std::unique_ptr<DramPoolRuntime> runtime_;
    std::unique_ptr<CompletionPoller> poller_;
    DramPoolConfig savedConfig_;
};

TEST_F(CompletionPollerTest, FillPendingWindowHonorsConfiguredDepthAndQueueOrder)
{
    for (std::uint64_t index = 1; index <= 3; ++index) {
        auto record = MakeResponseRecord();
        record.remote_resp_addr = index;
        completionQueue_.Push(std::move(record));
    }

    poller_->FillPendingWindow();
    ASSERT_EQ(poller_->pending_.size(), 2U);
    EXPECT_EQ(poller_->pending_[0].remote_resp_addr, 1U);
    EXPECT_EQ(poller_->pending_[1].remote_resp_addr, 2U);

    CompletionRecord remaining;
    ASSERT_TRUE(completionQueue_.TryPop(remaining));
    EXPECT_EQ(remaining.remote_resp_addr, 3U);
}

TEST_F(CompletionPollerTest, FillPendingWindowDoesNothingWhenDepthIsZero)
{
    g_config.pollerPendingDepth = 0;
    completionQueue_.Push(MakeResponseRecord());

    poller_->FillPendingWindow();
    EXPECT_TRUE(poller_->pending_.empty());
    CompletionRecord remaining;
    EXPECT_TRUE(completionQueue_.TryPop(remaining));
}

TEST_F(CompletionPollerTest, PollPendingRemovesEveryTerminalFailureAndInvalidStage)
{
    InitFlagBuffer(kFlagSlotSize, 4);

    CompletionRecord dataRecord;
    dataRecord.stage = CompletionStage::PollDataTransfer;
    dataRecord.data_handle = transport::kInvalidTransferHandle;

    auto submitRecord = MakeResponseRecord();
    submitRecord.peer_one_sided_id.clear();

    CompletionRecord responseRecord;
    responseRecord.stage = CompletionStage::PollResponseTransfer;
    responseRecord.response_handle = transport::kInvalidTransferHandle;
    ASSERT_TRUE(flagBufferPool_.Allocate(responseRecord.local_resp_slot).Success());

    CompletionRecord invalidRecord;
    invalidRecord.stage = static_cast<CompletionStage>(255);

    poller_->pending_.push_back(std::move(dataRecord));
    poller_->pending_.push_back(std::move(submitRecord));
    poller_->pending_.push_back(std::move(responseRecord));
    poller_->pending_.push_back(std::move(invalidRecord));
    poller_->PollPendingCompletions();
    EXPECT_TRUE(poller_->pending_.empty());
    BufferPool::Slot reused;
    ASSERT_TRUE(flagBufferPool_.Allocate(reused).Success());
    EXPECT_TRUE(flagBufferPool_.Free(reused.slotIndex).Success());
}

TEST_F(CompletionPollerTest, RunWithStopSetDrainsAllQueuedFailures)
{
    constexpr std::size_t kRecordCount = 5;
    InitFlagBuffer(kFlagSlotSize, kRecordCount);
    for (std::size_t index = 0; index < kRecordCount; ++index) {
        auto record = MakeResponseRecord();
        record.peer_one_sided_id.clear();
        completionQueue_.Push(std::move(record));
    }
    const std::atomic_bool stop{true};

    poller_->Run(stop);

    EXPECT_TRUE(poller_->pending_.empty());
    CompletionRecord remaining;
    EXPECT_FALSE(completionQueue_.TryPop(remaining));
}

TEST_F(CompletionPollerTest, DataStatusApiFailureAbortsDumpAndAdvancesToResponse)
{
    const auto key = KeyFromHex("a1");
    ReserveEntry(key);
    CompletionRecord record;
    record.stage = CompletionStage::PollDataTransfer;
    record.opcode = KvOpcode::Dump;
    record.data_handle = transport::kInvalidTransferHandle;
    record.results = {ResultCode(DumpLoadResult::Ok)};
    record.transfer_items = {
        TransferItem{0, key}
    };

    EXPECT_TRUE(poller_->PollDataTransfer(record));

    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_EQ(record.data_handle, transport::kInvalidTransferHandle);
    EXPECT_TRUE(record.transfer_items.empty());
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Failed)}));
    EXPECT_FALSE(metadata_->Query(key));
}

TEST_F(CompletionPollerTest, CompletedDumpPublishesReservedMetadata)
{
    const auto key = KeyFromHex("a2");
    ReserveEntry(key);
    CompletionRecord record;
    record.opcode = KvOpcode::Dump;
    record.results = {ResultCode(DumpLoadResult::Failed)};
    record.transfer_items = {
        TransferItem{0, key}
    };

    poller_->SettleDataTransfer(record, transport::TransferStatus::Completed);

    EXPECT_TRUE(record.transfer_items.empty());
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Ok)}));
    EXPECT_TRUE(metadata_->Exist(key));
}

TEST_F(CompletionPollerTest, CompletedDumpDeletesEntryWhenStoreEndFails)
{
    const auto key = KeyFromHex("a3");
    PublishEntry(key);
    CompletionRecord record;
    record.opcode = KvOpcode::Dump;
    record.results = {ResultCode(DumpLoadResult::Ok)};
    record.transfer_items = {
        TransferItem{0, key}
    };

    poller_->SettleDataTransfer(record, transport::TransferStatus::Completed);

    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Failed)}));
    EXPECT_FALSE(metadata_->Query(key));
}

TEST_F(CompletionPollerTest, FailedDumpHandlesMissingMetadataAndOutOfRangeResult)
{
    const auto reservedKey = KeyFromHex("a4");
    const auto missingKey = KeyFromHex("a5");
    ReserveEntry(reservedKey);
    CompletionRecord record;
    record.opcode = KvOpcode::Dump;
    record.results = {ResultCode(DumpLoadResult::Ok)};
    record.transfer_items = {
        TransferItem{0, reservedKey},
        TransferItem{2, missingKey }
    };

    poller_->SettleDataTransfer(record, transport::TransferStatus::Failed);

    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Failed)}));
    EXPECT_FALSE(metadata_->Query(reservedKey));
    EXPECT_TRUE(record.transfer_items.empty());
}

TEST_F(CompletionPollerTest, CompletedLoadReleasesReaderAndReturnsSuccess)
{
    const auto key = KeyFromHex("b1");
    auto entry = PublishEntry(key);
    EntryPtr loaded;
    ASSERT_TRUE(metadata_->LoadBegin(key, loaded).Success());
    ASSERT_EQ(entry->refCnt, 1U);
    CompletionRecord record;
    record.opcode = KvOpcode::Load;
    record.results = {ResultCode(DumpLoadResult::Failed)};
    record.transfer_items = {
        TransferItem{0, key}
    };

    poller_->SettleDataTransfer(record, transport::TransferStatus::Completed);

    EXPECT_EQ(entry->refCnt, 0U);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Ok)}));
}

TEST_F(CompletionPollerTest, FailedLoadReleasesReaderButKeepsFailureResult)
{
    const auto key = KeyFromHex("b2");
    auto entry = PublishEntry(key);
    EntryPtr loaded;
    ASSERT_TRUE(metadata_->LoadBegin(key, loaded).Success());
    CompletionRecord record;
    record.opcode = KvOpcode::Load;
    record.results = {ResultCode(DumpLoadResult::Ok)};
    record.transfer_items = {
        TransferItem{0, key}
    };

    poller_->SettleDataTransfer(record, transport::TransferStatus::Failed);

    EXPECT_EQ(entry->refCnt, 0U);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Failed)}));
}

TEST_F(CompletionPollerTest, MissingLoadMetadataRemainsFailure)
{
    CompletionRecord record;
    record.opcode = KvOpcode::Load;
    record.results = {ResultCode(DumpLoadResult::Ok)};
    record.transfer_items = {
        TransferItem{0, KeyFromHex("b3")}
    };

    poller_->SettleDataTransfer(record, transport::TransferStatus::Completed);

    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{ResultCode(DumpLoadResult::Failed)}));
}

TEST_F(CompletionPollerTest, OperationTimeoutHandlesBoundaryAndClockRollback)
{
    CompletionRecord record;
    record.submit_ms = 1'000;
    record.peer_one_sided_id = kUnavailablePeer;
    g_config.opTimeoutMs = 100;

    EXPECT_FALSE(poller_->OperationTimedOut(record, 999));
    EXPECT_FALSE(poller_->OperationTimedOut(record, 1'099));
    EXPECT_TRUE(poller_->OperationTimedOut(record, 1'100));
    EXPECT_FALSE(MarkTimeoutReportedForTest(*poller_, record, 1'099));
    EXPECT_TRUE(MarkTimeoutReportedForTest(*poller_, record, 1'100));
    EXPECT_TRUE(record.timeout_reported);
    EXPECT_FALSE(MarkTimeoutReportedForTest(*poller_, record, 1'200));
}

TEST_F(CompletionPollerTest, SubmitResponseRejectsMissingPeerAndUnknownOpcode)
{
    InitFlagBuffer();
    auto missingPeer = MakeResponseRecord();
    missingPeer.peer_one_sided_id.clear();
    EXPECT_TRUE(poller_->SubmitResponse(missingPeer));

    auto unknownOpcode = MakeResponseRecord(KvOpcode::None);
    EXPECT_TRUE(poller_->SubmitResponse(unknownOpcode));
    EXPECT_EQ(unknownOpcode.local_resp_slot.localAddr, nullptr);
}

TEST_F(CompletionPollerTest, SubmitResponseRejectsAlreadyOwnedBuffer)
{
    InitFlagBuffer();
    auto record = MakeResponseRecord();
    ASSERT_TRUE(flagBufferPool_.Allocate(record.local_resp_slot).Success());

    EXPECT_FALSE(poller_->SubmitResponse(record));

    EXPECT_NE(record.local_resp_slot.localAddr, nullptr);
    EXPECT_TRUE(flagBufferPool_.Free(record.local_resp_slot.slotIndex).Success());
}

TEST_F(CompletionPollerTest, SubmitResponseReleasesUndersizedBuffer)
{
    InitFlagBuffer(1);
    auto record = MakeResponseRecord();
    ASSERT_GT(protocols_.GetPackedResponseSize(record.opcode, record.results.size()), 1U);

    EXPECT_TRUE(poller_->SubmitResponse(record));

    EXPECT_EQ(record.local_resp_slot.localAddr, nullptr);
    BufferPool::Slot reused;
    ASSERT_TRUE(flagBufferPool_.Allocate(reused).Success());
    EXPECT_TRUE(flagBufferPool_.Free(reused.slotIndex).Success());
}

TEST_F(CompletionPollerTest, SubmitResponseReleasesBufferWhenProtocolRejectsResults)
{
    InitFlagBuffer();
    const std::vector<std::vector<std::uint8_t>> invalidResults{{}, {2}};
    for (const auto& results : invalidResults) {
        auto record = MakeResponseRecord();
        record.results = results;

        EXPECT_TRUE(poller_->SubmitResponse(record));

        EXPECT_EQ(record.local_resp_slot.localAddr, nullptr);
        BufferPool::Slot reused;
        ASSERT_TRUE(flagBufferPool_.Allocate(reused).Success());
        EXPECT_TRUE(flagBufferPool_.Free(reused.slotIndex).Success());
    }
}

TEST_F(CompletionPollerTest, SubmitResponseReleasesBufferWhenRealTransportRejectsPeer)
{
    InitFlagBuffer();
    auto record = MakeResponseRecord();

    EXPECT_TRUE(poller_->SubmitResponse(record));

    EXPECT_EQ(record.local_resp_slot.localAddr, nullptr);
    EXPECT_EQ(record.response_handle, transport::kInvalidTransferHandle);
    BufferPool::Slot reused;
    ASSERT_TRUE(flagBufferPool_.Allocate(reused).Success());
    EXPECT_TRUE(flagBufferPool_.Free(reused.slotIndex).Success());
}

TEST_F(CompletionPollerTest, SubmitResponseReturnsFalseOnNoSpaceWithoutMutatingRecord)
{
    InitFlagBuffer();
    BufferPool::Slot occupied;
    ASSERT_TRUE(flagBufferPool_.Allocate(occupied).Success());
    auto record = MakeResponseRecord();

    EXPECT_FALSE(poller_->SubmitResponse(record));

    EXPECT_EQ(record.local_resp_slot.localAddr, nullptr);
    EXPECT_TRUE(flagBufferPool_.Free(occupied.slotIndex).Success());
}

TEST_F(CompletionPollerTest, ResponseStatusApiFailureReleasesOwnedBuffer)
{
    InitFlagBuffer();
    CompletionRecord record;
    record.stage = CompletionStage::PollResponseTransfer;
    record.response_handle = transport::kInvalidTransferHandle;
    ASSERT_TRUE(flagBufferPool_.Allocate(record.local_resp_slot).Success());

    EXPECT_TRUE(poller_->PollResponseTransfer(record));

    EXPECT_EQ(record.local_resp_slot.localAddr, nullptr);
    BufferPool::Slot reused;
    ASSERT_TRUE(flagBufferPool_.Allocate(reused).Success());
    EXPECT_TRUE(flagBufferPool_.Free(reused.slotIndex).Success());
}

TEST_F(CompletionPollerTest, ResponseBufferReleaseFailureIsStillTerminal)
{
    CompletionRecord record;
    record.response_handle = transport::kInvalidTransferHandle;

    EXPECT_TRUE(poller_->PollResponseTransfer(record));
    EXPECT_EQ(record.local_resp_slot.localAddr, nullptr);
}

}  // namespace
}  // namespace UC::DramPool
