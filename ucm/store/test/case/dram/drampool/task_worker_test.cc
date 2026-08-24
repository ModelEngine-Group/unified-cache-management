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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "buffer_manager.h"
#include "core/transport_manager.h"
#include "dram/dram_test_common.h"
#include "drampool_config.h"
#include "drampool_types.h"
#include "kv_protocol.h"
#include "metadata.h"
#include "status/status.h"
#include "trans/device.h"

// Keep white-box access entirely in the test translation unit. Production code uses the real
// TransportManager directly and exposes no test-only interface.
#define private public
#include "task_worker.h"
#undef private

namespace UC::DramPool {
namespace {

constexpr std::size_t kQueueCapacity = 16;
constexpr std::uint32_t kValueLength = 16;
constexpr std::uint64_t kResponseAddress = 0x9000;
constexpr char kTargetManager[] = "127.0.0.1:29000";

using UC::Test::Dram::Clock;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeBufferManager;
using UC::Test::Dram::MakeEntry;

std::uint8_t DumpLoadCode(DumpLoadResult result) { return static_cast<std::uint8_t>(result); }

std::uint8_t LookupCode(LookupResult result) { return static_cast<std::uint8_t>(result); }

class TaskWorkerTest : public ::testing::Test {
protected:
    static constexpr std::uint64_t kRequestId = 42;

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
        g_config.flagBufferSlotSizeBytes = 64;
        g_config.defaultDumpTtlMs = 60'000;

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
    }

    void TearDown() override
    {
        runtime_.reset();
        metadata_.reset();
        bufferManager_.reset();
        g_config = std::move(savedConfig_);
    }

    EntryPtr PublishEntry(const BlockId& key)
    {
        auto entry = MakeEntry(key, 0, Clock::now() + std::chrono::hours(1),
                               EntryStatus::INITIALIZED, 0, {}, kValueLength);
        EXPECT_TRUE(metadata_->StoreBegin(key, entry).Success());
        EXPECT_TRUE(metadata_->StoreEnd(key).Success());
        return entry;
    }

    Status ProcessDump(KvDumpRequest& request)
    {
        TaskWorker worker(*runtime_);
        return worker.ProcessDump(request, kTargetManager, StartedTiming());
    }

    Status ProcessLoad(KvLoadRequest& request)
    {
        TaskWorker worker(*runtime_);
        return worker.ProcessLoad(request, kTargetManager, StartedTiming());
    }

    Status ProcessLookup(KvLookupRequest& request)
    {
        TaskWorker worker(*runtime_);
        return worker.ProcessLookup(request, kTargetManager, StartedTiming());
    }

    static RequestTiming StartedTiming()
    {
        RequestTiming timing;
        timing.received_us = SteadyNowUs();
        timing.received_ts_us = UnixNowUs();
        timing.worker_started_us = SteadyNowUs();
        timing.worker_started_ts_us = UnixNowUs();
        return timing;
    }

    CompletionRecord PopCompletion()
    {
        CompletionRecord record;
        EXPECT_TRUE(completionQueue_.TryPop(record));
        return record;
    }

    void ExpectNoCompletion()
    {
        CompletionRecord record;
        EXPECT_FALSE(completionQueue_.TryPop(record));
    }

    RequestQueue requestQueue_;
    CompletionQueue completionQueue_;
    std::unique_ptr<BufferManager> bufferManager_;
    std::unique_ptr<MetadataManager> metadata_;
    BufferPool flagBufferPool_;
    ProtocolManager protocols_;
    transport::TransportManager manager_{"127.0.0.1:28000"};
    std::unique_ptr<DramPoolRuntime> runtime_;
    DramPoolConfig savedConfig_;
};

TEST_F(TaskWorkerTest, RejectsMalformedTasksBeforeTransport)
{
    TaskWorker worker(*runtime_);
    EXPECT_TRUE(worker.ProcessOneRequest(nullptr).Failure());

    auto missingRequest = std::make_unique<RequestTask>();
    missingRequest->peer_one_sided_id = kTargetManager;
    EXPECT_TRUE(worker.ProcessOneRequest(std::move(missingRequest)).Failure());

    auto missingPeer = std::make_unique<RequestTask>();
    missingPeer->request = std::make_unique<KvLookupRequest>();
    EXPECT_TRUE(worker.ProcessOneRequest(std::move(missingPeer)).Failure());
    ExpectNoCompletion();
}

TEST_F(TaskWorkerTest, ProcessesRequestWithoutInitiatingPeerConnection)
{
    TaskWorker worker(*runtime_);
    auto task = std::make_unique<RequestTask>();
    task->peer_one_sided_id = kTargetManager;
    task->request = std::make_unique<KvLookupRequest>();
    task->request->opcode = OpType::LOOKUP;
    task->request->request_id = kRequestId;
    task->timing.received_us = SteadyNowUs();
    task->timing.received_ts_us = UnixNowUs();

    EXPECT_TRUE(worker.ProcessOneRequest(std::move(task)).Success());
    const auto record = PopCompletion();
    EXPECT_EQ(record.peer_one_sided_id, kTargetManager);
    EXPECT_EQ(record.request_id, kRequestId);
    EXPECT_NE(record.timing.worker_started_us, 0U);
    EXPECT_NE(record.timing.worker_started_ts_us, 0U);
    EXPECT_GE(record.timing.completion_queued_us, record.timing.worker_started_us);
}

TEST_F(TaskWorkerTest, LookupReturnsHitAndMiss)
{
    const auto hitKey = KeyFromHex("a1");
    PublishEntry(hitKey);

    KvLookupRequest request;
    request.opcode = OpType::LOOKUP;
    request.request_id = kRequestId;
    request.resp_addr = kResponseAddress;
    request.entries = {{hitKey}, {KeyFromHex("a2")}};
    request.batch_size = static_cast<std::uint16_t>(request.entries.size());
    ASSERT_TRUE(ProcessLookup(request).Success());

    const auto record = PopCompletion();
    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_EQ(record.opcode, OpType::LOOKUP);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{LookupCode(LookupResult::Exists),
                                                         LookupCode(LookupResult::NotFound)}));
}

TEST_F(TaskWorkerTest, DuplicateDumpIsIdempotent)
{
    const auto key = KeyFromHex("a1");
    PublishEntry(key);

    KvDumpRequest request;
    request.opcode = OpType::DUMP;
    request.request_id = kRequestId;
    request.resp_addr = kResponseAddress;
    request.entries = {
        {key, 0x1000, kValueLength, 0}
    };
    request.batch_size = static_cast<std::uint16_t>(request.entries.size());
    ASSERT_TRUE(ProcessDump(request).Success());

    const auto record = PopCompletion();
    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_FALSE(record.data_transfer_required);
    EXPECT_EQ(record.data_bytes, 0U);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{DumpLoadCode(DumpLoadResult::Ok)}));
    EXPECT_TRUE(metadata_->Exist(key));
}

TEST_F(TaskWorkerTest, DumpStopsAfterFirstStoreBeginFailure)
{
    const auto duplicateKey = KeyFromHex("a1");
    const auto failedKey = KeyFromHex("a2");
    const auto skippedKey = KeyFromHex("a3");
    PublishEntry(duplicateKey);

    KvDumpRequest request;
    request.opcode = OpType::DUMP;
    request.request_id = kRequestId;
    request.resp_addr = kResponseAddress;
    request.entries = {
        {duplicateKey, 0x1000, kValueLength,     0},
        {failedKey,    0x2000, kValueLength * 2, 1},
        {skippedKey,   0x3000, kValueLength,     2},
    };
    request.batch_size = static_cast<std::uint16_t>(request.entries.size());
    ASSERT_TRUE(ProcessDump(request).Success());

    const auto record = PopCompletion();
    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{DumpLoadCode(DumpLoadResult::Ok),
                                                         DumpLoadCode(DumpLoadResult::Failed),
                                                         DumpLoadCode(DumpLoadResult::Failed)}));
    EXPECT_TRUE(metadata_->Exist(duplicateKey));
    EXPECT_FALSE(metadata_->Query(failedKey));
    EXPECT_FALSE(metadata_->Query(skippedKey));
}

TEST_F(TaskWorkerTest, DumpSubmitFailureDeletesReservedMetadata)
{
    const auto key = KeyFromHex("a1");

    KvDumpRequest request;
    request.opcode = OpType::DUMP;
    request.request_id = kRequestId;
    request.resp_addr = kResponseAddress;
    request.entries = {
        {key, 0x1000, kValueLength, 0}
    };
    request.batch_size = static_cast<std::uint16_t>(request.entries.size());
    ASSERT_TRUE(ProcessDump(request).Success());

    const auto record = PopCompletion();
    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_TRUE(record.data_transfer_required);
    EXPECT_FALSE(record.data_transfer_submitted);
    EXPECT_EQ(record.data_bytes, kValueLength);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{DumpLoadCode(DumpLoadResult::Failed)}));
    EXPECT_FALSE(metadata_->Query(key));
    EXPECT_NE(record.timing.data_execute_async.manager_entered_us, 0U);
    EXPECT_NE(record.timing.data_execute_async.manager_entered_ts_us, 0U);
    EXPECT_EQ(record.timing.data_execute_async.backend_called_us, 0U);
}

TEST_F(TaskWorkerTest, LoadReportsMissingAndOversizedItems)
{
    const auto missingKey = KeyFromHex("a1");
    const auto oversizedKey = KeyFromHex("a2");
    const auto oversizedEntry = PublishEntry(oversizedKey);

    KvLoadRequest request;
    request.opcode = OpType::LOAD;
    request.request_id = kRequestId;
    request.resp_addr = kResponseAddress;
    request.entries = {
        {missingKey,   0x1000, kValueLength,     0},
        {oversizedKey, 0x2000, kValueLength + 1, 1},
    };
    request.batch_size = static_cast<std::uint16_t>(request.entries.size());
    ASSERT_TRUE(ProcessLoad(request).Success());

    const auto record = PopCompletion();
    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_FALSE(record.data_transfer_required);
    EXPECT_FALSE(record.data_transfer_submitted);
    EXPECT_EQ(record.data_bytes, 0U);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{DumpLoadCode(DumpLoadResult::Failed),
                                                         DumpLoadCode(DumpLoadResult::Failed)}));
    EXPECT_EQ(oversizedEntry->refCnt, 0U);
}

TEST_F(TaskWorkerTest, LoadSubmitFailureEndsAllPinnedItems)
{
    const auto firstKey = KeyFromHex("a1");
    const auto secondKey = KeyFromHex("a2");
    const auto firstEntry = PublishEntry(firstKey);
    const auto secondEntry = PublishEntry(secondKey);

    KvLoadRequest request;
    request.opcode = OpType::LOAD;
    request.request_id = kRequestId;
    request.resp_addr = kResponseAddress;
    request.entries = {
        {firstKey,  0x1000, kValueLength, 0},
        {secondKey, 0x2000, kValueLength, 1},
    };
    request.batch_size = static_cast<std::uint16_t>(request.entries.size());
    ASSERT_TRUE(ProcessLoad(request).Success());

    const auto record = PopCompletion();
    EXPECT_EQ(record.stage, CompletionStage::SubmitResponse);
    EXPECT_TRUE(record.data_transfer_required);
    EXPECT_FALSE(record.data_transfer_submitted);
    EXPECT_EQ(record.data_bytes, kValueLength * 2);
    EXPECT_EQ(record.results, (std::vector<std::uint8_t>{DumpLoadCode(DumpLoadResult::Failed),
                                                         DumpLoadCode(DumpLoadResult::Failed)}));
    EXPECT_EQ(firstEntry->refCnt, 0U);
    EXPECT_EQ(secondEntry->refCnt, 0U);
    EXPECT_TRUE(metadata_->Exist(firstKey));
    EXPECT_TRUE(metadata_->Exist(secondKey));
    EXPECT_NE(record.timing.data_execute_async.manager_entered_us, 0U);
    EXPECT_NE(record.timing.data_execute_async.manager_entered_ts_us, 0U);
    EXPECT_EQ(record.timing.data_execute_async.backend_called_us, 0U);
}

TEST_F(TaskWorkerTest, RejectsResponsesLargerThanFlagBufferSlot)
{
    g_config.flagBufferSlotSizeBytes = 0;

    KvDumpRequest dump;
    dump.opcode = OpType::DUMP;
    dump.request_id = kRequestId;
    dump.batch_size = 1;
    dump.entries = {
        {KeyFromHex("a1"), 0x1000, kValueLength, 0}
    };
    EXPECT_TRUE(ProcessDump(dump).Failure());

    KvLoadRequest load;
    load.opcode = OpType::LOAD;
    load.request_id = kRequestId;
    load.batch_size = 1;
    load.entries = {
        {KeyFromHex("a2"), 0x2000, kValueLength, 0}
    };
    EXPECT_TRUE(ProcessLoad(load).Failure());

    KvLookupRequest lookup;
    lookup.opcode = OpType::LOOKUP;
    lookup.request_id = kRequestId;
    lookup.batch_size = 1;
    lookup.entries = {{KeyFromHex("a3")}};
    EXPECT_TRUE(ProcessLookup(lookup).Failure());
    ExpectNoCompletion();
}

}  // namespace
}  // namespace UC::DramPool
