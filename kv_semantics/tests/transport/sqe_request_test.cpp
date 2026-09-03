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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#define private public
#include "asu_transport_impl.h"
#undef private
#include "trans_provider.h"
#include "buffer_manager.h"
#include "kv_protocol.h"
#include "transport_config_parser.h"

namespace UC::ASU {

namespace {

TEST(TransportConfigParserTest, LoadsMaxErrorCount)
{
    constexpr const char* kConfigPath = "asu_transport_max_error_count_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "max_error_count=7\n"
                   << "kernel_count=1\n"
                   << "quiet_count=1\n";
    }

    TransportConfig config;
    auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.maxErrorCount, std::uint32_t{7});
}

TEST(TransportConfigParserTest, LoadsCompletionPollSpinLimit)
{
    constexpr const char* kConfigPath = "asu_transport_completion_poll_spin_limit_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "completion_poll_spin_limit=23\n"
                   << "kernel_count=1\n"
                   << "quiet_count=1\n";
    }

    TransportConfig config;
    const auto status = LoadTransportConfig(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.completionPollSpinLimit, std::size_t{23});
}

CacheKey MakeCacheKey(std::string_view text)
{
    CacheKey key{};
    const auto size = std::min(text.size(), key.size());
    if (size > 0) { std::memcpy(key.data(), text.data(), size); }
    return key;
}

class StubTransProvider : public TransProvider {
public:
    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t, uint32_t,
                            std::vector<ConnectionHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>&) override
    {
        return {};
    }
    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>&, uint32_t,
                             uint32_t) override
    {
        return {};
    }
    Status RegisterMemory(const std::vector<RegisterMemoryDesc>&,
                          std::vector<MRHandle>& handles) override
    {
        handles.push_back(reinterpret_cast<MRHandle>(static_cast<uintptr_t>(1)));
        return Status::OK();
    }
    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>&) override
    {
        return {};
    }
    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> FreeThread(const std::vector<ThreadHandle>&) override { return {}; }
    Status GetMemTokenId(MRHandle, uint32_t& tokenId) override
    {
        tokenId = 1;
        return Status::OK();
    }
};

void CreateTaskExecutor(AsuTransportImpl& transport)
{
    transport.taskExecutor_ = std::make_unique<TransportTaskExecutor>(
        transport.config_, transport.transProvider_, transport.connManager_);
}

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
        entries[index].key = MakeCacheKey("key_" + std::to_string(index));
        entries[index].buffer.region.addr = 0x100000 + index * 0x1000;
        entries[index].buffer.region.size = 4096;
        entries[index].buffer.handle = 0x20 + index;
    }
    return entries;
}

void SetMrKeys(std::vector<KVBuffer>& entries, std::uint32_t tokenBase)
{
    for (std::size_t index = 0; index < entries.size(); ++index) {
        entries[index].mrKey = tokenBase + static_cast<std::uint32_t>(index);
    }
}

std::uint32_t PackedBatchEntryMrKey(const std::uint32_t* sqe, std::size_t entryIndex)
{
    const auto base = kSqeDwordCount + entryIndex * kBatchEntryDwordCount;
    return ((sqe[base + 7] >> 24) & 0xFF) | ((sqe[base + 8] & 0xFFFFFF) << 8);
}

}  // namespace

class SqeRequestTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        transport_ = std::make_unique<AsuTransportImpl>();
        transport_->SetTransProvider(std::make_unique<StubTransProvider>());
        transport_->config_.attrs = DefaultAttrs();
        CreateTaskExecutor(*transport_);
        auto status = transport_->taskExecutor_->flagBufferManager_.Init(
            "test flag buffer", MemoryType::HOST_PINNED, kFlagBufferSlotSize, kFlagBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->taskExecutor_->sendBufferManager_.Init(
            "test send buffer", MemoryType::HOST_PINNED, kTestSendBufferSlotSize,
            kTestSendBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->taskExecutor_->RegisterBufferMemory(
            transport_->taskExecutor_->sendBufferManager_,
            transport_->taskExecutor_->sendBufferMrHandle_);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->taskExecutor_->RegisterBufferMemory(
            transport_->taskExecutor_->flagBufferManager_,
            transport_->taskExecutor_->flagBufferMrHandle_);
        ASSERT_TRUE(status.ok()) << status.message;
    }

    std::unique_ptr<AsuTransportImpl> transport_;
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
    entries[0].offset = kAlignmentBytes;
    entries[1].offset = kAlignmentBytes * 2;
    entries[2].offset = kAlignmentBytes * 3;
    SetMrKeys(entries, 0xABCD0000);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    transport_->taskExecutor_->nextRequestCid_.store(41, std::memory_order_relaxed);
    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        AsuOpType::BATCH_STORE, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + (entries.size() + 1) / 2);
    EXPECT_EQ(subBatchContext.cid, std::uint32_t{41});
    EXPECT_EQ(subBatchContext.opType, AsuOpType::BATCH_STORE);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    EXPECT_NE(subBatchContext.sendSge.device_addr, std::uint64_t{0});
    EXPECT_NE(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
    EXPECT_NE(subBatchContext.flagBuffer.device_addr, std::uint64_t{0});

    const auto* packedSqe =
        reinterpret_cast<const std::uint32_t*>(subBatchContext.sendSge.local_addr);
    const auto packedResponseAddr =
        static_cast<std::uint64_t>(packedSqe[3]) | (static_cast<std::uint64_t>(packedSqe[4]) << 32);
    EXPECT_EQ(packedResponseAddr, subBatchContext.flagBuffer.device_addr);
    ASSERT_EQ(subBatchContext.entryStatus.size(), entries.size());
    for (const auto& entryStatus : subBatchContext.entryStatus) { EXPECT_TRUE(entryStatus.ok()); }
    const auto* sqe = reinterpret_cast<const std::uint32_t*>(subBatchContext.sendSge.local_addr);
    EXPECT_EQ(sqe[kSqeDwordCount], entries[0].offset);
    EXPECT_EQ(sqe[kSqeDwordCount + kBatchEntryDwordCount], entries[1].offset);
    EXPECT_EQ(sqe[kSqeDwordCount + 2 * kBatchEntryDwordCount], entries[2].offset);
    EXPECT_EQ(PackedBatchEntryMrKey(sqe, 0), std::uint32_t{0xABCD0000});
    EXPECT_EQ(PackedBatchEntryMrKey(sqe, 1), std::uint32_t{0xABCD0001});
    EXPECT_EQ(PackedBatchEntryMrKey(sqe, 2), std::uint32_t{0xABCD0002});
}

TEST_F(SqeRequestTest, SubmitStoreUsesStoreOpcodeAndRequest)
{
    auto entries = MakeEntries(1);
    entries[0].offset = kAlignmentBytes;
    SetMrKeys(entries, 0xABCD0000);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        AsuOpType::STORE, subBatch, subBatchContext);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, AsuOpType::STORE);
    const auto* sqe = reinterpret_cast<const std::uint32_t*>(subBatchContext.sendSge.local_addr);
    EXPECT_EQ(sqe[0] & 0xFF, static_cast<std::uint32_t>(KvOpcode::Store));
    EXPECT_EQ(static_cast<std::uint64_t>(sqe[6]) | (static_cast<std::uint64_t>(sqe[7]) << 32),
              entries[0].buffer.region.addr);
    EXPECT_EQ(sqe[8] & 0xFFFFFF, entries[0].buffer.region.size);
    EXPECT_EQ(sqe[10], entries[0].offset);
    EXPECT_EQ(sqe[11] & 0xFFFFFF, entries[0].buffer.region.size);
}

TEST_F(SqeRequestTest, SubmitBatchRetrieveUsesRetrieveOpcodeAndRequest)
{
    auto entries = MakeEntries(2);
    entries[0].offset = kAlignmentBytes * 4;
    entries[1].offset = kAlignmentBytes * 5;
    SetMrKeys(entries, 0x76540000);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    transport_->taskExecutor_->nextRequestCid_.store(9, std::memory_order_relaxed);
    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        AsuOpType::BATCH_LOAD, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, AsuOpType::BATCH_LOAD);
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{9});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    const auto* sqe = reinterpret_cast<const std::uint32_t*>(subBatchContext.sendSge.local_addr);
    EXPECT_EQ(sqe[kSqeDwordCount], entries[0].offset);
    EXPECT_EQ(sqe[kSqeDwordCount + kBatchEntryDwordCount], entries[1].offset);
    EXPECT_EQ(PackedBatchEntryMrKey(sqe, 0), std::uint32_t{0x76540000});
    EXPECT_EQ(PackedBatchEntryMrKey(sqe, 1), std::uint32_t{0x76540001});
}

TEST_F(SqeRequestTest, SubmitLoadUsesRetrieveOpcodeAndRequest)
{
    auto entries = MakeEntries(1);
    entries[0].offset = kAlignmentBytes * 4;
    SetMrKeys(entries, 0x76540000);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        AsuOpType::LOAD, subBatch, subBatchContext);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, AsuOpType::LOAD);
    const auto* sqe = reinterpret_cast<const std::uint32_t*>(subBatchContext.sendSge.local_addr);
    EXPECT_EQ(sqe[0] & 0xFF, static_cast<std::uint32_t>(KvOpcode::Retrieve));
    EXPECT_EQ(sqe[10], entries[0].offset);
    EXPECT_EQ(sqe[11] & 0xFFFFFF, entries[0].buffer.region.size);
}

TEST_F(SqeRequestTest, SubmitBatchStoreRejectsUnregisteredEntryBuffer)
{
    auto entries = MakeEntries(1);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        AsuOpType::BATCH_STORE, subBatch, subBatchContext);

    EXPECT_EQ(status.code, StatusCode::BUFFER_NOT_REGISTERED);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::BUFFER_NOT_REGISTERED);
    ASSERT_EQ(subBatchContext.entryStatus.size(), entries.size());
    EXPECT_EQ(subBatchContext.entryStatus[0].code, StatusCode::BUFFER_NOT_REGISTERED);
    EXPECT_EQ(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.sendSge.local_addr, std::uint64_t{0});
}

TEST_F(SqeRequestTest, SubmitDeleteCopiesKeysAndBuildsFlagBackedRequest)
{
    std::vector<CacheKey> keys = {MakeCacheKey("k0"), MakeCacheKey("k1"), MakeCacheKey("k2"),
                                  MakeCacheKey("k3"), MakeCacheKey("k4"), MakeCacheKey("k5"),
                                  MakeCacheKey("k6"), MakeCacheKey("k7"), MakeCacheKey("k8")};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;
    transport_->taskExecutor_->nextRequestCid_.store(55, std::memory_order_relaxed);

    const auto status = transport_->taskExecutor_->SubmitKeySubBatchRequest(
        AsuOpType::DELETE, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, AsuOpType::DELETE);
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{55});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + (keys.size() + 7) / 8);
    EXPECT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), keys.size());
    for (const auto& entryStatus : subBatchContext.entryStatus) { EXPECT_TRUE(entryStatus.ok()); }
}

TEST_F(SqeRequestTest, SubmitExistReadsScAttribute)
{
    std::vector<CacheKey> keys = {MakeCacheKey("k0"), MakeCacheKey("k1"), MakeCacheKey("k2"),
                                  MakeCacheKey("k3"), MakeCacheKey("k4"), MakeCacheKey("k5"),
                                  MakeCacheKey("k6"), MakeCacheKey("k7"), MakeCacheKey("k8")};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;
    transport_->taskExecutor_->nextRequestCid_.store(13, std::memory_order_relaxed);

    const auto status = transport_->taskExecutor_->SubmitKeySubBatchRequest(
        AsuOpType::QUERY, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, AsuOpType::QUERY);
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{13});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_TRUE(subBatchContext.useSeekControl);
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + (keys.size() + 7) / 8);
}

TEST_F(SqeRequestTest, SubmitExistDisablesSeekControlWhenScDisabled)
{
    auto attrs = DefaultAttrs();
    attrs["sc"] = "false";
    transport_->config_.attrs = attrs;
    transport_->taskExecutor_->nextRequestCid_.store(13, std::memory_order_relaxed);

    std::vector<CacheKey> keys = {MakeCacheKey("k0")};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = transport_->taskExecutor_->SubmitKeySubBatchRequest(
        AsuOpType::QUERY, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_FALSE(subBatchContext.useSeekControl);
}

TEST_F(SqeRequestTest, AllocationFailureMarksWholeSubBatchFailed)
{
    auto entries = MakeEntries(2);
    SetMrKeys(entries, 0x45670000);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    AsuTransportImpl uninitializedFlagTransport;
    uninitializedFlagTransport.SetTransProvider(std::make_unique<StubTransProvider>());
    uninitializedFlagTransport.config_.attrs = DefaultAttrs();
    CreateTaskExecutor(uninitializedFlagTransport);
    ASSERT_TRUE(uninitializedFlagTransport.taskExecutor_->sendBufferManager_
                    .Init("test send buffer", MemoryType::HOST, kTestSendBufferSlotSize,
                          kTestSendBufferSlotNum)
                    .ok());
    uninitializedFlagTransport.taskExecutor_->nextRequestCid_.store(3, std::memory_order_relaxed);
    const auto status = uninitializedFlagTransport.taskExecutor_->BuildEntrySubBatchRequest(
        AsuOpType::BATCH_STORE, subBatch, subBatchContext);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), entries.size());
    for (const auto& entryStatus : subBatchContext.entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::NOT_INITIALIZED);
    }
}

TEST_F(SqeRequestTest, SubmitKeepAliveBuildsFlagBackedRequest)
{
    TransportSubBatchContext subBatchContext;
    transport_->taskExecutor_->nextRequestCid_.store(77, std::memory_order_relaxed);

    const auto status = transport_->taskExecutor_->SubmitKeepAliveRequest(subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{77});
    EXPECT_EQ(subBatchContext.opType, AsuOpType::KEEP_ALIVE);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + 1);
    EXPECT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), 1);
    EXPECT_TRUE(subBatchContext.entryStatus[0].ok());
}

}  // namespace UC::ASU
