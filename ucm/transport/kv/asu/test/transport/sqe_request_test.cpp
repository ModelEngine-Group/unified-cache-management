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
#include <acl/acl.h>
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
#include "buffer_manager.h"
#include "kv_protocol.h"
#include "trans_provider.h"
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
        transport.config_, transport.ioScheduler_, transport.transProvider_,
        transport.sendBufferManager_, transport.flagBufferManager_, transport.protocolManager_,
        transport.connManager_, transport.nextRequestCid_, transport.registeredRegionsMu_,
        transport.registeredRegions_);
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

void BindEntries(AsuTransportImpl& transport, const std::vector<KVBuffer>& entries,
                 std::uint32_t tokenBase)
{
    for (std::size_t index = 0; index < entries.size(); ++index) {
        RegisteredMemory memory;
        memory.region = entries[index].buffer.region;
        memory.handle = entries[index].buffer.handle;
        memory.tokenId = tokenBase + static_cast<std::uint32_t>(index);
        transport.registeredRegions_[memory.handle] = memory;
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
        transport_ = std::make_unique<AsuTransportImpl>();
        transport_->SetTransProvider(std::make_unique<StubTransProvider>());
        transport_->config_.attrs = DefaultAttrs();
        transport_->nextRequestCid_.store(1, std::memory_order_relaxed);
        auto* provider = transport_->transProvider_.get();
        auto status =
            transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST_PINNED,
                                                kFlagBufferSlotSize, kFlagBufferSlotNum, provider);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST_PINNED,
                                                     kTestSendBufferSlotSize,
                                                     kTestSendBufferSlotNum, provider);
        ASSERT_TRUE(status.ok()) << status.message;
        transport_->protocolManager_ = std::make_unique<ProtocolManager>();
        CreateTaskExecutor(*transport_);
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
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    transport_->nextRequestCid_.store(41, std::memory_order_relaxed);
    BindEntries(*transport_, entries, 0xABCD0000);

    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        TransportOpType::BATCH_STORE, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + (entries.size() + 1) / 2);
    EXPECT_EQ(subBatchContext.cid, std::uint32_t{41});
    EXPECT_EQ(subBatchContext.opType, TransportOpType::BATCH_STORE);
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

TEST_F(SqeRequestTest, SubmitBatchStorePacksSqeIntoDeviceSendBuffer)
{
    AsuTransportImpl deviceTransport;
    deviceTransport.SetTransProvider(std::make_unique<StubTransProvider>());
    deviceTransport.config_.attrs = DefaultAttrs();
    deviceTransport.nextRequestCid_.store(41, std::memory_order_relaxed);
    auto* provider = deviceTransport.transProvider_.get();
    ASSERT_TRUE(deviceTransport.flagBufferManager_
                    .Init("test flag buffer", MemoryType::HOST_PINNED, kFlagBufferSlotSize,
                          kFlagBufferSlotNum, provider)
                    .ok());
    ASSERT_TRUE(deviceTransport.sendBufferManager_
                    .Init("test send buffer", MemoryType::ASCEND_DEVICE, kTestSendBufferSlotSize,
                          kTestSendBufferSlotNum, provider)
                    .ok());
    deviceTransport.protocolManager_ = std::make_unique<ProtocolManager>();
    CreateTaskExecutor(deviceTransport);

    auto entries = MakeEntries(3);
    BindEntries(deviceTransport, entries, 0x12340000);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = deviceTransport.taskExecutor_->BuildEntrySubBatchRequest(
        TransportOpType::BATCH_STORE, subBatch, subBatchContext);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.sendSge.memory_type, MemoryType::ASCEND_DEVICE);
    EXPECT_NE(subBatchContext.sendSge.device_addr, std::uint64_t{0});
    std::vector<std::uint8_t> packed(subBatchContext.sendSge.length);
    ASSERT_EQ(aclrtMemcpy(packed.data(), packed.size(),
                          reinterpret_cast<void*>(subBatchContext.sendSge.device_addr),
                          packed.size(), ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    EXPECT_TRUE(
        deviceTransport.protocolManager_->VerifyPackedBuffer(packed.data(), packed.size()).ok());
}

TEST_F(SqeRequestTest, SubmitBatchRetrieveUsesRetrieveOpcodeAndRequest)
{
    auto entries = MakeEntries(2);
    entries[0].offset = kAlignmentBytes * 4;
    entries[1].offset = kAlignmentBytes * 5;
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;
    transport_->nextRequestCid_.store(9, std::memory_order_relaxed);
    BindEntries(*transport_, entries, 0x76540000);

    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        TransportOpType::BATCH_LOAD, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, TransportOpType::BATCH_LOAD);
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

TEST_F(SqeRequestTest, SubmitBatchStoreUsesDefaultMrKeyForUnregisteredEntryBuffer)
{
    auto entries = MakeEntries(1);
    IoScheduler::ScheduledIoBatch subBatch{
        BatchView<KVBuffer>{entries.data(), entries.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = transport_->taskExecutor_->BuildEntrySubBatchRequest(
        TransportOpType::BATCH_STORE, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    const auto* sqe = reinterpret_cast<const std::uint32_t*>(subBatchContext.sendSge.local_addr);
    ASSERT_NE(sqe, nullptr);
    EXPECT_EQ(PackedBatchEntryMrKey(sqe, 0), std::uint32_t{1});
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
    transport_->nextRequestCid_.store(55, std::memory_order_relaxed);

    const auto status = transport_->taskExecutor_->SubmitKeySubBatchRequest(
        TransportOpType::DELETE, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, TransportOpType::DELETE);
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
    transport_->nextRequestCid_.store(13, std::memory_order_relaxed);

    const auto status = transport_->taskExecutor_->SubmitKeySubBatchRequest(
        TransportOpType::QUERY, subBatch, subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.opType, TransportOpType::QUERY);
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
    transport_->nextRequestCid_.store(13, std::memory_order_relaxed);

    std::vector<CacheKey> keys = {MakeCacheKey("k0")};
    IoScheduler::ScheduledKeyBatch subBatch{
        BatchView<CacheKey>{keys.data(), keys.size()}
    };
    TransportSubBatchContext subBatchContext;

    const auto status = transport_->taskExecutor_->SubmitKeySubBatchRequest(
        TransportOpType::QUERY, subBatch, subBatchContext);

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
    AsuTransportImpl uninitializedFlagTransport;
    uninitializedFlagTransport.SetTransProvider(std::make_unique<StubTransProvider>());
    uninitializedFlagTransport.config_.attrs = DefaultAttrs();
    uninitializedFlagTransport.nextRequestCid_.store(3, std::memory_order_relaxed);
    ASSERT_TRUE(uninitializedFlagTransport.sendBufferManager_
                    .Init("test send buffer", MemoryType::HOST, kTestSendBufferSlotSize,
                          kTestSendBufferSlotNum)
                    .ok());
    uninitializedFlagTransport.protocolManager_ = std::make_unique<ProtocolManager>();
    CreateTaskExecutor(uninitializedFlagTransport);
    BindEntries(uninitializedFlagTransport, entries, 0x45670000);

    const auto status = uninitializedFlagTransport.taskExecutor_->BuildEntrySubBatchRequest(
        TransportOpType::BATCH_STORE, subBatch, subBatchContext);

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
    transport_->nextRequestCid_.store(77, std::memory_order_relaxed);

    const auto status = transport_->taskExecutor_->SubmitKeepAliveRequest(subBatchContext);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(subBatchContext.cid, std::uint16_t{77});
    EXPECT_EQ(subBatchContext.opType, TransportOpType::KEEP_ALIVE);
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContext.status.ok());
    EXPECT_EQ(subBatchContext.flagBuffer.length, kFlagBufferHeaderSize + 1);
    EXPECT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    ASSERT_EQ(subBatchContext.entryStatus.size(), 1);
    EXPECT_TRUE(subBatchContext.entryStatus[0].ok());
}

}  // namespace UC::ASU
