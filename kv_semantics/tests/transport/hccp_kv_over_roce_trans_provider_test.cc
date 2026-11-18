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
#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include "hccp_kv_over_roce/hccp_kv_over_roce_trans_provider.h"
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "../support/fake_hccp_kv_over_roce_abi.h"
#include "hccp_kv_over_roce/hbm_slot_allocator.h"
#include "hccp_kv_over_roce/hccp_kv_over_roce_abi.h"
#include "trans_provider.h"

namespace kv::test {
namespace {

constexpr std::uint32_t kTestCapacity = 4;

// Every control symbol exported by the fake library, resolved through dlsym so
// the test always talks to the exact library instance the provider loaded.
#define KV_FAKE_CONTROL_FUNCTIONS(X)                              \
    X(reset, HccpFakeReset)                                       \
    X(setCreateResult, HccpFakeSetCreateResult)                   \
    X(setCreateReportedCount, HccpFakeSetCreateReportedCount)     \
    X(setCreateManagerNull, HccpFakeSetCreateManagerNull)         \
    X(setDeleteResult, HccpFakeSetDeleteResult)                   \
    X(setRegisterResult, HccpFakeSetRegisterResult)               \
    X(setRegisterZeroRkeyIndex, HccpFakeSetRegisterZeroRkeyIndex) \
    X(setUnregisterResult, HccpFakeSetUnregisterResult)           \
    X(setSendResult, HccpFakeSetSendResult)                       \
    X(setSendBatchErrorIndex, HccpFakeSetSendBatchErrorIndex)     \
    X(setAbiVersion, HccpFakeSetAbiVersion)                       \
    X(createCalls, HccpFakeCreateCalls)                           \
    X(deleteCalls, HccpFakeDeleteCalls)                           \
    X(registerCalls, HccpFakeRegisterCalls)                       \
    X(unregisterCalls, HccpFakeUnregisterCalls)                   \
    X(sendCalls, HccpFakeSendCalls)                               \
    X(capabilitiesCalls, HccpFakeCapabilitiesCalls)               \
    X(handlesPrefilled, HccpFakeHandlesPrefilled)                 \
    X(qpcByte, HccpFakeQpcByte)                                   \
    X(sendSlotCount, HccpFakeSendSlotCount)                       \
    X(sendSlot, HccpFakeSendSlot)                                 \
    X(deleteManagerCount, HccpFakeDeleteManagerCount)             \
    X(deleteManager, HccpFakeDeleteManager)                       \
    X(lastManager, HccpFakeLastManager)                           \
    X(capabilitiesManager, HccpFakeCapabilitiesManager)           \
    X(registerDescCount, HccpFakeRegisterDescCount)               \
    X(registerDescMemoryType, HccpFakeRegisterDescMemoryType)     \
    X(registerDescAddr, HccpFakeRegisterDescAddr)                 \
    X(registerDescSize, HccpFakeRegisterDescSize)                 \
    X(registerDescLocalAddr, HccpFakeRegisterDescLocalAddr)       \
    X(unregisterInfoCount, HccpFakeUnregisterInfoCount)           \
    X(unregisterInfoRkey, HccpFakeUnregisterInfoRkey)

struct FakeApi {
#define KV_FAKE_DECLARE(member, symbol) decltype(&::symbol) member{nullptr};
    KV_FAKE_CONTROL_FUNCTIONS(KV_FAKE_DECLARE)
#undef KV_FAKE_DECLARE

    ~FakeApi() { Close(); }

    bool Open(const char* path)
    {
        Close();
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) { return false; }
#define KV_FAKE_LOAD(member, symbol)                                        \
    member = reinterpret_cast<decltype(&::symbol)>(dlsym(handle, #symbol)); \
    if (member == nullptr) {                                                \
        Close();                                                            \
        return false;                                                       \
    }
        KV_FAKE_CONTROL_FUNCTIONS(KV_FAKE_LOAD)
#undef KV_FAKE_LOAD
        reset();
        return true;
    }

    void Close()
    {
        if (handle == nullptr) { return; }
        (void)dlclose(handle);
        handle = nullptr;
    }

    void* handle{nullptr};
};

class HccpKvOverRoceTransProviderTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(aclrtGetDevice(&device_), ACL_SUCCESS);
        ASSERT_TRUE(fake_.Open(UCM_HCCP_FAKE_ABI_FULL_PATH)) << dlerror();
    }

    void TearDown() override
    {
        provider_.reset();
        fake_.Close();
    }

    bool UseLibrary(const char* path)
    {
        provider_.reset();
        return fake_.Open(path);
    }

    TransportConfig MakeConfig(std::uint32_t capacity = kTestCapacity,
                               const std::string& libraryPath = UCM_HCCP_FAKE_ABI_FULL_PATH) const
    {
        TransportConfig config;
        config.deviceId = device_;
        config.nodeId = 1;
        config.attrs["hccp_kv_roce_lib_path"] = libraryPath;
        config.attrs["hccp_kv_roce_max_connections"] = std::to_string(capacity);
        return config;
    }

    void CreateProvider(std::uint32_t capacity = kTestCapacity,
                        const std::string& libraryPath = UCM_HCCP_FAKE_ABI_FULL_PATH)
    {
        provider_ =
            std::make_unique<HccpKvOverRoceTransProvider>(MakeConfig(capacity, libraryPath));
    }

    std::vector<TransProvider::ConnectionHandle> CreateConnections(
        std::uint32_t qpNum, const std::string& remoteIp = "192.168.1.2")
    {
        std::vector<TransProvider::ConnectionHandle> handles;
        const auto status =
            provider_->CreateConnection("192.168.1.1", remoteIp, 12345, qpNum, 100, handles);
        EXPECT_TRUE(status.ok()) << status.message;
        return handles;
    }

    FakeApi fake_;
    std::unique_ptr<HccpKvOverRoceTransProvider> provider_;
    std::int32_t device_{0};
};

TEST_F(HccpKvOverRoceTransProviderTest, FactoryRejectsMissingLibrary)
{
    auto config = MakeConfig();
    config.providerType = TransProviderType::HCCP_KV_OVER_ROCE;
    config.attrs["hccp_kv_roce_lib_path"] = "/nonexistent/libhccp_fake_missing.so";

    std::shared_ptr<TransProvider> provider;
    const auto status = CreateTransProvider(config, provider);
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_FALSE(provider);
    EXPECT_NE(status.message.find("failed to create HCCP KV-over-RoCE provider"),
              std::string::npos);
}

TEST_F(HccpKvOverRoceTransProviderTest, ConstructorRejectsMissingLibrary)
{
    EXPECT_THROW(
        {
            HccpKvOverRoceTransProvider provider(
                MakeConfig(kTestCapacity, "/nonexistent/libhccp_fake.so"));
            (void)provider;
        },
        std::runtime_error);
}

TEST_F(HccpKvOverRoceTransProviderTest, ConstructorRejectsZeroSlotCapacity)
{
    EXPECT_THROW(
        {
            HccpKvOverRoceTransProvider provider(MakeConfig(0));
            (void)provider;
        },
        std::runtime_error);
}

TEST_F(HccpKvOverRoceTransProviderTest, ConstructorRejectsAbiVersionMismatch)
{
    fake_.setAbiVersion(HCCP_KV_ROCE_ABI_VERSION + 1);
    EXPECT_THROW(
        {
            HccpKvOverRoceTransProvider provider(MakeConfig());
            (void)provider;
        },
        std::runtime_error);
}

TEST_F(HccpKvOverRoceTransProviderTest, ConstructorRejectsMissingRequiredSymbol)
{
    ASSERT_TRUE(UseLibrary(UCM_HCCP_FAKE_ABI_MISSING_DELETE_PATH));
    EXPECT_THROW(
        {
            HccpKvOverRoceTransProvider provider(
                MakeConfig(kTestCapacity, UCM_HCCP_FAKE_ABI_MISSING_DELETE_PATH));
            (void)provider;
        },
        std::runtime_error);
}

TEST_F(HccpKvOverRoceTransProviderTest, FactoryCreatesProvider)
{
    auto config = MakeConfig();
    config.providerType = TransProviderType::HCCP_KV_OVER_ROCE;

    std::shared_ptr<TransProvider> provider;
    ASSERT_TRUE(CreateTransProvider(config, provider).ok());
    ASSERT_TRUE(provider != nullptr);

    std::vector<TransProvider::ConnectionHandle> handles;
    ASSERT_TRUE(provider->CreateConnection("1.1.1.1", "2.2.2.2", 1, 1, 10, handles).ok());
    EXPECT_EQ(handles.size(), 1U);
    EXPECT_EQ(fake_.createCalls(), 1U);

    provider.reset();
    EXPECT_EQ(fake_.deleteCalls(), 1U);
}

TEST_F(HccpKvOverRoceTransProviderTest, CreateConnectionValidatesArguments)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    std::vector<TransProvider::ConnectionHandle> handles{reinterpret_cast<void*>(0x1)};
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "2.2.2.2", 1, 0, 10, handles).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(handles.empty());

    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "", 1, 1, 10, handles).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.createCalls(), 0U);
}

TEST_F(HccpKvOverRoceTransProviderTest, CreateConnectionRejectsCapacityOverflow)
{
    CreateProvider(2);
    ASSERT_TRUE(provider_ != nullptr);

    std::vector<TransProvider::ConnectionHandle> handles;
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "2.2.2.2", 1, 3, 10, handles).code,
              StatusCode::RESOURCE_BUSY);
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.createCalls(), 0U);

    handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);

    // Every slot is now owned by the first pair, so one more QP cannot fit.
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "3.3.3.3", 1, 1, 10, handles).code,
              StatusCode::RESOURCE_BUSY);
    EXPECT_EQ(fake_.createCalls(), 1U);
}

TEST_F(HccpKvOverRoceTransProviderTest, CreateConnectionAllocatesOneQpcPerQp)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);
    EXPECT_NE(handles[0], nullptr);
    EXPECT_NE(handles[1], nullptr);
    EXPECT_NE(handles[0], handles[1]);
    EXPECT_EQ(fake_.createCalls(), 1U);
    EXPECT_EQ(fake_.handlesPrefilled(), 1);
    EXPECT_EQ(fake_.deleteCalls(), 0U);

    void* firstManager = fake_.lastManager();
    EXPECT_NE(firstManager, nullptr);

    auto other = CreateConnections(1, "192.168.1.3");
    ASSERT_EQ(other.size(), 1U);
    EXPECT_NE(other[0], handles[0]);
    EXPECT_NE(other[0], handles[1]);
    EXPECT_EQ(fake_.createCalls(), 2U);
    EXPECT_NE(fake_.lastManager(), nullptr);
    EXPECT_NE(fake_.lastManager(), firstManager);
}

TEST_F(HccpKvOverRoceTransProviderTest, CreateConnectionRejectsMissingManager)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);
    fake_.setCreateManagerNull(1);

    std::vector<TransProvider::ConnectionHandle> handles;
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "2.2.2.2", 1, 2, 10, handles).code,
              StatusCode::INTERNAL_ERROR);
    EXPECT_TRUE(handles.empty());
    // Without a manager there is nothing the operator could delete.
    EXPECT_EQ(fake_.deleteCalls(), 0U);

    // All slots must have been released by the failed call.
    fake_.setCreateManagerNull(0);
    handles = CreateConnections(kTestCapacity);
    EXPECT_EQ(handles.size(), kTestCapacity);
}

TEST_F(HccpKvOverRoceTransProviderTest, CreateConnectionRejectsReportedCountMismatch)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    fake_.setCreateReportedCount(2, 0);
    std::vector<TransProvider::ConnectionHandle> handles;
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "2.2.2.2", 1, 3, 10, handles).code,
              StatusCode::INTERNAL_ERROR);
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.deleteCalls(), 1U);
    EXPECT_EQ(fake_.deleteManager(0), fake_.lastManager());

    fake_.setCreateReportedCount(5, 0);
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "2.2.2.2", 1, 3, 10, handles).code,
              StatusCode::INTERNAL_ERROR);
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.deleteCalls(), 2U);

    fake_.setCreateReportedCount(0, 1);
    handles = CreateConnections(kTestCapacity);
    EXPECT_EQ(handles.size(), kTestCapacity);
    EXPECT_EQ(fake_.deleteCalls(), 2U);
}

TEST_F(HccpKvOverRoceTransProviderTest, CreateConnectionReportsOperatorFailure)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);
    fake_.setCreateResult(HCCP_KV_ROCE_CONNECTION_ERROR);

    std::vector<TransProvider::ConnectionHandle> handles;
    EXPECT_EQ(provider_->CreateConnection("1.1.1.1", "2.2.2.2", 1, 2, 10, handles).code,
              StatusCode::CONNECTION_ERROR);
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.deleteCalls(), 0U);

    // The failed call must release every slot it acquired.
    fake_.setCreateResult(0);
    handles = CreateConnections(kTestCapacity);
    EXPECT_EQ(handles.size(), kTestCapacity);
}

TEST_F(HccpKvOverRoceTransProviderTest, DeleteConnectionsRequiresWholeIpPair)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);

    const auto partial = provider_->DeleteConnections({handles[0]});
    ASSERT_EQ(partial.size(), 1U);
    EXPECT_EQ(partial[0].code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(fake_.deleteCalls(), 0U);

    // The rejected subset must leave the pair fully usable.
    ServerKvCapabilities capabilities;
    EXPECT_TRUE(provider_->GetServerCapabilities(handles[0], capabilities).ok());
    EXPECT_TRUE(provider_->GetServerCapabilities(handles[1], capabilities).ok());

    const auto complete = provider_->DeleteConnections(handles);
    ASSERT_EQ(complete.size(), 2U);
    EXPECT_TRUE(complete[0].ok());
    EXPECT_TRUE(complete[1].ok());
    EXPECT_EQ(fake_.deleteCalls(), 1U);
}

TEST_F(HccpKvOverRoceTransProviderTest, DeleteConnectionsRejectsInvalidHandles)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);

    const auto duplicates = provider_->DeleteConnections({handles[0], handles[0], handles[1]});
    ASSERT_EQ(duplicates.size(), 3U);
    for (const auto& status : duplicates) { EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT); }
    EXPECT_EQ(fake_.deleteCalls(), 0U);

    const auto unknown = provider_->DeleteConnections({nullptr, reinterpret_cast<void*>(0x1)});
    ASSERT_EQ(unknown.size(), 2U);
    EXPECT_EQ(unknown[0].code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(unknown[1].code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(fake_.deleteCalls(), 0U);

    EXPECT_TRUE(provider_->DeleteConnections({}).empty());
}

TEST_F(HccpKvOverRoceTransProviderTest, DeleteConnectionsDeletesMultipleIpPairs)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto pairA = CreateConnections(1, "192.168.1.2");
    ASSERT_EQ(pairA.size(), 1U);
    void* managerA = fake_.lastManager();
    auto pairB = CreateConnections(2, "192.168.1.3");
    ASSERT_EQ(pairB.size(), 2U);
    void* managerB = fake_.lastManager();
    ASSERT_NE(managerA, nullptr);
    ASSERT_NE(managerB, nullptr);
    ASSERT_NE(managerA, managerB);

    const auto results = provider_->DeleteConnections({pairA[0], pairB[0], pairB[1]});
    ASSERT_EQ(results.size(), 3U);
    for (const auto& status : results) { EXPECT_TRUE(status.ok()); }

    EXPECT_EQ(fake_.deleteCalls(), 1U);
    ASSERT_EQ(fake_.deleteManagerCount(), 2U);
    const std::unordered_set<void*> managers{fake_.deleteManager(0), fake_.deleteManager(1)};
    EXPECT_EQ(managers.count(managerA), 1U);
    EXPECT_EQ(managers.count(managerB), 1U);
}

TEST_F(HccpKvOverRoceTransProviderTest, DeleteConnectionsRollsBackOnOperatorFailure)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(1);
    ASSERT_EQ(handles.size(), 1U);
    fake_.setDeleteResult(HCCP_KV_ROCE_INTERNAL_ERROR);

    const auto failed = provider_->DeleteConnections(handles);
    ASSERT_EQ(failed.size(), 1U);
    EXPECT_EQ(failed[0].code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(fake_.deleteCalls(), 1U);

    // The record is restored so a retry can still reach the operator.
    ServerKvCapabilities capabilities;
    EXPECT_TRUE(provider_->GetServerCapabilities(handles[0], capabilities).ok());

    fake_.setDeleteResult(0);
    const auto retried = provider_->DeleteConnections(handles);
    ASSERT_EQ(retried.size(), 1U);
    EXPECT_TRUE(retried[0].ok());
    EXPECT_EQ(fake_.deleteCalls(), 2U);
}

TEST_F(HccpKvOverRoceTransProviderTest, DestructorDeletesOpenConnections)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);
    void* manager = fake_.lastManager();
    ASSERT_NE(manager, nullptr);

    provider_.reset();
    EXPECT_EQ(fake_.deleteCalls(), 1U);
    ASSERT_EQ(fake_.deleteManagerCount(), 1U);
    EXPECT_EQ(fake_.deleteManager(0), manager);
}

TEST_F(HccpKvOverRoceTransProviderTest, SendForwardsSlotAddresses)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);

    std::uint8_t sendBuffer[16]{};
    std::uint8_t flagBuffer[8]{};
    const std::vector<TransProvider::SendIoBatch> batches{
        {handles[1], sendBuffer, flagBuffer, sizeof(sendBuffer)},
        {handles[0], sendBuffer, flagBuffer, sizeof(sendBuffer)}
    };
    const auto statuses = provider_->Send(batches, 1, 0);
    ASSERT_EQ(statuses.size(), 2U);
    EXPECT_TRUE(statuses[0].ok());
    EXPECT_TRUE(statuses[1].ok());
    EXPECT_EQ(fake_.sendCalls(), 1U);

    ASSERT_EQ(fake_.sendSlotCount(), 2U);
    auto* slotForQp1 = static_cast<std::uint8_t*>(fake_.sendSlot(0));
    auto* slotForQp0 = static_cast<std::uint8_t*>(fake_.sendSlot(1));
    ASSERT_NE(slotForQp1, nullptr);
    ASSERT_NE(slotForQp0, nullptr);
    EXPECT_EQ(
        reinterpret_cast<std::uintptr_t>(slotForQp1) - reinterpret_cast<std::uintptr_t>(slotForQp0),
        kHccpKvRoceConnSlotBytes);

    // The operator reads the QPC out of the slot, so the slot must hold exactly
    // the bytes filled by CreateConnection.
    std::vector<std::uint8_t> qpc(HCCP_KV_ROCE_QPC_BYTES, 0);
    ASSERT_EQ(
        aclrtMemcpy(qpc.data(), qpc.size(), slotForQp0, qpc.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    for (std::size_t index = 0; index < qpc.size(); ++index) {
        EXPECT_EQ(qpc[index], static_cast<std::uint8_t>(fake_.qpcByte(0))) << index;
    }

    auto* tailAddress =
        reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(slotForQp0) + qpc.size());
    std::uint8_t tail = 0xFF;
    ASSERT_EQ(
        aclrtMemcpy(&tail, sizeof(tail), tailAddress, sizeof(tail), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(tail, 0);
}

TEST_F(HccpKvOverRoceTransProviderTest, SendSkipsInvalidHandles)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);

    const std::vector<TransProvider::SendIoBatch> batches{
        {handles[0], nullptr, nullptr, 0},
        {nullptr,    nullptr, nullptr, 0},
        {handles[1], nullptr, nullptr, 0}
    };
    const auto statuses = provider_->Send(batches, 1, 0);
    ASSERT_EQ(statuses.size(), 3U);
    EXPECT_TRUE(statuses[0].ok());
    EXPECT_EQ(statuses[1].code, StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(statuses[2].ok());
    EXPECT_EQ(fake_.sendCalls(), 1U);
    EXPECT_EQ(fake_.sendSlotCount(), 2U);

    // A batch made only of invalid handles never reaches the operator.
    const auto rejected = provider_->Send(
        {
            {nullptr, nullptr, nullptr, 0}
    },
        1, 0);
    ASSERT_EQ(rejected.size(), 1U);
    EXPECT_EQ(rejected[0].code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(fake_.sendCalls(), 1U);

    EXPECT_TRUE(provider_->Send({}, 1, 0).empty());
    EXPECT_EQ(fake_.sendCalls(), 1U);
}

TEST_F(HccpKvOverRoceTransProviderTest, SendMapsOperatorAndBatchFailures)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);
    const std::vector<TransProvider::SendIoBatch> batches{
        {handles[0], nullptr, nullptr, 0},
        {handles[1], nullptr, nullptr, 0}
    };

    fake_.setSendBatchErrorIndex(1);
    const auto partial = provider_->Send(batches, 1, 0);
    ASSERT_EQ(partial.size(), 2U);
    EXPECT_TRUE(partial[0].ok());
    EXPECT_EQ(partial[1].code, StatusCode::IO_ERROR);

    fake_.setSendBatchErrorIndex(-1);
    fake_.setSendResult(HCCP_KV_ROCE_IO_ERROR);
    const auto failed = provider_->Send(batches, 1, 0);
    ASSERT_EQ(failed.size(), 2U);
    EXPECT_EQ(failed[0].code, StatusCode::IO_ERROR);
    EXPECT_EQ(failed[1].code, StatusCode::IO_ERROR);
    EXPECT_EQ(fake_.sendCalls(), 2U);
}

TEST_F(HccpKvOverRoceTransProviderTest, GetServerCapabilitiesUsesConnectionManager)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(2);
    ASSERT_EQ(handles.size(), 2U);
    void* manager = fake_.lastManager();
    ASSERT_NE(manager, nullptr);

    ServerKvCapabilities capabilities;
    ASSERT_TRUE(provider_->GetServerCapabilities(handles[1], capabilities).ok());
    EXPECT_EQ(capabilities.queueNum, 8U);
    EXPECT_EQ(capabilities.ioQueueDepth, 32U);
    EXPECT_EQ(capabilities.ioQueueKeyConcurrency, 3U);
    EXPECT_EQ(capabilities.connectionKeyConcurrency, 4U);
    EXPECT_EQ(capabilities.singleValueMaxBytes, 4096U);
    EXPECT_EQ(capabilities.batchValueMaxBytes, 8192U);
    EXPECT_EQ(capabilities.batchStoreKeys, 16U);
    EXPECT_EQ(capabilities.batchLoadKeys, 17U);
    EXPECT_EQ(capabilities.deleteKeys, 18U);
    EXPECT_EQ(capabilities.queryKeys, 19U);
    EXPECT_EQ(capabilities.keyLength, 64U);
    EXPECT_EQ(capabilities.kvCapabilities, 5U);
    EXPECT_EQ(fake_.capabilitiesCalls(), 1U);
    EXPECT_EQ(fake_.capabilitiesManager(), manager);
}

TEST_F(HccpKvOverRoceTransProviderTest, GetServerCapabilitiesRejectsUnknownHandle)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(1);
    ASSERT_EQ(handles.size(), 1U);

    ServerKvCapabilities capabilities;
    capabilities.queueNum = 99;
    EXPECT_EQ(provider_->GetServerCapabilities(nullptr, capabilities).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(capabilities.queueNum, 0U);

    EXPECT_EQ(provider_->GetServerCapabilities(reinterpret_cast<void*>(0x1), capabilities).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(fake_.capabilitiesCalls(), 0U);
}

TEST_F(HccpKvOverRoceTransProviderTest, RegisterMemoryRegistersDeviceMemory)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096, 0x8000},
        {TransProvider::MemType::MEM_DEVICE, 0x2000, 8192, 0x9000}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->RegisterMemory(descs, handles).ok());

    ASSERT_EQ(handles.size(), 2U);
    EXPECT_NE(handles[0], kInvalidMRHandle);
    EXPECT_NE(handles[1], kInvalidMRHandle);
    EXPECT_NE(handles[0], handles[1]);
    EXPECT_EQ(fake_.registerCalls(), 1U);

    ASSERT_EQ(fake_.registerDescCount(), 2U);
    EXPECT_EQ(fake_.registerDescMemoryType(0), HCCP_KV_ROCE_MEM_DEVICE);
    EXPECT_EQ(fake_.registerDescAddr(0), 0x1000U);
    EXPECT_EQ(fake_.registerDescSize(0), 4096U);
    EXPECT_EQ(fake_.registerDescLocalAddr(0), 0x8000U);
    EXPECT_EQ(fake_.registerDescAddr(1), 0x2000U);
    EXPECT_EQ(fake_.registerDescSize(1), 8192U);
    EXPECT_EQ(fake_.registerDescLocalAddr(1), 0x9000U);

    std::uint32_t tokenId = 0;
    ASSERT_TRUE(provider_->GetMemTokenId(handles[0], tokenId).ok());
    EXPECT_EQ(tokenId, 1000U);
    ASSERT_TRUE(provider_->GetMemTokenId(handles[1], tokenId).ok());
    EXPECT_EQ(tokenId, 1001U);
}

TEST_F(HccpKvOverRoceTransProviderTest, RegisterMemoryRejectsInvalidDescriptors)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    std::vector<MRHandle> handles{7, 8};
    ASSERT_TRUE(provider_->RegisterMemory({}, handles).ok());
    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.registerCalls(), 0U);

    const std::vector<TransProvider::RegisterMemoryDesc> host{
        {TransProvider::MemType::MEM_HOST, 0x1000, 4096}
    };
    EXPECT_EQ(provider_->RegisterMemory(host, handles).code, StatusCode::INVALID_ARGUMENT);

    const std::vector<TransProvider::RegisterMemoryDesc> zeroAddr{
        {TransProvider::MemType::MEM_DEVICE, 0, 4096}
    };
    EXPECT_EQ(provider_->RegisterMemory(zeroAddr, handles).code, StatusCode::INVALID_ARGUMENT);

    const std::vector<TransProvider::RegisterMemoryDesc> zeroSize{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 0}
    };
    EXPECT_EQ(provider_->RegisterMemory(zeroSize, handles).code, StatusCode::INVALID_ARGUMENT);

    EXPECT_TRUE(handles.empty());
    EXPECT_EQ(fake_.registerCalls(), 0U);
}

TEST_F(HccpKvOverRoceTransProviderTest, RegisterMemoryReportsOperatorFailure)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);
    fake_.setRegisterResult(HCCP_KV_ROCE_UNSUPPORTED);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096}
    };
    std::vector<MRHandle> handles;
    EXPECT_EQ(provider_->RegisterMemory(descs, handles).code, StatusCode::UNSUPPORTED);
    EXPECT_TRUE(handles.empty());
    // No rkey was reported, so there is nothing to roll back.
    EXPECT_EQ(fake_.unregisterCalls(), 0U);
}

TEST_F(HccpKvOverRoceTransProviderTest, RegisterMemoryRollsBackOnZeroRkey)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);
    fake_.setRegisterZeroRkeyIndex(1);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096},
        {TransProvider::MemType::MEM_DEVICE, 0x2000, 4096}
    };
    std::vector<MRHandle> handles;
    EXPECT_EQ(provider_->RegisterMemory(descs, handles).code, StatusCode::INTERNAL_ERROR);
    EXPECT_TRUE(handles.empty());

    // Only the entry that reported a usable rkey is deregistered.
    EXPECT_EQ(fake_.unregisterCalls(), 1U);
    ASSERT_EQ(fake_.unregisterInfoCount(), 1U);
    EXPECT_EQ(fake_.unregisterInfoRkey(0), 1000U);
}

TEST_F(HccpKvOverRoceTransProviderTest, UnregisterMemoryDropsRegisteredRecords)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096},
        {TransProvider::MemType::MEM_DEVICE, 0x2000, 4096}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->RegisterMemory(descs, handles).ok());
    ASSERT_EQ(handles.size(), 2U);

    const auto results = provider_->UnregisterMemory({{handles[0]}, {handles[1]}});
    ASSERT_EQ(results.size(), 2U);
    EXPECT_TRUE(results[0].ok());
    EXPECT_TRUE(results[1].ok());
    EXPECT_EQ(fake_.unregisterCalls(), 1U);
    ASSERT_EQ(fake_.unregisterInfoCount(), 2U);
    EXPECT_EQ(fake_.unregisterInfoRkey(0), 1000U);
    EXPECT_EQ(fake_.unregisterInfoRkey(1), 1001U);

    std::uint32_t tokenId = 99;
    EXPECT_EQ(provider_->GetMemTokenId(handles[0], tokenId).code,
              StatusCode::BUFFER_NOT_REGISTERED);
    EXPECT_EQ(tokenId, 0U);
}

TEST_F(HccpKvOverRoceTransProviderTest, UnregisterMemoryRejectsInvalidHandles)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->RegisterMemory(descs, handles).ok());
    ASSERT_EQ(handles.size(), 1U);

    const auto duplicates = provider_->UnregisterMemory({{handles[0]}, {handles[0]}});
    ASSERT_EQ(duplicates.size(), 2U);
    EXPECT_TRUE(duplicates[0].ok());
    EXPECT_EQ(duplicates[1].code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(fake_.unregisterCalls(), 1U);
    EXPECT_EQ(fake_.unregisterInfoCount(), 1U);

    const auto unknown =
        provider_->UnregisterMemory({{kInvalidMRHandle}, {static_cast<MRHandle>(999)}});
    ASSERT_EQ(unknown.size(), 2U);
    EXPECT_EQ(unknown[0].code, StatusCode::BUFFER_NOT_REGISTERED);
    EXPECT_EQ(unknown[1].code, StatusCode::BUFFER_NOT_REGISTERED);
    EXPECT_EQ(fake_.unregisterCalls(), 1U);

    EXPECT_TRUE(provider_->UnregisterMemory({}).empty());
    EXPECT_EQ(fake_.unregisterCalls(), 1U);
}

TEST_F(HccpKvOverRoceTransProviderTest, UnregisterMemoryKeepsRecordOnOperatorFailure)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->RegisterMemory(descs, handles).ok());
    ASSERT_EQ(handles.size(), 1U);

    fake_.setUnregisterResult(HCCP_KV_ROCE_INTERNAL_ERROR);
    const auto failed = provider_->UnregisterMemory({{handles[0]}});
    ASSERT_EQ(failed.size(), 1U);
    EXPECT_EQ(failed[0].code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(fake_.unregisterCalls(), 1U);

    std::uint32_t tokenId = 0;
    ASSERT_TRUE(provider_->GetMemTokenId(handles[0], tokenId).ok());
    EXPECT_EQ(tokenId, 1000U);

    fake_.setUnregisterResult(0);
    const auto retried = provider_->UnregisterMemory({{handles[0]}});
    ASSERT_EQ(retried.size(), 1U);
    EXPECT_TRUE(retried[0].ok());
    EXPECT_EQ(fake_.unregisterCalls(), 2U);
    EXPECT_EQ(provider_->GetMemTokenId(handles[0], tokenId).code,
              StatusCode::BUFFER_NOT_REGISTERED);
}

TEST_F(HccpKvOverRoceTransProviderTest, BindMemoryCreatesLocalRecords)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    const std::vector<TransProvider::BindMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096, 5000},
        {TransProvider::MemType::MEM_DEVICE, 0x2000, 8192, 5001}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->BindMemory(descs, handles).ok());

    ASSERT_EQ(handles.size(), 2U);
    EXPECT_NE(handles[0], kInvalidMRHandle);
    EXPECT_NE(handles[1], kInvalidMRHandle);
    EXPECT_NE(handles[0], handles[1]);
    // Binding reuses an existing registration: the operator is never called.
    EXPECT_EQ(fake_.registerCalls(), 0U);

    std::uint32_t tokenId = 0;
    ASSERT_TRUE(provider_->GetMemTokenId(handles[0], tokenId).ok());
    EXPECT_EQ(tokenId, 5000U);
    ASSERT_TRUE(provider_->GetMemTokenId(handles[1], tokenId).ok());
    EXPECT_EQ(tokenId, 5001U);
}

TEST_F(HccpKvOverRoceTransProviderTest, BindMemoryRejectsInvalidDescriptors)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    std::vector<MRHandle> handles{7};
    ASSERT_TRUE(provider_->BindMemory({}, handles).ok());
    EXPECT_TRUE(handles.empty());

    const std::vector<TransProvider::BindMemoryDesc> host{
        {TransProvider::MemType::MEM_HOST, 0x1000, 4096, 5000}
    };
    EXPECT_EQ(provider_->BindMemory(host, handles).code, StatusCode::INVALID_ARGUMENT);

    const std::vector<TransProvider::BindMemoryDesc> zeroAddr{
        {TransProvider::MemType::MEM_DEVICE, 0, 4096, 5000}
    };
    EXPECT_EQ(provider_->BindMemory(zeroAddr, handles).code, StatusCode::INVALID_ARGUMENT);

    const std::vector<TransProvider::BindMemoryDesc> zeroSize{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 0, 5000}
    };
    EXPECT_EQ(provider_->BindMemory(zeroSize, handles).code, StatusCode::INVALID_ARGUMENT);

    const std::vector<TransProvider::BindMemoryDesc> zeroToken{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096, 0}
    };
    EXPECT_EQ(provider_->BindMemory(zeroToken, handles).code, StatusCode::INVALID_ARGUMENT);

    EXPECT_TRUE(handles.empty());
}

TEST_F(HccpKvOverRoceTransProviderTest, UnregisterMemorySkipsOperatorForBoundRecords)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    const std::vector<TransProvider::BindMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096, 5000}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->BindMemory(descs, handles).ok());
    ASSERT_EQ(handles.size(), 1U);

    const auto results = provider_->UnregisterMemory({{handles[0]}});
    ASSERT_EQ(results.size(), 1U);
    EXPECT_TRUE(results[0].ok());
    EXPECT_EQ(fake_.unregisterCalls(), 0U);

    std::uint32_t tokenId = 0;
    EXPECT_EQ(provider_->GetMemTokenId(handles[0], tokenId).code,
              StatusCode::BUFFER_NOT_REGISTERED);
}

TEST_F(HccpKvOverRoceTransProviderTest, DestructorUnregistersOperatorMemory)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    auto connections = CreateConnections(1);
    ASSERT_EQ(connections.size(), 1U);
    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096}
    };
    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider_->RegisterMemory(descs, handles).ok());
    ASSERT_EQ(handles.size(), 1U);

    provider_.reset();
    EXPECT_EQ(fake_.deleteCalls(), 1U);
    EXPECT_EQ(fake_.unregisterCalls(), 1U);
    ASSERT_EQ(fake_.unregisterInfoCount(), 1U);
    EXPECT_EQ(fake_.unregisterInfoRkey(0), 1000U);
}

TEST_F(HccpKvOverRoceTransProviderTest, ThreadOperationsAreNoOps)
{
    CreateProvider();
    ASSERT_TRUE(provider_ != nullptr);

    std::vector<TransProvider::ThreadHandle> threads{reinterpret_cast<void*>(0x1)};
    ASSERT_TRUE(provider_->AllocThread(2, {1, 2}, threads).ok());
    EXPECT_TRUE(threads.empty());

    const std::vector<TransProvider::ThreadHandle> freeThreads{reinterpret_cast<void*>(0x1),
                                                               reinterpret_cast<void*>(0x2)};
    const auto statuses = provider_->FreeThread(freeThreads);
    ASSERT_EQ(statuses.size(), 2U);
    EXPECT_TRUE(statuses[0].ok());
    EXPECT_TRUE(statuses[1].ok());
    EXPECT_TRUE(provider_->FreeThread({}).empty());
}

TEST_F(HccpKvOverRoceTransProviderTest, MinimalLibraryFallsBackToUnsupported)
{
    ASSERT_TRUE(UseLibrary(UCM_HCCP_FAKE_ABI_MINIMAL_PATH));
    CreateProvider(kTestCapacity, UCM_HCCP_FAKE_ABI_MINIMAL_PATH);
    ASSERT_TRUE(provider_ != nullptr);

    auto handles = CreateConnections(1);
    ASSERT_EQ(handles.size(), 1U);

    const std::vector<TransProvider::SendIoBatch> batches{
        {handles[0], nullptr, nullptr, 0}
    };
    const auto send = provider_->Send(batches, 1, 0);
    ASSERT_EQ(send.size(), 1U);
    EXPECT_EQ(send[0].code, StatusCode::UNSUPPORTED);
    EXPECT_EQ(fake_.sendCalls(), 0U);

    ServerKvCapabilities capabilities;
    EXPECT_EQ(provider_->GetServerCapabilities(handles[0], capabilities).code,
              StatusCode::UNSUPPORTED);
    EXPECT_EQ(fake_.capabilitiesCalls(), 0U);

    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096}
    };
    std::vector<MRHandle> memoryHandles;
    ASSERT_TRUE(provider_->RegisterMemory(descs, memoryHandles).ok());
    ASSERT_EQ(memoryHandles.size(), 1U);

    // Without UnregisterMemory the local record is dropped and the operator is
    // never called.
    const auto unregistered = provider_->UnregisterMemory({{memoryHandles[0]}});
    ASSERT_EQ(unregistered.size(), 1U);
    EXPECT_TRUE(unregistered[0].ok());
    EXPECT_EQ(fake_.unregisterCalls(), 0U);

    std::uint32_t tokenId = 0;
    EXPECT_EQ(provider_->GetMemTokenId(memoryHandles[0], tokenId).code,
              StatusCode::BUFFER_NOT_REGISTERED);
}

}  // namespace
}  // namespace kv::test

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
