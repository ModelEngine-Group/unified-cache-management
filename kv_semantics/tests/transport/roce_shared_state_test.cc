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

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <vector>
#include "roce/cpu_kv_over_roce_trans_provider.h"
#include "roce/roce_process_state.h"

namespace kv::test {
namespace {
TransportConfig MakeSimulatedConfig()
{
    TransportConfig config;
    config.providerType = TransProviderType::CPU;
    config.attrs["roce_qp_backend"] = "simulated";
    return config;
}

constexpr std::uint32_t kPeerToken = 0xABCDEF01U;
}  // namespace

// The store side builds one provider per ASU node, so memory registered through
// one of them has to stay reachable from the others: the address table is
// process wide, not provider private.
TEST(CpuKvOverRoceSharedStateTest, ASecondProviderBindsWhatTheFirstOneRegistered)
{
    roce::CpuKvOverRoceTransProvider first(MakeSimulatedConfig());
    roce::CpuKvOverRoceTransProvider second(MakeSimulatedConfig());
    std::uint8_t buffer[4096] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         0}
    };
    std::vector<MRHandle> registered;
    ASSERT_TRUE(first.RegisterMemory(registerDescs, registered).ok());
    ASSERT_EQ(1U, registered.size());

    // The second provider never saw the buffer, only its address.
    const std::vector<TransProvider::BindMemoryDesc> bindDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer) + 512, 1024,
         kPeerToken}
    };
    std::vector<MRHandle> bindings;
    const Status bound = second.BindMemory(bindDescs, bindings);
    ASSERT_TRUE(bound.ok()) << bound.message;
    ASSERT_EQ(1U, bindings.size());

    std::uint32_t token = 0;
    ASSERT_TRUE(second.GetMemTokenId(bindings[0], token).ok());
    EXPECT_EQ(kPeerToken, token);

    // The registration itself resolves through either provider.
    std::uint32_t fromSecond = 0;
    ASSERT_TRUE(second.GetMemTokenId(registered[0], fromSecond).ok());
    std::uint32_t fromFirst = 0;
    ASSERT_TRUE(first.GetMemTokenId(registered[0], fromFirst).ok());
    EXPECT_EQ(fromFirst, fromSecond);
}

TEST(CpuKvOverRoceSharedStateTest, DeviceBucketsAreKeyedByAdapterAndPort)
{
    const roce::DeviceKey first{"mlx5_0", 1};
    const roce::DeviceKey same{"mlx5_0", 1};
    const roce::DeviceKey otherPort{"mlx5_0", 2};
    const roce::DeviceKey otherAdapter{"mlx5_1", 1};

    EXPECT_TRUE(first == same);
    EXPECT_FALSE(first == otherPort);
    EXPECT_FALSE(first == otherAdapter);
    EXPECT_EQ(roce::DeviceKeyHash{}(first), roce::DeviceKeyHash{}(same));

    // The hash has to move with the port and with the GID index, otherwise
    // two adapters or two GIDs collapse into the same bucket.
    const roce::DeviceKey otherGid{"mlx5_0", 1, 1};
    EXPECT_FALSE(first == otherGid);
    EXPECT_NE(roce::DeviceKeyHash{}(first), roce::DeviceKeyHash{}(otherPort));
    EXPECT_NE(roce::DeviceKeyHash{}(first), roce::DeviceKeyHash{}(otherAdapter));
    EXPECT_NE(roce::DeviceKeyHash{}(first), roce::DeviceKeyHash{}(otherGid));
}

TEST(CpuKvOverRoceSharedStateTest, RejectsRangesRegisteredWithAnotherBackend)
{
    TransportConfig verbsConfig;
    verbsConfig.providerType = TransProviderType::CPU;
    verbsConfig.attrs["roce_qp_backend"] = "verbs";

    roce::CpuKvOverRoceTransProvider simulated(MakeSimulatedConfig());
    roce::CpuKvOverRoceTransProvider verbs(verbsConfig);
    std::uint8_t buffer[2048] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         0}
    };
    std::vector<MRHandle> registered;
    ASSERT_TRUE(simulated.RegisterMemory(registerDescs, registered).ok());
    ASSERT_EQ(1U, registered.size());

    // The address is known, but the registration behind it has no verbs access
    // record, so a verbs provider would hand out a token it cannot honour.
    const std::vector<TransProvider::BindMemoryDesc> bindDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), 512,
         kPeerToken}
    };
    std::vector<MRHandle> bindings;
    const Status bound = verbs.BindMemory(bindDescs, bindings);
    EXPECT_EQ(StatusCode::UNSUPPORTED, bound.code);
    EXPECT_TRUE(bindings.empty());
}

TEST(CpuKvOverRoceSharedStateTest, HandlesOfTwoProvidersNeverAlias)
{
    roce::CpuKvOverRoceTransProvider first(MakeSimulatedConfig());
    roce::CpuKvOverRoceTransProvider second(MakeSimulatedConfig());
    std::uint8_t firstBuffer[1024] = {0};
    std::uint8_t secondBuffer[1024] = {0};
    std::vector<MRHandle> firstHandles;
    std::vector<MRHandle> secondHandles;
    const std::vector<TransProvider::RegisterMemoryDesc> firstDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(firstBuffer),
         sizeof(firstBuffer), 0}
    };
    const std::vector<TransProvider::RegisterMemoryDesc> secondDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(secondBuffer),
         sizeof(secondBuffer), 0}
    };
    ASSERT_TRUE(first.RegisterMemory(firstDescs, firstHandles).ok());
    ASSERT_TRUE(second.RegisterMemory(secondDescs, secondHandles).ok());
    ASSERT_EQ(1U, firstHandles.size());
    ASSERT_EQ(1U, secondHandles.size());
    EXPECT_NE(firstHandles[0], secondHandles[0]);
}

TEST(CpuKvOverRoceSharedStateTest, OnlyTheOwningProviderReleasesARegion)
{
    roce::CpuKvOverRoceTransProvider owner(MakeSimulatedConfig());
    roce::CpuKvOverRoceTransProvider other(MakeSimulatedConfig());
    std::uint8_t buffer[2048] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         0}
    };
    std::vector<MRHandle> registered;
    ASSERT_TRUE(owner.RegisterMemory(registerDescs, registered).ok());
    ASSERT_EQ(1U, registered.size());

    const std::vector<TransProvider::UnregisterMemoryDesc> releaseDescs{{registered[0]}};
    const std::vector<Status> results = other.UnregisterMemory(releaseDescs);
    ASSERT_EQ(1U, results.size());
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, results[0].code);

    // The region is untouched, so its token still resolves.
    std::uint32_t token = 0;
    EXPECT_TRUE(owner.GetMemTokenId(registered[0], token).ok());
    EXPECT_NE(0U, token);
}

TEST(CpuKvOverRoceSharedStateTest, DestroyingTheOwnerRetiresItsBindings)
{
    auto owner = std::make_unique<roce::CpuKvOverRoceTransProvider>(MakeSimulatedConfig());
    roce::CpuKvOverRoceTransProvider other(MakeSimulatedConfig());
    std::uint8_t buffer[4096] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         0}
    };
    std::vector<MRHandle> registered;
    ASSERT_TRUE(owner->RegisterMemory(registerDescs, registered).ok());
    ASSERT_EQ(1U, registered.size());

    const std::vector<TransProvider::BindMemoryDesc> bindDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), 512,
         kPeerToken}
    };
    std::vector<MRHandle> bindings;
    ASSERT_TRUE(other.BindMemory(bindDescs, bindings).ok());
    ASSERT_EQ(1U, bindings.size());

    owner.reset();

    // The registration went away with its owner and the binding that pointed at
    // it cannot describe memory any more.
    std::uint32_t token = 7;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, other.GetMemTokenId(registered[0], token).code);
    EXPECT_EQ(0U, token);
    std::uint32_t bindingToken = 7;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, other.GetMemTokenId(bindings[0], bindingToken).code);
    EXPECT_EQ(0U, bindingToken);
}

// The device a provider adopts comes from the regions it binds, so a batch that
// fails half way must not leave the provider pinned to the device of the
// descriptor that happened to pass: the retry the caller wants is the region
// that failed, and it lives on another adapter.
TEST(CpuKvOverRoceSharedStateTest, AFailedBatchDoesNotPinTheProviderToItsFirstRegion)
{
    roce::ProcessState& state = roce::ProcessState::Instance();
    // Stand-ins for two protection domains.  The bind path only compares these
    // pointers, it never dereferences them.
    roce::VerbDevice adapterA;
    roce::VerbDevice adapterB;
    std::uint8_t buffer[8192] = {0};
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(buffer);

    MRHandle onA = 0;
    MRHandle onB = 0;
    {
        std::lock_guard<std::mutex> guard(state.memoryMutex);
        onA = state.nextBookkeepingId++;
        roce::Registration first;
        first.addr = base;
        first.size = 1024;
        first.device = &adapterA;
        state.registrations[onA] = first;
        onB = state.nextBookkeepingId++;
        roce::Registration second;
        second.addr = base + 4096;
        second.size = 1024;
        second.device = &adapterB;
        state.registrations[onB] = second;
    }

    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    // The first descriptor resolves on A, the second matches nothing at all.
    const std::vector<TransProvider::BindMemoryDesc> batch{
        {TransProvider::MemType::MEM_HOST, base + 128,  256, kPeerToken},
        {TransProvider::MemType::MEM_HOST, base + 6144, 256, kPeerToken}
    };
    std::vector<MRHandle> bindings;
    EXPECT_FALSE(provider.BindMemory(batch, bindings).ok());
    EXPECT_TRUE(bindings.empty());

    // Binding B alone has to work now: had the failed batch pinned A, this would
    // come back as a device mismatch.
    const std::vector<TransProvider::BindMemoryDesc> retry{
        {TransProvider::MemType::MEM_HOST, base + 4096 + 128, 256, kPeerToken}
    };
    const Status bound = provider.BindMemory(retry, bindings);
    EXPECT_TRUE(bound.ok()) << bound.message;
    EXPECT_EQ(1U, bindings.size());

    {
        std::lock_guard<std::mutex> guard(state.memoryMutex);
        state.registrations.erase(onA);
        state.registrations.erase(onB);
    }
}

}  // namespace kv::test
