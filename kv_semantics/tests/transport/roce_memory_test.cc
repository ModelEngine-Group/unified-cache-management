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
#include <vector>
#include "roce/cpu_kv_over_roce_trans_provider.h"

namespace kv::test {
namespace {

TransportConfig MakeSimulatedConfig()
{
    TransportConfig config;
    config.providerType = TransProviderType::CPU;
    config.attrs["roce_qp_backend"] = "simulated";
    return config;
}

TEST(CpuKvOverRoceMemoryTest, RegistersBuffersAndReportsTheirRemoteKeys)
{
    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    std::uint8_t first[4096] = {0};
    std::uint8_t second[8192] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(first),  sizeof(first),
         0},
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(second), sizeof(second),
         0}
    };

    std::vector<MRHandle> handles;
    const Status status = provider.RegisterMemory(descs, handles);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(2U, handles.size());

    std::uint32_t firstToken = 0;
    std::uint32_t secondToken = 0;
    ASSERT_TRUE(provider.GetMemTokenId(handles[0], firstToken).ok());
    ASSERT_TRUE(provider.GetMemTokenId(handles[1], secondToken).ok());
    EXPECT_NE(0U, firstToken);
    EXPECT_NE(0U, secondToken);
    EXPECT_NE(firstToken, secondToken);

    const std::vector<TransProvider::UnregisterMemoryDesc> unregisterDescs{{handles[0]},
                                                                           {handles[1]}};
    const std::vector<Status> results = provider.UnregisterMemory(unregisterDescs);
    ASSERT_EQ(2U, results.size());
    EXPECT_TRUE(results[0].ok()) << results[0].message;
    EXPECT_TRUE(results[1].ok()) << results[1].message;

    // Once released, the handle must not resolve any more.
    std::uint32_t stale = 7;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, provider.GetMemTokenId(handles[0], stale).code);
    EXPECT_EQ(0U, stale);
}

TEST(CpuKvOverRoceMemoryTest, RejectsEmptyAndZeroSizedDescriptors)
{
    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    std::vector<MRHandle> handles;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, provider.RegisterMemory({}, handles).code);
    EXPECT_TRUE(handles.empty());

    std::uint8_t buffer[64] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> zeroAddress{
        {TransProvider::MemType::MEM_HOST, 0, sizeof(buffer), 0}
    };
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, provider.RegisterMemory(zeroAddress, handles).code);
    EXPECT_TRUE(handles.empty());

    const std::vector<TransProvider::RegisterMemoryDesc> zeroSize{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), 0, 0}
    };
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, provider.RegisterMemory(zeroSize, handles).code);
    EXPECT_TRUE(handles.empty());
}

TEST(CpuKvOverRoceMemoryTest, BindsARegisteredRangeAndCarriesThePeerToken)
{
    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    std::uint8_t buffer[4096] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         0}
    };
    std::vector<MRHandle> registrations;
    ASSERT_TRUE(provider.RegisterMemory(registerDescs, registrations).ok());
    ASSERT_EQ(1U, registrations.size());

    // A sub-range of a registered region is bindable; the store token travels
    // through untouched, because the client compares it after the call.
    std::vector<MRHandle> bindings;
    const std::vector<TransProvider::BindMemoryDesc> bindDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer) + 512, 1024,
         0xABCDEF01U}
    };
    const Status status = provider.BindMemory(bindDescs, bindings);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(1U, bindings.size());

    std::uint32_t tokenId = 0;
    ASSERT_TRUE(provider.GetMemTokenId(bindings.front(), tokenId).ok());
    EXPECT_EQ(0xABCDEF01U, tokenId);

    // Releasing the binding leaves the registration alone.
    const std::vector<Status> released = provider.UnregisterMemory({{bindings.front()}});
    ASSERT_EQ(1U, released.size());
    EXPECT_TRUE(released.front().ok()) << released.front().message;
    std::uint32_t registrationToken = 0;
    EXPECT_TRUE(provider.GetMemTokenId(registrations.front(), registrationToken).ok());
}

TEST(CpuKvOverRoceMemoryTest, RejectsBindingMemoryItDidNotRegister)
{
    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    std::uint8_t buffer[1024] = {0};
    std::vector<MRHandle> bindings{42};

    // Nothing has been registered at all.
    const std::vector<TransProvider::BindMemoryDesc> foreign{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         7}
    };
    EXPECT_EQ(StatusCode::UNSUPPORTED, provider.BindMemory(foreign, bindings).code);
    EXPECT_TRUE(bindings.empty());

    // A registered region is not a licence to bind a range that runs past its
    // end either.
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), 512, 0}
    };
    std::vector<MRHandle> registrations;
    ASSERT_TRUE(provider.RegisterMemory(registerDescs, registrations).ok());

    const std::vector<TransProvider::BindMemoryDesc> overflowing{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer) + 256, 512, 7}
    };
    EXPECT_EQ(StatusCode::UNSUPPORTED, provider.BindMemory(overflowing, bindings).code);
    EXPECT_TRUE(bindings.empty());
}

TEST(CpuKvOverRoceMemoryTest, ReleasingARegistrationRetiresItsBindings)
{
    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    std::uint8_t buffer[2048] = {0};
    const std::vector<TransProvider::RegisterMemoryDesc> registerDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         0}
    };
    std::vector<MRHandle> registrations;
    ASSERT_TRUE(provider.RegisterMemory(registerDescs, registrations).ok());
    ASSERT_EQ(1U, registrations.size());

    std::vector<MRHandle> bindings;
    const std::vector<TransProvider::BindMemoryDesc> bindDescs{
        {TransProvider::MemType::MEM_HOST, reinterpret_cast<std::uintptr_t>(buffer), sizeof(buffer),
         11}
    };
    ASSERT_TRUE(provider.BindMemory(bindDescs, bindings).ok());
    ASSERT_EQ(1U, bindings.size());

    ASSERT_TRUE(provider.UnregisterMemory({{registrations.front()}}).front().ok());

    // The memory is gone, so a token that described it must not be handed out
    // any more.
    std::uint32_t tokenId = 0;
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, provider.GetMemTokenId(bindings.front(), tokenId).code);
    EXPECT_EQ(0U, tokenId);
}

TEST(CpuKvOverRoceMemoryTest, UnregisterReportsUnknownHandles)
{
    roce::CpuKvOverRoceTransProvider provider(MakeSimulatedConfig());

    const std::vector<TransProvider::UnregisterMemoryDesc> descs{{0xDEADBEEFULL}};
    const std::vector<Status> results = provider.UnregisterMemory(descs);

    ASSERT_EQ(1U, results.size());
    EXPECT_EQ(StatusCode::INVALID_ARGUMENT, results.front().code);
    EXPECT_NE(std::string::npos, results.front().message.find("unknown memory registration"));
}

}  // namespace
}  // namespace kv::test
