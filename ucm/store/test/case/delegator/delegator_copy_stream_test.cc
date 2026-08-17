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
 */
#include "delegator/cc/delegator_copy_stream.h"
#include <acl/acl.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include "pool/buffer_pool.h"

namespace UC::Delegator {
namespace {

class CopyStreamTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        const auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            FAIL() << "aclInit failed: " << ret;
        }
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        (void)aclrtResetDevice(0);
        (void)aclFinalize();
    }
};

TEST_F(CopyStreamTest, CreatesAndCyclesStreams)
{
    CopyStream streams;
    EXPECT_EQ(streams.NextStream(), nullptr);

    EXPECT_EQ(streams.Setup(1, 1), Status::InvalidParam());
    ASSERT_TRUE(streams.Setup(0, 2).Success());
    EXPECT_EQ(streams.Size(), std::size_t{2});

    const auto first = streams.NextStream();
    const auto second = streams.NextStream();
    EXPECT_NE(first, nullptr);
    EXPECT_NE(second, nullptr);
    EXPECT_NE(first, second);
    EXPECT_EQ(streams.NextStream(), first);
    EXPECT_TRUE(streams.Synchronize(first).Success());
    EXPECT_TRUE(streams.SynchronizeAll().Success());
    EXPECT_EQ(streams.Setup(0, 1), Status::Error());
}

TEST_F(CopyStreamTest, CopiesDeviceMemoryAsynchronously)
{
    constexpr std::size_t kCopySize = 256;

    BufferPool pool;
    ASSERT_TRUE(
        pool.Init("delegator_copy_stream_test", BufferPool::MemoryType::Device, kCopySize, 2)
            .Success());

    BufferPool::Slot source;
    BufferPool::Slot destination;
    ASSERT_TRUE(pool.Allocate(source).Success());
    ASSERT_TRUE(pool.Allocate(destination).Success());

    std::array<std::uint8_t, kCopySize> input{};
    std::array<std::uint8_t, kCopySize> output{};
    for (std::size_t i = 0; i < input.size(); ++i) { input[i] = static_cast<std::uint8_t>(i); }

    ASSERT_EQ(aclrtMemcpy(source.device_addr, source.length, input.data(), input.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);

    CopyStream streams;
    ASSERT_TRUE(streams.Setup(0, 2).Success());
    const auto stream = streams.NextStream();
    ASSERT_TRUE(streams
                    .DeviceToDeviceAsync(stream, destination.device_addr, destination.length,
                                         source.device_addr, input.size())
                    .Success());
    ASSERT_TRUE(streams.Synchronize(stream).Success());

    ASSERT_EQ(aclrtMemcpy(output.data(), output.size(), destination.device_addr, input.size(),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    EXPECT_EQ(output, input);
}

TEST_F(CopyStreamTest, RejectsInvalidConfigurationAndCopies)
{
    CopyStream streams;
    EXPECT_EQ(streams.Setup(-1, 1), Status::InvalidParam());
    EXPECT_EQ(streams.Setup(0, 0), Status::InvalidParam());
    EXPECT_EQ(streams.Synchronize(nullptr), Status::InvalidParam());

    auto stream = reinterpret_cast<aclrtStream>(std::uintptr_t{1});
    auto* source = reinterpret_cast<void*>(std::uintptr_t{1});
    auto* destination = reinterpret_cast<void*>(std::uintptr_t{2});

    EXPECT_EQ(streams.Synchronize(stream), Status::InvalidParam());

    EXPECT_EQ(streams.DeviceToDeviceAsync(nullptr, destination, 1, source, 1),
              Status::InvalidParam());
    EXPECT_EQ(streams.DeviceToDeviceAsync(stream, nullptr, 1, source, 1), Status::InvalidParam());
    EXPECT_EQ(streams.DeviceToDeviceAsync(stream, destination, 1, nullptr, 1),
              Status::InvalidParam());
    EXPECT_EQ(streams.DeviceToDeviceAsync(stream, destination, 1, source, 0),
              Status::InvalidParam());
    EXPECT_EQ(streams.DeviceToDeviceAsync(stream, destination, 1, source, 2),
              Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Delegator
