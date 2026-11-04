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
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include "pool/buffer_pool.h"
#include "trans/device.h"

namespace UC::Delegator {
namespace {

class CopyStreamTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        const auto status = device_.Init();
        deviceRuntimeOwned_ = status.Success();
        ASSERT_TRUE(deviceRuntimeOwned_ || status == Status::DuplicateKey()) << status.ToString();
        ASSERT_TRUE(device_.Setup(0).Success());
    }

    static void TearDownTestSuite()
    {
        if (!deviceRuntimeOwned_) { return; }
        EXPECT_TRUE(device_.Reset(0).Success());
        EXPECT_TRUE(device_.Finalize().Success());
        deviceRuntimeOwned_ = false;
    }

    static Status HostToDevice(void* host, void* device, std::size_t size)
    {
        auto stream = device_.MakeStream();
        if (!stream) { return Status::Error("failed to create test stream"); }
        return stream->HostToDevice(host, device, size);
    }

    static Status DeviceToHost(void* device, void* host, std::size_t size)
    {
        auto stream = device_.MakeStream();
        if (!stream) { return Status::Error("failed to create test stream"); }
        return stream->DeviceToHost(device, host, size);
    }

    inline static Trans::Device device_;
    inline static bool deviceRuntimeOwned_{false};
};

TEST_F(CopyStreamTest, CreatesAndCyclesStreams)
{
    CopyStream streams;
    EXPECT_EQ(streams.NextStream(), nullptr);

    EXPECT_EQ(streams.Setup(-1, 1), Status::InvalidParam());
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

    ASSERT_TRUE(HostToDevice(input.data(), source.deviceAddr, input.size()).Success());

    CopyStream streams;
    ASSERT_TRUE(streams.Setup(0, 2).Success());
    const auto stream = streams.NextStream();
    ASSERT_TRUE(streams
                    .DeviceToDeviceAsync(stream, destination.deviceAddr, destination.length,
                                         source.deviceAddr, input.size())
                    .Success());
    ASSERT_TRUE(streams.Synchronize(stream).Success());

    ASSERT_TRUE(DeviceToHost(destination.deviceAddr, output.data(), input.size()).Success());
    EXPECT_EQ(output, input);
}

TEST_F(CopyStreamTest, RejectsInvalidConfigurationAndCopies)
{
    CopyStream streams;
    EXPECT_EQ(streams.Setup(-1, 1), Status::InvalidParam());
    EXPECT_EQ(streams.Setup(0, 0), Status::InvalidParam());
    EXPECT_EQ(streams.Synchronize(nullptr), Status::InvalidParam());

    auto stream = device_.MakeSharedStream();
    ASSERT_NE(stream, nullptr);
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
