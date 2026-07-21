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
#include "delegator/cc/delegator_buffer_manager.h"
#include <acl/acl.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace UC::Delegator {
namespace {

class BufferManagerTest : public ::testing::Test {
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

TEST_F(BufferManagerTest, AlignsShardCapacityAddressesAndOffsetsTo16KB)
{
    constexpr std::size_t kShardSize = BufferManager::kAlignmentBytes + 1;
    constexpr std::size_t kAlignedSize = 2 * BufferManager::kAlignmentBytes;

    BufferManager manager;
    auto status = manager.Init(kShardSize, 2);
    ASSERT_TRUE(status.Success()) << status.ToString();

    EXPECT_EQ(manager.AlignedSize(), kAlignedSize);
    ASSERT_NE(manager.DeviceAddress(), nullptr);

    BufferPool::Slot first;
    BufferPool::Slot second;
    ASSERT_TRUE(manager.Acquire(first).Success());
    ASSERT_TRUE(manager.Acquire(second).Success());

    EXPECT_EQ(first.length, kAlignedSize);
    EXPECT_EQ(second.length, kAlignedSize);
    EXPECT_EQ(manager.Size(), 2 * kAlignedSize);
    EXPECT_EQ(manager.Offset(first), std::size_t{0});
    EXPECT_EQ(manager.Offset(second), kAlignedSize);
    EXPECT_EQ(manager.Offset(second) % BufferManager::kAlignmentBytes,
              std::size_t{0});
    EXPECT_EQ(static_cast<std::byte*>(manager.DeviceAddress()) + manager.Offset(first),
              first.device_addr);
    EXPECT_EQ(static_cast<std::byte*>(manager.DeviceAddress()) + manager.Offset(second),
              second.device_addr);

    EXPECT_TRUE(manager.Release(first).Success());
    EXPECT_TRUE(manager.Release(second).Success());
}

TEST_F(BufferManagerTest, ReleaseReturnsSlot)
{
    BufferManager manager;
    ASSERT_TRUE(manager.Init(1, 1).Success());

    BufferPool::Slot first;
    ASSERT_TRUE(manager.Acquire(first).Success());

    BufferPool::Slot exhausted;
    EXPECT_EQ(manager.Acquire(exhausted), Status::NoSpace());

    ASSERT_TRUE(manager.Release(first).Success());

    BufferPool::Slot reused;
    ASSERT_TRUE(manager.Acquire(reused).Success());
    EXPECT_EQ(reused.device_addr, first.device_addr);
    EXPECT_TRUE(manager.Release(reused).Success());
}

TEST_F(BufferManagerTest, RejectsInvalidInitialization)
{
    BufferManager manager;

    BufferPool::Slot slot;
    EXPECT_EQ(manager.Acquire(slot), Status::Error());

    EXPECT_EQ(manager.Init(0, 1), Status::InvalidParam());
    EXPECT_EQ(manager.Init(1, 0), Status::InvalidParam());
    EXPECT_EQ(manager.Init(std::numeric_limits<std::size_t>::max(), 1),
              Status::InvalidParam());

    ASSERT_TRUE(manager.Init(1, 1).Success());
    EXPECT_EQ(manager.Init(1, 1), Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Delegator
