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
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "dram/cc/drampool/buffer_manager.h"
#include "status/status.h"

namespace UC::DramPool {
namespace {

// The per-slot-size usage gauges are revealed on first successful allocation
// (see BufferManager::IsRevealed and the GC report loop), so configured pools
// that never serve traffic stay out of exposition entirely.
class UCDrampoolBufferManagerRevealTest : public testing::Test {
protected:
    std::unique_ptr<BufferManager> manager_ =
        std::make_unique<BufferManager>(std::vector<std::pair<std::size_t, std::size_t>>{
            {64,   4},
            {4096, 2},
    });
};

TEST_F(UCDrampoolBufferManagerRevealTest, ConfiguredButUnusedPoolsStayUnrevealed)
{
    EXPECT_FALSE(manager_->IsRevealed(64));
    EXPECT_FALSE(manager_->IsRevealed(4096));
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.0);
}

TEST_F(UCDrampoolBufferManagerRevealTest, FirstSuccessfulAllocationRevealsSize)
{
    Buffer buf;
    ASSERT_TRUE(manager_->Allocate(64, buf).Success());

    EXPECT_TRUE(manager_->IsRevealed(64));
    EXPECT_FALSE(manager_->IsRevealed(4096));
}

TEST_F(UCDrampoolBufferManagerRevealTest, RevealIsStickyAfterPoolDrains)
{
    Buffer buf;
    ASSERT_TRUE(manager_->Allocate(64, buf).Success());
    ASSERT_TRUE(manager_->Free(64, buf.slot).Success());

    EXPECT_TRUE(manager_->IsRevealed(64));
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.0);
}

TEST_F(UCDrampoolBufferManagerRevealTest, UnregisteredSizeIsNeverRevealed)
{
    Buffer buf;
    EXPECT_TRUE(manager_->Allocate(128, buf) == Status::NotFound());

    EXPECT_FALSE(manager_->IsRevealed(128));
}

TEST_F(UCDrampoolBufferManagerRevealTest, ResetClearsRevealState)
{
    Buffer buf;
    ASSERT_TRUE(manager_->Allocate(64, buf).Success());
    manager_->Reset();

    EXPECT_FALSE(manager_->IsRevealed(64));
}

// The used-slot counter now lives in BufferManager (per size) instead of
// shared BufferPool, so its Allocate/Free semantics are asserted here.

TEST_F(UCDrampoolBufferManagerRevealTest, AllocateBumpsUsedSlotRatio)
{
    Buffer first;
    Buffer second;
    ASSERT_TRUE(manager_->Allocate(64, first).Success());
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.25);
    ASSERT_TRUE(manager_->Allocate(64, second).Success());
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.5);
}

TEST_F(UCDrampoolBufferManagerRevealTest, FreeDecreasesUsedSlotRatio)
{
    Buffer buf;
    ASSERT_TRUE(manager_->Allocate(64, buf).Success());
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.25);

    ASSERT_TRUE(manager_->Free(64, buf.slot).Success());
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.0);
}

TEST_F(UCDrampoolBufferManagerRevealTest, FailedAllocateDoesNotChangeRatio)
{
    // The 4096 pool has 2 slots; exhausting it then over-allocating must leave
    // the used ratio pinned at 1.0.
    Buffer first;
    Buffer second;
    Buffer third;
    ASSERT_TRUE(manager_->Allocate(4096, first).Success());
    ASSERT_TRUE(manager_->Allocate(4096, second).Success());
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(4096), 1.0);

    EXPECT_TRUE(manager_->Allocate(4096, third).Failure());
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(4096), 1.0);
}

TEST_F(UCDrampoolBufferManagerRevealTest, UsedSlotCountersAreIndependentPerSize)
{
    Buffer buf;
    ASSERT_TRUE(manager_->Allocate(64, buf).Success());

    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.25);
    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(4096), 0.0);
}

TEST_F(UCDrampoolBufferManagerRevealTest, ResetClearsUsedSlotRatio)
{
    Buffer buf;
    ASSERT_TRUE(manager_->Allocate(64, buf).Success());
    manager_->Reset();

    EXPECT_DOUBLE_EQ(manager_->GetUsedSlotRatio(64), 0.0);
}

}  // namespace
}  // namespace UC::DramPool
