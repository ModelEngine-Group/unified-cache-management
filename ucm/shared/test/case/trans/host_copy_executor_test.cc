/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "trans/host/host_copy_executor.h"
#include <array>
#include <atomic>
#include <gtest/gtest.h>
#include <list>
#include <utility>

namespace {

using UC::Trans::HostCopyExecutor;

TEST(HostCopyExecutorTest, GatherAndScatterVariableSegments)
{
    std::array<unsigned char, 2> first{1, 2};
    std::array<unsigned char, 3> second{3, 4, 5};
    std::array<unsigned char, 5> contiguous{};
    std::vector<HostCopyExecutor::Segment> sources{{first.data(), first.size()},
                                                   {second.data(), second.size()}};
    size_t bytes = 0;
    ASSERT_TRUE(HostCopyExecutor::Gather(sources, contiguous.data(), &bytes).Success());
    EXPECT_EQ(bytes, contiguous.size());
    EXPECT_EQ(contiguous, (std::array<unsigned char, 5>{1, 2, 3, 4, 5}));

    first.fill(0);
    second.fill(0);
    ASSERT_TRUE(HostCopyExecutor::Scatter(contiguous.data(), sources, &bytes).Success());
    EXPECT_EQ(first, (std::array<unsigned char, 2>{1, 2}));
    EXPECT_EQ(second, (std::array<unsigned char, 3>{3, 4, 5}));
}

TEST(HostCopyExecutorTest, ReservationIsAtomicAndCompletionRunsOnce)
{
    HostCopyExecutor executor;
    ASSERT_TRUE(executor.Setup(2, 2).Success());

    auto reservationResult = executor.Reserve(2);
    ASSERT_TRUE(reservationResult.HasValue());
    EXPECT_FALSE(executor.Reserve(1).HasValue());
    auto reservation = std::move(reservationResult).Value();

    std::array<unsigned char, 4> source{1, 2, 3, 4};
    std::array<unsigned char, 4> first{};
    std::array<unsigned char, 4> second{};
    std::atomic<size_t> completions{0};
    std::list<HostCopyExecutor::Job> jobs;
    for (auto* destination : {first.data(), second.data()}) {
        HostCopyExecutor::Job job;
        job.direction = HostCopyExecutor::Direction::SCATTER;
        job.contiguous = source.data();
        job.segments = {{destination, source.size()}};
        job.completion = [&completions](const auto& result) {
            if (result.status.Success()) { completions.fetch_add(1); }
        };
        jobs.push_back(std::move(job));
    }
    ASSERT_TRUE(reservation.Submit(jobs).Success());
    executor.Synchronize();

    EXPECT_EQ(completions.load(), 2U);
    EXPECT_EQ(first, source);
    EXPECT_EQ(second, source);
    EXPECT_TRUE(executor.Reserve(2).HasValue());

    std::atomic<bool> completionOnly{false};
    ASSERT_TRUE(executor.PostCompletion([&completionOnly](const auto& result) {
        completionOnly.store(result.status.Success());
    }).Success());
    executor.Synchronize();
    EXPECT_TRUE(completionOnly.load());
}

TEST(HostCopyExecutorTest, PrerequisiteFailureSkipsCopy)
{
    HostCopyExecutor executor;
    ASSERT_TRUE(executor.Setup(1, 1).Success());
    std::array<unsigned char, 1> source{1};
    std::array<unsigned char, 1> destination{0};
    std::atomic<bool> failed{false};
    std::list<HostCopyExecutor::Job> jobs;
    HostCopyExecutor::Job job;
    job.direction = HostCopyExecutor::Direction::SCATTER;
    job.contiguous = source.data();
    job.segments = {{destination.data(), destination.size()}};
    job.prerequisite = [] { return UC::Status::Error("not ready"); };
    job.completion = [&failed](const auto& result) { failed.store(result.status.Failure()); };
    jobs.push_back(std::move(job));

    ASSERT_TRUE(executor.Submit(jobs).Success());
    executor.Synchronize();
    EXPECT_TRUE(failed.load());
    EXPECT_EQ(destination[0], 0);
}

}  // namespace
