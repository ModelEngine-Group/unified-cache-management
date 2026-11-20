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
#include "m800_live_smoke_watch.h"
#include <chrono>
#include <gtest/gtest.h>

namespace kv::test {
namespace {

// 测试场景：仅指定 --watch-seconds，不配置 --expect-add、--expect-remove 或
// --expect-asu-counts。预期结果：监听窗口到达 deadline 前不退出，到达后退出。
TEST(M800LiveSmokeWatchTest, WatchSecondsRunsToDeadlineWithoutExpectations)
{
    const M800LiveSmokeWatchOptions options{1};
    const auto deadline = std::chrono::steady_clock::now();
    EXPECT_FALSE(WatchShouldStop(deadline - std::chrono::seconds(1), deadline, options));
    EXPECT_TRUE(WatchShouldStop(deadline, deadline, options));
}

// 测试场景：--watch-seconds 为 0。预期结果：不进入监听窗口。
TEST(M800LiveSmokeWatchTest, ZeroWatchSecondsIsDisabled)
{
    const M800LiveSmokeWatchOptions options{0};
    const auto now = std::chrono::steady_clock::now();
    EXPECT_TRUE(WatchShouldStop(now, now + std::chrono::seconds(1), options));
}

}  // namespace
}  // namespace kv::test
