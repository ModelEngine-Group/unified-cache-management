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
#include "view/threaded_refresh_scheduler.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kv::test {
namespace {

using namespace std::chrono_literals;

TEST(ThreadedRefreshScheduler, ValidatesStartAndSupportsRestart)
{
    // 测试场景：Start 入参非法、重复启动、Stop 后重启以及重复 Stop。
    // 期望调度器拒绝非法 interval/callback，运行中不允许再次 Start，Stop 后可重新启动且 Stop 幂等。
    ThreadedRefreshScheduler scheduler;
    const auto refresh = [] { return Status::OK(); };

    auto status = scheduler.Start(0ms, refresh);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    status = scheduler.Start(-1ms, refresh);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    status = scheduler.Start(1000ms, nullptr);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);

    status = scheduler.Start(1000ms, refresh);
    ASSERT_TRUE(status.ok()) << status.message;
    status = scheduler.Start(1000ms, refresh);
    EXPECT_EQ(status.code, StatusCode::RESOURCE_BUSY);

    scheduler.Stop();
    status = scheduler.Start(1000ms, refresh);
    ASSERT_TRUE(status.ok()) << status.message;
    scheduler.Stop();
    scheduler.Stop();
}

TEST(ThreadedRefreshScheduler, TimerRefreshesSerially)
{
    // 测试场景：定时器按配置周期自动触发刷新。
    // 期望 worker 线程多次执行同一个回调，并通过互斥锁验证回调执行是串行的。
    ThreadedRefreshScheduler scheduler;
    std::mutex mutex;
    std::condition_variable cv;
    std::uint32_t refreshCount = 0;

    const auto status = scheduler.Start(10ms, [&] {
        std::lock_guard<std::mutex> lock{mutex};
        ++refreshCount;
        cv.notify_all();
        return Status::OK();
    });
    ASSERT_TRUE(status.ok()) << status.message;

    {
        std::unique_lock<std::mutex> lock{mutex};
        ASSERT_TRUE(cv.wait_for(lock, 2s, [&] { return refreshCount >= 2; }));
    }
    scheduler.Stop();

    std::lock_guard<std::mutex> lock{mutex};
    EXPECT_GE(refreshCount, std::uint32_t{2});
}

TEST(ThreadedRefreshScheduler, TriggersAreMergedWhileRefreshIsActive)
{
    // 测试场景：刷新回调尚未结束时，定时和 IO 失败触发同时到达。
    // 期望这些触发合并到当前刷新，不会产生并发或排队后的第二次回调。
    ThreadedRefreshScheduler scheduler;
    std::atomic_uint32_t refreshCount{0};
    std::promise<void> entered;
    std::promise<void> release;
    auto releaseFuture = release.get_future();
    const auto enteredFuture = entered.get_future();

    const auto status = scheduler.Start(1000ms, [&] {
        ++refreshCount;
        entered.set_value();
        releaseFuture.get();
        return Status::OK();
    });
    ASSERT_TRUE(status.ok()) << status.message;

    scheduler.TriggerByIoFailure(Status::Error(StatusCode::IO_ERROR, "io failed"));
    ASSERT_EQ(enteredFuture.wait_for(1s), std::future_status::ready);

    scheduler.TriggerByTimer();
    scheduler.TriggerByIoFailure(Status::Error(StatusCode::CONNECTION_ERROR, "connection failed"));

    release.set_value();
    scheduler.Stop();
    EXPECT_EQ(refreshCount.load(), std::uint32_t{1});
}

TEST(ThreadedRefreshScheduler, ManualRefreshRunsImmediatelyAndReturnsCallbackResult)
{
    // 测试场景：调度器已启动且未到定时周期时调用 RefreshNow，随后 Stop 后再次调用。
    // 期望手动刷新立即执行并返回回调的失败结果；调度器停止后返回 NOT_INITIALIZED。
    ThreadedRefreshScheduler scheduler;
    std::atomic_uint32_t refreshCount{0};

    const auto status = scheduler.Start(1000ms, [&] {
        ++refreshCount;
        return Status::Error(StatusCode::IO_ERROR, "expected refresh failure");
    });
    ASSERT_TRUE(status.ok()) << status.message;

    const auto result = scheduler.RefreshNow();
    EXPECT_EQ(result.code, StatusCode::IO_ERROR);
    EXPECT_EQ(result.message, "expected refresh failure");
    EXPECT_EQ(refreshCount.load(), std::uint32_t{1});

    scheduler.Stop();
    const auto stoppedResult = scheduler.RefreshNow();
    EXPECT_EQ(stoppedResult.code, StatusCode::NOT_INITIALIZED);
}

TEST(ThreadedRefreshScheduler, SuccessfulIoStatusDoesNotTriggerRefresh)
{
    // 测试场景：IO 失败触发接口收到成功状态。
    // 期望成功状态被忽略，不唤醒刷新线程，也不执行刷新回调。
    ThreadedRefreshScheduler scheduler;
    std::atomic_uint32_t refreshCount{0};

    const auto status = scheduler.Start(1000ms, [&] {
        ++refreshCount;
        return Status::OK();
    });
    ASSERT_TRUE(status.ok()) << status.message;

    scheduler.TriggerByIoFailure(Status::OK());
    std::this_thread::sleep_for(20ms);
    EXPECT_EQ(refreshCount.load(), std::uint32_t{0});

    scheduler.Stop();
}

TEST(ThreadedRefreshScheduler, CallbackExceptionBecomesInternalError)
{
    // 测试场景：刷新回调抛出异常。
    // 期望 worker 捕获异常并将其转换为 INTERNAL_ERROR，异常不会穿透调度线程。
    ThreadedRefreshScheduler scheduler;

    const auto status =
        scheduler.Start(1000ms, []() -> Status { throw std::runtime_error("callback failed"); });
    ASSERT_TRUE(status.ok()) << status.message;

    const auto result = scheduler.RefreshNow();
    EXPECT_EQ(result.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(result.message, "refresh callback threw an exception: callback failed");

    scheduler.Stop();
}

TEST(ThreadedRefreshScheduler, ConcurrentManualRefreshesAreMerged)
{
    // 测试场景：三个线程在同一个回调执行期间并发调用 RefreshNow。
    // 期望并发手动刷新合并为一次回调执行，所有调用方拿到同一个成功结果。
    ThreadedRefreshScheduler scheduler;
    std::atomic_uint32_t refreshCount{0};
    std::atomic_int manualCallersReady{0};
    std::promise<void> entered;
    std::promise<void> release;
    auto releaseFuture = release.get_future();
    const auto enteredFuture = entered.get_future();

    const auto status = scheduler.Start(1000ms, [&] {
        ++refreshCount;
        entered.set_value();
        releaseFuture.get();
        return Status::OK();
    });
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::future<Status>> results;
    for (int i = 0; i < 3; ++i) {
        results.push_back(std::async(std::launch::async, [&] {
            ++manualCallersReady;
            return scheduler.RefreshNow();
        }));
    }

    const auto readyDeadline = std::chrono::steady_clock::now() + 1s;
    while (manualCallersReady.load() < 3 && std::chrono::steady_clock::now() < readyDeadline) {
        std::this_thread::sleep_for(1ms);
    }
    ASSERT_EQ(manualCallersReady.load(), 3);

    ASSERT_EQ(enteredFuture.wait_for(1s), std::future_status::ready);
    std::this_thread::sleep_for(20ms);
    release.set_value();

    for (auto& result : results) {
        const auto refreshResult = result.get();
        EXPECT_TRUE(refreshResult.ok()) << refreshResult.message;
    }
    scheduler.Stop();
    EXPECT_EQ(refreshCount.load(), std::uint32_t{1});
}

TEST(ThreadedRefreshScheduler, StopWaitsForActiveRefresh)
{
    // 测试场景：回调尚未返回时调用 Stop，并释放回调等待条件。
    // 期望 Stop 不中断活动回调，等待回调完成后退出，且回调只执行一次。
    ThreadedRefreshScheduler scheduler;
    std::atomic_uint32_t refreshCount{0};
    std::promise<void> entered;
    std::promise<void> release;
    auto releaseFuture = release.get_future();
    const auto enteredFuture = entered.get_future();

    const auto status = scheduler.Start(1000ms, [&] {
        ++refreshCount;
        entered.set_value();
        releaseFuture.get();
        return Status::OK();
    });
    ASSERT_TRUE(status.ok()) << status.message;

    scheduler.TriggerByIoFailure(Status::Error(StatusCode::IO_ERROR, "io failed"));
    ASSERT_EQ(enteredFuture.wait_for(1s), std::future_status::ready);

    auto stopFuture = std::async(std::launch::async, [&] {
        scheduler.Stop();
        return true;
    });
    EXPECT_EQ(stopFuture.wait_for(20ms), std::future_status::timeout);

    release.set_value();
    ASSERT_TRUE(stopFuture.get());
    EXPECT_EQ(refreshCount.load(), std::uint32_t{1});
}

}  // namespace
}  // namespace kv::test
