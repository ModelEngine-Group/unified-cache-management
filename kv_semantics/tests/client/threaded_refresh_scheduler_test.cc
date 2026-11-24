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

// 测试场景：分别以零或负 interval、空 callback 调用 Start，并在合法启动后重复 Start、Stop
// 后重启及重复 Stop。 预期结果：非法参数返回 INVALID_ARGUMENT，运行中重复启动返回
// RESOURCE_BUSY，停止后可成功重启且重复停止安全。
TEST(ThreadedRefreshScheduler, ValidatesStartAndSupportsRestart)
{
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

// 测试场景：以 10ms 周期启动调度器，刷新回调在互斥锁保护下累计执行次数并通知等待线程。
// 预期结果：定时器自动触发串行刷新，2 秒内至少执行 2 次回调，停止后保留正确的累计次数。
TEST(ThreadedRefreshScheduler, TimerRefreshesSerially)
{
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

// 测试场景：IO 失败触发一个阻塞中的刷新回调，在回调结束前再同时发起定时触发和另一次 IO 失败触发。
// 预期结果：活动刷新期间的触发被合并，不会并发执行或在释放回调后补跑第二次，最终回调计数为 1。
TEST(ThreadedRefreshScheduler, TriggersAreMergedWhileRefreshIsActive)
{
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

// 测试场景：调度器启动且尚未到定时周期时调用 RefreshNow，回调返回指定 IO 错误，停止后再次手动刷新。
// 预期结果：首次手动刷新立即执行一次并原样返回回调错误码和消息，停止后的调用返回 NOT_INITIALIZED。
TEST(ThreadedRefreshScheduler, ManualRefreshRunsImmediatelyAndReturnsCallbackResult)
{
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

// 测试场景：调度器以较长定时间隔启动后，向 TriggerByIoFailure 传入 Status::OK 并短暂等待。
// 预期结果：成功状态被忽略，不唤醒刷新线程，等待期间刷新回调计数保持为 0。
TEST(ThreadedRefreshScheduler, SuccessfulIoStatusDoesNotTriggerRefresh)
{
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

// 测试场景：刷新回调抛出内容为 callback failed 的 runtime_error，随后通过 RefreshNow 获取执行结果。
// 预期结果：异常被调度器捕获并转换为 INTERNAL_ERROR，返回消息包含固定前缀和原异常内容。
TEST(ThreadedRefreshScheduler, CallbackExceptionBecomesInternalError)
{
    ThreadedRefreshScheduler scheduler;

    const auto status =
        scheduler.Start(1000ms, []() -> Status { throw std::runtime_error("callback failed"); });
    ASSERT_TRUE(status.ok()) << status.message;

    const auto result = scheduler.RefreshNow();
    EXPECT_EQ(result.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(result.message, "refresh callback threw an exception: callback failed");

    scheduler.Stop();
}

// 测试场景：三个线程并发调用 RefreshNow，首个刷新回调进入后阻塞，待所有调用方就绪后再释放回调。
// 预期结果：并发手动刷新合并为一次回调执行，三个调用方均获得成功结果，最终刷新计数为 1。
TEST(ThreadedRefreshScheduler, ConcurrentManualRefreshesAreMerged)
{
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

// 测试场景：IO 失败触发一个阻塞中的刷新回调，另一线程在回调返回前调用 Stop，随后释放回调。
// 预期结果：Stop 在回调活动期间持续等待而不中断它，回调释放后 Stop 完成且刷新仅执行一次。
TEST(ThreadedRefreshScheduler, StopWaitsForActiveRefresh)
{
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

// 测试场景：触发 IO 失败刷新后立即调用 Stop，刷新回调进入后阻塞等待释放。
// 预期结果：无论 Stop 时刷新已处于活动态还是待执行态，Stop 都会执行该已接收刷新；
// Stop 完成后新触发被拒绝，最终回调计数为 1。
TEST(ThreadedRefreshScheduler, StopDrainsAcceptedRefresh)
{
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
    auto stopFuture = std::async(std::launch::async, [&] {
        scheduler.Stop();
        return true;
    });
    ASSERT_EQ(enteredFuture.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(stopFuture.wait_for(20ms), std::future_status::timeout);

    release.set_value();
    ASSERT_TRUE(stopFuture.get());
    EXPECT_EQ(refreshCount.load(), std::uint32_t{1});

    scheduler.TriggerByIoFailure(Status::Error(StatusCode::IO_ERROR, "after stop"));
    EXPECT_EQ(refreshCount.load(), std::uint32_t{1});
}

// 测试场景：活动刷新期间两个线程并发 Stop，首个 Stop 清理后另一线程立即重新 Start，最后释放原回调。
// 预期结果：两个 Stop 均能完成，等待中的 Stop 不会被新一轮 running
// 状态卡住，调度器可成功重启并再次停止。
TEST(ThreadedRefreshScheduler, ConcurrentStopDoesNotWaitForRestartedScheduler)
{
    ThreadedRefreshScheduler scheduler;
    std::promise<void> entered;
    std::promise<void> release;
    auto releaseFuture = release.get_future();
    const auto enteredFuture = entered.get_future();

    const auto status = scheduler.Start(1000ms, [&] {
        entered.set_value();
        releaseFuture.get();
        return Status::OK();
    });
    ASSERT_TRUE(status.ok()) << status.message;

    scheduler.TriggerByIoFailure(Status::Error(StatusCode::IO_ERROR, "io failed"));
    ASSERT_EQ(enteredFuture.wait_for(1s), std::future_status::ready);

    auto firstStop = std::async(std::launch::async, [&] {
        scheduler.Stop();
        return true;
    });
    std::this_thread::sleep_for(20ms);
    auto secondStop = std::async(std::launch::async, [&] {
        scheduler.Stop();
        return true;
    });
    std::this_thread::sleep_for(20ms);

    std::thread restartThread([&] {
        while (true) {
            if (scheduler.Start(1000ms, [] { return Status::OK(); }).ok()) { break; }
            std::this_thread::sleep_for(1ms);
        }
    });

    release.set_value();
    ASSERT_TRUE(firstStop.get());
    ASSERT_TRUE(secondStop.get());
    restartThread.join();
    scheduler.Stop();
}

}  // namespace
}  // namespace kv::test
