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
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include "kv_common/router.h"
#include "task_manager.h"

namespace UC::Dram {
namespace {

void PublishCompletion(TaskManager& manager, RequestCompleted event)
{
    std::vector<RequestCompleted> events;
    events.push_back(std::move(event));
    manager.Publish(events);
}

class CountingRouter final : public UC::KV::Router {
public:
    CountingRouter() : Router([](const std::string&) { return std::uint64_t{0}; }) {}

    std::unordered_map<UC::KV::NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<UC::KV::CacheKey>& keys) const override
    {
        calls_.fetch_add(1, std::memory_order_relaxed);
        std::vector<EntryIndex> indexes(keys.size());
        for (std::size_t index = 0; index < keys.size(); ++index) { indexes[index] = index; }
        return {
            {NodeId{7}, std::move(indexes)}
        };
    }

    std::size_t calls() const noexcept { return calls_.load(std::memory_order_relaxed); }

private:
    UC::KV::NodeId RouteKey(const UC::KV::CacheKey&) const override { return NodeId{7}; }

    mutable std::atomic<std::size_t> calls_{0};
};

class BlockingRequests final {
public:
    Status Post(Request& request)
    {
        std::unique_lock lock(mutex_);
        requests_.push_back(request);
        changed_.notify_all();
        if (requests_.size() == 1) {
            changed_.wait(lock, [this] { return released_; });
        }
        return Status::OK();
    }

    Request WaitForSubmission(std::size_t index = 0)
    {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [this, index] { return requests_.size() > index; });
        return requests_[index];
    }

    void Release()
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<Request> requests_;
    bool released_{false};
};

class FailSecondRequest final {
public:
    Status Post(Request& request)
    {
        std::lock_guard lock(mutex_);
        requests_.push_back(request);
        changed_.notify_all();
        return requests_.size() == 2 ? Status::Error("second request rejected") : Status::OK();
    }

    std::vector<Request> WaitForRequests(std::size_t count)
    {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [this, count] { return requests_.size() >= count; });
        return requests_;
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<Request> requests_;
};

class CompletingRequests final {
public:
    void Bind(TaskManager& manager) noexcept { manager_ = &manager; }

    Status Post(Request& request)
    {
        std::vector<EntryResult> results;
        for (std::size_t index = 0; index < request.entries.size(); ++index) {
            results.push_back(EntryResult{index, true, 0});
        }
        PublishCompletion(*manager_,
                          RequestCompleted{request.taskId, request.requestId, request.nodeId,
                                           Status::OK(), std::move(results)});
        return Status::OK();
    }

private:
    TaskManager* manager_{nullptr};
};

class TaskManagerAsyncTest : public testing::Test {
protected:
    TaskManagerConfig Config(std::size_t batchSize = 8) const
    {
        return TaskManagerConfig{
            {0x100},
            32,
            batchSize,
            TimeoutConfig{std::chrono::milliseconds{100}, std::chrono::milliseconds{100},
             std::chrono::milliseconds{100}}
        };
    }

    TaskManagerDependencies Dependencies(RequestSubmitter submitRequest) const
    {
        return TaskManagerDependencies{router_, std::move(submitRequest)};
    }

    static Detail::TaskDesc ValidTask()
    {
        Detail::TaskDesc task;
        Detail::Shard shard{};
        shard.index = 0;
        shard.addrs.push_back(reinterpret_cast<void*>(kLayerBase));
        task.push_back(std::move(shard));
        return task;
    }

    static Detail::TaskDesc TwoEntryTask()
    {
        Detail::TaskDesc task;
        task.push_back(Detail::Shard{{}, 0, {reinterpret_cast<void*>(kLayerBase)}});
        task.push_back(Detail::Shard{{}, 1, {reinterpret_cast<void*>(kSecondLayerBase)}});
        return task;
    }

    static Detail::TaskDesc TwoTensorShardTask()
    {
        Detail::TaskDesc task;
        task.push_back(Detail::Shard{
            {},
            1,
            {reinterpret_cast<void*>(kLayerBase), reinterpret_cast<void*>(kSecondLayerBase)}
        });
        return task;
    }

    static constexpr std::uintptr_t kLayerBase = 0x10000;
    static constexpr std::uintptr_t kSecondLayerBase = 0x12000;
    std::shared_ptr<UC::KV::Router> router_{
        UC::KV::CreateRouter({NodeId{7}}, {}, UC::KV::RouterConfig{})};
};

TEST_F(TaskManagerAsyncTest, CheckDoesNotWaitForTaskWorker)
{
    BlockingRequests commands;
    TaskManager manager(
        Config(), Dependencies([&commands](Request& request) { return commands.Post(request); }));
    ASSERT_TRUE(manager.Start().Success());

    auto submitted = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(submitted);
    const auto task_id = std::move(submitted).Value();
    const auto request = commands.WaitForSubmission();
    ASSERT_EQ(request.entries.size(), std::size_t{1});
    EXPECT_EQ(request.entries[0].buffer.address, kLayerBase);
    EXPECT_EQ(request.entries[0].buffer.length, std::uint64_t{0x100});

    auto complete = manager.Check(task_id);
    commands.Release();
    ASSERT_TRUE(complete);
    EXPECT_FALSE(std::move(complete).Value());

    PublishCompletion(
        manager,
        RequestCompleted{request.taskId, request.requestId, request.nodeId, Status::OK(), {}});
    EXPECT_TRUE(manager.WaitTransfer(task_id).Success());
    EXPECT_FALSE(manager.Check(task_id));
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, FlattensTensorIndexesWithinShard)
{
    BlockingRequests commands;
    auto config = Config();
    config.tensorSizes = {0x100, 0x200};
    TaskManager manager(std::move(config), Dependencies([&commands](Request& request) {
                            return commands.Post(request);
                        }));
    ASSERT_TRUE(manager.Start().Success());

    auto submitted = manager.SubmitTransfer(OpType::LOAD, TwoTensorShardTask());
    ASSERT_TRUE(submitted);
    const auto taskId = std::move(submitted).Value();
    const auto request = commands.WaitForSubmission();
    ASSERT_EQ(request.entries.size(), std::size_t{2});
    EXPECT_EQ(request.entries[0].shardId, std::uint32_t{2});
    EXPECT_EQ(request.entries[0].buffer.address, kLayerBase);
    EXPECT_EQ(request.entries[0].buffer.length, std::uint64_t{0x100});
    EXPECT_EQ(request.entries[1].shardId, std::uint32_t{3});
    EXPECT_EQ(request.entries[1].buffer.address, kSecondLayerBase);
    EXPECT_EQ(request.entries[1].buffer.length, std::uint64_t{0x200});

    commands.Release();
    PublishCompletion(
        manager,
        RequestCompleted{request.taskId, request.requestId, request.nodeId, Status::OK(), {}});
    EXPECT_TRUE(manager.WaitTransfer(taskId).Success());
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, CompletionIsProcessedBeforeQueuedSubmission)
{
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Request> requests;
    bool releaseFirstPost = false;
    bool secondPosted = false;
    bool firstCompletionObserved = false;
    TaskId firstTaskId = 0;
    TaskManager* manager = nullptr;

    auto dependencies = Dependencies([&](Request& request) {
        std::unique_lock lock(mutex);
        requests.push_back(request);
        changed.notify_all();
        if (requests.size() == 1) {
            changed.wait(lock, [&] { return releaseFirstPost; });
        } else {
            lock.unlock();
            auto checked = manager->Check(firstTaskId);
            const auto completed = checked && checked.Value();
            lock.lock();
            firstCompletionObserved = completed;
            secondPosted = true;
            changed.notify_all();
        }
        return Status::OK();
    });
    TaskManager instance(Config(), std::move(dependencies));
    manager = &instance;
    ASSERT_TRUE(instance.Start().Success());

    auto first = instance.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(first);
    firstTaskId = std::move(first).Value();
    Request firstRequest;
    {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return requests.size() == 1; });
        firstRequest = requests.front();
    }

    auto second = instance.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(second);
    const auto secondTaskId = std::move(second).Value();
    PublishCompletion(
        instance,
        RequestCompleted{
            firstRequest.taskId, firstRequest.requestId, firstRequest.nodeId, Status::OK(), {}});
    {
        std::lock_guard lock(mutex);
        releaseFirstPost = true;
    }
    changed.notify_all();

    Request secondRequest;
    {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return secondPosted; });
        ASSERT_EQ(requests.size(), std::size_t{2});
        secondRequest = requests.back();
    }
    EXPECT_TRUE(firstCompletionObserved);
    PublishCompletion(
        instance,
        RequestCompleted{
            secondRequest.taskId, secondRequest.requestId, secondRequest.nodeId, Status::OK(), {}});
    EXPECT_TRUE(instance.WaitTransfer(firstTaskId).Success());
    EXPECT_TRUE(instance.WaitTransfer(secondTaskId).Success());
    instance.Shutdown();
}

TEST_F(TaskManagerAsyncTest, TerminalCheckPreservesHandleAndReleasesIoEntryCapacity)
{
    CompletingRequests commands;
    auto config = Config();
    config.maxIoEntries = 2;
    TaskManager manager(std::move(config), Dependencies([&commands](Request& request) {
                            return commands.Post(request);
                        }));
    commands.Bind(manager);
    ASSERT_TRUE(manager.Start().Success());

    auto first = manager.SubmitTransfer(OpType::LOAD, TwoEntryTask());
    ASSERT_TRUE(first);
    const auto firstTaskId = std::move(first).Value();

    bool terminal = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while (!terminal && std::chrono::steady_clock::now() < deadline) {
        auto checked = manager.Check(firstTaskId);
        ASSERT_TRUE(checked);
        terminal = checked.Value();
        if (!terminal) { std::this_thread::sleep_for(std::chrono::milliseconds{1}); }
    }
    ASSERT_TRUE(terminal);

    auto checkedAgain = manager.Check(firstTaskId);
    ASSERT_TRUE(checkedAgain);
    EXPECT_TRUE(checkedAgain.Value());

    auto second = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(second);
    EXPECT_TRUE(manager.WaitTransfer(std::move(second).Value()).Success());
    EXPECT_TRUE(manager.WaitTransfer(firstTaskId).Success());
    auto consumed = manager.Check(firstTaskId);
    ASSERT_FALSE(consumed);
    EXPECT_EQ(consumed.Error(), Status::NotFound());
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, CapacityFailureIsReportedThroughAcceptedHandle)
{
    BlockingRequests commands;
    auto config = Config(1);
    config.maxIoEntries = 2;
    TaskManager manager(std::move(config), Dependencies([&commands](Request& request) {
                            return commands.Post(request);
                        }));
    ASSERT_TRUE(manager.Start().Success());

    auto first = manager.SubmitTransfer(OpType::LOAD, TwoEntryTask());
    ASSERT_TRUE(first);
    const auto firstTaskId = std::move(first).Value();
    const auto firstRequest = commands.WaitForSubmission();

    auto accepted = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(accepted);
    const auto acceptedTaskId = std::move(accepted).Value();

    commands.Release();
    const auto secondRequest = commands.WaitForSubmission(1);
    EXPECT_EQ(manager.WaitTransfer(acceptedTaskId), Status::NoSpace());

    std::vector<RequestCompleted> completions;
    completions.push_back(RequestCompleted{
        firstRequest.taskId, firstRequest.requestId, firstRequest.nodeId, Status::OK(), {}});
    completions.push_back(RequestCompleted{
        secondRequest.taskId, secondRequest.requestId, secondRequest.nodeId, Status::OK(), {}});
    manager.Publish(completions);

    EXPECT_TRUE(manager.WaitTransfer(firstTaskId).Success());
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, ShutdownStopsNewSubmissions)
{
    TaskManager manager(Config(), Dependencies([](Request&) { return Status::OK(); }));
    ASSERT_TRUE(manager.Start().Success());
    manager.Shutdown();
    EXPECT_FALSE(manager.SubmitTransfer(OpType::LOAD, ValidTask()));
}

TEST_F(TaskManagerAsyncTest, ShutdownRejectsNewTasksAndDropsLateCompletions)
{
    BlockingRequests commands;
    TaskManager manager(
        Config(), Dependencies([&commands](Request& request) { return commands.Post(request); }));
    ASSERT_TRUE(manager.Start().Success());
    auto submitted = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(submitted);
    const auto request = commands.WaitForSubmission();

    auto stopping = std::async(std::launch::async, [&manager] { return manager.Shutdown(); });
    EXPECT_EQ(stopping.wait_for(std::chrono::milliseconds{10}), std::future_status::timeout);
    commands.Release();
    stopping.get();

    EXPECT_FALSE(manager.SubmitTransfer(OpType::LOAD, ValidTask()));
    PublishCompletion(
        manager,
        RequestCompleted{request.taskId, request.requestId, request.nodeId, Status::OK(), {}});
}

TEST_F(TaskManagerAsyncTest, PublishingWhileStoppedIsIgnored)
{
    TaskManager manager(Config(), Dependencies([](Request&) { return Status::OK(); }));
    PublishCompletion(manager, RequestCompleted{1, 1, 7, Status::OK(), {}});
    ASSERT_TRUE(manager.Start().Success());
    manager.Shutdown();
    PublishCompletion(manager, RequestCompleted{1, 1, 7, Status::OK(), {}});
}

TEST_F(TaskManagerAsyncTest, WorkerExceptionFailsAcceptedTasksAndStopsNewSubmissions)
{
    std::promise<void> dispatchStarted;
    std::promise<void> failDispatch;
    auto failureReady = failDispatch.get_future();
    TaskManager manager(Config(), Dependencies([&](Request&) -> Status {
                            dispatchStarted.set_value();
                            failureReady.wait();
                            throw std::runtime_error("node command publisher failed");
                        }));
    ASSERT_TRUE(manager.Start().Success());

    auto active = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(active);
    dispatchStarted.get_future().wait();
    auto queued = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(queued);
    failDispatch.set_value();

    EXPECT_TRUE(manager.WaitTransfer(std::move(active).Value()).Failure());
    EXPECT_TRUE(manager.WaitTransfer(std::move(queued).Value()).Failure());

    bool stopped = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while (!stopped && std::chrono::steady_clock::now() < deadline) {
        auto attempt = manager.SubmitTransfer(OpType::LOAD, ValidTask());
        stopped = !attempt && attempt.Error() != Status::NoSpace();
        if (!stopped) { std::this_thread::sleep_for(std::chrono::milliseconds{1}); }
    }
    EXPECT_TRUE(stopped);

    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, PartialDispatchWaitsForPostedRequest)
{
    FailSecondRequest commands;
    TaskManager manager(
        Config(1), Dependencies([&commands](Request& request) { return commands.Post(request); }));
    ASSERT_TRUE(manager.Start().Success());
    auto submitted = manager.SubmitTransfer(OpType::DUMP, TwoEntryTask());
    ASSERT_TRUE(submitted);
    const auto task_id = std::move(submitted).Value();
    const auto requests = commands.WaitForRequests(2);
    ASSERT_EQ(requests.size(), std::size_t{2});
    ASSERT_EQ(requests[0].entries.size(), std::size_t{1});
    ASSERT_EQ(requests[1].entries.size(), std::size_t{1});
    EXPECT_EQ(requests[0].taskId, task_id);
    EXPECT_EQ(requests[1].taskId, task_id);
    EXPECT_NE(requests[0].requestId, kInvalidRequestId);
    EXPECT_NE(requests[1].requestId, kInvalidRequestId);
    EXPECT_NE(requests[0].requestId, requests[1].requestId);
    EXPECT_EQ(requests[0].nodeId, NodeId{7});
    EXPECT_EQ(requests[1].nodeId, NodeId{7});
    EXPECT_EQ(requests[0].entries[0].shardId, std::uint32_t{0});
    EXPECT_EQ(requests[1].entries[0].shardId, std::uint32_t{1});
    PublishCompletion(
        manager,
        RequestCompleted{
            requests[0].taskId, requests[0].requestId, requests[0].nodeId, Status::OK(), {}});
    EXPECT_TRUE(manager.WaitTransfer(task_id).Failure());
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, ShutdownWaitsForInProgressNodeDispatch)
{
    BlockingRequests commands;
    TaskManager manager(
        Config(), Dependencies([&commands](Request& request) { return commands.Post(request); }));
    ASSERT_TRUE(manager.Start().Success());
    auto submitted = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(submitted);
    (void)commands.WaitForSubmission();

    auto stopping = std::async(std::launch::async, [&manager] { return manager.Shutdown(); });
    EXPECT_EQ(stopping.wait_for(std::chrono::milliseconds{10}), std::future_status::timeout);
    commands.Release();
    stopping.get();
}

TEST_F(TaskManagerAsyncTest, LookupCanCompleteBeforeNodePostReturns)
{
    CompletingRequests commands;
    TaskManager manager(
        Config(), Dependencies([&commands](Request& request) { return commands.Post(request); }));
    commands.Bind(manager);
    ASSERT_TRUE(manager.Start().Success());
    Detail::BlockId block{};
    auto submitted = manager.SubmitLookup(&block, 1);
    ASSERT_TRUE(submitted);
    auto result = manager.WaitLookup(std::move(submitted).Value());
    ASSERT_TRUE(result);
    ASSERT_EQ(result.Value().size(), std::size_t{1});
    EXPECT_EQ(result.Value()[0], std::uint8_t{1});
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, BatchedRequestsCanCompleteSynchronouslyWithOneLiveTask)
{
    CompletingRequests commands;
    auto config = Config(1);
    config.maxIoEntries = 2;
    TaskManager manager(std::move(config), Dependencies([&commands](Request& request) {
                            return commands.Post(request);
                        }));
    commands.Bind(manager);
    ASSERT_TRUE(manager.Start().Success());

    auto submitted = manager.SubmitTransfer(OpType::LOAD, TwoEntryTask());
    ASSERT_TRUE(submitted);
    EXPECT_TRUE(manager.WaitTransfer(std::move(submitted).Value()).Success());
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, NodeNoSpaceIsTerminal)
{
    std::size_t attempts = 0;
    TaskManager manager(Config(), Dependencies([&attempts](Request&) {
                            ++attempts;
                            return Status::NoSpace();
                        }));
    ASSERT_TRUE(manager.Start().Success());

    auto submitted = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(submitted);
    auto status = manager.WaitTransfer(std::move(submitted).Value());
    EXPECT_EQ(status, Status::NoSpace());
    EXPECT_EQ(attempts, std::size_t{1});
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, LookupOwnsBlockIdsBeforeReturningHandle)
{
    BlockingRequests commands;
    TaskManager manager(
        Config(), Dependencies([&commands](Request& request) { return commands.Post(request); }));
    ASSERT_TRUE(manager.Start().Success());

    auto transfer = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(transfer);
    const auto transfer_id = std::move(transfer).Value();
    const auto transfer_request = commands.WaitForSubmission();

    Detail::BlockId block{};
    block[0] = std::byte{1};
    auto lookup = manager.SubmitLookup(&block, 1);
    ASSERT_TRUE(lookup);
    const auto lookup_id = std::move(lookup).Value();
    block[0] = std::byte{2};

    commands.Release();
    const auto lookup_request = commands.WaitForSubmission(1);
    ASSERT_EQ(lookup_request.entries.size(), std::size_t{1});
    EXPECT_EQ(lookup_request.entries[0].blockId[0], std::byte{1});

    PublishCompletion(manager, RequestCompleted{transfer_request.taskId,
                                                transfer_request.requestId,
                                                transfer_request.nodeId,
                                                Status::OK(),
                                                {}});
    PublishCompletion(manager, RequestCompleted{lookup_request.taskId,
                                                lookup_request.requestId,
                                                lookup_request.nodeId,
                                                Status::OK(),
                                                {}});
    EXPECT_TRUE(manager.WaitTransfer(transfer_id).Success());
    EXPECT_TRUE(manager.WaitLookup(lookup_id));
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, ExpiredQueuedSubmissionSkipsRouting)
{
    BlockingRequests commands;
    auto router = std::make_shared<CountingRouter>();
    auto config = Config();
    config.timeouts = TimeoutConfig{std::chrono::milliseconds{20}, std::chrono::milliseconds{20},
                                    std::chrono::milliseconds{20}};
    auto dependencies =
        Dependencies([&commands](Request& request) { return commands.Post(request); });
    dependencies.router = router;
    TaskManager manager(std::move(config), std::move(dependencies));
    ASSERT_TRUE(manager.Start().Success());

    auto first = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(first);
    const auto firstTaskId = std::move(first).Value();
    const auto firstRequest = commands.WaitForSubmission();

    auto expired = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(expired);
    const auto expiredTaskId = std::move(expired).Value();
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    commands.Release();

    PublishCompletion(
        manager,
        RequestCompleted{
            firstRequest.taskId, firstRequest.requestId, firstRequest.nodeId, Status::OK(), {}});
    auto expiredStatus = manager.WaitTransfer(expiredTaskId);
    EXPECT_EQ(expiredStatus, Status::Timeout());
    EXPECT_EQ(router->calls(), std::size_t{1});
    (void)manager.WaitTransfer(firstTaskId);
    manager.Shutdown();
}

TEST_F(TaskManagerAsyncTest, DeadlineWaitsForPostedRequestCompletion)
{
    FailSecondRequest commands;
    auto config = Config();
    config.timeouts = TimeoutConfig{std::chrono::milliseconds{20}, std::chrono::milliseconds{20},
                                    std::chrono::milliseconds{20}};
    TaskManager manager(std::move(config), Dependencies([&commands](Request& request) {
                            return commands.Post(request);
                        }));
    ASSERT_TRUE(manager.Start().Success());

    auto submitted = manager.SubmitTransfer(OpType::LOAD, ValidTask());
    ASSERT_TRUE(submitted);
    const auto taskId = std::move(submitted).Value();
    const auto requests = commands.WaitForRequests(1);
    std::this_thread::sleep_for(std::chrono::milliseconds{30});

    PublishCompletion(
        manager,
        RequestCompleted{taskId, requests[0].requestId, requests[0].nodeId, Status::Timeout(), {}});
    auto status = manager.WaitTransfer(taskId);
    EXPECT_EQ(status, Status::Timeout());
    manager.Shutdown();
}

}  // namespace
}  // namespace UC::Dram
