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
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include "node_actor.h"
#include "node_scheduler.h"
#include "router/router.h"
#include "task_manager.h"

namespace UC::Dram {
namespace {
using namespace std::chrono_literals;
NodeEndpoint Peer() { return {1, "127.0.0.1", 10000, "mock"}; }

Request DumpRequest(std::uintptr_t handle = 123)
{
    Request request;
    request.taskId = request.requestId = request.nodeId = 1;
    request.op = OpType::DUMP;
    request.deadline = std::chrono::steady_clock::now() + 1h;
    request.prerequisiteHandle = handle;
    request.entries.resize(1);
    request.entries[0].blockId[0] = std::byte{1};
    request.entries[0].buffer = {0x1000, 64};
    return request;
}

struct TestNode {
    NodeEventPublisher publish;
    std::array<std::uint8_t, 64> slot{};
    std::atomic<std::size_t> sends{0}, acquisitions{0};
    std::vector<RequestCompleted> completions;

    NodeDependencies Dependencies()
    {
        NodeDependencies result;
        result.publishCompletion = [this](std::vector<RequestCompleted>& events) {
            for (auto& event : events) { completions.push_back(std::move(event)); }
        };
        result.submitTransport = [this](TransportCommand& command) {
            if (const auto* connect = std::get_if<Connect>(&command)) {
                publish(1, ConnectCompleted{1, connect->laneId, connect->epoch, Status::OK()});
            } else if (const auto* transmit = std::get_if<Transmit>(&command)) {
                ++sends;
                publish(1, TransmitCompleted{transmit->token, Status::OK()});
                publish(1, ReplyObserved{transmit->token, Status::OK(), {}});
            } else {
                ADD_FAILURE() << "unsent requests must not require a fence";
            }
            return Status::OK();
        };
        result.acquireReplySlot = [this](const RequestToken&, OpType,
                                         std::size_t) -> Expected<ReplySlot> {
            ++acquisitions;
            return ReplySlot{slot.data(), slot.data(), slot.size(), 0};
        };
        result.releaseReplySlot = [](const RequestToken&, const ReplySlot&) {
            return Status::OK();
        };
        return result;
    }
};

class UCDramPrerequisiteActorTest : public testing::Test {
protected:
    void SetUp() override
    {
        node.publish = [this](NodeId, NodeEvent event) { events.push_back(std::move(event)); };
        auto dependencies = node.Dependencies();
        dependencies.queryPrerequisite = [this](std::uintptr_t handle) {
            EXPECT_EQ(handle, 123U);
            ++queries;
            return query();
        };
        NodeActor::Config config{
            Peer(), {1, 1},
             1ms, 175us
        };
        actor = std::make_unique<NodeActor>(config, std::move(dependencies));
        Advance();
    }
    void Advance()
    {
        actor->Advance(now);
        while (!events.empty()) {
            auto event = std::move(events.front());
            events.pop_front();
            actor->Handle(std::move(event), now);
            actor->Advance(now);
        }
    }
    void Post(Request request)
    {
        actor->Handle(std::move(request), now);
        Advance();
    }
    TestNode node;
    std::deque<NodeEvent> events;
    std::unique_ptr<NodeActor> actor;
    NodeActor::TimePoint now{std::chrono::steady_clock::now()};
    std::size_t queries{0};
    bool ready{false};
    std::function<Expected<bool>()> query = [this] { return ready; };
};

TEST_F(UCDramPrerequisiteActorTest, PollsAtConfiguredIntervalAndSendsOnlyWhenReady)
{
    Post(DumpRequest());
    EXPECT_EQ(node.acquisitions, 0U);
    EXPECT_EQ(actor->NextWakeup(), now + 175us);
    ready = true;
    now += 174us;
    for (int i = 0; i < 10; ++i) { Advance(); }
    EXPECT_EQ(queries, 1U);
    EXPECT_EQ(node.sends, 0U);
    EXPECT_EQ(actor->NextWakeup(), now + 1us);
    now += 1us;
    Advance();
    EXPECT_EQ(queries, 2U);
    ASSERT_EQ(node.completions.size(), 1U);
    EXPECT_TRUE(node.completions[0].status.Success());
    Post(DumpRequest(0));
    EXPECT_EQ(node.sends, 2U);
    now += 1h;
    Advance();
    EXPECT_EQ(queries, 2U);
}

TEST_F(UCDramPrerequisiteActorTest, QueryErrorsCompleteWithoutSendingOrAcquiringSlots)
{
    for (bool throws : {false, true}) {
        query = [throws]() -> Expected<bool> {
            if (throws) { throw std::runtime_error("query failed"); }
            return Status::Error("query failed");
        };
        Post(DumpRequest());
    }
    EXPECT_EQ(node.acquisitions, 0U);
    EXPECT_EQ(node.sends, 0U);
    ASSERT_EQ(node.completions.size(), 2U);
    for (const auto& event : node.completions) { EXPECT_TRUE(event.status.Failure()); }
    Advance();
    EXPECT_EQ(queries, 2U);
}

TEST_F(UCDramPrerequisiteActorTest, PendingTimeoutStopsQueryingWithoutSending)
{
    auto request = DumpRequest();
    request.deadline = now + 1h;
    Post(std::move(request));
    now += 1h;
    Advance();
    ASSERT_EQ(node.completions.size(), 1U);
    EXPECT_EQ(node.completions[0].status, Status::Timeout());
    ready = true;
    now += 1h;
    Advance();
    EXPECT_EQ(queries, 1U);
    EXPECT_EQ(node.sends, 0U);
    EXPECT_EQ(node.acquisitions, 0U);
}

TEST(UCDramOwnerEventPollingTest, RetriesWithoutTrafficAndStopsQueryingAfterCompletion)
{
    TestNode node;
    std::atomic<bool> ready{false};
    std::atomic<std::size_t> queries{0};
    std::promise<void> polled, completed;
    auto dependencies = node.Dependencies();
    dependencies.queryPrerequisite = [&](std::uintptr_t) -> Expected<bool> {
        if (++queries == 2) { polled.set_value(); }
        return ready.load();
    };
    dependencies.publishCompletion = [&](std::vector<RequestCompleted>& events) {
        EXPECT_EQ(events.size(), 1U);
        EXPECT_TRUE(events[0].status.Success());
        completed.set_value();
    };
    NodeSchedulerConfig config{
        {Peer()},
        {1, 1},
        1ms, 1
    };
    NodeScheduler scheduler(config, std::move(dependencies));
    node.publish = [&](NodeId id, NodeEvent event) { scheduler.Publish(id, std::move(event)); };
    ASSERT_TRUE(scheduler.Start().Success());
    auto request = DumpRequest();
    ASSERT_TRUE(scheduler.Post(request).Success());
    EXPECT_EQ(polled.get_future().wait_for(3s), std::future_status::ready);
    EXPECT_EQ(node.sends, 0U);
    ready = true;
    EXPECT_EQ(completed.get_future().wait_for(3s), std::future_status::ready);
    const auto count = queries.load();
    scheduler.Shutdown();
    EXPECT_EQ(queries, count);
    EXPECT_EQ(node.sends, 1U);
}

TEST(UCDramOwnerEventPollingTest, TaskManagerExceptionJoinsQueryBeforeReturningFailure)
{
    TestNode node;
    TaskManager* managerPtr = nullptr;
    std::promise<void> entered, release, stopping;
    auto released = release.get_future();
    std::atomic<bool> destroyed{false};
    auto dependencies = node.Dependencies();
    dependencies.queryPrerequisite = [&](std::uintptr_t handle) -> Expected<bool> {
        EXPECT_EQ(handle, 123U);
        EXPECT_FALSE(destroyed.load());
        entered.set_value();
        released.wait();
        EXPECT_FALSE(destroyed.load());
        return Status::Error("query failed");
    };
    dependencies.publishCompletion = [&](std::vector<RequestCompleted>& events) {
        managerPtr->Publish(events);
    };
    NodeSchedulerConfig config{
        {Peer()},
        {1, 1},
        1ms, 1
    };
    NodeScheduler scheduler(config, std::move(dependencies));
    node.publish = [&](NodeId id, NodeEvent event) { scheduler.Publish(id, std::move(event)); };
    TaskManagerConfig taskConfig{
        {64},
        32, 8, TimeoutConfig{1h, 1h, 1h}
    };
    TaskManager manager(taskConfig,
                        TaskManagerDependencies{
                            UC::Router::CreateRouter({NodeId{1}}, {}, UC::Router::RouterConfig{}),
                            [&](Request& request) -> Status {
                                if (request.op == OpType::LOAD) {
                                    throw std::runtime_error("dispatch failed");
                                }
                                return scheduler.Post(request);
                            },
                            [&] {
                                stopping.set_value();
                                scheduler.Shutdown();
                            }});
    managerPtr = &manager;
    ASSERT_TRUE(scheduler.Start().Success());
    ASSERT_TRUE(manager.Start().Success());
    Detail::TaskDesc task;
    task.push_back(Detail::Shard{{}, 0, {reinterpret_cast<void*>(0x1000)}});
    task.prerequisiteHandle = 123;
    auto dump = manager.SubmitTransfer(OpType::DUMP, task);
    // Always release the blocked query before teardown, including on test failure.
    EXPECT_EQ(entered.get_future().wait_for(3s), std::future_status::ready);
    auto load = manager.SubmitTransfer(OpType::LOAD, task);
    EXPECT_TRUE(dump && load);
    EXPECT_EQ(stopping.get_future().wait_for(3s), std::future_status::ready);
    for (auto* submitted : {&dump, &load}) {
        if (!*submitted) { continue; }
        auto done = manager.Check(submitted->Value());
        EXPECT_TRUE(done && !done.Value());
    }
    release.set_value();
    if (dump) { EXPECT_TRUE(manager.WaitTransfer(dump.Value()).Failure()); }
    if (load) { EXPECT_TRUE(manager.WaitTransfer(load.Value()).Failure()); }
    destroyed = true;
    manager.Shutdown();
    EXPECT_EQ(node.sends, 0U);
    auto following = DumpRequest();
    EXPECT_TRUE(scheduler.Post(following).Failure());
    EXPECT_FALSE(manager.SubmitTransfer(OpType::LOAD, task));
}

}  // namespace
}  // namespace UC::Dram
