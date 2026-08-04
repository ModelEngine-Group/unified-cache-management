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
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <gtest/gtest.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>
#include "node_actor.h"
#include "node_scheduler.h"

namespace UC::Dram {
namespace {

using namespace std::chrono_literals;

IoEntry Entry(std::uint8_t value)
{
    IoEntry entry;
    entry.blockId[0] = static_cast<std::byte>(value);
    return entry;
}

NodeEndpoint Endpoint(NodeId nodeId)
{
    return NodeEndpoint{nodeId, "127.0.0.1", static_cast<std::uint16_t>(10000 + nodeId),
                        "127.0.0.1:" + std::to_string(20000 + nodeId)};
}

NodeLimits Limits(std::size_t maxInflightRequests) { return NodeLimits{maxInflightRequests, 8}; }

void MoveCompletions(std::vector<RequestCompleted>& source,
                     std::vector<RequestCompleted>& destination)
{
    for (auto& event : source) { destination.push_back(std::move(event)); }
}

NodeSchedulerConfig SchedulerConfig(std::vector<NodeEndpoint> nodes, std::size_t runnerCount,
                                    std::size_t maxInflightRequests,
                                    std::chrono::milliseconds reconnectInterval)
{
    return NodeSchedulerConfig{std::move(nodes), Limits(maxInflightRequests), reconnectInterval,
                               runnerCount};
}

NodeDependencies SchedulerDependencies(TransportCommandSubmitter submit)
{
    NodeDependencies dependencies;
    dependencies.publishCompletion = [](std::vector<RequestCompleted>&) {};
    dependencies.submitTransport = std::move(submit);
    dependencies.acquireReplySlot = [](const RequestToken&, OpType,
                                       std::size_t) -> Expected<ReplySlot> {
        static std::uint8_t replyByte = 0;
        return ReplySlot{&replyByte, &replyByte, 1, 0};
    };
    dependencies.releaseReplySlot = [](const RequestToken&, const ReplySlot&) {
        return Status::OK();
    };
    return dependencies;
}

NodeActor::Config ActorConfig(std::size_t maxInflightRequests,
                              std::chrono::milliseconds reconnectInterval)
{
    return NodeActor::Config{Endpoint(1), Limits(maxInflightRequests), reconnectInterval};
}

Request MakeRequest(
    TaskId taskId = 1, RequestId requestId = 1, NodeId nodeId = 1, std::size_t entryCount = 1,
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + 1h,
    std::uint8_t entryValue = 1)
{
    Request request;
    request.taskId = taskId;
    request.requestId = requestId;
    request.nodeId = nodeId;
    request.entries.assign(entryCount, Entry(entryValue));
    request.deadline = deadline;
    return request;
}

void ConnectActor(NodeActor& actor, NodeActor::TimePoint now)
{
    actor.Advance(now);
    actor.Handle(
        NodeEvent{
            ConnectCompleted{1, kDefaultLaneId, 1, Status::OK()}
    },
        now);
}

void SubmitToActor(NodeActor& actor, Request request, NodeActor::TimePoint now)
{
    actor.Handle(std::move(request), now);
}

void PublishConnected(NodeScheduler& scheduler)
{
    scheduler.Publish(1, NodeEvent{
                             ConnectCompleted{1, kDefaultLaneId, 1, Status::OK()}
    });
}

template <typename Predicate>
bool WaitUntil(std::mutex& mutex, std::condition_variable& changed, Predicate predicate,
               std::chrono::milliseconds timeout = 1s)
{
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, timeout, std::move(predicate));
}

TEST(NodeActorTest, PacksLookupWithTargetDramPoolProtocol)
{
    std::array<std::uint8_t, 64> reply{};
    Status unpackStatus = Status::Error();
    std::unique_ptr<DramPool::KvRequest> unpacked;
    DramPool::ProtocolManager serverProtocol;
    auto dependencies = NodeDependencies{
        [](std::vector<RequestCompleted>&) {},
        [&](TransportCommand& command) {
            const auto* transmit = std::get_if<Transmit>(&command);
            if (transmit == nullptr) { return Status::OK(); }
            unpackStatus = serverProtocol.UnpackRequest(transmit->payload.data(),
                                                        transmit->payload.size(), unpacked);
            return Status::Retry();
        },
        [&](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            return ReplySlot{reply.data(), reply.data(), reply.size(), 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(ActorConfig(1, 1ms), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    ConnectActor(actor, now);
    SubmitToActor(actor, MakeRequest(1, 1, 1, 1, now + 1h, 7), now);
    actor.Advance(now);

    ASSERT_TRUE(unpackStatus.Success());
    const auto* lookup = dynamic_cast<const DramPool::KvLookupRequest*>(unpacked.get());
    ASSERT_NE(lookup, nullptr);
    ASSERT_EQ(lookup->entries.size(), std::size_t{1});
    EXPECT_EQ(lookup->entries[0].key[0], std::byte{7});
    EXPECT_EQ(lookup->resp_addr, reinterpret_cast<std::uintptr_t>(reply.data()));
}

TEST(NodeActorTest, ReplyReleaseFailureDoesNotBlockCompletion)
{
    std::array<std::uint8_t, 64> reply{};
    std::optional<Status> completion;
    std::size_t releaseCount = 0;
    const auto releaseFailure = Status::Error("reply release failed");
    auto dependencies = NodeDependencies{
        [&](std::vector<RequestCompleted>& events) {
            completion.emplace(std::move(events.front().status));
        },
        [](TransportCommand& command) {
            if (std::get_if<Transmit>(&command) != nullptr) { return Status::Retry(); }
            return Status::OK();
        },
        [&](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            return ReplySlot{reply.data(), reply.data(), reply.size(), 0};
        },
        [&](const RequestToken&, const ReplySlot&) {
            ++releaseCount;
            return releaseFailure;
        },
    };
    NodeActor actor(ActorConfig(1, 1ms), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    ConnectActor(actor, now);
    SubmitToActor(actor, MakeRequest(1, 1, 1, 1, now + 1h), now);
    actor.Advance(now);

    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(*completion, Status::Retry());
    EXPECT_EQ(releaseCount, std::size_t{1});
}

TEST(NodeActorTest, MapsOrderedReplyResultsToOriginalTaskIndexes)
{
    std::optional<RequestCompleted> completion;
    auto dependencies = NodeDependencies{
        [&](std::vector<RequestCompleted>& events) {
            completion.emplace(std::move(events.front()));
        },
        [](TransportCommand&) { return Status::OK(); },
        [](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            static std::uint8_t replyByte = 0;
            return ReplySlot{&replyByte, &replyByte, 1, 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(ActorConfig(1, 1ms), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    ConnectActor(actor, now);

    auto request = MakeRequest(1, 1, 1, 2, now + 1h);
    request.entries[0].originalIndex = 4;
    request.entries[1].originalIndex = 1;
    SubmitToActor(actor, std::move(request), now);
    actor.Advance(now);
    actor.Handle(
        NodeEvent{
            ReplyObserved{RequestToken{1, kDefaultLaneId, 1, 1},
                          Status::OK(),
                          {EntryResult{0, true, 0}, EntryResult{0, false, 0}}}
    },
        now);
    actor.Advance(now);

    ASSERT_TRUE(completion.has_value());
    ASSERT_EQ(completion->entryResults.size(), std::size_t{2});
    EXPECT_EQ(completion->entryResults[0].originalIndex, std::size_t{4});
    EXPECT_EQ(completion->entryResults[1].originalIndex, std::size_t{1});
}

TEST(NodeActorTest, SynchronousRepliesCompleteInOneBatch)
{
    std::vector<RequestCompleted> completions;
    std::size_t completionBatchCount = 0;
    std::size_t transmitCount = 0;
    NodeActor* actorPointer = nullptr;
    const auto now = std::chrono::steady_clock::now();
    auto dependencies = NodeDependencies{
        [&](std::vector<RequestCompleted>& events) {
            ++completionBatchCount;
            MoveCompletions(events, completions);
        },
        [&](TransportCommand& command) {
            const auto* transmit = std::get_if<Transmit>(&command);
            if (transmit == nullptr) { return Status::OK(); }
            ++transmitCount;
            actorPointer->Handle(
                NodeEvent{
                    ReplyObserved{transmit->token, Status::OK(), {EntryResult{0, true, 0}}}
            },
                now);
            return Status::OK();
        },
        [](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            static std::uint8_t replyByte = 0;
            return ReplySlot{&replyByte, &replyByte, 1, 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(ActorConfig(2, 1ms), std::move(dependencies));
    actorPointer = &actor;
    ConnectActor(actor, now);

    SubmitToActor(actor, MakeRequest(1, 1, 1, 1, now + 1h), now);
    SubmitToActor(actor, MakeRequest(2, 2, 1, 1, now + 1h), now);
    actor.Advance(now);

    EXPECT_EQ(transmitCount, std::size_t{2});
    EXPECT_EQ(completionBatchCount, std::size_t{1});
    EXPECT_EQ(completions.size(), std::size_t{2});
}

TEST(NodeActorTest, FailedRecoveryFenceIsFatal)
{
    std::size_t fenceAttempts = 0;
    auto dependencies = NodeDependencies{
        [](std::vector<RequestCompleted>&) { std::_Exit(0); },
        [&](TransportCommand& command) {
            if (std::get_if<FenceEpoch>(&command) != nullptr) { ++fenceAttempts; }
            return Status::OK();
        },
        [](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            static std::uint8_t replyByte = 0;
            return ReplySlot{&replyByte, &replyByte, 1, 0};
        },
        [](const RequestToken&, const ReplySlot&) -> Status { std::_Exit(0); },
    };
    NodeActor actor(ActorConfig(1, 1ms), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    ConnectActor(actor, now);
    SubmitToActor(actor, MakeRequest(1, 1, 1, 1, now + 1ms), now);
    actor.Advance(now);
    actor.Advance(now + 2ms);
    actor.Advance(now + 3ms);
    ASSERT_EQ(fenceAttempts, std::size_t{1});

    EXPECT_DEATH(
        actor.Handle(
            NodeEvent{
                FenceCompleted{1, kDefaultLaneId, 1, Status::Error("recovery fence failed")}
    },
            now + 2ms),
        "");
}

TEST(NodeActorTest, ExposedTimeoutFencesActiveRequests)
{
    std::vector<RequestCompleted> completions;
    std::vector<RequestId> transmitted;
    std::size_t fenceAttempts = 0;
    auto dependencies = NodeDependencies{
        [&](std::vector<RequestCompleted>& events) { MoveCompletions(events, completions); },
        [&](TransportCommand& command) {
            if (const auto* transmit = std::get_if<Transmit>(&command)) {
                transmitted.push_back(transmit->token.requestId);
            }
            if (std::get_if<FenceEpoch>(&command) != nullptr) { ++fenceAttempts; }
            return Status::OK();
        },
        [](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            static std::uint8_t replyByte = 0;
            return ReplySlot{&replyByte, &replyByte, 1, 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(ActorConfig(2, 1ms), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    ConnectActor(actor, now);

    SubmitToActor(actor, MakeRequest(1, 1, 1, 1, now + 1ms), now);
    SubmitToActor(actor, MakeRequest(2, 2, 1, 1, now + 1h), now);
    actor.Advance(now);
    actor.Advance(now + 2ms);

    EXPECT_TRUE(completions.empty());
    EXPECT_EQ(fenceAttempts, std::size_t{1});

    actor.Handle(
        NodeEvent{
            FenceCompleted{1, kDefaultLaneId, 1, Status::OK()}
    },
        now + 2ms);
    actor.Advance(now + 2ms);
    ASSERT_EQ(completions.size(), std::size_t{2});
    EXPECT_EQ(completions[0].status, Status::Timeout());
    EXPECT_EQ(completions[1].status, Status::Timeout());
    EXPECT_EQ(transmitted, (std::vector<RequestId>{1, 2}));
}

TEST(NodeActorTest, PendingRequestTimesOutWithoutRemoteAccess)
{
    std::optional<RequestCompleted> completion;
    std::size_t transmitCount = 0;
    std::size_t fenceCount = 0;
    auto dependencies = NodeDependencies{
        [&](std::vector<RequestCompleted>& events) {
            completion.emplace(std::move(events.front()));
        },
        [&](TransportCommand& command) {
            transmitCount += std::get_if<Transmit>(&command) != nullptr;
            fenceCount += std::get_if<FenceEpoch>(&command) != nullptr;
            return Status::Retry();
        },
        [](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            static std::uint8_t replyByte = 0;
            return ReplySlot{&replyByte, &replyByte, 1, 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(ActorConfig(1, 1h), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    actor.Advance(now);
    actor.Handle(MakeRequest(1, 1, 1, 1, now + 1ms), now);
    actor.Advance(now + 2ms);

    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->status, Status::Timeout());
    EXPECT_EQ(transmitCount, std::size_t{0});
    EXPECT_EQ(fenceCount, std::size_t{0});
}

TEST(NodeActorTest, RecomputesPendingWakeupAfterAStaleDeadline)
{
    std::vector<RequestToken> transmitted;
    auto dependencies = NodeDependencies{
        [](std::vector<RequestCompleted>&) {},
        [&](TransportCommand& command) {
            if (const auto* transmit = std::get_if<Transmit>(&command)) {
                transmitted.push_back(transmit->token);
            }
            return Status::OK();
        },
        [](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            static std::uint8_t replyByte = 0;
            return ReplySlot{&replyByte, &replyByte, 1, 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(ActorConfig(1, 1h), std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    ConnectActor(actor, now);
    SubmitToActor(actor, MakeRequest(1, 1, 1, 1, now + 10ms), now);
    SubmitToActor(actor, MakeRequest(2, 2, 1, 1, now + 20ms), now);
    SubmitToActor(actor, MakeRequest(3, 3, 1, 1, now + 30ms), now);
    actor.Advance(now);

    ASSERT_EQ(transmitted.size(), std::size_t{1});
    actor.Handle(
        NodeEvent{
            ReplyObserved{transmitted.front(), Status::OK(), {}}
    },
        now + 1ms);
    actor.Advance(now + 1ms);

    ASSERT_EQ(transmitted.size(), std::size_t{2});
    EXPECT_EQ(actor.NextWakeup(), now + 10ms);
    actor.Advance(now + 10ms);
    EXPECT_EQ(actor.NextWakeup(), now + 20ms);
}

TEST(NodeSchedulerTest, ShutdownStopsNewCommands)
{
    NodeScheduler scheduler(
        SchedulerConfig({Endpoint(1)}, 1, 1, 1h),
        SchedulerDependencies([](TransportCommand&) { return Status::Retry(); }));
    ASSERT_TRUE(scheduler.Start().Success());
    ASSERT_TRUE(scheduler.Shutdown().Success());

    auto command = MakeRequest();
    EXPECT_TRUE(scheduler.Post(command).Failure());
}

TEST(NodeSchedulerTest, CompletionPublisherExceptionIsFatal)
{
    EXPECT_DEATH(
        {
            std::mutex mutex;
            std::condition_variable changed;
            bool connectSubmitted = false;
            bool transmitEntered = false;
            bool releaseTransmit = false;
            auto submit = [&](TransportCommand& command) {
                if (std::get_if<Connect>(&command) != nullptr) {
                    {
                        std::lock_guard lock(mutex);
                        connectSubmitted = true;
                    }
                    changed.notify_all();
                    return Status::OK();
                }
                if (std::get_if<Transmit>(&command) != nullptr) {
                    std::unique_lock lock(mutex);
                    transmitEntered = true;
                    changed.notify_all();
                    changed.wait(lock, [&] { return releaseTransmit; });
                    return Status::Retry();
                }
                return Status::OK();
            };
            auto config = SchedulerConfig({Endpoint(1)}, 1, 1, 1h);
            config.limits.maxBatchEntries = 1;
            auto dependencies = SchedulerDependencies(std::move(submit));
            dependencies.publishCompletion = [](std::vector<RequestCompleted>&) {
                throw std::runtime_error("completion receiver failed");
            };
            NodeScheduler scheduler(std::move(config), std::move(dependencies));
            (void)scheduler.Start();

            (void)WaitUntil(mutex, changed, [&] { return connectSubmitted; });
            PublishConnected(scheduler);
            auto command = MakeRequest();
            (void)scheduler.Post(command);
            (void)WaitUntil(mutex, changed, [&] { return transmitEntered; });
            {
                std::lock_guard lock(mutex);
                releaseTransmit = true;
            }
            changed.notify_all();
            (void)scheduler.Shutdown();
        },
        "");
}

TEST(NodeSchedulerTest, DisconnectedNodeQueuesRequests)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool connectAttempted = false;
    auto submit = [&](TransportCommand& command) {
        if (std::get_if<Connect>(&command) != nullptr) {
            std::lock_guard lock(mutex);
            connectAttempted = true;
            changed.notify_all();
        }
        return Status::Retry();
    };
    NodeScheduler scheduler(SchedulerConfig({Endpoint(1)}, 1, 1, 1h),
                            SchedulerDependencies(submit));
    ASSERT_TRUE(scheduler.Start().Success());
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return connectAttempted; }));

    auto command = MakeRequest();
    EXPECT_TRUE(scheduler.Post(command).Success());
    EXPECT_TRUE(scheduler.Shutdown().Success());
}

TEST(NodeSchedulerTest, RequestsWaitForInflightCapacity)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool connectSubmitted = false;
    std::vector<RequestId> transmitted;
    RequestToken token;
    auto dependencies = SchedulerDependencies([&](TransportCommand& command) {
        std::lock_guard lock(mutex);
        if (std::get_if<Connect>(&command) != nullptr) { connectSubmitted = true; }
        if (const auto* transmit = std::get_if<Transmit>(&command)) {
            if (transmitted.empty()) { token = transmit->token; }
            transmitted.push_back(transmit->token.requestId);
        }
        changed.notify_all();
        return Status::OK();
    });

    NodeScheduler scheduler(SchedulerConfig({Endpoint(1)}, 1, 1, 1h), std::move(dependencies));
    ASSERT_TRUE(scheduler.Start().Success());
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return connectSubmitted; }));
    PublishConnected(scheduler);

    auto firstCommand = MakeRequest(1, 1);
    ASSERT_TRUE(scheduler.Post(firstCommand).Success());
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return transmitted.size() == 1; }));

    auto secondCommand = MakeRequest(2, 2, 1);
    EXPECT_TRUE(scheduler.Post(secondCommand).Success());
    {
        std::unique_lock lock(mutex);
        EXPECT_FALSE(changed.wait_for(lock, 20ms, [&] { return transmitted.size() == 2; }));
    }

    scheduler.Publish(1, NodeEvent{
                             ReplyObserved{token, Status::OK(), {}}
    });
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return transmitted.size() == 2; }));
    EXPECT_EQ(transmitted, (std::vector<RequestId>{1, 2}));
    EXPECT_TRUE(scheduler.Shutdown().Success());
}

TEST(NodeSchedulerTest, DifferentRunnersRunConcurrently)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool firstConnectEntered = false;
    bool secondConnectEntered = false;
    bool releaseFirstConnect = false;
    auto submit = [&](TransportCommand& command) {
        const auto* connect = std::get_if<Connect>(&command);
        if (connect == nullptr) { return Status::Retry(); }
        std::unique_lock lock(mutex);
        if (connect->nodeId == 1) {
            firstConnectEntered = true;
            changed.notify_all();
            changed.wait(lock, [&] { return releaseFirstConnect; });
        } else if (connect->nodeId == 2) {
            secondConnectEntered = true;
            changed.notify_all();
        }
        return Status::Retry();
    };
    NodeScheduler scheduler(SchedulerConfig({Endpoint(1), Endpoint(2)}, 2, 1, 1h),
                            SchedulerDependencies(submit));
    ASSERT_TRUE(scheduler.Start().Success());

    bool bothEntered = false;
    {
        std::unique_lock lock(mutex);
        bothEntered =
            changed.wait_for(lock, 1s, [&] { return firstConnectEntered && secondConnectEntered; });
        releaseFirstConnect = true;
    }
    changed.notify_all();

    EXPECT_TRUE(bothEntered);
    EXPECT_TRUE(scheduler.Shutdown().Success());
}

TEST(NodeSchedulerTest, NodesOnTheSameRunnerRunSerially)
{
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t connectCount = 0;
    bool releaseFirstConnect = false;
    auto submit = [&](TransportCommand& command) {
        const auto* connect = std::get_if<Connect>(&command);
        if (connect == nullptr) { return Status::Retry(); }
        std::unique_lock lock(mutex);
        ++connectCount;
        changed.notify_all();
        if (connectCount == 1) {
            changed.wait(lock, [&] { return releaseFirstConnect; });
        }
        return Status::Retry();
    };
    NodeScheduler scheduler(SchedulerConfig({Endpoint(1), Endpoint(2)}, 1, 1, 1h),
                            SchedulerDependencies(submit));
    ASSERT_TRUE(scheduler.Start().Success());

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(changed.wait_for(lock, 1s, [&] { return connectCount == 1; }));
        EXPECT_FALSE(changed.wait_for(lock, 20ms, [&] { return connectCount == 2; }));
        releaseFirstConnect = true;
    }
    changed.notify_all();

    {
        std::unique_lock lock(mutex);
        EXPECT_TRUE(changed.wait_for(lock, 1s, [&] { return connectCount == 2; }));
    }
    EXPECT_TRUE(scheduler.Shutdown().Success());
}

TEST(NodeSchedulerTest, DisconnectedActorRetriesConnectionOnSchedule)
{
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t attempts = 0;
    auto submit = [&](TransportCommand& command) {
        if (std::get_if<Connect>(&command) != nullptr) {
            std::lock_guard lock(mutex);
            ++attempts;
            changed.notify_all();
        }
        return Status::Retry();
    };
    NodeScheduler scheduler(SchedulerConfig({Endpoint(1)}, 1, 1, 2ms),
                            SchedulerDependencies(submit));
    ASSERT_TRUE(scheduler.Start().Success());

    const bool retried = WaitUntil(mutex, changed, [&] { return attempts >= 2; });

    EXPECT_TRUE(retried);
    EXPECT_TRUE(scheduler.Shutdown().Success());
}

TEST(NodeSchedulerTest, TransportAdmissionFailureDoesNotParkActor)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool releaseTransmit = false;
    std::size_t transmitCount = 0;
    std::size_t completionCount = 0;
    std::size_t releases = 0;
    NodeScheduler* scheduler = nullptr;
    auto dependencies = SchedulerDependencies([&](TransportCommand& command) {
        if (const auto* connect = std::get_if<Connect>(&command)) {
            scheduler->Publish(connect->nodeId,
                               NodeEvent{
                                   ConnectCompleted{connect->nodeId, connect->laneId,
                                                    connect->epoch, Status::OK()}
            });
            return Status::OK();
        }
        if (const auto* fence = std::get_if<FenceEpoch>(&command)) {
            scheduler->Publish(fence->nodeId, NodeEvent{
                                                  FenceCompleted{fence->nodeId, fence->laneId,
                                                                 fence->epoch, Status::OK()}
            });
            return Status::OK();
        }
        if (std::get_if<Transmit>(&command) != nullptr) {
            std::unique_lock lock(mutex);
            ++transmitCount;
            changed.notify_all();
            if (transmitCount == 1) {
                changed.wait(lock, [&] { return releaseTransmit; });
            }
        }
        return Status::Retry();
    });
    dependencies.publishCompletion = [&](std::vector<RequestCompleted>& events) {
        std::lock_guard lock(mutex);
        completionCount += events.size();
        changed.notify_all();
    };
    dependencies.releaseReplySlot = [&](const RequestToken&, const ReplySlot&) {
        std::lock_guard lock(mutex);
        ++releases;
        return Status::OK();
    };

    NodeScheduler instance(SchedulerConfig({Endpoint(1)}, 1, 1, 1h), std::move(dependencies));
    scheduler = &instance;
    ASSERT_TRUE(instance.Start().Success());

    auto command = MakeRequest();
    ASSERT_TRUE(instance.Post(command).Success());
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return transmitCount == 1; }));

    auto pending = MakeRequest(2, 2);
    EXPECT_TRUE(instance.Post(pending).Success());
    {
        std::lock_guard lock(mutex);
        releaseTransmit = true;
    }
    changed.notify_all();

    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return completionCount == 2; }));
    EXPECT_EQ(transmitCount, std::size_t{2});
    EXPECT_EQ(releases, std::size_t{2});
    EXPECT_TRUE(instance.Shutdown().Success());
}

TEST(NodeSchedulerTest, ShutdownAbandonsActiveRequest)
{
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t connectAttempts = 0;
    std::size_t transmitAttempts = 0;
    std::size_t fenceAttempts = 0;
    std::size_t releases = 0;
    bool completed = false;
    auto submit = [&](TransportCommand& command) {
        std::lock_guard lock(mutex);
        if (std::get_if<Connect>(&command) != nullptr) {
            ++connectAttempts;
        } else if (std::get_if<Transmit>(&command) != nullptr) {
            ++transmitAttempts;
        } else if (std::get_if<FenceEpoch>(&command) != nullptr) {
            ++fenceAttempts;
        }
        changed.notify_all();
        return Status::OK();
    };
    auto dependencies = SchedulerDependencies(submit);
    dependencies.publishCompletion = [&](std::vector<RequestCompleted>&) {
        std::lock_guard lock(mutex);
        completed = true;
        changed.notify_all();
    };
    dependencies.releaseReplySlot = [&](const RequestToken&, const ReplySlot&) {
        std::lock_guard lock(mutex);
        ++releases;
        return Status::OK();
    };
    NodeScheduler scheduler(SchedulerConfig({Endpoint(1)}, 1, 1, 1ms), std::move(dependencies));
    ASSERT_TRUE(scheduler.Start().Success());
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return connectAttempts == 1; }));
    PublishConnected(scheduler);

    auto command = MakeRequest();
    ASSERT_TRUE(scheduler.Post(command).Success());
    ASSERT_TRUE(WaitUntil(mutex, changed, [&] { return transmitAttempts == 1; }));
    auto pending = MakeRequest(2, 2);
    EXPECT_TRUE(scheduler.Post(pending).Success());

    EXPECT_TRUE(scheduler.Shutdown().Success());
    EXPECT_FALSE(completed);
    EXPECT_EQ(transmitAttempts, std::size_t{1});
    EXPECT_EQ(releases, std::size_t{0});
    EXPECT_EQ(fenceAttempts, std::size_t{0});
}

TEST(NodeSchedulerTest, LateEventsAfterShutdownAreDiscarded)
{
    NodeScheduler scheduler(
        SchedulerConfig({Endpoint(1)}, 1, 1, 1ms),
        SchedulerDependencies([](TransportCommand&) { return Status::Retry(); }));
    ASSERT_TRUE(scheduler.Start().Success());
    ASSERT_TRUE(scheduler.Shutdown().Success());

    for (std::size_t index = 0; index < 16; ++index) {
        scheduler.Publish(1, NodeEvent{
                                 ConnectCompleted{1, kDefaultLaneId, 1, Status::OK()}
        });
    }
}

TEST(NodeSchedulerTest, HundredsOfNodesShareAFewRunners)
{
    constexpr std::size_t kNodeCount = 256;
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t attempts = 0;
    auto submit = [&](TransportCommand& command) {
        if (std::get_if<Connect>(&command) != nullptr) {
            std::lock_guard lock(mutex);
            ++attempts;
            changed.notify_all();
        }
        return Status::Retry();
    };

    std::vector<NodeEndpoint> nodes;
    nodes.reserve(kNodeCount);
    for (std::size_t index = 0; index < kNodeCount; ++index) { nodes.push_back(Endpoint(index)); }
    NodeScheduler scheduler(SchedulerConfig(std::move(nodes), 4, 1, 1h),
                            SchedulerDependencies(submit));
    ASSERT_TRUE(scheduler.Start().Success());

    const bool allAttempted = WaitUntil(mutex, changed, [&] { return attempts == kNodeCount; }, 2s);

    EXPECT_TRUE(allAttempted);
    EXPECT_TRUE(scheduler.Shutdown().Success());
}

}  // namespace
}  // namespace UC::Dram
