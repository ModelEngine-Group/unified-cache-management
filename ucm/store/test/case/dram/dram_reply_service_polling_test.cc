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
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>
#include "reply_service.h"

namespace UC::Dram {
namespace {

Status PackReply(const ReplySlot& slot, DramPool::KvOpcode opcode,
                 std::vector<std::uint8_t> results)
{
    DramPool::ProtocolManager protocol;
    return protocol.PackResponse(slot.local_addr, opcode, DramPool::KvResponse{std::move(results)});
}

class EventCollector final {
public:
    void Publish(NodeId node_id, NodeEvent event)
    {
        std::lock_guard lock(mutex_);
        node_id_ = node_id;
        event_ = std::move(event);
        changed_.notify_all();
    }

    std::optional<NodeEvent> Wait(std::chrono::milliseconds timeout)
    {
        std::unique_lock lock(mutex_);
        if (!changed_.wait_for(lock, timeout, [this] { return event_.has_value(); })) {
            return std::nullopt;
        }
        return std::move(event_);
    }

    NodeId node_id() const
    {
        std::lock_guard lock(mutex_);
        return node_id_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    NodeId node_id_{0};
    std::optional<NodeEvent> event_;
};

TEST(ReplyServicePollingTest, PublishesObservedReplyAndKeepsLeaseUntilActorRelease)
{
    EventCollector collector;
    auto created = ReplyService::Create(ReplyService::Options{
        64,
        1,
        std::chrono::microseconds{50},
        [&collector](NodeId node, NodeEvent event) { collector.Publish(node, std::move(event)); },
    });
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());

    const RequestToken token{7, kDefaultLaneId, 9, 42};
    auto acquired = service->Acquire(token, OpType::LOOKUP, 2);
    ASSERT_TRUE(acquired);
    auto slot = acquired.Value();
    ASSERT_TRUE(PackReply(slot, DramPool::KvOpcode::Lookup, {1, 0}).Success());

    auto event = collector.Wait(std::chrono::milliseconds{500});
    ASSERT_TRUE(event.has_value());
    ASSERT_TRUE(std::holds_alternative<ReplyObserved>(*event));
    auto observed = std::get<ReplyObserved>(std::move(*event));
    EXPECT_EQ(collector.node_id(), token.nodeId);
    EXPECT_TRUE(observed.token == token);
    EXPECT_TRUE(observed.status.Success());
    EXPECT_EQ(observed.entryResults.size(), std::size_t{2});
    EXPECT_TRUE(observed.entryResults[0].found);
    EXPECT_FALSE(observed.entryResults[1].found);

    // Observation does not free the slot. NodeActor releases it only after consuming
    // this event, preserving the safe-terminal/fence rule without a generation field.
    EXPECT_EQ(service->Available(), std::size_t{0});
    EXPECT_TRUE(service->Release(token, slot).Success());
    EXPECT_EQ(service->Available(), std::size_t{1});
    service->Shutdown();
}

TEST(ReplyServicePollingTest, ReusesSlotWhilePreviousReplyEventIsBeingPublished)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool firstPublishEntered = false;
    bool allowFirstPublish = false;
    std::vector<RequestToken> published;
    const RequestToken firstToken{7, kDefaultLaneId, 9, 42};
    const RequestToken secondToken{7, kDefaultLaneId, 9, 43};

    auto created = ReplyService::Create(ReplyService::Options{
        64,
        1,
        std::chrono::microseconds{50},
        [&](NodeId, NodeEvent event) {
            const auto& observed = std::get<ReplyObserved>(event);
            std::unique_lock lock(mutex);
            published.push_back(observed.token);
            if (observed.token == firstToken) {
                firstPublishEntered = true;
                changed.notify_all();
                changed.wait(lock, [&] { return allowFirstPublish; });
            }
            changed.notify_all();
        },
    });
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());

    auto first = service->Acquire(firstToken, OpType::LOOKUP, 1);
    ASSERT_TRUE(first);
    ASSERT_TRUE(PackReply(first.Value(), DramPool::KvOpcode::Lookup, {1}).Success());
    bool publishEntered = false;
    {
        std::unique_lock lock(mutex);
        publishEntered = changed.wait_for(lock, std::chrono::milliseconds{500},
                                          [&] { return firstPublishEntered; });
    }

    auto firstRelease = Status::Error();
    Expected<ReplySlot> second{Status::Error()};
    auto secondPack = Status::Error();
    if (publishEntered) {
        firstRelease = service->Release(firstToken, first.Value());
        second = service->Acquire(secondToken, OpType::LOOKUP, 1);
        if (second) { secondPack = PackReply(second.Value(), DramPool::KvOpcode::Lookup, {1}); }
    }

    {
        std::lock_guard lock(mutex);
        allowFirstPublish = true;
    }
    changed.notify_all();

    bool secondPublished = false;
    if (secondPack.Success()) {
        std::unique_lock lock(mutex);
        secondPublished = changed.wait_for(lock, std::chrono::milliseconds{500}, [&] {
            return std::find(published.begin(), published.end(), secondToken) != published.end();
        });
    }

    Status cleanup = Status::OK();
    if (second) {
        cleanup = service->Release(secondToken, second.Value());
    } else if (firstRelease.Failure()) {
        cleanup = service->Release(firstToken, first.Value());
    }
    service->Shutdown();

    EXPECT_TRUE(publishEntered);
    EXPECT_TRUE(firstRelease.Success());
    ASSERT_TRUE(second);
    EXPECT_EQ(second.Value().slot_index, first.Value().slot_index);
    EXPECT_TRUE(secondPack.Success());
    EXPECT_TRUE(secondPublished);
    EXPECT_TRUE(cleanup.Success());
}

TEST(ReplyServicePollingTest, AllowsShutdownWithAbandonedLease)
{
    auto created = ReplyService::Create(ReplyService::Options{
        64,
        1,
        std::chrono::microseconds{50},
        [](NodeId, NodeEvent) {},
    });
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());
    auto acquired = service->Acquire(RequestToken{1, kDefaultLaneId, 1, 1}, OpType::LOOKUP, 1);
    ASSERT_TRUE(acquired);
    service->Shutdown();
}

TEST(ReplyServicePollingTest, EventPublisherExceptionIsFatal)
{
    EXPECT_DEATH(
        {
            std::promise<void> publishAttempted;
            auto publishAttemptedFuture = publishAttempted.get_future();
            auto created = ReplyService::Create(ReplyService::Options{
                64,
                1,
                std::chrono::microseconds{50},
                [&publishAttempted](NodeId, NodeEvent) {
                    publishAttempted.set_value();
                    throw std::runtime_error("event receiver failed");
                },
            });
            if (created) {
                auto service = std::move(created).Value();
                auto acquired =
                    service->Start().Success()
                        ? service->Acquire(RequestToken{3, kDefaultLaneId, 1, 9}, OpType::LOOKUP, 1)
                        : Expected<ReplySlot>{Status::Error()};
                if (acquired &&
                    PackReply(acquired.Value(), DramPool::KvOpcode::Lookup, {1}).Success()) {
                    publishAttemptedFuture.wait();
                }
                service->Shutdown();
            }
        },
        "");
}

}  // namespace
}  // namespace UC::Dram
