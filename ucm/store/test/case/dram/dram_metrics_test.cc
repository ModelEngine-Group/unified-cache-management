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
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>
#include "metrics_api.h"
#include "node_actor.h"
#include "router/router.h"
#include "task_manager.h"
#include "time/now_time.h"

namespace UC::Dram {
namespace {

using namespace std::chrono_literals;

std::uint64_t HistogramCount(const Metrics::HistogramStat& histogram)
{
    return std::accumulate(histogram.bucketCounts.begin(), histogram.bucketCounts.end(),
                           std::uint64_t{0});
}

class SingleRouter final : public Router::Router {
public:
    SingleRouter() : Router([](const std::string&) { return std::uint64_t{0}; }) {}

private:
    UC::Router::NodeId RouteKey(const UC::Router::CacheKey&) const override { return NodeId{1}; }
};

class UCDramMetricsTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        Metrics::SetUp();
        for (const auto* prefix : {"dramstore_dump_", "dramstore_lookup_", "dramstore_load_"}) {
            for (const auto* suffix :
                 {"tasks_submitted_total", "tasks_rejected_total", "tasks_succeeded_total",
                  "tasks_failed_total", "requests_completed_total", "requests_failed_total",
                  "task_timeouts_total", "request_timeouts_total"}) {
                Metrics::CreateStats(std::string(prefix) + suffix, "counter");
            }
            for (const auto* suffix :
                 {"duration_ms", "task_queue_duration_ms", "task_to_request_duration_ms",
                  "request_duration_ms", "request_queue_duration_ms", "request_pending_duration_ms",
                  "request_setup_duration_ms", "request_transport_queue_duration_ms",
                  "request_transport_send_duration_ms", "request_remote_duration_ms"}) {
                Metrics::CreateStats(std::string(prefix) + suffix, "histogram",
                                     {0.1, 1, 100, 5000});
            }
        }
        Metrics::CreateStats("dramstore_stale_replies_total", "counter");
        Metrics::CreateStats("dramstore_reply_slot_nospace_total", "counter");
        for (const auto* name : {"dramstore_task_queue_size", "dramstore_task_queue_capacity",
                                 "dramstore_completion_queue_size",
                                 "dramstore_completion_queue_capacity", "dramstore_tasks_active",
                                 "dramstore_io_entries_used", "dramstore_io_entries_capacity"}) {
            Metrics::CreateStats(name, "gauge");
        }
    }

    void SetUp() override { Metrics::GetAllStatsAndClear(); }
};

TEST_F(UCDramMetricsTest, NodeActorRecordsCompletedFailedAndStaleRequest)
{
    std::array<std::uint8_t, 64> reply{};
    std::vector<RequestCompleted> completed;
    RequestToken token;
    NodeDependencies dependencies{
        [&](std::vector<RequestCompleted>& events) {
            for (auto& event : events) { completed.push_back(std::move(event)); }
            events.clear();
        },
        [&](TransportCommand& command) {
            if (auto* transmit = std::get_if<Transmit>(&command)) { token = transmit->token; }
            return Status::OK();
        },
        [&](const RequestToken&, OpType, std::size_t) -> Expected<ReplySlot> {
            return ReplySlot{reply.data(), reply.data(), reply.size(), 0};
        },
        [](const RequestToken&, const ReplySlot&) { return Status::OK(); },
    };
    NodeActor actor(
        {
            {1, "127.0.0.1", 12345, "127.0.0.1:23456"},
            {4, 8},
            1ms
    },
        std::move(dependencies));
    const auto now = std::chrono::steady_clock::now();
    actor.Advance(now);
    actor.Handle(
        NodeEvent{
            ConnectCompleted{1, kDefaultLaneId, 1, Status::OK()}
    },
        now);

    Request request;
    request.taskId = 1;
    request.requestId = 1;
    request.nodeId = 1;
    request.op = OpType::DUMP;
    request.deadline = now + 1h;
    request.metricsStarted = NowTime::Now();
    for (std::uint8_t index = 0; index < 2; ++index) {
        IoEntry entry;
        entry.blockId[0] = static_cast<std::byte>(index + 1);
        entry.buffer = {0x1000, 64};
        request.entries.push_back(entry);
    }

    actor.Handle(std::move(request), now);
    actor.Advance(now);
    actor.Handle(
        NodeEvent{
            TransmitCompleted{token, Status::OK(), NowTime::Now()}
    },
        now);
    actor.Handle(
        NodeEvent{
            ReplyObserved{token, Status::OK(), {{0, true, 0}, {1, false, 1}}}
    },
        now);
    actor.Advance(now);
    actor.Handle(
        NodeEvent{
            ReplyObserved{token, Status::OK(), {{0, true, 0}, {1, false, 1}}}
    },
        now);
    actor.Advance(now);

    ASSERT_EQ(completed.size(), std::size_t{1});
    EXPECT_TRUE(completed[0].status.Failure());
    auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(counters.at("dramstore_dump_requests_completed_total"), 1);
    EXPECT_EQ(counters.at("dramstore_dump_requests_failed_total"), 1);
    EXPECT_EQ(counters.count("dramstore_dump_request_timeouts_total"), 0);
    EXPECT_EQ(counters.at("dramstore_stale_replies_total"), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_dump_request_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_dump_request_queue_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_dump_request_pending_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_dump_request_setup_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_dump_request_remote_duration_ms")), 1);
    // Scheduler timestamps may precede actual dispatch; metrics use their own clock samples.
    for (const auto& [name, histogram] : histograms) { EXPECT_GE(histogram.sum, 0.0) << name; }
}

TEST_F(UCDramMetricsTest, ReplySlotNoSpaceExcludesOtherAcquisitionFailures)
{
    const std::array<Status, 3> errors = {
        Status(Status::NoSpace().Underlying(), "dram_reply_slots: no free slots"),
        Status::InvalidParam("invalid reply lease request"),
        Status::Error("ReplyService is stopping")};
    std::size_t completed = 0;
    for (const auto& error : errors) {
        NodeDependencies dependencies;
        dependencies.submitTransport = [](TransportCommand& command) {
            EXPECT_TRUE(std::holds_alternative<Connect>(command));
            return Status::OK();
        };
        dependencies.acquireReplySlot = [&](const RequestToken&, OpType,
                                            std::size_t) -> Expected<ReplySlot> { return error; };
        dependencies.releaseReplySlot = [](const RequestToken&, const ReplySlot&) {
            ADD_FAILURE() << "Failed acquisition must not release a reply slot";
            return Status::OK();
        };
        dependencies.publishCompletion = [&](std::vector<RequestCompleted>& events) {
            for (const auto& event : events) {
                EXPECT_EQ(event.status, error);
                ++completed;
            }
        };
        NodeActor actor(
            {
                {1, "127.0.0.1", 12345, "127.0.0.1:23456"},
                {4, 8},
                1ms
        },
            std::move(dependencies));
        const auto now = std::chrono::steady_clock::now();
        actor.Advance(now);
        actor.Handle(
            NodeEvent{
                ConnectCompleted{1, kDefaultLaneId, 1, Status::OK()}
        },
            now);
        Request request;
        request.taskId = 1;
        request.requestId = 1;
        request.nodeId = 1;
        request.op = OpType::DUMP;
        request.entries.emplace_back();
        request.deadline = now + 1h;
        request.metricsStarted = NowTime::Now();
        actor.Handle(std::move(request), now);
        actor.Advance(now);
        actor.Advance(now);
    }
    EXPECT_EQ(completed, errors.size());
    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("dramstore_reply_slot_nospace_total"), 1);
    EXPECT_EQ(counters.at("dramstore_dump_requests_failed_total"), 3);
    const auto& histograms = std::get<2>(stats);
    const auto setup = histograms.find("dramstore_dump_request_setup_duration_ms");
    if (setup != histograms.end()) { EXPECT_EQ(HistogramCount(setup->second), 0); }
}

TEST_F(UCDramMetricsTest, RequestTimeoutsCountOnceAtAdmissionAndWhilePending)
{
    for (const auto op : {OpType::LOOKUP, OpType::DUMP, OpType::LOAD}) {
        std::size_t completed = 0;
        NodeDependencies dependencies;
        dependencies.submitTransport = [](TransportCommand&) { return Status::Retry(); };
        dependencies.publishCompletion = [&](std::vector<RequestCompleted>& events) {
            for (const auto& event : events) {
                EXPECT_EQ(event.status, Status::Timeout());
                ++completed;
            }
        };
        NodeActor actor(
            {
                {1, "127.0.0.1", 12345, "127.0.0.1:23456"},
                {4, 8},
                1ms
        },
            std::move(dependencies));
        const auto now = std::chrono::steady_clock::now();
        for (RequestId id = 1; id <= 2; ++id) {
            Request request;
            request.taskId = id;
            request.requestId = id;
            request.nodeId = 1;
            request.op = op;
            request.metricsStarted = NowTime::Now();
            request.deadline = id == 1 ? now : now + 1ms;
            actor.Handle(std::move(request), now);
        }
        actor.Advance(now);
        actor.Advance(now + 2ms);
        actor.Advance(now + 3ms);
        EXPECT_EQ(completed, 2);
    }
    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    for (const auto* prefix : {"dramstore_lookup_", "dramstore_dump_", "dramstore_load_"}) {
        EXPECT_EQ(counters.at(std::string(prefix) + "request_timeouts_total"), 2);
        EXPECT_EQ(counters.at(std::string(prefix) + "requests_failed_total"), 2);
        EXPECT_EQ(counters.at(std::string(prefix) + "requests_completed_total"), 2);
    }
}

TEST_F(UCDramMetricsTest, TaskManagerSettlesAcceptedTaskOnceAndRecordsRejectedSubmission)
{
    TaskManager* managerPointer = nullptr;
    TaskManagerConfig config{
        {64},
        16, 8, {1s, 1s, 1s}
    };
    TaskManager manager(
        config, {std::make_shared<SingleRouter>(), [&](Request& request) {
                     std::vector<RequestCompleted> events{
                         {request.taskId, request.requestId, 1, Status::OK(), {{0, true, 0}}}
                     };
                     managerPointer->Publish(events);
                     return Status::OK();
                 }});
    managerPointer = &manager;
    ASSERT_TRUE(manager.Start().Success());

    Detail::BlockId key{};
    key[0] = std::byte{1};
    auto submitted = manager.SubmitLookup(&key, 1);
    ASSERT_TRUE(submitted);
    const auto taskId = submitted.Value();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (true) {
        auto checked = manager.Check(taskId);
        ASSERT_TRUE(checked);
        if (checked.Value()) { break; }
        ASSERT_LT(std::chrono::steady_clock::now(), deadline);
        std::this_thread::yield();
    }
    for (int index = 0; index < 10; ++index) { EXPECT_TRUE(manager.Check(taskId).Value()); }
    EXPECT_TRUE(manager.WaitLookup(taskId));
    manager.Shutdown();
    EXPECT_FALSE(manager.SubmitLookup(&key, 1));

    auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(counters.at("dramstore_lookup_tasks_submitted_total"), 1);
    EXPECT_EQ(counters.at("dramstore_lookup_tasks_succeeded_total"), 1);
    EXPECT_EQ(counters.at("dramstore_lookup_tasks_rejected_total"), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_lookup_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("dramstore_lookup_task_to_request_duration_ms")), 1);
}

TEST_F(UCDramMetricsTest, TaskCapacityGaugesPublishInitialSnapshot)
{
    TaskManagerConfig config{
        {64},
        16, 8, {10s, 10s, 10s}
    };
    TaskManager manager(config,
                        {std::make_shared<SingleRouter>(), [](Request&) { return Status::OK(); }});
    ASSERT_TRUE(manager.Start().Success());

    const auto deadline = std::chrono::steady_clock::now() + 2s;
    std::unordered_map<std::string, double> gauges;
    while (gauges.size() != 7 && std::chrono::steady_clock::now() < deadline) {
        const auto stats = Metrics::GetAllStatsAndClear();
        for (const auto& [name, value] : std::get<1>(stats)) { gauges[name] = value; }
        std::this_thread::yield();
    }
    manager.Shutdown();

    ASSERT_EQ(gauges.size(), 7);
    EXPECT_EQ(gauges.at("dramstore_tasks_active"), 0);
    EXPECT_EQ(gauges.at("dramstore_io_entries_used"), 0);
    EXPECT_EQ(gauges.at("dramstore_io_entries_capacity"), 16);
    EXPECT_EQ(gauges.at("dramstore_task_queue_size"), 0);
    EXPECT_EQ(gauges.at("dramstore_task_queue_capacity"), 16);
    EXPECT_EQ(gauges.at("dramstore_completion_queue_size"), 0);
    EXPECT_EQ(gauges.at("dramstore_completion_queue_capacity"), 16);
}

}  // namespace
}  // namespace UC::Dram
