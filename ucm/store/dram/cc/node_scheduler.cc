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
#include "node_scheduler.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include "node_actor.h"

namespace UC::Dram {
namespace {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
}  // namespace

struct NodeScheduler::Runner {
    using EventMessage = std::pair<NodeId, NodeEvent>;

    explicit Runner(std::size_t nodeCount) { actors.reserve(nodeCount); }

    std::deque<Request> commands;
    std::deque<EventMessage> events;
    std::deque<Request> commandBatch;
    std::deque<EventMessage> eventBatch;
    std::unordered_map<NodeId, std::unique_ptr<NodeActor>> actors;
    std::mutex mutex;
    std::condition_variable wake;
    std::thread thread;
};

NodeScheduler::NodeScheduler(NodeSchedulerConfig config, NodeDependencies dependencies)
    : config_(std::move(config)), dependencies_(std::move(dependencies))
{
}

NodeScheduler::~NodeScheduler() { (void)Shutdown(); }

NodeScheduler::Runner& NodeScheduler::GetRunner(NodeId nodeId) const noexcept
{
    return *nodes_.find(nodeId)->second;
}

Status NodeScheduler::Start()
{
    if (!nodes_.empty() || !runners_.empty()) { return Status::DuplicateKey(); }

    std::promise<bool> launch;
    auto launchGate = launch.get_future().share();
    try {
        const auto& config = config_;
        const auto nodeCount = config.nodes.size();
        const auto runnerCount = std::min(config.runnerCount, nodeCount);
        nodes_.reserve(nodeCount);
        runners_.reserve(runnerCount);
        for (std::size_t index = 0; index < runnerCount; ++index) {
            const auto nodesPerRunner = nodeCount / runnerCount + (index < nodeCount % runnerCount);
            runners_.push_back(std::make_unique<Runner>(nodesPerRunner));
        }

        std::size_t index = 0;
        for (const auto& endpoint : config.nodes) {
            auto* runner = runners_[index % runnerCount].get();
            NodeActor::Config actorConfig{endpoint, config.limits, config.reconnectInterval};
            auto actor = std::make_unique<NodeActor>(std::move(actorConfig), dependencies_);
            runner->actors.emplace(endpoint.nodeId, std::move(actor));
            nodes_.emplace(endpoint.nodeId, runner);
            ++index;
        }

        for (auto& runner : runners_) {
            runner->thread = std::thread([this, runner = runner.get(), launchGate] {
                if (!launchGate.get()) { return; }
                RunActors(*runner);
            });
        }
    } catch (...) {
        launch.set_value(false);
        JoinAll();
        return Status::Error("failed to start NodeScheduler");
    }

    acceptingMessages_.store(true, std::memory_order_release);
    launch.set_value(true);
    return Status::OK();
}

Status NodeScheduler::Post(Request& request)
{
    if (!acceptingMessages_.load(std::memory_order_acquire)) {
        return Status::Error("NodeScheduler is not accepting commands");
    }
    auto& runner = GetRunner(request.nodeId);
    {
        std::lock_guard lock(runner.mutex);
        runner.commands.push_back(std::move(request));
    }
    runner.wake.notify_one();
    return Status::OK();
}

void NodeScheduler::Publish(NodeId nodeId, NodeEvent event)
{
    if (!acceptingMessages_.load(std::memory_order_acquire)) { return; }
    auto& runner = GetRunner(nodeId);
    {
        std::lock_guard lock(runner.mutex);
        runner.events.emplace_back(nodeId, std::move(event));
    }
    runner.wake.notify_one();
}

void NodeScheduler::RunActors(Runner& runner) noexcept
{
    try {
        auto nextWakeup = TimePoint::min();
        for (;;) {
            {
                std::unique_lock lock(runner.mutex);
                const auto ready = [this, &runner] {
                    return !acceptingMessages_.load(std::memory_order_acquire) ||
                           !runner.events.empty() || !runner.commands.empty();
                };
                if (nextWakeup == TimePoint::max()) {
                    runner.wake.wait(lock, ready);
                } else {
                    runner.wake.wait_until(lock, nextWakeup, ready);
                }
                if (!acceptingMessages_.load(std::memory_order_acquire)) { break; }

                runner.eventBatch.swap(runner.events);
                runner.commandBatch.swap(runner.commands);
            }

            const auto now = Clock::now();
            for (auto& message : runner.eventBatch) {
                runner.actors.find(message.first)->second->Handle(std::move(message.second), now);
            }
            for (auto& request : runner.commandBatch) {
                runner.actors.find(request.nodeId)->second->Handle(std::move(request), now);
            }
            runner.eventBatch.clear();
            runner.commandBatch.clear();

            nextWakeup = TimePoint::max();
            for (auto& [_, actor] : runner.actors) {
                actor->Advance(now);
                nextWakeup = std::min(nextWakeup, actor->NextWakeup());
            }
        }
    } catch (...) {
        AbortDramStore(Status::Error("NodeScheduler runner stopped unexpectedly"));
    }
}

void NodeScheduler::JoinAll()
{
    for (auto& runner : runners_) {
        if (runner->thread.joinable()) { runner->thread.join(); }
    }
}

Status NodeScheduler::Shutdown()
{
    acceptingMessages_.store(false, std::memory_order_release);
    for (auto& runner : runners_) {
        std::lock_guard lock(runner->mutex);
        runner->wake.notify_all();
    }
    JoinAll();
    return Status::OK();
}

}  // namespace UC::Dram
