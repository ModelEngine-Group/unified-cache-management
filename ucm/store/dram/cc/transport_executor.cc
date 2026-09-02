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
#include "transport_executor.h"
#include <exception>
#include <limits>
#include <optional>
#include <type_traits>

namespace UC::Dram {
namespace {

NodeId CommandNode(const TransportCommand& command) noexcept
{
    return std::visit(
        [](const auto& value) -> NodeId {
            using Command = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Command, Transmit>) {
                return value.token.nodeId;
            } else {
                return value.nodeId;
            }
        },
        command);
}

bool IsFence(const TransportCommand& command) noexcept
{
    return std::holds_alternative<FenceEpoch>(command);
}

}  // namespace

TransportExecutor::TransportExecutor(Options options) : options_(std::move(options)) {}

TransportExecutor::~TransportExecutor() { Shutdown(); }

void TransportExecutor::Execute(TransportCommand command) noexcept
{
    try {
        std::visit(
            [this](auto&& value) {
                using Command = std::decay_t<decltype(value)>;
                NodeId nodeId = 0;
                NodeEvent event;
                if constexpr (std::is_same_v<Command, Transmit>) {
                    nodeId = value.token.nodeId;
                    event = NodeEvent{options_.backend->Transmit(value)};
                } else if constexpr (std::is_same_v<Command, Connect>) {
                    nodeId = value.nodeId;
                    event = NodeEvent{
                        ConnectCompleted{value.nodeId, value.laneId, value.epoch,
                                         options_.backend->Connect(value)}
                    };
                } else if constexpr (std::is_same_v<Command, FenceEpoch>) {
                    nodeId = value.nodeId;
                    event = NodeEvent{
                        FenceCompleted{value.nodeId, value.laneId, value.epoch,
                                       options_.backend->Fence(value)}
                    };
                }
                options_.publishEvent(nodeId, std::move(event));
            },
            std::move(command));
    } catch (...) {
        AbortDramStore(
            Status::Error("TransportExecutor failed while publishing a command completion"));
    }
}

void TransportExecutor::Run(Worker& worker) noexcept
{
    for (;;) {
        std::optional<TransportCommand> command;
        {
            std::unique_lock lock(worker.mutex);
            worker.wake.wait(lock, [this, &worker] {
                return !worker.queue.Empty() || !acceptingCommands_.load(std::memory_order_acquire);
            });
            if (worker.queue.Empty() && !acceptingCommands_.load(std::memory_order_acquire)) {
                return;
            }
            command.emplace(worker.queue.Pop());
        }
        {
            std::lock_guard lock(admissionMutex_);
            if (IsFence(*command)) {
                --queuedFences_;
            } else {
                --queuedCommands_;
            }
        }
        Execute(std::move(*command));
    }
}

Status TransportExecutor::Start()
{
    if (options_.workerCount == 0 || options_.nodeCount == 0 ||
        options_.maxInflightRequestsPerNode == 0 ||
        options_.maxInflightRequestsPerNode > std::numeric_limits<std::size_t>::max() - 2 ||
        options_.nodeCount >
            std::numeric_limits<std::size_t>::max() / (options_.maxInflightRequestsPerNode + 2) ||
        !options_.backend || !options_.publishEvent) {
        return Status::InvalidParam("invalid TransportExecutor options");
    }
    commandQueueCapacity_ = options_.nodeCount * (options_.maxInflightRequestsPerNode + 1);
    fenceQueueCapacity_ = options_.nodeCount;
    if (acceptingCommands_.exchange(true, std::memory_order_acq_rel)) {
        return Status::DuplicateKey();
    }
    try {
        workers_.reserve(options_.workerCount);
        for (std::size_t index = 0; index < options_.workerCount; ++index) {
            workers_.push_back(
                std::make_unique<Worker>(commandQueueCapacity_ + fenceQueueCapacity_));
        }
        for (auto& worker : workers_) {
            worker->thread = std::thread([this, worker = worker.get()] { Run(*worker); });
        }
        return Status::OK();
    } catch (const std::exception& error) {
        acceptingCommands_.store(false, std::memory_order_release);
        for (auto& worker : workers_) { worker->wake.notify_all(); }
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) { worker->thread.join(); }
        }
        workers_.clear();
        return Status::Error(fmt::format("failed to start TransportExecutor: {}", error.what()));
    }
}

Status TransportExecutor::TryPost(TransportCommand& command)
{
    if (!acceptingCommands_.load(std::memory_order_acquire)) {
        return Status::Error("TransportExecutor is stopping");
    }
    const auto nodeId = CommandNode(command);
    const bool isFence = IsFence(command);
    const auto workerIndex = std::hash<NodeId>{}(nodeId) % workers_.size();
    auto& worker = *workers_[workerIndex];

    std::lock_guard admission_lock(admissionMutex_);
    if (!acceptingCommands_.load(std::memory_order_acquire)) {
        return Status::Error("TransportExecutor is stopping");
    }
    if ((isFence && queuedFences_ >= fenceQueueCapacity_) ||
        (!isFence && queuedCommands_ >= commandQueueCapacity_)) {
        return Status::Error("TransportExecutor queue full");
    }

    auto staged = std::move(command);
    {
        std::lock_guard queue_lock(worker.mutex);
        if (!worker.queue.Push(staged)) {
            command = std::move(staged);
            return Status::Error("TransportExecutor worker queue invariant violated");
        }
    }
    if (isFence) {
        ++queuedFences_;
    } else {
        ++queuedCommands_;
    }
    worker.wake.notify_one();
    return Status::OK();
}

void TransportExecutor::Shutdown()
{
    acceptingCommands_.store(false, std::memory_order_release);
    for (auto& worker : workers_) { worker->wake.notify_all(); }
    for (auto& worker : workers_) {
        if (worker->thread.joinable()) { worker->thread.join(); }
    }
}

}  // namespace UC::Dram
