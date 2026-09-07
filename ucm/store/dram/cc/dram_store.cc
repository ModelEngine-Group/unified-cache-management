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
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "config.h"
#include "logger/logger.h"
#include "node_scheduler.h"
#include "reply_service.h"
#include "router/router.h"
#include "task_manager.h"
#include "trans/device.h"
#include "transport_executor.h"
#include "transport_manager_backend.h"
#include "ucmstore_v1.h"

namespace UC::Dram {

// Synchronize an optional compute-stream event before a dump task is submitted.
// DramStore owns no NPU stream; the real D2H is an RDMA Read executed on the
// remote DramPool directly against this client's device memory.
// RDMA and the compute stream are unordered, so the only safe point is to
// block on the prerequisite event here, before the control message is sent.
static Status WaitPrerequisiteEvent(std::uintptr_t eventHandle)
{
    return Trans::Event{eventHandle}.Synchronize();
}

class DramStore final : public StoreV1 {
public:
    DramStore() = default;
    ~DramStore() override { StopGraph(); }

    DramStore(const DramStore&) = delete;
    DramStore& operator=(const DramStore&) = delete;

    Status Setup(const Detail::Dictionary& config) override;
    std::string Readme() const override;
    Expected<std::vector<std::uint8_t>> Lookup(const Detail::BlockId* blocks,
                                               std::size_t num) override;
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, std::size_t num) override;
    void Prefetch(const Detail::BlockId* blocks, std::size_t num) override;
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override;
    Expected<bool> Check(Detail::TaskHandle taskId) override;
    Status Wait(Detail::TaskHandle taskId) override;

private:
    Status RegisterKVCaches(const KVCacheRegistration* registrations, std::size_t count);
    Expected<Detail::TaskHandle> SubmitTransfer(OpType op, Detail::TaskDesc task);
    Status Start();

    Status SetupParsed(DramConfig parsed)
    {
        config_ = std::make_unique<DramConfig>(std::move(parsed));
        auto status = Compose();

        if (status.Success() && config_->role == Role::WORKER) {
            std::vector<KVCacheRegistration> registrations;
            registrations.reserve(config_->gpuKvBufferAddrs.size());
            for (std::size_t index = 0; index < config_->gpuKvBufferAddrs.size(); ++index) {
                registrations.push_back(
                    {config_->gpuKvBufferAddrs[index], config_->gpuKvBufferSizes[index]});
            }
            status = RegisterKVCaches(registrations.data(), registrations.size());
        }
        if (status.Success()) { status = Start(); }

        if (status.Failure()) {
            StopGraph();
            return status;
        }
        return Status::OK();
    }

    Status Compose()
    {
        UC::Trans::Device device;
        const auto initStatus = device.Init();
        if (initStatus.Failure() && initStatus != Status::DuplicateKey()) {
            return Status::Error("aclInit failed: " + initStatus.ToString());
        }
        const auto setupStatus = device.Setup(config_->deviceId);
        if (setupStatus.Failure()) {
            return Status::Error("aclrtSetDevice failed: " + setupStatus.ToString());
        }
#ifdef UC_DRAM_ASCEND_BACKEND
        // load and dump are timed separately per-task in TaskManager. The transport backend
        // enforces a single deadline
        const auto transferTimeout =
            std::max(config_->taskTimeouts.load, config_->taskTimeouts.dump);
        TransportManagerBackendOptions backendOpts{
            config_->localControlHost,
            config_->localControlPort,
            config_->localTransportManagerId,
            config_->localHost,
            config_->deviceId,
            config_->hixlListenPort,
            config_->enableHixlCs,
            1000,
            static_cast<std::int32_t>(std::min<std::int64_t>(
                transferTimeout.count(), std::numeric_limits<std::int32_t>::max())),
            config_->nodeScheduler.nodes};
        auto createdBackend = CreateTransportManagerBackend(std::move(backendOpts));
        if (!createdBackend) { return createdBackend.Error(); }
        transportBackend_ = std::move(createdBackend).Value();

        auto createdReplies = ReplyService::Create(ReplyService::Options{
            config_->deviceId, config_->replySlotSize, config_->replySlotCount,
            std::chrono::microseconds{50}, [this](NodeId nodeId, NodeEvent event) {
                nodeScheduler_->Publish(nodeId, std::move(event));
            }});
        if (!createdReplies) { return createdReplies.Error(); }
        replyService_ = std::move(createdReplies).Value();

        memoryHandles_.reserve(1);
        const auto replyMemory = replyService_->MemoryRegion();
        auto registeredReply = transportBackend_->RegisterMemory(
            replyMemory.address, replyMemory.length, MemoryRegionType::DEVICE);
        if (!registeredReply) { return registeredReply.Error(); }
        memoryHandles_.push_back(std::move(registeredReply).Value());

        std::vector<UC::Router::NodeId> nodeIds;
        nodeIds.reserve(config_->nodeScheduler.nodes.size());
        for (const auto& node : config_->nodeScheduler.nodes) { nodeIds.push_back(node.nodeId); }
        UC::Router::RouterConfig routerConfig;
        routerConfig.type = config_->routerType;
        router_ = UC::Router::CreateRouter(nodeIds, {}, routerConfig);
        if (!router_) { return Status::Error("failed to create DramStore router"); }

        transport_ = std::make_unique<TransportExecutor>(TransportExecutor::Options{
            config_->transportRuntime.workerCount, config_->nodeScheduler.nodes.size(),
            config_->nodeScheduler.limits.maxInflightRequests, transportBackend_,
            [this](NodeId nodeId, NodeEvent event) {
                nodeScheduler_->Publish(nodeId, std::move(event));
            }});

        nodeScheduler_ = std::make_unique<NodeScheduler>(
            config_->nodeScheduler,
            NodeDependencies{
                [this](std::vector<RequestCompleted>& events) { taskManager_->Publish(events); },
                [this](TransportCommand& command) {
                    return transport_ ? transport_->TryPost(command)
                                      : Status::Error("TransportExecutor is unavailable");
                },
                [this](const RequestToken& token, OpType op,
                       std::size_t entryCount) -> Expected<ReplySlot> {
                    return replyService_
                               ? replyService_->Acquire(token, op, entryCount)
                               : Expected<ReplySlot>{Status::Error("ReplyService is unavailable")};
                },
                [this](const RequestToken& token, const ReplySlot& slot) {
                    return replyService_ ? replyService_->Release(token, slot)
                                         : Status::Error("ReplyService is unavailable");
                }});

        taskManager_ = std::make_unique<TaskManager>(
            TaskManagerConfig{config_->tensorSizes, config_->maxIoEntries,
                              config_->nodeScheduler.limits.maxBatchEntries, config_->taskTimeouts},
            TaskManagerDependencies{router_, [this](Request& request) {
                                        return nodeScheduler_
                                                   ? nodeScheduler_->Post(request)
                                                   : Status::Error("NodeScheduler is unavailable");
                                    }});
        return Status::OK();
#else
        return Status::Unsupported();
#endif
    }

    void StopGraph()
    {
        if (!config_) { return; }

        UC_INFO("DramStore cleaning up");

        // Shutdown is terminal application teardown. TaskManager first stops all
        // admission and drops late completions while its callback target remains alive.
        if (taskManager_) { taskManager_->Shutdown(); }
        if (nodeScheduler_) { nodeScheduler_->Shutdown(); }
        if (replyService_) { replyService_->Shutdown(); }
        if (transport_) { transport_->Shutdown(); }
        if (transportBackend_) { transportBackend_->Stop(); }

        memoryHandles_.clear();
        transport_.reset();
        replyService_.reset();
        nodeScheduler_.reset();
        taskManager_.reset();
        transportBackend_.reset();
        router_.reset();
        config_.reset();
    }

    std::unique_ptr<DramConfig> config_;
    std::shared_ptr<ITransportBackend> transportBackend_;
    std::vector<MemoryHandle> memoryHandles_;
    std::shared_ptr<UC::Router::Router> router_;
    std::unique_ptr<TransportExecutor> transport_;
    std::unique_ptr<ReplyService> replyService_;
    std::unique_ptr<NodeScheduler> nodeScheduler_;
    std::unique_ptr<TaskManager> taskManager_;
};

Status DramStore::Setup(const Detail::Dictionary& config)
{
    auto parsed = DramConfig::Parse(config);
    if (!parsed) {
        UC_ERROR("DramStore setup failed while parsing configuration: {}", parsed.Error());
        return parsed.Error();
    }
    auto status = SetupParsed(std::move(parsed).Value());
    if (status.Failure()) {
        UC_ERROR("DramStore setup failed: {}", status);
        return status;
    }
    UC_INFO(
        "DramStore setup succeeded, role={} device_id={} nodes={} max_io_entries={} "
        "max_batch_entries={} max_inflight_per_node={} reply_slots={}",
        config_->role == Role::SCHEDULER ? "scheduler" : "worker", config_->deviceId,
        config_->nodeScheduler.nodes.size(), config_->maxIoEntries,
        config_->nodeScheduler.limits.maxBatchEntries,
        config_->nodeScheduler.limits.maxInflightRequests, config_->replySlotCount);
    return Status::OK();
}

Status DramStore::Start()
{
    auto status = transport_->Start();
    if (status.Failure()) {
        UC_ERROR("DramStore start failed, stage=TransportExecutor status={}", status);
        return status;
    }
    status = taskManager_->Start();
    if (status.Failure()) {
        UC_ERROR("DramStore start failed, stage=TaskManager status={}", status);
        return status;
    }
    status = replyService_->Start();
    if (status.Failure()) {
        UC_ERROR("DramStore start failed, stage=ReplyService status={}", status);
        return status;
    }
    status = nodeScheduler_->Start();
    if (status.Failure()) {
        UC_ERROR("DramStore start failed, stage=NodeScheduler status={}", status);
        return status;
    }
    UC_INFO("DramStore started, role={} nodes={}",
            config_->role == Role::SCHEDULER ? "scheduler" : "worker",
            config_->nodeScheduler.nodes.size());
    return Status::OK();
}

std::string DramStore::Readme() const
{
    return "DramStore: serialized task coordination and per-node remote-access safety";
}

Expected<std::vector<std::uint8_t>> DramStore::Lookup(const Detail::BlockId* blocks,
                                                      std::size_t num)
{
    if (num == 0) { return std::vector<std::uint8_t>{}; }
    if (blocks == nullptr || num > config_->maxIoEntries) {
        return Status::InvalidParam("invalid lookup input");
    }
    auto submitted = taskManager_->SubmitLookup(blocks, num);
    if (!submitted) { return submitted.Error(); }
    return taskManager_->WaitLookup(std::move(submitted).Value());
}

Expected<ssize_t> DramStore::LookupOnPrefix(const Detail::BlockId* blocks, std::size_t num)
{
    auto looked = Lookup(blocks, num);
    if (!looked) { return looked.Error(); }
    const auto& founds = looked.Value();
    for (std::size_t i = 0; i < founds.size(); ++i) {
        if (founds[i] == 0) { return static_cast<ssize_t>(i) - 1; }
    }
    return static_cast<ssize_t>(num) - 1;
}

void DramStore::Prefetch(const Detail::BlockId*, std::size_t) {}

Expected<Detail::TaskHandle> DramStore::Load(Detail::TaskDesc task)
{
    return SubmitTransfer(OpType::LOAD, std::move(task));
}

Expected<Detail::TaskHandle> DramStore::Dump(Detail::TaskDesc task)
{
    auto status = WaitPrerequisiteEvent(task.prerequisiteHandle);
    if (status.Failure()) {
        UC_ERROR("DramStore dump prerequisite wait failed, prerequisite_handle={} status={}",
                 task.prerequisiteHandle, status);
        return status;
    }
    task.prerequisiteHandle = 0;
    return SubmitTransfer(OpType::DUMP, std::move(task));
}

Expected<Detail::TaskHandle> DramStore::SubmitTransfer(OpType op, Detail::TaskDesc task)
{
    if (task.empty()) { return Status::InvalidParam("invalid transfer task"); }

    return taskManager_->SubmitTransfer(op, std::move(task));
}

Expected<bool> DramStore::Check(Detail::TaskHandle taskId) { return taskManager_->Check(taskId); }

Status DramStore::Wait(Detail::TaskHandle taskId) { return taskManager_->WaitTransfer(taskId); }

Status DramStore::RegisterKVCaches(const KVCacheRegistration* registrations, std::size_t count)
{
    if (count != 0 && registrations == nullptr) {
        return Status::InvalidParam("KV cache registrations are null");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (registrations[index].addr == 0 || registrations[index].size == 0) {
            return Status::InvalidParam("KV cache registration has an invalid memory range");
        }
    }

    const auto firstHandle = memoryHandles_.size();
    if (count > memoryHandles_.max_size() - firstHandle) {
        return Status::InvalidParam("too many KV cache registrations");
    }
    try {
        memoryHandles_.reserve(firstHandle + count);
    } catch (...) {
        return Status::InvalidParam("too many KV cache registrations");
    }

    for (std::size_t index = 0; index < count; ++index) {
        auto registered =
            transportBackend_->RegisterMemory(reinterpret_cast<void*>(registrations[index].addr),
                                              registrations[index].size, MemoryRegionType::DEVICE);
        if (registered) {
            memoryHandles_.push_back(std::move(registered).Value());
            continue;
        }

        auto result = registered.Error();
        while (memoryHandles_.size() > firstHandle) {
            const auto cleanup = transportBackend_->UnregisterMemory(memoryHandles_.back());
            if (cleanup.Failure()) {
                result = cleanup;
                break;
            }
            memoryHandles_.pop_back();
        }
        UC_ERROR("DramStore KV cache registration failed, failed_index={} count={} status={}",
                 index, count, result);
        return result;
    }

    UC_INFO("DramStore KV caches registered, count={}", count);
    return Status::OK();
}

}  // namespace UC::Dram

extern "C" UC::StoreV1* MakeDramStore() { return new UC::Dram::DramStore(); }
