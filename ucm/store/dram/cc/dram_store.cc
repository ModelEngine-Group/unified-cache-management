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
#include "kv_common/router.h"
#include "logger/logger.h"
#include "node_scheduler.h"
#include "reply_service.h"
#include "task_manager.h"
#include "transport_executor.h"
#include "transport_manager_backend.h"
#include "ucmstore_v1.h"
#ifdef UC_DRAM_ASCEND_BACKEND
#include <acl/acl.h>
#endif

namespace UC::Dram {

// Synchronize an optional compute-stream event before a dump task is submitted.
// DramStore owns no NPU stream; the real D2H is an RDMA Read executed on the
// remote DramPool directly against this client's device memory.
// RDMA and the compute stream are unordered, so the only safe point is to
// block on the prerequisite event here, before the control message is sent.
static Status WaitPrerequisiteEvent(std::uintptr_t eventHandle)
{
    if (eventHandle == 0) { return Status::OK(); }
#ifdef UC_DRAM_ASCEND_BACKEND
    const auto ret = aclrtSynchronizeEvent(reinterpret_cast<aclrtEvent>(eventHandle));
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status::Error("aclrtSynchronizeEvent failed: " + std::to_string(ret));
#else
    return Status::OK();
#endif
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
    bool NeedRegisterKVCaches() const override;
    Status RegisterKVCaches(const KVCacheRegistration* registrations, std::size_t count) override;

private:
    Expected<Detail::TaskHandle> SubmitTransfer(OpType op, Detail::TaskDesc task);

    Status SetupParsed(DramConfig parsed)
    {
        config = std::make_unique<DramConfig>(std::move(parsed));
        auto status = Compose();

        if (status.Failure()) {
            StopGraph();
            return status;
        }
        return Status::OK();
    }

    Status Compose()
    {
#ifdef UC_DRAM_ASCEND_BACKEND
        const auto transferTimeout = std::max(config->taskTimeouts.load, config->taskTimeouts.dump);
        auto createdBackend = CreateTransportManagerBackend(TransportManagerBackendOptions{
            config->localControlHost, config->localControlPort, config->localTransportManagerId,
            config->localHost, config->transportDeviceId, 1000,
            static_cast<std::int32_t>(std::min<std::int64_t>(
                transferTimeout.count(), std::numeric_limits<std::int32_t>::max())),
            config->nodeScheduler.nodes});
        if (!createdBackend) { return createdBackend.Error(); }
        transportBackend = std::move(createdBackend).Value();

        auto createdReplies = ReplyService::Create(ReplyService::Options{
            config->replySlotSize, config->replySlotCount, std::chrono::microseconds{50},
            [this](NodeId nodeId, NodeEvent event) {
                nodeScheduler->Publish(nodeId, std::move(event));
            }});
        if (!createdReplies) { return createdReplies.Error(); }
        replyService = std::move(createdReplies).Value();

        memoryHandles.reserve(1);
        const auto replyMemory = replyService->MemoryRegion();
        auto registeredReply = transportBackend->RegisterMemory(
            replyMemory.address, replyMemory.length, MemoryRegionType::HOST);
        if (!registeredReply) { return registeredReply.Error(); }
        memoryHandles.push_back(std::move(registeredReply).Value());

        std::vector<UC::KV::NodeId> nodeIds;
        nodeIds.reserve(config->nodeScheduler.nodes.size());
        for (const auto& node : config->nodeScheduler.nodes) { nodeIds.push_back(node.nodeId); }
        UC::KV::RouterConfig routerConfig;
        routerConfig.type = config->routerType;
        router = UC::KV::CreateRouter(nodeIds, {}, routerConfig);
        if (!router) { return Status::Error("failed to create DramStore router"); }

        transport = std::make_unique<TransportExecutor>(TransportExecutor::Options{
            config->transportRuntime.workerCount, config->nodeScheduler.nodes.size(),
            config->nodeScheduler.limits.maxInflightRequests, transportBackend,
            [this](NodeId nodeId, NodeEvent event) {
                nodeScheduler->Publish(nodeId, std::move(event));
            }});

        nodeScheduler = std::make_unique<NodeScheduler>(
            config->nodeScheduler,
            NodeDependencies{
                [this](std::vector<RequestCompleted>& events) { taskManager->Publish(events); },
                [this](TransportCommand& command) {
                    return transport ? transport->TryPost(command)
                                     : Status::Error("TransportExecutor is unavailable");
                },
                [this](const RequestToken& token, OpType op,
                       std::size_t entryCount) -> Expected<ReplySlot> {
                    return replyService
                               ? replyService->Acquire(token, op, entryCount)
                               : Expected<ReplySlot>{Status::Error("ReplyService is unavailable")};
                },
                [this](const RequestToken& token, const ReplySlot& slot) {
                    return replyService ? replyService->Release(token, slot)
                                        : Status::Error("ReplyService is unavailable");
                }});

        taskManager = std::make_unique<TaskManager>(
            TaskManagerConfig{config->tensorSizes, config->maxIoEntries,
                              config->nodeScheduler.limits.maxBatchEntries, config->taskTimeouts},
            TaskManagerDependencies{router, [this](Request& request) {
                                        return nodeScheduler
                                                   ? nodeScheduler->Post(request)
                                                   : Status::Error("NodeScheduler is unavailable");
                                    }});

        auto status = transport->Start();
        if (status.Failure()) { return status; }
        status = taskManager->Start();
        if (status.Failure()) { return status; }
        status = replyService->Start();
        if (status.Failure()) { return status; }
        return nodeScheduler->Start();
#else
        return Status::Unsupported();
#endif
    }

    void StopGraph()
    {
        if (!config) { return; }

        // Shutdown is terminal application teardown. TaskManager first stops all
        // admission and drops late completions while its callback target remains alive.
        if (taskManager) { taskManager->Shutdown(); }
        if (nodeScheduler) { nodeScheduler->Shutdown(); }
        if (replyService) { replyService->Shutdown(); }
        if (transport) { transport->Shutdown(); }
        if (transportBackend) { transportBackend->Stop(); }

        memoryHandles.clear();
        transport.reset();
        replyService.reset();
        nodeScheduler.reset();
        taskManager.reset();
        transportBackend.reset();
        router.reset();
        config.reset();
    }

    std::unique_ptr<DramConfig> config;
    std::shared_ptr<ITransportBackend> transportBackend;
    std::vector<MemoryHandle> memoryHandles;
    std::shared_ptr<UC::KV::Router> router;
    std::unique_ptr<TransportExecutor> transport;
    std::unique_ptr<ReplyService> replyService;
    std::unique_ptr<NodeScheduler> nodeScheduler;
    std::unique_ptr<TaskManager> taskManager;
};

Status DramStore::Setup(const Detail::Dictionary& config)
{
    auto parsed = DramConfig::Parse(config);
    if (!parsed) { return parsed.Error(); }
    return SetupParsed(std::move(parsed).Value());
}

std::string DramStore::Readme() const
{
    return "DramStore: serialized task coordination and per-node remote-access safety";
}

Expected<std::vector<std::uint8_t>> DramStore::Lookup(const Detail::BlockId* blocks,
                                                      std::size_t num)
{
    if (num == 0) { return std::vector<std::uint8_t>{}; }
    if (blocks == nullptr || num > config->maxIoEntries) {
        return Status::InvalidParam("invalid lookup input");
    }
    auto submitted = taskManager->SubmitLookup(blocks, num);
    if (!submitted) { return submitted.Error(); }
    return taskManager->WaitLookup(std::move(submitted).Value());
}

Expected<ssize_t> DramStore::LookupOnPrefix(const Detail::BlockId*, std::size_t)
{
    return Status::Unsupported();
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
        UC_ERROR("Failed({}) to wait prerequisite event for dump.", status);
        return status;
    }
    task.prerequisiteHandle = 0;
    return SubmitTransfer(OpType::DUMP, std::move(task));
}

Expected<Detail::TaskHandle> DramStore::SubmitTransfer(OpType op, Detail::TaskDesc task)
{
    if (task.empty()) { return Status::InvalidParam("invalid transfer task"); }

    return taskManager->SubmitTransfer(op, std::move(task));
}

Expected<bool> DramStore::Check(Detail::TaskHandle taskId) { return taskManager->Check(taskId); }

Status DramStore::Wait(Detail::TaskHandle taskId) { return taskManager->WaitTransfer(taskId); }

bool DramStore::NeedRegisterKVCaches() const { return true; }

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

    const auto firstHandle = memoryHandles.size();
    if (count > memoryHandles.max_size() - firstHandle) {
        return Status::InvalidParam("too many KV cache registrations");
    }
    try {
        memoryHandles.reserve(firstHandle + count);
    } catch (...) {
        return Status::InvalidParam("too many KV cache registrations");
    }

    for (std::size_t index = 0; index < count; ++index) {
        auto registered =
            transportBackend->RegisterMemory(reinterpret_cast<void*>(registrations[index].addr),
                                             registrations[index].size, MemoryRegionType::DEVICE);
        if (registered) {
            memoryHandles.push_back(std::move(registered).Value());
            continue;
        }

        auto result = registered.Error();
        while (memoryHandles.size() > firstHandle) {
            const auto cleanup = transportBackend->UnregisterMemory(memoryHandles.back());
            if (cleanup.Failure()) {
                result = cleanup;
                break;
            }
            memoryHandles.pop_back();
        }
        return result;
    }
    return Status::OK();
}

}  // namespace UC::Dram

extern "C" UC::StoreV1* MakeDramStore() { return new UC::Dram::DramStore(); }
