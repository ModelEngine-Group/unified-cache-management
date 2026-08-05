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
#include "transport_manager_backend.h"
#include <utility>
#include "core/transport_init_attrs.h"
#include "logger/logger.h"

namespace UC::Dram {

TransportManagerBackend::TransportManagerBackend(TransportManagerBackendOptions options)
    : options_(std::move(options)), manager_(options_.localTransportManagerId)
{
}

TransportManagerBackend::~TransportManagerBackend() = default;

Status TransportManagerBackend::Init()
{
    if (options_.deviceId < 0 || options_.connectTimeoutMs <= 0 ||
        options_.transferTimeoutMs <= 0 || options_.localHost.empty() ||
        options_.localControlHost.empty() || options_.localControlPort == 0 ||
        options_.localTransportManagerId.empty() || options_.nodes.empty()) {
        return Status::InvalidParam("invalid TransportManager backend options");
    }
    localControl_ = transport::Endpoint{options_.localControlHost, options_.localControlPort};

    // TODO: make transport backend configurable
    transport::HixlInitAttrs attrs;
    attrs.ip = options_.localHost;
    attrs.instances.push_back(transport::HixlInitAttrs::Instance{-1, options_.deviceId, {}});
    attrs.connect_timeout_ms = options_.connectTimeoutMs;
    attrs.transfer_timeout_ms = options_.transferTimeoutMs;
    auto transportStatus = manager_.Init();
    if (transportStatus.Failure()) { return transportStatus; }
    transportStatus = manager_.InstallTransport(transport::TransportProtocol::Hixl, attrs);
    if (transportStatus.Failure()) { return transportStatus; }
    transportStatus = control_.Init(localControl_);
    if (transportStatus.Failure()) { return transportStatus; }
    for (const auto& node : options_.nodes) {
        if (!nodes_.emplace(node.nodeId, node).second) {
            return Status::InvalidParam("duplicate DramPool node");
        }
    }
    return Status::OK();
}

Expected<MemoryHandle> TransportManagerBackend::RegisterMemory(void* address, std::size_t length,
                                                               MemoryRegionType type)
{
    transport::MemoryRegion region;
    region.addr = address;
    region.length = length;
    region.type = type == MemoryRegionType::DEVICE ? transport::MemoryType::Device
                                                   : transport::MemoryType::Host;
    region.device_id = type == MemoryRegionType::DEVICE ? options_.deviceId : -1;
    transport::MemoryHandle handle = transport::kInvalidMemoryHandle;
    const auto status = manager_.RegisterMemory(region, handle);
    if (status.Failure()) { return status; }
    return static_cast<MemoryHandle>(handle);
}

Status TransportManagerBackend::UnregisterMemory(MemoryHandle handle)
{
    return manager_.UnregisterMemory(static_cast<transport::MemoryHandle>(handle));
}

TransmitCompleted TransportManagerBackend::Transmit(const ::UC::Dram::Transmit& command) noexcept
{
    const auto found = nodes_.find(command.token.nodeId);
    if (found == nodes_.end() || command.payload.empty()) {
        return TransmitCompleted{command.token, Status::InvalidParam("invalid transmit")};
    }
    try {
        const transport::Endpoint endpoint{found->second.controlHost, found->second.controlPort};
        const auto status = control_.Send(endpoint, command.payload.data(), command.payload.size());
        if (status.Success()) { return TransmitCompleted{command.token, Status::OK()}; }
        return TransmitCompleted{command.token, status};
    } catch (...) {
        return TransmitCompleted{command.token, Status::Error("TCP transmit threw an exception")};
    }
}

Status TransportManagerBackend::Connect(const ::UC::Dram::Connect& command) noexcept
{
    if (command.transportManagerId.empty()) {
        return Status::InvalidParam("remote TransportManager id is missing");
    }
    try {
        auto status = manager_.ExchangeMetadata(command.transportManagerId);
        if (status.Failure()) { return status; }
        status = manager_.Connect(transport::TransportProtocol::Hixl, command.transportManagerId);
        return status;
    } catch (...) {
        return Status::Error("TransportManager connect threw an exception");
    }
}

Status TransportManagerBackend::Fence(const ::UC::Dram::FenceEpoch& command) noexcept
{
    const auto found = nodes_.find(command.nodeId);
    if (found == nodes_.end()) { return Status::InvalidParam("unknown DramPool node"); }
    try {
        // The transport Manager contract guarantees that successful Disconnect
        // synchronously revokes old-connection access to local registered memory.
        return manager_.Disconnect(transport::TransportProtocol::Hixl,
                                   found->second.transportManagerId);
    } catch (...) {
        return Status::Error("TransportManager disconnect threw an exception");
    }
}

void TransportManagerBackend::Stop()
{
    std::lock_guard lock(stopMutex_);
    if (stopped_) { return; }
    const auto controlStatus = control_.Shutdown();
    const auto managerStatus = manager_.Shutdown();
    stopped_ = true;
    if (controlStatus.Failure()) {
        UC_ERROR("DramStore transport control shutdown failed: {}", controlStatus);
    }
    if (managerStatus.Failure()) {
        UC_ERROR("DramStore transport manager shutdown failed: {}", managerStatus);
    }
}

Expected<std::shared_ptr<ITransportBackend>> CreateTransportManagerBackend(
    TransportManagerBackendOptions options)
{
    auto backend =
        std::shared_ptr<TransportManagerBackend>(new TransportManagerBackend(std::move(options)));
    const auto status = backend->Init();
    if (status.Failure()) { return status; }
    std::shared_ptr<ITransportBackend> result = std::move(backend);
    return result;
}

}  // namespace UC::Dram
