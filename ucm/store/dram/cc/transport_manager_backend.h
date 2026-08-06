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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_TRANSPORT_MANAGER_BACKEND_H
#define UNIFIEDCACHE_DRAM_STORE_CC_TRANSPORT_MANAGER_BACKEND_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "channels/tcp/tcp_message_channel.h"
#include "core/transport_manager.h"
#include "transport_executor.h"
#include "types.h"

namespace UC::Dram {

struct TransportManagerBackendOptions {
    std::string localControlHost;
    std::uint16_t localControlPort{0};
    std::string localTransportManagerId;
    std::string localHost;
    std::int32_t deviceId{0};
    std::int32_t connectTimeoutMs{1000};
    std::int32_t transferTimeoutMs{5000};
    std::vector<NodeEndpoint> nodes;
};

Expected<std::shared_ptr<ITransportBackend>> CreateTransportManagerBackend(
    TransportManagerBackendOptions options);

class TransportManagerBackend final : public ITransportBackend {
public:
    ~TransportManagerBackend() override;

    TransportManagerBackend(const TransportManagerBackend&) = delete;
    TransportManagerBackend& operator=(const TransportManagerBackend&) = delete;

    Expected<MemoryHandle> RegisterMemory(void* address, std::size_t length,
                                          MemoryRegionType type) override;
    Status UnregisterMemory(MemoryHandle handle) override;
    TransmitCompleted Transmit(const ::UC::Dram::Transmit& command) noexcept override;
    Status Connect(const ::UC::Dram::Connect& command) noexcept override;
    Status Fence(const ::UC::Dram::FenceEpoch& command) noexcept override;
    void Stop() override;

private:
    friend Expected<std::shared_ptr<ITransportBackend>> CreateTransportManagerBackend(
        TransportManagerBackendOptions options);

    explicit TransportManagerBackend(TransportManagerBackendOptions options);
    Status Init();

    TransportManagerBackendOptions options_;
    transport::Endpoint localControl_;
    transport::TransportManager manager_;
    transport::TcpMessageChannel control_;
    std::unordered_map<NodeId, NodeEndpoint> nodes_;
    std::mutex stopMutex_;
    bool stopped_{false};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_TRANSPORT_MANAGER_BACKEND_H
