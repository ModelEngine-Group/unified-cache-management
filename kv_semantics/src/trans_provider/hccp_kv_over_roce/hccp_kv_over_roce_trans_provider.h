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
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "trans_provider.h"

namespace kv {

// TransProvider backed by an external HCCP KV-over-RoCE library loaded with
// dlopen. The library owns the actual connection/memory implementation; this
// provider adds an HBM slot allocator so every live connection reserves a
// fixed-size connection-information slot (tracked by slotid). It also owns the
// MR handle space and keeps a provider-local handle-to-memory-info map for both
// registered and bound regions.
class HccpKvOverRoceTransProvider : public TransProvider {
public:
    explicit HccpKvOverRoceTransProvider(const TransportConfig& config);
    ~HccpKvOverRoceTransProvider() override;

    Status CreateConnection(const std::string& localIp, const std::string& remoteIp, uint32_t port,
                            uint32_t qpNum, uint32_t timeout,
                            std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) override;

    Status GetServerCapabilities(ConnectionHandle connectionHandle,
                                 ServerKvCapabilities& capabilities) override;

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t kernelCount,
                             uint32_t quietCount) override;

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MRHandle>& mrHandles) override;

    Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                      std::vector<MRHandle>& mrHandles) override;

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override;

    Status AllocThread(uint32_t threadNum, const std::vector<uint32_t>& notifyNumPerThread,
                       std::vector<ThreadHandle>& threads) override;

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override;

    Status GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kv
