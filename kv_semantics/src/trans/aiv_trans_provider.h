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
#include "aiv_transport.h"
#include "trans_provider.h"

namespace UC::ASU {

class AIVTransProviderAdapter : public TransProvider {
public:
    AIVTransProviderAdapter() : impl_(CreateAIVTransProvider()) {}
    explicit AIVTransProviderAdapter(uint32_t deviceId) : impl_(CreateAIVTransProvider(deviceId)) {}

    Status CreateConnection(const std::string& localIp, const std::string& remoteIp, uint32_t port,
                            uint32_t qpNum, uint32_t timeout,
                            std::vector<ConnectionHandle>& connectionHandles) override
    {
        return FromImpl(
            impl_->CreateConnection(localIp, remoteIp, port, qpNum, timeout, connectionHandles));
    }

    std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) override
    {
        return FromImpl(impl_->DeleteConnections(connectionHandles));
    }

    Status GetServerCapabilities(ConnectionHandle connectionHandle,
                                 ServerKvCapabilities& capabilities) override
    {
        return FromImpl(impl_->GetServerCapabilities(connectionHandle, capabilities));
    }

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t kernelCount,
                             uint32_t quietCount) override
    {
        std::vector<AIVTransport::SendIoBatch> batches;
        batches.reserve(ioBatches.size());
        for (const auto& io : ioBatches) {
            batches.push_back({io.connectionHandle, io.sendBuffer, io.flagBuffer, io.len});
        }
        return FromImpl(impl_->Send(batches, kernelCount, quietCount));
    }

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MRHandle>& mrHandles) override
    {
        std::vector<AIVTransport::RegisterMemoryDesc> descs;
        descs.reserve(memoryDescs.size());
        for (const auto& desc : memoryDescs) {
            descs.push_back(
                {static_cast<AIVTransport::MemType>(desc.memoryType), desc.addr, desc.size});
        }
        return FromImpl(impl_->RegisterMemory(descs, mrHandles));
    }

    Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                      std::vector<MRHandle>& mrHandles) override
    {
        std::vector<AIVTransport::BindMemoryDesc> descs;
        descs.reserve(memoryDescs.size());
        for (const auto& desc : memoryDescs) {
            descs.push_back({static_cast<AIVTransport::MemType>(desc.memoryType), desc.addr,
                             desc.size, desc.tokenId});
        }
        return FromImpl(impl_->BindMemory(descs, mrHandles));
    }

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override
    {
        std::vector<AIVTransport::UnregisterMemoryDesc> descs;
        descs.reserve(memoryDescs.size());
        for (const auto& desc : memoryDescs) { descs.push_back({desc.mrHandle}); }
        return FromImpl(impl_->UnregisterMemory(descs));
    }

    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override
    {
        return std::vector<Status>(threads.size(), Status::OK());
    }

    Status GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId) override
    {
        return FromImpl(impl_->GetMemTokenId(mrHandle, tokenId));
    }

private:
    static Status FromImpl(const Status& status)
    {
        return status.ok() ? Status::OK() : Status::Error(status.code, status.message);
    }

    static std::vector<Status> FromImpl(const std::vector<Status>& statuses)
    {
        std::vector<Status> out;
        out.reserve(statuses.size());
        for (const auto& status : statuses) { out.push_back(FromImpl(status)); }
        return out;
    }

    std::unique_ptr<AIVTransport> impl_;
};

}  // namespace UC::ASU
