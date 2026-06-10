#pragma once

#include "trans_provider.h"

namespace UC::ASU {

using AICPUTransProviderSendHook =
    std::vector<Status> (*)(const std::vector<TransProvider::SendIoBatch>& ioBatches,
                            uint32_t kernelCount, uint32_t quietCount);

void SetAICPUTransProviderSendHook(AICPUTransProviderSendHook hook);
AICPUTransProviderSendHook GetAICPUTransProviderSendHook();

class AICPUTransProvider : public TransProvider {
public:
    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override
    {
        handles.clear();
        handles.reserve(qpNum);
        for (uint32_t index = 0; index < qpNum; ++index) {
            handles.push_back(reinterpret_cast<ConnectionHandle>(
                static_cast<std::uintptr_t>(index) + static_cast<std::uintptr_t>(1)));
        }
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>& ioBatches,
                             uint32_t kernelCount, uint32_t quietCount) override
    {
        auto hook = GetAICPUTransProviderSendHook();
        if (hook != nullptr) { return hook(ioBatches, kernelCount, quietCount); }
        return std::vector<Status>(ioBatches.size(), Status::OK());
    }

    Status RegisterMemory(ConnectionHandle, const std::vector<RegisterMemoryDesc>&,
                          std::vector<MemHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>& handles) override
    {
        return std::vector<Status>(handles.size(), Status::OK());
    }

    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override
    {
        return std::vector<Status>(threads.size(), Status::OK());
    }

    Status GetMemTokenId(MemHandle, uint32_t& tokenId) override
    {
        tokenId = 0;
        return Status::OK();
    }
};

}  // namespace UC::ASU
