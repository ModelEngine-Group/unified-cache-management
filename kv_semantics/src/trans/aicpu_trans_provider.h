#pragma once

#include "trans_provider.h"

namespace UC::ASU {

class AICPUTransProvider : public TransProvider {
public:
    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t, uint32_t,
                            std::vector<ConnectionHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t, uint32_t) override
    {
        return std::vector<Status>(ioBatches.size(), Status::OK());
    }

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>&, std::vector<MRHandle>&) override
    {
        return Status::OK();
    }

    Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                      std::vector<MRHandle>& mrHandles) override
    {
        mrHandles.assign(memoryDescs.size(), kInvalidMRHandle);
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

    Status GetMemTokenId(MRHandle, uint32_t& tokenId) override
    {
        tokenId = 0;
        return Status::OK();
    }
};

}  // namespace UC::ASU
