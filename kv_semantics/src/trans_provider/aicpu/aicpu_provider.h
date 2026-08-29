#pragma once

#include <memory>
#include <vector>
#include "kv_transport.h"
#include "trans_provider.h"

namespace kv {

class AICPUProvider : public TransProvider {
public:
    explicit AICPUProvider(const TransportConfig& config);
    ~AICPUProvider() override;

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

    Status AllocThread(uint32_t, const std::vector<uint32_t>&,
                       std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override
    {
        return std::vector<Status>(threads.size(), Status::OK());
    }

    Status GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId) override;

private:
    enum class RegistrationMode {
        REGISTER,
        BIND,
    };

    Status RegisterMemoryImpl(const std::vector<RegisterMemoryDesc>& memoryDescs,
                              RegistrationMode mode, const char* operation,
                              std::vector<MRHandle>& mrHandles,
                              const std::vector<std::uint32_t>* expectedTokenIds = nullptr);
    std::vector<Status> ReleaseMemory(const std::vector<MRHandle>& mrHandles,
                                      const char* operation);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kv
