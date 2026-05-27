#pragma once

#include "trans_provider.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace UC::ASU {

class AICPUTransProvider : public TransProvider {
public:
    AICPUTransProvider();
    ~AICPUTransProvider() override;

    Status CreateConnection(
        const std::string& localIp,
        const std::string& remoteIp,
        uint32_t port,
        uint32_t qpNum,
        uint32_t timeout,
        std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> Send(
        const std::vector<SendIoBatch>& ioBatches,
        uint32_t kernelCount,
        uint32_t quietCount) override;

    Status RegisterMemory(
        ConnectionHandle connectionHandle,
        const std::vector<RegisterMemoryDesc>& memoryDescs,
        std::vector<MemHandle>& memoryHandles) override;

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override;

    Status AllocThread(
        uint32_t threadNum,
        const std::vector<uint32_t>& notifyNumPerThread,
        std::vector<ThreadHandle>& threads) override;

    std::vector<Status> FreeThread(
        const std::vector<ThreadHandle>& threads) override;

private:
    struct LinkContext {
        uint32_t phyDev;
        uint64_t channel;
        uint64_t thread;
        std::string remoteIp;
        uint16_t remotePort;
    };

    struct EndpointContext {
        void* endpoint;
        uint32_t refCount;
    };

    std::mutex mutex_;
    std::unordered_map<ConnectionHandle, std::shared_ptr<LinkContext>> linkContexts_;
    std::unordered_map<uint32_t, EndpointContext> endpointMap_;  // phyDev -> endpoint

    LinkContext* GetLinkContext(ConnectionHandle handle);
    void* GetOrCreateEndpoint(uint32_t phyDev, const std::string& localIp);
    void ReleaseEndpoint(uint32_t phyDev);
};

}  // namespace UC::ASU
