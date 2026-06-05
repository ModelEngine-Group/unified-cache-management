#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "acl/acl.h"
#include "hcomm/hcomm_res_defs.h"
#include "hixl_kernel/hixl_send.h"
#include "trans_provider.h"

namespace UC::ASU {

class AICPUTransProvider : public TransProvider {
public:
    explicit AICPUTransProvider(const std::string& kernelJsonPath = "",
                                const std::string& ipMapPath = "");
    ~AICPUTransProvider() override;

    Status CreateConnection(const std::string& localIp, const std::string& remoteIp, uint32_t port,
                            uint32_t qpNum, uint32_t timeout,
                            std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t kernelCount,
                             uint32_t quietCount) override;

    Status RegisterMemory(ConnectionHandle connectionHandle,
                          const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MemHandle>& memoryHandles) override;

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override;

    Status AllocThread(uint32_t threadNum, const std::vector<uint32_t>& notifyNumPerThread,
                       std::vector<ThreadHandle>& threads) override;

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override;

    Status GetMemTokenId(MemHandle memHandle, uint32_t& tokenId) override;

    // Memory export/import for cross-process sharing (demo only, not used by asu_transport)
    Status ExportMemory(ConnectionHandle connectionHandle, MemHandle memHandle, void** exportDesc,
                        uint32_t* exportLen);

    Status ImportMemory(ConnectionHandle connectionHandle, const void* importDesc,
                        uint32_t importLen, MemHandle* importedHandle);

    Status GetImportedMemoryInfo(ConnectionHandle connectionHandle, MemHandle importedHandle,
                                 uint64_t* addr, uint64_t* size);

    Status UnimportMemory(ConnectionHandle connectionHandle, MemHandle importedHandle);

    Status InitEndpoint(const std::string& localIp);

    const EndpointDesc& GetLocalEndpointDesc() const { return localEndpointDesc_; }

    uint64_t GetChannelHandle(ConnectionHandle connectionHandle)
    {
        auto* ctx = GetLinkContext(connectionHandle);
        return ctx ? ctx->channel : 0;
    }

private:
    struct LinkContext {
        std::string localIp;
        uint64_t channel;
        uint64_t thread;
        aclrtStream stream{nullptr};
        std::string remoteIp;
        uint16_t remotePort;
    };

    struct ImportedMemInfo {
        void* addr;
        uint64_t size;
        std::vector<uint8_t> memDesc;  // 保存 memDesc 用于 unimport
    };

    std::mutex mutex_;
    std::unordered_map<MemHandle, ImportedMemInfo> importedMemMap_;  // importedHandle -> info

    void* endpoint_{nullptr};
    std::string localIp_;
    uint32_t endpointRefCount_{0};
    EndpointDesc localEndpointDesc_{};

    aclrtBinHandle kernelBin_{nullptr};
    aclrtFuncHandle kernelFunc_{nullptr};

    static LinkContext* GetLinkContext(ConnectionHandle handle)
    { return static_cast<LinkContext*>(handle); }
    void* GetOrCreateEndpoint(const std::string& localIp);
    void ReleaseEndpoint(const std::string& localIp);
    void LoadIpToDeviceMap();
    uint32_t LookupDeviceByIp(const std::string& ip);

    std::string ipMapPath_;
    std::unordered_map<std::string, uint32_t> ipToDeviceMap_;
    bool ipToDeviceMapLoaded_{false};
};

}  // namespace UC::ASU
