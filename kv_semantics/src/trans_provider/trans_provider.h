#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "types.h"
#include "trans_types.h"

namespace kv {

struct TransportConfig;

class TransProvider {
public:
    using ConnectionHandle = void*;
    using ThreadHandle = void*;

    virtual ~TransProvider() = default;

    // On success, connectionHandles contains exactly qpNum handles. On failure, it is empty.
    virtual Status CreateConnection(const std::string& localIp, const std::string& remoteIp,
                                    uint32_t port, uint32_t qpNum, uint32_t timeout,
                                    std::vector<ConnectionHandle>& connectionHandles) = 0;

    virtual std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) = 0;

    // Returns provider-neutral KV limits for the server behind this connection. Providers that
    // do not support capability discovery keep the default UNSUPPORTED result.
    virtual Status GetServerCapabilities(ConnectionHandle connectionHandle,
                                         ServerKvCapabilities& capabilities)
    {
        (void)connectionHandle;
        capabilities = {};
        return Status::Error(StatusCode::UNSUPPORTED, "server capability query is not supported");
    }

    struct SendIoBatch {
        ConnectionHandle connectionHandle;
        void* sendBuffer;
        void* flagBuffer;
        uint64_t len;
    };

    virtual std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches,
                                     uint32_t kernelCount, uint32_t quietCount) = 0;

    enum class MemType { MEM_DEVICE, MEM_HOST };

    struct RegisterMemoryDesc {
        MemType memoryType;
        uintptr_t addr;
        size_t size;
        uintptr_t localAddr{0};
    };

    virtual Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                                  std::vector<MRHandle>& mrHandles) = 0;

    struct BindMemoryDesc {
        MemType memoryType;
        uintptr_t addr;
        size_t size;
        std::uint32_t tokenId;
    };

    // Creates a provider-local handle for an existing shared registration.
    virtual Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                              std::vector<MRHandle>& mrHandles)
    {
        (void)memoryDescs;
        mrHandles.clear();
        return Status::Error(StatusCode::UNSUPPORTED, "memory binding is not supported");
    }

    struct UnregisterMemoryDesc {
        MRHandle mrHandle;
    };

    virtual std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) = 0;

    virtual Status AllocThread(uint32_t threadNum, const std::vector<uint32_t>& notifyNumPerThread,
                               std::vector<ThreadHandle>& threads) = 0;

    virtual std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) = 0;

    virtual Status GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId) = 0;
};

Status CreateTransProvider(const TransportConfig& config,
                           std::shared_ptr<TransProvider>& transProvider);

}  // namespace kv
