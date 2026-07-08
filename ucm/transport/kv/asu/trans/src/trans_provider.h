#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "asu_transport/types.h"

namespace UC::ASU {

class TransProvider {
public:
    using ConnectionHandle = void*;
    using ThreadHandle = void*;
    using MemHandle = void*;

    virtual ~TransProvider() = default;

    virtual Status CreateConnection(const std::string& localIp, const std::string& remoteIp,
                                    uint32_t port, uint32_t qpNum, uint32_t timeout,
                                    std::vector<ConnectionHandle>& connectionHandles) = 0;

    virtual std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) = 0;

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

    virtual Status RegisterMemory(ConnectionHandle connectionHandle,
                                  const std::vector<RegisterMemoryDesc>& memoryDescs,
                                  std::vector<MemHandle>& memoryHandles) = 0;

    struct UnregisterMemoryDesc {
        ConnectionHandle connectionHandle;
        MemHandle memoryHandle;
    };

    virtual std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) = 0;

    virtual Status AllocThread(uint32_t threadNum, const std::vector<uint32_t>& notifyNumPerThread,
                               std::vector<ThreadHandle>& threads) = 0;

    virtual std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) = 0;

    virtual Status GetMemTokenId(MemHandle memHandle, uint32_t& tokenId) = 0;
};

}  // namespace UC::ASU
