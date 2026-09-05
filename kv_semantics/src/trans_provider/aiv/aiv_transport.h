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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "types.h"

namespace kv {

class AIVTransport {
public:
    using ConnectionHandle = void*;

    virtual ~AIVTransport() = default;

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
    };

    virtual Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                                  std::vector<MRHandle>& mrHandles) = 0;

    struct BindMemoryDesc {
        MemType memoryType;
        uintptr_t addr;
        size_t size;
        uint32_t tokenId;
    };

    virtual Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                              std::vector<MRHandle>& mrHandles) = 0;

    struct UnregisterMemoryDesc {
        MRHandle mrHandle;
    };

    virtual std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) = 0;

    virtual Status GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId) = 0;

    virtual Status GetServerCapabilities(ConnectionHandle connectionHandle,
                                         ServerKvCapabilities& capabilities) = 0;
};

std::unique_ptr<AIVTransport> CreateAIVTransProvider();
std::unique_ptr<AIVTransport> CreateAIVTransProvider(uint32_t deviceId);

}  // namespace kv
