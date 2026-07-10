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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include "asu_transport/asu_transport.h"
#include "trans_provider.h"

namespace UC::ASU {

struct FakeTransProviderConfig {
    std::string storePath{"./asu-fake-backend-store"};
    std::uint64_t latencyMs{1};
    std::int32_t deviceId{0};
};

class FakeTransProvider : public TransProvider {
public:
    explicit FakeTransProvider(FakeTransProviderConfig config);

    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override;

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override;

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t kernelCount,
                             uint32_t quietCount) override;

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MRHandle>& mrHandles) override;

    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>& handles) override;

    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override;

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override;

    Status GetMemTokenId(MRHandle, uint32_t& tokenId) override;

private:
    struct RegisteredMemory {
        std::uintptr_t providerAddr{0};
        std::uintptr_t localAddr{0};
        std::size_t size{0};
    };

    Status SetUpAclRuntime();
    Status ResolveLocalAddress(const void* providerAddr, std::size_t size, void*& localAddr);

    FakeTransProviderConfig config_;
    std::atomic<std::uintptr_t> nextMrHandle_{1};
    std::mutex registeredMemoryMu_;
    std::unordered_map<MRHandle, RegisteredMemory> registeredMemories_;
};

FakeTransProviderConfig MakeFakeTransProviderConfig(const TransportConfig& config);

}  // namespace UC::ASU
