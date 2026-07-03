/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_GLOBAL_CONFIG_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_GLOBAL_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

namespace UC {
class StoreV1;
}

namespace UC::MooncakeStore {

struct Config {
    std::string localHostname{};
    std::string metadataServer{"P2PHANDSHAKE"};
    std::string masterServerAddress{"127.0.0.1:50088"};
    std::string protocol{"ascend"};
    std::string deviceName{};

    uint64_t globalSegmentSize{32212254720};
    uint64_t localBufferSize{1073741824};
    uint32_t replicaNum{1};
    bool withSoftPin{false};

    std::vector<uint64_t> tensorSizeList{};

    int32_t deviceId{-1};
    uint32_t loadQueueDepth{524288};
    uint32_t dumpQueueDepth{8192};
    uint32_t hostBufPoolSize{1024};
    size_t timeoutMs{0};

    size_t streamNumber{4};
    std::vector<ssize_t> cpuAffinityCores{};

    bool ioDirect{false};

    size_t localRankSize{1};
    std::string uniqueId{};
    uint64_t shareBufferCapacity{64ULL << 30};
    size_t shareBufferNumber{0};

    StoreV1* storeBackend{nullptr};

    std::vector<uintptr_t> gpuKvBufferAddrs{};
    std::vector<size_t> gpuKvBufferSizes{};
};

}  // namespace UC::MooncakeStore

#endif
