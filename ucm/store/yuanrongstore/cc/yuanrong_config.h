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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_YUANRONG_CONFIG_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_YUANRONG_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>
#include "ucmstore_v1.h"

namespace UC::YuanRongStore {

inline constexpr size_t kPersistenceQueueDepth = 128;
inline constexpr size_t kDefaultMemoryAlignment = 64;
inline constexpr size_t kDirectIoMemoryAlignment = 4096;

struct Config {
    std::string host{};
    int32_t port{0};
    std::string nameSpace{};
    bool enableRemoteH2D{true};
    bool enableDeviceMemoryPreRegistration{false};

    int32_t deviceId{-1};
    std::vector<size_t> tensorSizes{};
    size_t shardSize{0};
    size_t blockSize{0};
    size_t objectSize{0};
    size_t memoryAlignment{kDefaultMemoryAlignment};

    size_t timeoutMs{60000};
    size_t waitingQueueDepth{8192};
    size_t loadWorkerCount{4};
    size_t dumpPrerequisiteWorkerCount{2};
    size_t recoveryBatchSize{32};
    size_t hostBufferCount{0};
    size_t hostBufferCapacityGb{8};
    bool hostBufferCountExplicit{false};
    size_t h2dStreamCount{4};
    size_t backfillWorkerCount{1};
    size_t backfillQueueDepth{128};
    size_t posixDumpBatchSize{0};
    size_t posixMaxInflightGb{1};
    std::vector<ssize_t> cpuAffinityCores{};
    bool ioDirect{true};
    std::string posixIoEngine{"psync"};

    std::vector<uintptr_t> gpuKvBufferAddrs{};
    std::vector<size_t> gpuKvBufferSizes{};

    StoreV1* storeBackend{nullptr};
};

}  // namespace UC::YuanRongStore

#endif
