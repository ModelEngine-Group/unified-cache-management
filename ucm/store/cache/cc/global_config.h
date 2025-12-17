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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_GLOBAL_CONFIG_H
#define UNIFIEDCACHE_CACHE_STORE_CC_GLOBAL_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace UC::CacheStore {

struct Config {
    uintptr_t storeBackend{};
    std::string uniqueId{};
    int32_t deviceId{-1};
    size_t tensorSize{0};
    size_t shardSize{0};
    size_t blockSize{0};
    size_t bufferSize{size_t(1) << 36};
    bool shareBufferEnable{false};
    size_t waitingQueueDepth{1024};
    size_t runningQueueDepth{32768};
    size_t timeoutMs{30000};
};

}  // namespace UC::CacheStore

#endif
