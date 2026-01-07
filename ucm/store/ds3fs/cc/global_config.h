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
#ifndef UNIFIEDCACHE_DS3FS_STORE_CC_GLOBAL_CONFIG_H
#define UNIFIEDCACHE_DS3FS_STORE_CC_GLOBAL_CONFIG_H

#include <string>
#include <vector>

namespace UC::Ds3fsStore {

struct Config {
    std::vector<std::string> storageBackends{};
    int32_t deviceId{-1};
    size_t tensorSize{0};
    size_t shardSize{0};
    size_t blockSize{0};
    bool ioDirect{true};
    size_t streamNumber{32};
    size_t timeoutMs{30000};
    size_t iorEntries{1};
    int32_t iorDepth{1};
    int32_t numaId{-1};
};

}  // namespace UC::Ds3fsStore

#endif
