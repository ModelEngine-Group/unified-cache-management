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
#ifndef UNIFIEDCACHE_TRANS_CUDA_GDR_CONFIG_H
#define UNIFIEDCACHE_TRANS_CUDA_GDR_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>
#include "status/status.h"

namespace UC::Trans {

class GdrNicConfig {
public:
    static Expected<std::string> ResolveNicName(int32_t deviceId);
};

class GdrKVBufferConfig {
public:
    GdrKVBufferConfig() = default;
    GdrKVBufferConfig(const GdrKVBufferConfig&) = delete;
    GdrKVBufferConfig& operator=(const GdrKVBufferConfig&) = delete;
    GdrKVBufferConfig(GdrKVBufferConfig&&) = default;
    GdrKVBufferConfig& operator=(GdrKVBufferConfig&&) = default;
    ~GdrKVBufferConfig();

    static Status Validate(const std::vector<uintptr_t>& addrs, const std::vector<size_t>& sizes);
    Status Register(const std::vector<uintptr_t>& addrs, const std::vector<size_t>& sizes);

private:
    struct RegisteredBuffer {
        uint64_t addr;
        size_t size;
    };
    std::vector<RegisteredBuffer> buffers_;
};

}  // namespace UC::Trans

#endif
