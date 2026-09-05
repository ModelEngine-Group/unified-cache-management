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
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include "../../include/types.h"

namespace kv {

inline std::string_view CacheKeyView(const CacheKey& key)
{
    return {reinterpret_cast<const char*>(key.data()), key.size()};
}

inline std::string CacheKeyToHex(const CacheKey& key)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string text;
    text.reserve(key.size() * 2);
    for (auto byte : key) {
        const auto value = static_cast<unsigned char>(std::to_integer<unsigned char>(byte));
        text.push_back(kHex[value >> 4]);
        text.push_back(kHex[value & 0x0F]);
    }
    return text;
}

constexpr std::uint32_t kAsuAlignmentBytes = 512;  // KV protocol requires 512B alignment

struct ServerKvCapabilities {
    // A zero limit means that the provider did not advertise that capability.
    std::uint32_t queueNum{0};
    std::uint32_t ioQueueDepth{0};
    std::uint32_t ioQueueKeyConcurrency{0};     // Placeholder
    std::uint32_t connectionKeyConcurrency{0};  // Placeholder
    std::uint64_t singleValueMaxBytes{0};
    std::uint64_t batchValueMaxBytes{0};
    std::uint32_t batchStoreKeys{0};
    std::uint32_t batchLoadKeys{0};
    std::uint32_t deleteKeys{0};
    std::uint32_t queryKeys{0};
    std::uint32_t keyLength{0};
    std::uint32_t kvCapabilities{0};  // Placeholder
};

using TaskCompletionCallback = std::function<void(TaskResult)>;

}  // namespace kv
