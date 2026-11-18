/**
 * MIT License
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include "kv_types.h"

namespace kv {

// Internal storage identity, normalized before routing.
struct ResolvedCacheKey {
    BinaryKey key{};
    std::uint32_t gnsid{0};
};

struct ResolvedKVBuffer {
    ResolvedCacheKey key;
    Buffer buffer;
    std::uint32_t offset{0};
    std::optional<std::uint32_t> mrKey;
};

// Resolves namespace and storage identity before routing. Input buffers and offsets
// remain unchanged. Implementations must be deterministic and thread-safe.
class KeyResolver {
public:
    virtual ~KeyResolver() = default;
    Status Resolve(const CacheKey& input, std::string_view baseKvns,
                   ResolvedCacheKey& result) const;

protected:
    struct ParsedKey {
        std::string identity;
        std::optional<std::string> component;
        std::optional<std::uint32_t> ppRank;
    };

    virtual Status ParseEngineKey(std::string_view input, ParsedKey& result) const = 0;
};

std::unique_ptr<KeyResolver> CreateDefaultKeyResolver(const KeyResolutionConfig& config);

std::uint32_t ResolveNamespace(std::string_view kvns);

}  // namespace kv
