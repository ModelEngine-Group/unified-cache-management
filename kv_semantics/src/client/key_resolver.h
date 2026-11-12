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
#include "kv_types.h"

#define HASH_SEED_VALUE (5381)

namespace kv {

struct ResolvedKey {
    BinaryKey key{};
    std::uint64_t baseOffset{0};
    // Present only for layerwise keys, including layer zero.
    std::optional<std::uint64_t> layerSlotBytes;
};

// Resolves object identity and an offset increment before routing. Implementations must be
// deterministic and safe for concurrent calls, and preserve FromBytes inputs unchanged.
class KeyResolver {
public:
    virtual ~KeyResolver() = default;
    virtual Status Resolve(const CacheKey& input, ResolvedKey& result) const = 0;
};

std::unique_ptr<KeyResolver> CreateDefaultKeyResolver(const KeyResolutionConfig& config);

}  // namespace kv
