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
#include "key_resolver.h"
#define XXH_INLINE_ALL
#include <charconv>
#include <string_view>
#include "utils/config_utils.h"
#include "xxhash.h"

namespace kv {
namespace {

class DefaultKeyResolver final : public KeyResolver {
public:
    explicit DefaultKeyResolver(KeyResolutionConfig config) : config_(config) {}

    Status Resolve(const CacheKey& input, ResolvedKey& result) const override
    {
        result = {};
        if (const auto* bytes = input.TryGetBinaryKey()) {
            result.key = *bytes;
            return Status::OK();
        }
        const auto& text = *input.TryGetEngineKey();
        std::string_view identity = text;
        if (config_.engine == InferenceEngine::VLLM_ASCEND) {
            auto status = ResolveVllmLayerKey(text, identity, result);
            if (!status.ok()) { return status; }
        }
        auto hash = XXH64(identity.data(), identity.size(), HASH_SEED_VALUE);
        for (auto& byte : result.key) {
            byte = static_cast<std::byte>(hash & 0xff);
            hash >>= 8;
        }
        return Status::OK();
    }

private:
    Status ResolveVllmLayerKey(std::string_view key, std::string_view& identity,
                               ResolvedKey& result) const;
    KeyResolutionConfig config_;
};

Status DefaultKeyResolver::ResolveVllmLayerKey(std::string_view key, std::string_view& identity,
                                               ResolvedKey& result) const
{
    identity = key;
    result.baseOffset = 0;
    // LayerPoolKey ends with @chunk_hash@layer. Preserve every other field.
    const auto layerSeparator = key.rfind('@');
    const auto hashSeparator = layerSeparator != std::string_view::npos && layerSeparator != 0
                                   ? key.rfind('@', layerSeparator - 1)
                                   : std::string_view::npos;
    if (hashSeparator != std::string_view::npos && hashSeparator != 0) {
        const auto hash = key.substr(hashSeparator + 1, layerSeparator - hashSeparator - 1);
        const auto layerText = key.substr(layerSeparator + 1);
        if (!hash.empty() && hash.find(':') == std::string_view::npos) {
            std::uint64_t layer = 0;
            const auto [end, error] =
                std::from_chars(layerText.data(), layerText.data() + layerText.size(), layer);
            if (error != std::errc{} || end != layerText.data() + layerText.size()) {
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "invalid layer in key: " + std::string(key));
            }
            auto status = ValidateKeyResolutionConfig(config_);
            if (!status.ok()) { return status; }
            if (layer >= config_.layerCount) {
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "invalid layer in key: " + std::string(key));
            }
            identity = key.substr(0, layerSeparator);
            result.baseOffset = layer * config_.layerSlotBytes;
            result.layerSlotBytes = config_.layerSlotBytes;
        }
    }
    return Status::OK();
}

}  // namespace

std::unique_ptr<KeyResolver> CreateDefaultKeyResolver(const KeyResolutionConfig& config)
{
    return std::make_unique<DefaultKeyResolver>(config);
}

}  // namespace kv
