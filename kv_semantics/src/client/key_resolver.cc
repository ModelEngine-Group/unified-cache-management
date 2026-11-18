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
#include <charconv>
#define XXH_INLINE_ALL
#include "xxhash.h"

#define HASH_SEED_VALUE (5381)

namespace kv {
namespace {

std::uint32_t HashNamespace(std::string_view kvns)
{
    return XXH32(kvns.data(), kvns.size(), HASH_SEED_VALUE);
}

Status ParseRank(std::string_view value, std::uint32_t& rank)
{
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), rank);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "invalid namespace rank: " + std::string(value));
    }
    return Status::OK();
}

class VllmKeyResolver final : public KeyResolver {
private:
    Status ParseEngineKey(std::string_view input, ParsedKey& result) const override
    {
        // Only named group/PP fields select the namespace; preserve all other fields.
        constexpr std::string_view groupPrefix = "group:";
        constexpr std::string_view ppPrefix = "pp_rank:";
        std::size_t begin = 0;
        bool first = true;
        while (true) {
            const auto end = input.find('@', begin);
            const auto field = input.substr(
                begin, end == std::string_view::npos ? input.size() - begin : end - begin);
            const bool group = !first && field.substr(0, groupPrefix.size()) == groupPrefix;
            const bool pp = !first && field.substr(0, ppPrefix.size()) == ppPrefix;
            if (group || pp) {
                if ((group && result.component) || (pp && result.ppRank)) {
                    return Status::Error(StatusCode::INVALID_ARGUMENT,
                                         "duplicate namespace field in key: " + std::string(input));
                }
                std::uint32_t rank = 0;
                auto status =
                    ParseRank(field.substr(group ? groupPrefix.size() : ppPrefix.size()), rank);
                if (!status.ok()) { return status; }
                if (group) {
                    result.component = "group" + std::to_string(rank);
                } else {
                    result.ppRank = rank;
                }
            } else {
                if (!first) { result.identity.push_back('@'); }
                result.identity.append(field);
            }
            first = false;
            if (end == std::string_view::npos) { break; }
            begin = end + 1;
        }
        return Status::OK();
    }
};

class SglangKeyResolver final : public KeyResolver {
private:
    Status ParseEngineKey(std::string_view input, ParsedKey& result) const override
    {
        result.identity = input;
        // Direct-linker suffix: _cp<rank>_pp<rank>_<pool>. Pool names may contain '_'.
        constexpr std::string_view cpPrefix = "_cp";
        constexpr std::string_view ppPrefix = "_pp";
        std::size_t pp = std::string_view::npos;
        for (auto cp = input.find(cpPrefix); cp != std::string_view::npos;
             cp = input.find(cpPrefix, cp + cpPrefix.size())) {
            const auto end = input.find('_', cp + cpPrefix.size());
            if (end == std::string_view::npos || input.substr(end, ppPrefix.size()) != ppPrefix) {
                continue;
            }
            std::uint32_t cpRank = 0;
            if (ParseRank(input.substr(cp + cpPrefix.size(), end - cp - cpPrefix.size()), cpRank)
                    .ok()) {
                pp = end;
                break;
            }
        }
        if (pp == std::string_view::npos) { return Status::OK(); }
        const auto rankBegin = pp + ppPrefix.size();
        const auto pool = input.find('_', rankBegin);
        if (pool == std::string_view::npos || pool + 1 == input.size()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "missing pool in direct-linker key: " + std::string(input));
        }
        std::uint32_t ppRank = 0;
        auto status = ParseRank(input.substr(rankBegin, pool - rankBegin), ppRank);
        if (!status.ok()) { return status; }
        result.ppRank = ppRank;
        result.component = std::string(input.substr(pool + 1));
        result.identity = input.substr(0, pp);
        return Status::OK();
    }
};

class GenericKeyResolver final : public KeyResolver {
private:
    Status ParseEngineKey(std::string_view input, ParsedKey& result) const override
    {
        result.identity = input;
        return Status::OK();
    }
};

}  // namespace

Status KeyResolver::Resolve(const CacheKey& input, std::string_view baseKvns,
                            ResolvedCacheKey& result) const
{
    ResolvedCacheKey resolvedKey;
    std::string kvns(baseKvns);
    if (const auto* bytes = input.TryGetBinaryKey()) {
        resolvedKey.keyBytes = *bytes;
    } else {
        ParsedKey parsed;
        auto status = ParseEngineKey(*input.TryGetEngineKey(), parsed);
        if (!status.ok()) { return status; }
        if (parsed.component) { kvns += "_" + *parsed.component; }
        if (parsed.ppRank) { kvns += "_pp" + std::to_string(*parsed.ppRank); }
        auto hash = XXH64(parsed.identity.data(), parsed.identity.size(), HASH_SEED_VALUE);
        for (auto& byte : resolvedKey.keyBytes) {
            byte = static_cast<std::byte>(hash & 0xff);
            hash >>= 8;
        }
    }
    resolvedKey.gnsid = HashNamespace(kvns);
    result = resolvedKey;
    return Status::OK();
}

std::unique_ptr<KeyResolver> CreateDefaultKeyResolver(const KeyResolutionConfig& config)
{
    switch (config.engine) {
        case InferenceEngine::VLLM_ASCEND: return std::make_unique<VllmKeyResolver>();
        case InferenceEngine::SGLANG: return std::make_unique<SglangKeyResolver>();
        default: return std::make_unique<GenericKeyResolver>();
    }
}

}  // namespace kv
