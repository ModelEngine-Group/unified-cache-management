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
#include "config.h"
#include <algorithm>
#include <cctype>
#include <string>
#include "status_utils.h"

namespace kv {
namespace {

std::string NormalizeAttrValue(std::string value)
{
    std::replace(value.begin(), value.end(), '-', '_');
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return value;
}

bool GetAttr(const std::unordered_map<std::string, std::string>& attrs, const std::string& key,
             std::string& value)
{
    auto iter = attrs.find(key);
    if (iter == attrs.end()) { return false; }
    value = iter->second;
    return true;
}

Status GetUint64Attr(const std::unordered_map<std::string, std::string>& attrs,
                     const std::string& key, std::uint64_t& value)
{
    std::string text;
    if (!GetAttr(attrs, key, text)) { return Status::OK(); }

    try {
        std::size_t parsed{0};
        const auto parsedValue = std::stoull(text, &parsed, 0);
        if (parsed != text.size()) {
            return ASU_LOG_ERROR_STATUS(StatusCode::INVALID_ARGUMENT,
                                        "invalid router config " + key + "=" + text);
        }
        value = parsedValue;
        return Status::OK();
    } catch (const std::exception&) {
        return ASU_LOG_ERROR_STATUS(StatusCode::INVALID_ARGUMENT,
                                    "invalid router config " + key + "=" + text);
    }
}

Status GetBoolAttr(const std::unordered_map<std::string, std::string>& attrs,
                   const std::string& key, bool& value)
{
    std::string text;
    if (!GetAttr(attrs, key, text)) { return Status::OK(); }

    const auto normalized = NormalizeAttrValue(text);
    if (normalized == "1" || normalized == "TRUE" || normalized == "ON" || normalized == "YES") {
        value = true;
        return Status::OK();
    }
    if (normalized == "0" || normalized == "FALSE" || normalized == "OFF" || normalized == "NO") {
        value = false;
        return Status::OK();
    }
    return ASU_LOG_ERROR_STATUS(StatusCode::INVALID_ARGUMENT,
                                "invalid router config " + key + "=" + text);
}

kv::RouterType ParseRouterType(const std::string& value, kv::RouterType fallback)
{
    const auto type = NormalizeAttrValue(value);
    if (type == "RING_HASH" || type == "RING_HASH_FULL_SPREAD") {
        return kv::RouterType::RING_HASH_FULL_SPREAD;
    }
    if (type == "MAGLEV" || type == "MAGLEV_FULL_SPREAD") {
        return kv::RouterType::MAGLEV_FULL_SPREAD;
    }
    if (type == "CONTIGUOUS_BLOCK_AFFINITY") {
        return kv::RouterType::CONTIGUOUS_BLOCK_AFFINITY;
    }
    if (type == "BATCH_TOPK_AFFINITY") { return kv::RouterType::BATCH_TOPK_AFFINITY; }
    return fallback;
}

}  // namespace

Status BuildRouterConfigFromAttrs(const std::unordered_map<std::string, std::string>& attrs,
                                  kv::RouterConfig& config)
{
    config = kv::RouterConfig{};

    std::string type;
    if (GetAttr(attrs, "hash_table.type", type)) {
        config.type = ParseRouterType(type, config.type);
    }

    auto status =
        GetUint64Attr(attrs, "ring_hash.virtual_node_count", config.ringHash.virtualNodeCount);
    if (!status.ok()) { return status; }
    status = GetUint64Attr(attrs, "maglev.table_size", config.maglev.tableSize);
    if (!status.ok()) { return status; }
    status = GetUint64Attr(attrs, "contiguous_block_affinity.block_count",
                           config.contiguousBlockAffinity.blockCount);
    if (!status.ok()) { return status; }
    status = GetUint64Attr(attrs, "batch_topk_affinity.top_k", config.batchTopKAffinity.topK);
    if (!status.ok()) { return status; }

    std::string fullSpreadType;
    if (GetAttr(attrs, "contiguous_block_affinity.full_spread_type", fullSpreadType)) {
        config.contiguousBlockAffinity.fullSpreadType =
            ParseRouterType(fullSpreadType, config.contiguousBlockAffinity.fullSpreadType);
    }

    status = GetBoolAttr(attrs, "contiguous_block_affinity.dynamic_adjust_enabled",
                         config.contiguousBlockAffinity.dynamicAdjustEnabled);
    if (!status.ok()) { return status; }
    status = GetBoolAttr(attrs, "batch_topk_affinity.dynamic_adjust_enabled",
                         config.batchTopKAffinity.dynamicAdjustEnabled);
    if (!status.ok()) { return status; }
    return Status::OK();
}

}  // namespace kv
