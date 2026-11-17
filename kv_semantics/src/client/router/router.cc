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
#include "router/router.h"
#include <algorithm>
#include <array>
#include <string>
#include <unordered_set>
#include <utility>

namespace kv {

using RingNode = std::pair<std::uint64_t, NodeId>;
using RingData = std::vector<RingNode>;

constexpr std::uint64_t kMaxVirtualNodeSaltAttempts = 4096;

std::uint64_t Crc32IEEE(std::string_view data)
{
    static const auto table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                if ((crc & 1U) != 0) {
                    crc = 0xEDB88320U ^ (crc >> 1U);
                } else {
                    crc >>= 1U;
                }
            }
            values[i] = crc;
        }
        return values;
    }();

    std::uint32_t crc = 0xFFFFFFFFU;
    for (unsigned char ch : data) { crc = table[(crc ^ ch) & 0xFFU] ^ (crc >> 8U); }
    return crc ^ 0xFFFFFFFFU;
}

std::string BuildVirtualNodeKey(NodeId nodeId, std::uint64_t index, std::uint64_t salt)
{
    auto key = "vn-" + std::to_string(index) + "#node-" + std::to_string(nodeId);
    if (salt == 0) { return key; }
    return key + "#" + std::to_string(salt);
}

bool InsertVirtualNode(RingData& ringData, std::unordered_set<std::uint64_t>& usedHashes,
                       NodeId nodeId, std::uint64_t index, const HashFunction& hash)
{
    std::uint64_t salt = 0;
    auto hashValue = hash(BuildVirtualNodeKey(nodeId, index, salt));
    while (!usedHashes.emplace(hashValue).second) {
        if (salt >= kMaxVirtualNodeSaltAttempts) { return false; }
        ++salt;
        hashValue = hash(BuildVirtualNodeKey(nodeId, index, salt));
    }
    ringData.emplace_back(hashValue, nodeId);
    return true;
}

bool IsPrime(std::uint64_t value)
{
    if (value < 2) { return false; }
    if (value == 2) { return true; }
    if (value % 2 == 0) { return false; }
    for (std::uint64_t factor = 3; factor <= value / factor; factor += 2) {
        if (value % factor == 0) { return false; }
    }
    return true;
}

std::vector<NodeId> NormalizeNodeIds(const std::vector<NodeId>& nodeIds)
{
    std::vector<NodeId> activeNodeIds;
    for (auto nodeId : nodeIds) {
        if (nodeId == kInvalidNodeId ||
            std::find(activeNodeIds.begin(), activeNodeIds.end(), nodeId) != activeNodeIds.end()) {
            continue;
        }
        activeNodeIds.emplace_back(nodeId);
    }
    return activeNodeIds;
}

std::string BuildBatchKey(const std::vector<Key>& keys)
{
    std::string batchKey = "batch-size#" + std::to_string(keys.size());
    for (std::size_t index = 0; index < keys.size(); ++index) {
        batchKey += "#";
        batchKey += std::to_string(index);
        batchKey += ":";
        batchKey += keys[index];
    }
    return batchKey;
}

std::shared_ptr<Router> CreateFullSpreadRouter(const std::vector<NodeId>& nodeIds,
                                               HashFunction hash, RouterConfig config)
{
    switch (config.type) {
        case RouterType::MAGLEV_FULL_SPREAD:
            return std::make_shared<MaglevRouter>(nodeIds, std::move(hash), config);
        case RouterType::RING_HASH_FULL_SPREAD:
            return std::make_shared<RingHashRouter>(nodeIds, std::move(hash), config);
        default: return nullptr;
    }
}

Router::Router(HashFunction hash) : hash_(std::move(hash))
{
    if (!hash_) { hash_ = Crc32IEEE; }
}

std::unordered_map<NodeId, std::vector<Router::EntryIndex>> Router::RouteKeys(
    const std::vector<Key>& keys) const
{
    std::unordered_map<NodeId, std::vector<EntryIndex>> routes;
    routes.reserve(keys.size());
    for (EntryIndex index = 0; index < keys.size(); ++index) {
        auto nodeId = RouteKey(keys[index]);
        if (nodeId != kInvalidNodeId) { routes[nodeId].emplace_back(index); }
    }
    return routes;
}

RingHashRouter::RingHashRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                               RouterConfig config)
    : Router(std::move(hash)), config_(config)
{
    Build(nodeIds);
}

void RingHashRouter::Build(const std::vector<NodeId>& nodeIds)
{
    RingData ringData;
    const auto activeNodeIds = NormalizeNodeIds(nodeIds);
    ringData.reserve(activeNodeIds.size() * config_.ringHash.virtualNodeCount);
    std::unordered_set<std::uint64_t> usedHashes;
    usedHashes.reserve(activeNodeIds.size() * config_.ringHash.virtualNodeCount);
    for (auto nodeId : activeNodeIds) {
        for (std::uint64_t index = 0; index < config_.ringHash.virtualNodeCount; ++index) {
            if (!InsertVirtualNode(ringData, usedHashes, nodeId, index, hash_)) { break; }
        }
    }

    std::sort(ringData.begin(), ringData.end());
    ring_ = std::move(ringData);
}

NodeId RingHashRouter::RouteKey(const Key& key) const
{
    if (ring_.empty()) { return kInvalidNodeId; }

    const auto hashValue = hash_(key);
    auto iter = std::lower_bound(
        ring_.begin(), ring_.end(), hashValue,
        [](const RingNode& ringNode, std::uint64_t value) { return ringNode.first < value; });
    if (iter == ring_.end()) { iter = ring_.begin(); }
    return iter->second;
}

MaglevRouter::MaglevRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                           RouterConfig config)
    : Router(std::move(hash)), config_(config)
{
    Build(nodeIds);
}

void MaglevRouter::Build(const std::vector<NodeId>& nodeIds)
{
    if (!IsPrime(config_.maglev.tableSize)) { config_.maglev.tableSize = kDefaultMaglevTableSize; }

    auto activeNodeIds = NormalizeNodeIds(nodeIds);
    if (activeNodeIds.empty()) { return; }

    lookupTable_.assign(config_.maglev.tableSize, kInvalidNodeId);
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> skips;
    std::vector<std::uint64_t> next;
    offsets.reserve(activeNodeIds.size());
    skips.reserve(activeNodeIds.size());
    next.assign(activeNodeIds.size(), 0);

    for (auto nodeId : activeNodeIds) {
        auto value = std::to_string(nodeId);
        offsets.emplace_back(hash_("maglev-offset#node-" + value) % config_.maglev.tableSize);
        skips.emplace_back(hash_("maglev-skip#node-" + value) % (config_.maglev.tableSize - 1) + 1);
    }

    std::uint64_t filled = 0;
    while (filled < config_.maglev.tableSize) {
        for (std::size_t index = 0;
             index < activeNodeIds.size() && filled < config_.maglev.tableSize; ++index) {
            auto candidate =
                (offsets[index] + next[index] * skips[index]) % config_.maglev.tableSize;
            ++next[index];
            while (lookupTable_[candidate] != kInvalidNodeId) {
                candidate =
                    (offsets[index] + next[index] * skips[index]) % config_.maglev.tableSize;
                ++next[index];
            }
            lookupTable_[candidate] = activeNodeIds[index];
            ++filled;
        }
    }
}

NodeId MaglevRouter::RouteKey(const Key& key) const
{
    if (lookupTable_.empty()) { return kInvalidNodeId; }
    return lookupTable_[hash_(key) % lookupTable_.size()];
}

ContiguousBlockAffinityRouter::ContiguousBlockAffinityRouter(const std::vector<NodeId>& nodeIds,
                                                             HashFunction hash, RouterConfig config)
    : Router(hash), config_(config)
{
    if (config_.contiguousBlockAffinity.blockCount == 0) {
        config_.contiguousBlockAffinity.blockCount = 1;
    }

    auto fullSpreadConfig = config_;
    fullSpreadConfig.type = config_.contiguousBlockAffinity.fullSpreadType;
    fullSpreadRouter_ = CreateFullSpreadRouter(nodeIds, std::move(hash), fullSpreadConfig);
}

std::unordered_map<NodeId, std::vector<Router::EntryIndex>>
ContiguousBlockAffinityRouter::RouteKeys(const std::vector<Key>& keys) const
{
    std::unordered_map<NodeId, std::vector<EntryIndex>> routes;
    if (!fullSpreadRouter_) { return routes; }

    const auto blockCount = std::max<std::uint64_t>(1, config_.contiguousBlockAffinity.blockCount);
    for (EntryIndex begin = 0; begin < keys.size(); begin += blockCount) {
        const auto nodeId = RouteKey(keys[begin]);
        if (nodeId == kInvalidNodeId) { continue; }

        const auto end = std::min<std::uint64_t>(keys.size(), begin + blockCount);
        auto& indices = routes[nodeId];
        for (EntryIndex index = begin; index < end; ++index) { indices.emplace_back(index); }
    }
    return routes;
}

NodeId ContiguousBlockAffinityRouter::RouteKey(const Key& key) const
{
    if (!fullSpreadRouter_) { return kInvalidNodeId; }
    auto routes = fullSpreadRouter_->RouteKeys({key});
    if (routes.empty()) { return kInvalidNodeId; }
    return routes.begin()->first;
}

BatchTopKAffinityRouter::BatchTopKAffinityRouter(const std::vector<NodeId>& nodeIds,
                                                 HashFunction hash, RouterConfig config)
    : Router(std::move(hash)), config_(config), nodeIds_(NormalizeNodeIds(nodeIds))
{
    if (config_.batchTopKAffinity.topK == 0) { config_.batchTopKAffinity.topK = 1; }
}

std::unordered_map<NodeId, std::vector<Router::EntryIndex>> BatchTopKAffinityRouter::RouteKeys(
    const std::vector<Key>& keys) const
{
    std::unordered_map<NodeId, std::vector<EntryIndex>> routes;
    auto candidates = SelectCandidates(BuildBatchKey(keys));
    if (candidates.empty()) { return routes; }

    for (EntryIndex index = 0; index < keys.size(); ++index) {
        auto nodeId = candidates[hash_("batch-topk-key#" + keys[index]) % candidates.size()];
        routes[nodeId].emplace_back(index);
    }
    return routes;
}

NodeId BatchTopKAffinityRouter::RouteKey(const Key& key) const
{
    if (nodeIds_.empty()) { return kInvalidNodeId; }
    return nodeIds_[hash_(key) % nodeIds_.size()];
}

std::vector<NodeId> BatchTopKAffinityRouter::SelectCandidates(const std::string& batchKey) const
{
    if (nodeIds_.empty()) { return {}; }

    std::vector<std::pair<std::uint64_t, NodeId>> scores;
    scores.reserve(nodeIds_.size());
    for (auto nodeId : nodeIds_) {
        scores.emplace_back(
            hash_("batch-topk-candidate#" + batchKey + "#node-" + std::to_string(nodeId)), nodeId);
    }

    std::sort(scores.begin(), scores.end());
    const auto candidateCount = std::min<std::size_t>(
        scores.size(), static_cast<std::size_t>(config_.batchTopKAffinity.topK));
    std::vector<NodeId> candidates;
    candidates.reserve(candidateCount);
    for (std::size_t index = 0; index < candidateCount; ++index) {
        candidates.emplace_back(scores[index].second);
    }
    return candidates;
}

std::shared_ptr<Router> CreateRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                                     RouterConfig config)
{
    switch (config.type) {
        case RouterType::MAGLEV_FULL_SPREAD:
            return CreateFullSpreadRouter(nodeIds, std::move(hash), config);
        case RouterType::CONTIGUOUS_BLOCK_AFFINITY:
            return std::make_shared<ContiguousBlockAffinityRouter>(nodeIds, std::move(hash),
                                                                   config);
        case RouterType::BATCH_TOPK_AFFINITY:
            return std::make_shared<BatchTopKAffinityRouter>(nodeIds, std::move(hash), config);
        case RouterType::RING_HASH_FULL_SPREAD:
            return CreateFullSpreadRouter(nodeIds, std::move(hash), config);
        default:
            config.type = RouterType::RING_HASH_FULL_SPREAD;
            return CreateFullSpreadRouter(nodeIds, std::move(hash), config);
    }
}

}  // namespace kv
