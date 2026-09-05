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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "router/config.h"

namespace kv {

using CacheKey = std::string;
using NodeId = std::uint64_t;
using HashFunction = std::function<std::uint64_t(const CacheKey&)>;

constexpr NodeId kInvalidNodeId = UINT64_MAX;

// Router maps cache keys to generic node identifiers.
class Router {
public:
    using EntryIndex = std::size_t;

    // Destroys the router interface.
    virtual ~Router() = default;
    // Routes cache keys to node identifiers.
    virtual std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<CacheKey>& keys) const;

protected:
    // Builds a router with the supplied hash function.
    explicit Router(HashFunction hash);
    // Returns the node that owns a cache key.
    virtual NodeId RouteKey(const CacheKey& key) const = 0;

    HashFunction hash_;
};

// RingHashRouter implements virtual-node consistent hashing.
class RingHashRouter final : public Router {
public:
    // Builds the ring from the provided node identifiers.
    RingHashRouter(const std::vector<NodeId>& nodeIds, HashFunction hash, RouterConfig config);

private:
    using RingNode = std::pair<std::uint64_t, NodeId>;

    // Returns the ring owner for a cache key.
    NodeId RouteKey(const CacheKey& key) const override;
    // Constructs the consistent-hash ring.
    void Build(const std::vector<NodeId>& nodeIds);

    RouterConfig config_;
    std::vector<RingNode> ring_;
};

// MaglevRouter implements Maglev lookup-table routing.
class MaglevRouter final : public Router {
public:
    // Builds the Maglev table from the provided node identifiers.
    MaglevRouter(const std::vector<NodeId>& nodeIds, HashFunction hash, RouterConfig config);

private:
    // Returns the lookup-table owner for a cache key.
    NodeId RouteKey(const CacheKey& key) const override;
    // Constructs the Maglev lookup table.
    void Build(const std::vector<NodeId>& nodeIds);

    RouterConfig config_;
    std::vector<NodeId> lookupTable_;
};

// ContiguousBlockAffinityRouter routes every K contiguous keys to the same node.
class ContiguousBlockAffinityRouter final : public Router {
public:
    // Builds a contiguous-block affinity strategy over a configured full-spread router.
    ContiguousBlockAffinityRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                                  RouterConfig config);

    // Routes keys by anchoring each K-sized range to its first key.
    std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<CacheKey>& keys) const override;

private:
    // Returns the owner selected by the underlying full-spread router.
    NodeId RouteKey(const CacheKey& key) const override;

    RouterConfig config_;
    std::shared_ptr<Router> fullSpreadRouter_;
};

// BatchTopKAffinityRouter limits each batch to a TopK ASU candidate set.
class BatchTopKAffinityRouter final : public Router {
public:
    // Builds a batch TopK affinity strategy over active node identifiers.
    BatchTopKAffinityRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                            RouterConfig config);

    // Routes one RouteKeys call through a batch-specific TopK node candidate set.
    std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<CacheKey>& keys) const override;

private:
    // Returns the owner selected from all active nodes.
    NodeId RouteKey(const CacheKey& key) const override;
    // Selects the TopK candidates for one batch fingerprint.
    std::vector<NodeId> SelectCandidates(const std::string& batchKey) const;

    RouterConfig config_;
    std::vector<NodeId> nodeIds_;
};

// Creates a router for the selected routing strategy configuration.
std::shared_ptr<Router> CreateRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                                     RouterConfig config);

}  // namespace kv
