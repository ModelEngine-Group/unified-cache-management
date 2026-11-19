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
#include <unordered_set>
#include <utility>
#include <vector>
#include "router/config.h"

namespace kv {

using Key = std::string;
using NodeId = std::uint64_t;
using HashFunction = std::function<std::uint64_t(const Key&)>;
using UnavailableNodeSet = std::unordered_set<NodeId>;

constexpr NodeId kInvalidNodeId = UINT64_MAX;

// Router maps cache keys to generic node identifiers.
class Router {
public:
    using EntryIndex = std::size_t;

    // Destroys the router interface.
    virtual ~Router() = default;
    // Routes cache keys on the single ring, ignoring node health.
    virtual std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<Key>& keys) const;
    // Routes cache keys while allowing a router implementation to exclude a
    // set of nodes. The default contract preserves legacy behavior by ignoring
    // unavailableNodes; RingHashRouter implements secondary-hash fallback.
    // A router that cannot find any available owner omits the key from the result
    // instead of grouping it under kInvalidNodeId. Task construction treats an
    // omitted key as unroutable and reports it per entry as NO_AVAILABLE_NODE.
    virtual std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<Key>& keys, const UnavailableNodeSet& unavailableNodes) const;

protected:
    // Builds a router with the supplied hash function.
    explicit Router(HashFunction hash);
    // Returns the node that owns a cache key.
    virtual NodeId RouteKey(const Key& key) const = 0;

    HashFunction hash_;
};

// RingHashRouter implements virtual-node consistent hashing with a secondary
// FNV-1a ring used for failover when the primary owner is unavailable.
class RingHashRouter final : public Router {
public:
    // Builds the primary and the secondary ring from the provided node identifiers.
    RingHashRouter(const std::vector<NodeId>& nodeIds, HashFunction hash, RouterConfig config);

    // Keeps the base overload visible when called through a RingHashRouter value.
    using Router::RouteKeys;

    // Routes with failover: the primary owner is used when available; otherwise the
    // secondary ring owner is used; when that owner is unavailable as well (or is the
    // same node) the key has no owner and is omitted from the result.
    std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
        const std::vector<Key>& keys, const UnavailableNodeSet& unavailableNodes) const override;

private:
    using RingNode = std::pair<std::uint64_t, NodeId>;

    // Returns the ring owner for a cache key.
    NodeId RouteKey(const Key& key) const override;
    // Returns the failover owner for a cache key, or kInvalidNodeId when no owner is
    // available. The secondary ring is consulted exactly once (no ring walk, no retry).
    NodeId RouteKeyWithFailover(const Key& key, const UnavailableNodeSet& unavailableNodes) const;
    // Constructs the consistent-hash ring.
    void Build(const std::vector<NodeId>& nodeIds);
    // Constructs the secondary failover ring with an independent hash function.
    void BuildSecondary(const std::vector<NodeId>& nodeIds);

    RouterConfig config_;
    std::vector<RingNode> ring_;
    HashFunction secondaryHash_;
    std::vector<RingNode> secondaryRing_;
};

// MaglevRouter implements Maglev lookup-table routing.
class MaglevRouter final : public Router {
public:
    // Builds the Maglev table from the provided node identifiers.
    MaglevRouter(const std::vector<NodeId>& nodeIds, HashFunction hash, RouterConfig config);

private:
    // Returns the lookup-table owner for a cache key.
    NodeId RouteKey(const Key& key) const override;
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
        const std::vector<Key>& keys) const override;

private:
    // Returns the owner selected by the underlying full-spread router.
    NodeId RouteKey(const Key& key) const override;

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
        const std::vector<Key>& keys) const override;

private:
    // Returns the owner selected from all active nodes.
    NodeId RouteKey(const Key& key) const override;
    // Selects the TopK candidates for one batch fingerprint.
    std::vector<NodeId> SelectCandidates(const std::string& batchKey) const;

    RouterConfig config_;
    std::vector<NodeId> nodeIds_;
};

// Creates a router for the selected routing strategy configuration.
std::shared_ptr<Router> CreateRouter(const std::vector<NodeId>& nodeIds, HashFunction hash,
                                     RouterConfig config);

}  // namespace kv
