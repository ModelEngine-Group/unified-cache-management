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
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "kv_types.h"

namespace kv {

class AsuTransport;

// Owns every AsuTransport of one client, keyed by node id.
//
// Ownership rules:
// - The pool is the only owner. Callers receive shared_ptr references through
//   GetOrCreate/Find/Snapshot and must never shut a transport down through them;
//   ShutdownAll() is the only place that closes connections.
// - A pooled transport is bound to the endpoints it was created with. When a
//   view presents different endpoints for the same node, the pool rebuilds that
//   connection instead of silently re-pointing it. The previous transport is
//   released, not shut down: in-flight tasks still hold a reference and the
//   destructor closes it once the last of them finishes.
class AsuTransportPool final {
public:
    // Creates one transport from a ready config. The pool never builds configs itself.
    using Creator = std::function<Status(const TransportConfig&, std::shared_ptr<AsuTransport>&)>;

    explicit AsuTransportPool(Creator creator);
    ~AsuTransportPool() = default;

    AsuTransportPool(const AsuTransportPool&) = delete;
    AsuTransportPool& operator=(const AsuTransportPool&) = delete;
    AsuTransportPool(AsuTransportPool&&) = delete;
    AsuTransportPool& operator=(AsuTransportPool&&) = delete;

    // Returns the pooled transport of a node, creating it on first use. Endpoints are
    // compared as a multiset, so reordering the same addresses is not a change. A real
    // endpoint change is reported as ENDPOINT_CHANGED and the pooled transport is kept:
    // the caller decides whether to Replace() it.
    Status GetOrCreate(const TransportConfig& config, std::shared_ptr<AsuTransport>& transport);
    // Rebuilds the connection of a node. The previous transport is released, not shut
    // down: in-flight tasks still hold a reference and the destructor closes it once the
    // last of them finishes.
    Status Replace(const TransportConfig& config, std::shared_ptr<AsuTransport>& transport);
    // Read-only accessors; the returned references keep transports alive but do not
    // transfer ownership.
    std::shared_ptr<AsuTransport> Find(NodeId nodeId) const;
    // Point-in-time copy of the pooled transports. The shared_ptrs keep them alive but
    // ownership stays with the pool: never shut a transport down through this result.
    std::unordered_map<NodeId, std::shared_ptr<AsuTransport>> GetTransports() const;
    std::size_t Size() const;
    // Shuts every pooled transport down once and empties the pool.
    Status ShutdownAll();

private:
    // Creates a transport and stores it as the current one for its node. The caller holds
    // mutex_; the transport this call replaced is handed back through `replacedTransport`
    // so that it can be released after the lock is dropped.
    Status CreateAndStoreLocked(const TransportConfig& config,
                                std::shared_ptr<AsuTransport>& transport,
                                std::shared_ptr<AsuTransport>& replacedTransport);

    Creator creator_;
    mutable std::mutex mutex_;
    std::unordered_map<NodeId, std::shared_ptr<AsuTransport>> transports_;
    // Endpoints recorded when a transport was created, for conflict detection.
    std::unordered_map<NodeId, std::vector<NodeEndpoint>> endpoints_;
};

}  // namespace kv
