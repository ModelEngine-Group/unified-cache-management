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
#include "transport/asu_transport_pool.h"
#include <string>
#include <utility>
#include "kv_transport.h"
#include "logger.h"

namespace kv {

namespace {

const char* ProtocolName(Protocol protocol)
{
    switch (protocol) {
        case Protocol::ROCE: return "ROCE";
        case Protocol::TCP: return "TCP";
        case Protocol::UB: return "UB";
    }
    return "UNKNOWN";
}

std::string FormatEndpoint(const NodeEndpoint& endpoint)
{
    std::string text =
        endpoint.ip + ':' + std::to_string(endpoint.port) + '/' + ProtocolName(endpoint.protocol);
    if (!endpoint.hcaName.empty()) {
        text += '@' + endpoint.hcaName + ':' + std::to_string(endpoint.hcaPort);
    }
    return text;
}

std::string FormatEndpoints(const std::vector<NodeEndpoint>& endpoints)
{
    std::string text;
    for (const auto& endpoint : endpoints) {
        if (!text.empty()) { text += ", "; }
        text += FormatEndpoint(endpoint);
    }
    return text.empty() ? "none" : text;
}

bool SameEndpoint(const NodeEndpoint& lhs, const NodeEndpoint& rhs)
{
    return lhs.ip == rhs.ip && lhs.port == rhs.port && lhs.protocol == rhs.protocol &&
           lhs.numaNode == rhs.numaNode && lhs.hcaName == rhs.hcaName &&
           lhs.hcaPort == rhs.hcaPort && lhs.attrs == rhs.attrs;
}

// Compares two endpoint lists as multisets: the same addresses in a different order are
// equal. A node exposes a handful of endpoints, so the quadratic match is cheaper than
// sorting copies and needs no temporary keys.
bool SameEndpoints(const std::vector<NodeEndpoint>& lhs, const std::vector<NodeEndpoint>& rhs)
{
    if (lhs.size() != rhs.size()) { return false; }

    std::vector<bool> matched(rhs.size(), false);
    for (const auto& left : lhs) {
        bool found = false;
        for (std::size_t index = 0; index < rhs.size(); ++index) {
            if (!matched[index] && SameEndpoint(left, rhs[index])) {
                matched[index] = true;
                found = true;
                break;
            }
        }
        if (!found) { return false; }
    }
    return true;
}

}  // namespace

AsuTransportPool::AsuTransportPool(Creator creator) : creator_(std::move(creator)) {}

Status AsuTransportPool::CreateAndStoreLocked(const TransportConfig& config,
                                              std::shared_ptr<AsuTransport>& transport,
                                              std::shared_ptr<AsuTransport>& replacedTransport)
{
    std::shared_ptr<AsuTransport> newTransport;
    auto status = creator_(config, newTransport);
    if (!status.ok()) { return status; }
    if (newTransport == nullptr) {
        return Status::Error(
            StatusCode::INTERNAL_ERROR,
            "transport creator returned null, nodeId=" + std::to_string(config.nodeId));
    }

    auto entry = transports_.find(config.nodeId);
    if (entry != transports_.end()) {
        replacedTransport = std::move(entry->second);
        entry->second = newTransport;
    } else {
        transports_.emplace(config.nodeId, newTransport);
    }
    endpoints_[config.nodeId] = config.endpoints;
    transport = std::move(newTransport);
    return Status::OK();
}

Status AsuTransportPool::GetOrCreate(const TransportConfig& config,
                                     std::shared_ptr<AsuTransport>& transport)
{
    if (!creator_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "transport creator is not set");
    }

    std::lock_guard<std::mutex> lock{mutex_};
    auto poolIter = transports_.find(config.nodeId);
    if (poolIter != transports_.end()) {
        auto endpointIter = endpoints_.find(config.nodeId);
        if (endpointIter == endpoints_.end() ||
            !SameEndpoints(endpointIter->second, config.endpoints)) {
            return Status::Error(
                StatusCode::ENDPOINT_CHANGED,
                "transport endpoints changed, nodeId=" + std::to_string(config.nodeId));
        }
        transport = poolIter->second;
        return Status::OK();
    }

    std::shared_ptr<AsuTransport> replacedTransport;
    return CreateAndStoreLocked(config, transport, replacedTransport);
}

Status AsuTransportPool::Replace(const TransportConfig& config,
                                 std::shared_ptr<AsuTransport>& transport)
{
    if (!creator_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "transport creator is not set");
    }

    // Declared before the lock so the replaced transport is destroyed after unlocking.
    std::shared_ptr<AsuTransport> replacedTransport;
    std::vector<NodeEndpoint> replacedEndpoints;
    Status status = Status::OK();
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto endpointIter = endpoints_.find(config.nodeId);
        if (endpointIter != endpoints_.end()) { replacedEndpoints = endpointIter->second; }
        status = CreateAndStoreLocked(config, transport, replacedTransport);
    }

    if (status.ok() && replacedTransport != nullptr) {
        KV_WARN("Transport endpoints changed, connection rebuilt: nodeId={}, old=[{}], new=[{}]",
                config.nodeId, FormatEndpoints(replacedEndpoints),
                FormatEndpoints(config.endpoints));
    }
    return status;
}

std::shared_ptr<AsuTransport> AsuTransportPool::Find(NodeId nodeId) const
{
    std::lock_guard<std::mutex> lock{mutex_};
    auto iter = transports_.find(nodeId);
    return iter == transports_.end() ? nullptr : iter->second;
}

std::unordered_map<NodeId, std::shared_ptr<AsuTransport>> AsuTransportPool::GetTransports() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    return transports_;
}

std::size_t AsuTransportPool::Size() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    return transports_.size();
}

Status AsuTransportPool::ShutdownAll()
{
    std::unordered_map<NodeId, std::shared_ptr<AsuTransport>> transports;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        transports.swap(transports_);
        endpoints_.clear();
    }

    Status failure = Status::OK();
    for (auto& item : transports) {
        if (item.second == nullptr) { continue; }
        auto status = item.second->Shutdown();
        if (!status.ok() && failure.ok()) { failure = status; }
    }
    return failure;
}

}  // namespace kv
