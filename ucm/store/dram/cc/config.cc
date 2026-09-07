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
#include <any>
#include <cctype>
#include <chrono>
#include <fmt/format.h>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <utility>
#include "kv_protocol.h"

namespace UC::Dram {
namespace {

constexpr std::size_t kTargetBatchEntries = 128;
constexpr std::size_t kMaxInflightRequestsPerNode = 128;

std::size_t MaxReplySize(std::size_t entryCount)
{
    DramPool::ProtocolManager protocol;
    return std::max({protocol.GetPackedResponseSize(OpType::LOOKUP, entryCount),
                     protocol.GetPackedResponseSize(OpType::DUMP, entryCount),
                     protocol.GetPackedResponseSize(OpType::LOAD, entryCount)});
}

Status RequiredString(const Detail::Dictionary& input, const char* key, std::string* output)
{
    if (!input.Contains(key)) {
        return Status::InvalidParam("missing DramStore config key({})", key);
    }
    input.Get(key, *output);
    return output->empty() ? Status::InvalidParam("config key({}) must not be empty", key)
                           : Status::OK();
}

Status ParseControlEndpoint(std::string_view value, std::string_view fieldName, std::string* host,
                            std::uint16_t* port)
{
    if (host == nullptr || port == nullptr) {
        return Status::InvalidParam("control endpoint output is null");
    }
    const auto separator = value.rfind(':');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 >= value.size()) {
        return Status::InvalidParam("{} must use host:port form", fieldName);
    }

    const auto parsedHost = value.substr(0, separator);
    for (const auto character : parsedHost) {
        if (std::isspace(static_cast<unsigned char>(character)) != 0) {
            return Status::InvalidParam("{} has an invalid host", fieldName);
        }
    }

    std::uint32_t parsedPort = 0;
    for (const auto character : value.substr(separator + 1)) {
        if (character < '0' || character > '9') {
            return Status::InvalidParam("{} has an invalid port", fieldName);
        }
        parsedPort = parsedPort * 10 + static_cast<std::uint32_t>(character - '0');
        if (parsedPort > std::numeric_limits<std::uint16_t>::max()) {
            return Status::InvalidParam("{} has an invalid port", fieldName);
        }
    }
    if (parsedPort == 0) { return Status::InvalidParam("{} has an invalid port", fieldName); }

    host->assign(parsedHost);
    *port = static_cast<std::uint16_t>(parsedPort);
    return Status::OK();
}

Status OptionalSize(const Detail::Dictionary& input, const char* key, std::size_t* output)
{
    if (!input.Contains(key)) { return Status::OK(); }
    ssize_t value = 0;
    input.GetNumber(key, value);
    if (value < 0) { return Status::InvalidParam("config key({}) must not be negative", key); }
    *output = static_cast<std::size_t>(value);
    return Status::OK();
}

Status NumberList(const Detail::Dictionary& input, const char* key,
                  std::vector<std::size_t>* output)
{
    if (!input.Contains(key)) {
        return Status::InvalidParam("missing DramStore config key({})", key);
    }
    std::vector<ssize_t> values;
    input.Get(key, values);
    output->reserve(values.size());
    for (const auto value : values) {
        if (value < 0) {
            return Status::InvalidParam("config key({}) contains a negative value", key);
        }
        output->push_back(static_cast<std::size_t>(value));
    }
    return Status::OK();
}

std::chrono::milliseconds Milliseconds(std::size_t value)
{
    using Rep = std::chrono::milliseconds::rep;
    return std::chrono::milliseconds{static_cast<Rep>(
        std::min<std::size_t>(value, static_cast<std::size_t>(std::numeric_limits<Rep>::max())))};
}

}  // namespace

Expected<DramConfig> DramConfig::Parse(const Detail::Dictionary& dictionary)
{
    try {
        DramConfig result;
        std::string localControlEndpoint;
        auto status = RequiredString(dictionary, "local_control_endpoint", &localControlEndpoint);
        if (status.Failure()) { return status; }
        status = ParseControlEndpoint(localControlEndpoint, "local_control_endpoint",
                                      &result.localControlHost, &result.localControlPort);
        if (status.Failure()) { return status; }
        status = RequiredString(dictionary, "local_host", &result.localHost);
        if (status.Failure()) { return status; }
        status = RequiredString(dictionary, "local_transport_manager_id",
                                &result.localTransportManagerId);
        if (status.Failure()) { return status; }

        std::string routerType{"ring_hash"};
        if (dictionary.Contains("router_type")) { dictionary.Get("router_type", routerType); }
        if (routerType == "ring_hash") {
            result.routerType = UC::Router::RouterType::RING_HASH_FULL_SPREAD;
        } else if (routerType == "maglev") {
            result.routerType = UC::Router::RouterType::MAGLEV_FULL_SPREAD;
        } else if (routerType == "contiguous_block_affinity") {
            result.routerType = UC::Router::RouterType::CONTIGUOUS_BLOCK_AFFINITY;
        } else if (routerType == "batch_topk_affinity") {
            result.routerType = UC::Router::RouterType::BATCH_TOPK_AFFINITY;
        } else {
            return Status::InvalidParam("unsupported router_type({})", routerType);
        }

        std::size_t deviceId = 0;
        status = OptionalSize(dictionary, "device_id", &deviceId);
        if (status.Failure() || deviceId > 65) {
            return status.Failure() ? status : Status::InvalidParam("device_id is out of range");
        }
        result.deviceId = static_cast<std::int32_t>(deviceId);
        result.nodeScheduler.deviceId = result.deviceId;

        if (dictionary.Contains("role")) {
            std::string roleStr;
            dictionary.Get("role", roleStr);
            if (roleStr == "scheduler") {
                result.role = Role::SCHEDULER;
            } else if (roleStr != "worker") {
                return Status::InvalidParam("unsupported role({})", roleStr);
            }
        }

        std::size_t hixlListenPort = result.hixlListenPort;
        status = OptionalSize(dictionary, "hixl_listen_port", &hixlListenPort);
        const auto hixlPortOffset = result.role == Role::WORKER ? 1U : 0U;
        if (status.Failure() || hixlListenPort == 0 ||
            hixlListenPort > std::numeric_limits<std::uint16_t>::max() - hixlPortOffset) {
            return status.Failure() ? status
                                    : Status::InvalidParam("hixl_listen_port is out of range");
        }
        result.hixlListenPort = static_cast<std::uint16_t>(hixlListenPort + hixlPortOffset);
        if (dictionary.Contains("enable_hixl_cs")) {
            dictionary.Get("enable_hixl_cs", result.enableHixlCs);
        }

        std::string managerHost;
        std::uint16_t managerPort = 0;
        status = ParseControlEndpoint(result.localTransportManagerId, "local_transport_manager_id",
                                      &managerHost, &managerPort);
        if (status.Failure()) { return status; }
        if (result.role == Role::WORKER) {
            const auto offset = static_cast<std::uint32_t>(result.deviceId) + 1;
            if (result.localControlPort > std::numeric_limits<std::uint16_t>::max() - offset ||
                managerPort > std::numeric_limits<std::uint16_t>::max() - offset) {
                return Status::InvalidParam("worker transport port is out of range");
            }
            result.localControlPort += static_cast<std::uint16_t>(offset);
            managerPort += static_cast<std::uint16_t>(offset);
        }
        result.localTransportManagerId = fmt::format("{}:{}", managerHost, managerPort);

        std::vector<std::string> controlEndpoints;
        std::vector<std::string> transportManagerIds;
        if (!dictionary.Contains("node_control_endpoints") ||
            !dictionary.Contains("node_transport_manager_ids")) {
            return Status::InvalidParam("missing DramStore node configuration arrays");
        }
        dictionary.Get("node_control_endpoints", controlEndpoints);
        dictionary.Get("node_transport_manager_ids", transportManagerIds);
        if (controlEndpoints.size() != transportManagerIds.size()) {
            return Status::InvalidParam("node config arrays must have equal lengths");
        }
        result.nodeScheduler.nodes.reserve(controlEndpoints.size());
        for (std::size_t index = 0; index < controlEndpoints.size(); ++index) {
            const auto controlField = fmt::format("node_control_endpoints[{}]", index);
            std::string controlHost;
            std::uint16_t controlPort = 0;
            status = ParseControlEndpoint(controlEndpoints[index], controlField, &controlHost,
                                          &controlPort);
            if (status.Failure()) { return status; }
            if (transportManagerIds[index].empty()) {
                return Status::InvalidParam("node_transport_manager_ids[{}] must not be empty",
                                            index);
            }
            result.nodeScheduler.nodes.push_back(
                NodeEndpoint{static_cast<NodeId>(index), std::move(controlHost), controlPort,
                             std::move(transportManagerIds[index])});
        }

        status = OptionalSize(dictionary, "max_io_entries", &result.maxIoEntries);
        if (status.Failure()) { return status; }

        const auto maxInflight = std::min(result.maxIoEntries, kMaxInflightRequestsPerNode);
        const auto maxBatch =
            std::min({result.maxIoEntries, kTargetBatchEntries, kMaxProtocolBatchEntries});
        result.nodeScheduler.limits = NodeLimits{maxInflight, maxBatch};
        result.nodeScheduler.runnerCount = 1;
        result.transportRuntime.workerCount = 1;
        if (dictionary.Contains("transport_worker_count")) {
            std::size_t wc = 0;
            if (OptionalSize(dictionary, "transport_worker_count", &wc).Success() && wc > 0) {
                result.transportRuntime.workerCount = wc;
            }
        }
        if (dictionary.Contains("node_runner_count")) {
            std::size_t rc = 0;
            if (OptionalSize(dictionary, "node_runner_count", &rc).Success() && rc > 0) {
                result.nodeScheduler.runnerCount = rc;
            }
        }

        if (maxInflight != 0 && result.nodeScheduler.nodes.size() >
                                    std::numeric_limits<std::size_t>::max() / maxInflight) {
            return Status::InvalidParam("derived reply slot count is out of range");
        }
        result.replySlotCount = result.nodeScheduler.nodes.size() * maxInflight;
        result.replySlotSize = static_cast<std::uint32_t>(MaxReplySize(maxBatch));

        std::size_t lookupTimeout = 1000;
        std::size_t dumpTimeout = 3000;
        std::size_t loadTimeout = 3000;
        std::size_t reconnectInterval = 3000;
        const std::pair<const char*, std::size_t*> durations[] = {
            {"lookup_timeout_ms",     &lookupTimeout    },
            {"dump_timeout_ms",       &dumpTimeout      },
            {"load_timeout_ms",       &loadTimeout      },
            {"reconnect_interval_ms", &reconnectInterval},
        };
        for (const auto& [key, target] : durations) {
            status = OptionalSize(dictionary, key, target);
            if (status.Failure()) { return status; }
        }

        result.taskTimeouts = TimeoutConfig{Milliseconds(lookupTimeout), Milliseconds(dumpTimeout),
                                            Milliseconds(loadTimeout)};
        result.nodeScheduler.reconnectInterval = Milliseconds(reconnectInterval);

        std::vector<std::size_t> tensorSizes;
        if (dictionary.Contains("tensor_size_list")) {
            status = NumberList(dictionary, "tensor_size_list", &tensorSizes);
            if (status.Failure()) { return status; }
        }
        result.tensorSizes.assign(tensorSizes.begin(), tensorSizes.end());

        std::vector<std::size_t> gpuKvBufferAddrs;
        if (dictionary.Contains("gpu_kv_buffer_addrs")) {
            status = NumberList(dictionary, "gpu_kv_buffer_addrs", &gpuKvBufferAddrs);
            if (status.Failure()) { return status; }
        }
        result.gpuKvBufferAddrs.assign(gpuKvBufferAddrs.begin(), gpuKvBufferAddrs.end());
        if (dictionary.Contains("gpu_kv_buffer_sizes")) {
            status = NumberList(dictionary, "gpu_kv_buffer_sizes", &result.gpuKvBufferSizes);
            if (status.Failure()) { return status; }
        }

        status = result.Validate();
        if (status.Failure()) { return status; }
        return result;
    } catch (const std::bad_any_cast&) {
        return Status::InvalidParam("DramStore config value has an unexpected type");
    }
}

Status DramConfig::Validate() const
{
    if (gpuKvBufferAddrs.size() != gpuKvBufferSizes.size()) {
        return Status::InvalidParam("GPU KV cache address/size list lengths differ");
    }
    for (std::size_t index = 0; index < gpuKvBufferAddrs.size(); ++index) {
        if (gpuKvBufferAddrs[index] == 0 || gpuKvBufferSizes[index] == 0) {
            return Status::InvalidParam("invalid GPU KV cache range at index({})", index);
        }
    }
    if (localControlHost.empty() || localControlPort == 0 || localHost.empty() ||
        localTransportManagerId.empty()) {
        return Status::InvalidParam("local DramStore transport configuration is invalid");
    }
    if (nodeScheduler.nodes.empty()) { return Status::InvalidParam("node list must not be empty"); }
    std::unordered_set<NodeId> ids;
    for (const auto& node : nodeScheduler.nodes) {
        if (node.nodeId == std::numeric_limits<NodeId>::max() || !ids.insert(node.nodeId).second ||
            node.controlHost.empty() || node.controlPort == 0 || node.transportManagerId.empty()) {
            return Status::InvalidParam("invalid or duplicate DramPool node endpoint");
        }
    }
    const auto& limits = nodeScheduler.limits;
    if (maxIoEntries == 0 || limits.maxInflightRequests == 0 || limits.maxBatchEntries == 0 ||
        replySlotCount == 0 || replySlotSize == 0 || nodeScheduler.runnerCount == 0 ||
        transportRuntime.workerCount == 0) {
        return Status::InvalidParam("DramStore capacities must be positive");
    }
    if (limits.maxBatchEntries > maxIoEntries ||
        limits.maxBatchEntries > kMaxProtocolBatchEntries) {
        return Status::InvalidParam("DramStore capacity relationship is invalid");
    }

    if (replySlotCount < nodeScheduler.nodes.size() * limits.maxInflightRequests) {
        return Status::InvalidParam(
            "reply slots must cover every node's maximum inflight requests");
    }
    const auto requiredReply = MaxReplySize(limits.maxBatchEntries);
    if (replySlotSize < requiredReply) {
        return Status::InvalidParam("reply slot cannot hold the configured maximum batch");
    }
    if (taskTimeouts.lookup.count() <= 0 || taskTimeouts.dump.count() <= 0 ||
        taskTimeouts.load.count() <= 0) {
        return Status::InvalidParam("DramStore timeouts must be positive");
    }
    if (nodeScheduler.reconnectInterval.count() <= 0) {
        return Status::InvalidParam("DramStore reconnect interval must be positive");
    }
    if (role != Role::SCHEDULER && tensorSizes.empty()) {
        return Status::InvalidParam("tensor_size_list must not be empty");
    }
    return Status::OK();
}

}  // namespace UC::Dram
