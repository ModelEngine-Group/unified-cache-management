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
#include "config_utils.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "parser_common.h"
#include "status_utils.h"

namespace kv {
namespace {

bool ParseClientBool(const std::string& value)
{
    const auto normalized = ToLower(value);
    return normalized == "1" || normalized == "true";
}

void ApplyClientViewEndpointField(NodeEndpoint& endpoint, const std::string& key,
                                  const std::string& value)
{
    if (key == "protocol") {
        endpoint.protocol = ParseConfigProtocol(value);
        endpoint.attrs["protocol"] = value;
    } else if (key == "placement") {
        endpoint.attrs["placement"] = value;
    } else if (key == "port") {
        endpoint.port = static_cast<std::uint16_t>(ParseConfigUint64(value));
    } else if (key == "local.comm_id" || key == "localCommId") {
        endpoint.ip = value;
    } else if (key == "tc") {
        endpoint.attrs["tc"] = value;
    } else if (key == "sl") {
        endpoint.attrs["sl"] = value;
    } else if (key == "send_size" || key == "sendSize") {
        endpoint.attrs["send_size"] = value;
    } else if (key == "flag_size" || key == "flagSize") {
        endpoint.attrs["flag_size"] = value;
    } else if (key == "remote_send_addr" || key == "remoteSendAddr") {
        endpoint.attrs["remote_send_addr"] = value;
    } else if (key == "remote_flag_addr" || key == "remoteFlagAddr") {
        endpoint.attrs["remote_flag_addr"] = value;
    }
}

}  // namespace

NodeInfo ParseAsuInfo(const std::string& value)
{
    NodeInfo info;
    for (const auto& endpointValue : SplitConfigValue(value, ';')) {
        info.endpoints.emplace_back(ParseClientViewEndpoint(endpointValue));
    }
    return info;
}

bool TryParseAsuInfoKey(const std::string& key, NodeId& nodeId)
{
    constexpr const char* kCamelPrefix = "asuInfo.";
    constexpr const char* kSnakePrefix = "asu_info.";
    if (key.rfind(kCamelPrefix, 0) == 0) {
        nodeId = std::stoull(key.substr(std::string{kCamelPrefix}.size()));
        return true;
    }
    if (key.rfind(kSnakePrefix, 0) == 0) {
        nodeId = std::stoull(key.substr(std::string{kSnakePrefix}.size()));
        return true;
    }
    return false;
}

bool TryGetTransportAttrKey(const std::string& key, std::string& attrKey)
{
    constexpr const char* kCamelPrefix = "transport.";
    if (key.rfind(kCamelPrefix, 0) == 0) {
        attrKey = key.substr(std::string{kCamelPrefix}.size());
        return !attrKey.empty();
    }
    return false;
}

NodeEndpoint ParseClientViewEndpoint(const std::string& value)
{
    NodeEndpoint endpoint;
    if (value.find('=') == std::string::npos) {
        auto parts = SplitConfigValue(value, ':');
        if (!parts.empty()) { endpoint.ip = parts[0]; }
        if (parts.size() > 1) {
            endpoint.port = static_cast<std::uint16_t>(ParseConfigUint64(parts[1]));
        }
        if (parts.size() > 2) {
            endpoint.protocol = ParseConfigProtocol(parts[2]);
            endpoint.attrs["protocol"] = parts[2];
        }
        return endpoint;
    }

    for (const auto& item : SplitConfigValue(value, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) { continue; }
        ApplyClientViewEndpointField(endpoint, TrimConfigValue(item.substr(0, pos)),
                                     TrimConfigValue(item.substr(pos + 1)));
    }
    return endpoint;
}

Status ValidateKeyResolutionConfig(const KeyResolutionConfig& config)
{
    if (config.layerCount == 0 || config.layerSlotBytes == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "vllm_ascend layerwise keys require positive "
                             "aggregation.layer_count and aggregation.layer_slot_bytes");
    }
    // KVBuffer carries a uint32 object offset.
    constexpr auto maxObjectBytes = std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1;
    if (config.layerCount > maxObjectBytes / config.layerSlotBytes) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "layer layout exceeds uint32 offsets");
    }
    return Status::OK();
}

Status KvClientConfig::LoadFromFile(const std::string& configPath, KvClientConfig& config)
{
    std::ifstream configFile{configPath};
    if (!configFile.is_open()) {
        const auto message = "failed to open asu client config, path=" + configPath;
        return ASU_LOG_ERROR_STATUS(StatusCode::NOT_FOUND, message);
    }

    config = KvClientConfig{};
    std::unordered_map<NodeId, NodeInfo> asuInfos;
    std::vector<std::pair<std::string, std::string>> transportFields;
    std::string line;
    while (std::getline(configFile, line)) {
        line = TrimConfigValue(line);
        if (line.empty() || line[0] == '#') { continue; }

        const auto pos = line.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = TrimConfigValue(line.substr(0, pos));
        const auto value = TrimConfigValue(line.substr(pos + 1));
        if (key == "client.clientId" || key == "client.client_id") {
            config.clientId = value;
        } else if (key == "client.maxInflightTasks" || key == "client.max_inflight_tasks") {
            config.maxInflightTasks = static_cast<std::uint32_t>(ParseConfigUint64(value));
        } else if (key == "viewServiceAddrs" || key == "view_service_addrs") {
            config.viewServiceAddrs = SplitConfigValue(value, ',');
        } else if (key == "view.config_path" || key == "viewConfigPath" ||
                   key == "view_config_path") {
            config.attrs["view.config_path"] = value;
        } else if (key == "waitTimeoutMs" || key == "wait_timeout_ms") {
            const auto waitTimeoutMs = ParseConfigUint64(value);
            config.defaultWaitTimeoutMs = waitTimeoutMs;
            config.timeoutMs = waitTimeoutMs;
        } else if (key == "inference.engine") {
            if (value == "vllm_ascend") {
                config.keyResolution.engine = InferenceEngine::VLLM_ASCEND;
            } else if (value == "sglang") {
                config.keyResolution.engine = InferenceEngine::SGLANG;
            } else {
                config.keyResolution.engine = InferenceEngine::GENERIC;
            }
        } else if (key == "aggregation.layer_count" || key == "aggregation.layer_slot_bytes") {
            std::uint64_t size = 0;
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), size);
            if (error != std::errc{} || end != value.data() + value.size()) {
                return ASU_LOG_ERROR_STATUS(StatusCode::INVALID_ARGUMENT,
                                            "expected a non-negative integer for " + key);
            }
            if (key == "aggregation.layer_count") {
                config.keyResolution.layerCount = size;
            } else {
                config.keyResolution.layerSlotBytes = size;
            }
        } else if (key == "client.useBatchIoForSingleKey" ||
                   key == "client.use_batch_io_for_single_key") {
            config.useBatchIoForSingleKey = ParseClientBool(value);
        } else if (key == "client.asuSharedProvider" || key == "client.asu_shared_provider") {
            config.sharedProviderMode = static_cast<SharedProviderMode>(ParseConfigUint64(value));
        } else if (key == "client.router.type" || key == "client.routerType" ||
                   key == "client.hashTable.type" || key == "client.hash_table.type") {
            auto type = value;
            std::transform(type.begin(), type.end(), type.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
            if (type == "MAGLEV" || type == "MAGLEV_FULL_SPREAD") {
                config.attrs["hash_table.type"] = "MAGLEV";
            } else if (type == "CONTIGUOUS_BLOCK_AFFINITY") {
                config.attrs["hash_table.type"] = "CONTIGUOUS_BLOCK_AFFINITY";
            } else if (type == "BATCH_TOPK_AFFINITY") {
                config.attrs["hash_table.type"] = "BATCH_TOPK_AFFINITY";
            } else {
                config.attrs["hash_table.type"] = "RING_HASH";
            }
        } else if (key == "client.hashTable.ringHash.virtualNodeCount" ||
                   key == "client.ring_hash.virtual_node_count") {
            config.attrs["ring_hash.virtual_node_count"] = value;
        } else if (key == "client.hashTable.maglev.tableSize" ||
                   key == "client.maglev.table_size") {
            config.attrs["maglev.table_size"] = value;
        } else if (key == "client.hashTable.contiguousBlockAffinity.blockCount" ||
                   key == "client.contiguous_block_affinity.block_count") {
            config.attrs["contiguous_block_affinity.block_count"] = value;
        } else if (key == "client.hashTable.contiguousBlockAffinity.fullSpreadType" ||
                   key == "client.contiguous_block_affinity.full_spread_type") {
            config.attrs["contiguous_block_affinity.full_spread_type"] = value;
        } else if (key == "client.hashTable.contiguousBlockAffinity.dynamicAdjustEnabled" ||
                   key == "client.contiguous_block_affinity.dynamic_adjust_enabled") {
            config.attrs["contiguous_block_affinity.dynamic_adjust_enabled"] = value;
        } else if (key == "client.hashTable.batchTopKAffinity.topK" ||
                   key == "client.batch_topk_affinity.top_k") {
            config.attrs["batch_topk_affinity.top_k"] = value;
        } else if (key == "client.hashTable.batchTopKAffinity.dynamicAdjustEnabled" ||
                   key == "client.batch_topk_affinity.dynamic_adjust_enabled") {
            config.attrs["batch_topk_affinity.dynamic_adjust_enabled"] = value;
        } else if (key == "transport.asuIds" || key == "transport.asu_ids" || key == "asuIds" ||
                   key == "asu_ids") {
            for (const auto& asuIdText : SplitConfigValue(value, ',')) {
                TransportConfig transportConfig;
                transportConfig.nodeId = ParseConfigUint64(asuIdText);
                config.transportConfigs.emplace_back(std::move(transportConfig));
            }
        } else {
            NodeId nodeId{0};
            std::string attrKey;
            if (TryParseAsuInfoKey(key, nodeId)) {
                asuInfos[nodeId] = ParseAsuInfo(value);
            } else if (TryGetTransportAttrKey(key, attrKey)) {
                transportFields.emplace_back(attrKey, value);
            }
        }
    }

    for (auto& transportConfig : config.transportConfigs) {
        transportConfig.timeoutMs = config.timeoutMs;
        for (const auto& field : transportFields) {
            if (ApplyTransportCompletionConfigField(transportConfig, field.first, field.second)) {
                continue;
            }
            if (ApplyTransportBufferConfigField(transportConfig, field.first, field.second)) {
                continue;
            }
            if (ApplyTransportIoNumConfigField(transportConfig, field.first, field.second)) {
                continue;
            }
            if (ApplyTransportProviderConfigField(transportConfig, field.first, field.second)) {
                continue;
            }
            if (ApplyTransportDeviceConfigField(transportConfig, field.first, field.second)) {
                continue;
            }
            if (field.first == "maxErrorCount" || field.first == "max_error_count") {
                transportConfig.maxErrorCount =
                    static_cast<std::uint32_t>(ParseConfigUint64(field.second));
                continue;
            }
            if (field.first == "maxInflightTasks" || field.first == "max_inflight_tasks") {
                transportConfig.maxInflightTasks =
                    static_cast<std::uint32_t>(ParseConfigUint64(field.second));
                continue;
            }
            transportConfig.attrs.emplace(field);
        }

        auto iter = asuInfos.find(transportConfig.nodeId);
        if (iter == asuInfos.end()) { continue; }
        ApplyAsuInfoToTransportConfig(iter->second, transportConfig);
    }
    return Status::OK();
}

void ApplyAsuInfoToTransportConfig(const NodeInfo& info, TransportConfig& config)
{
    if (info.endpoints.empty()) { return; }

    config.endpoints = info.endpoints;
}

}  // namespace kv
