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
#include "config_utils.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/utils/parser_common.h"
#include "common/utils/status_utils.h"

namespace kv {
namespace {

void ApplyClientViewEndpointField(AsuEndpoint& endpoint, const std::string& key,
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

AsuInfo ParseAsuInfo(const std::string& value)
{
    AsuInfo info;
    for (const auto& endpointValue : SplitConfigValue(value, ';')) {
        info.endpoints.emplace_back(ParseClientViewEndpoint(endpointValue));
    }
    return info;
}

}  // namespace

bool TryParseAsuInfoKey(const std::string& key, AsuId& asuId)
{
    constexpr const char* kCamelPrefix = "asuInfo.";
    constexpr const char* kSnakePrefix = "asu_info.";
    if (key.rfind(kCamelPrefix, 0) == 0) {
        asuId = std::stoull(key.substr(std::string{kCamelPrefix}.size()));
        return true;
    }
    if (key.rfind(kSnakePrefix, 0) == 0) {
        asuId = std::stoull(key.substr(std::string{kSnakePrefix}.size()));
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

AsuEndpoint ParseClientViewEndpoint(const std::string& value)
{
    AsuEndpoint endpoint;
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

Status LoadAsuClientConfig(const std::string& configPath, AsuClientConfig& config)
{
    std::ifstream configFile{configPath};
    if (!configFile.is_open()) {
        const auto message = "failed to open asu client config, path=" + configPath;
        return ASU_LOG_ERROR_STATUS(StatusCode::NOT_FOUND, message);
    }

    config = AsuClientConfig{};
    std::unordered_map<AsuId, AsuInfo> asuInfos;
    std::vector<std::pair<std::string, std::string>> transportFields;
    std::string line;
    while (std::getline(configFile, line)) {
        line = TrimConfigValue(line);
        if (line.empty() || line[0] == '#') { continue; }

        const auto pos = line.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = TrimConfigValue(line.substr(0, pos));
        const auto value = TrimConfigValue(line.substr(pos + 1));
        if (key == "clientId" || key == "client_id") {
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
        } else if (key == "asuSharedProvider" || key == "asu_shared_provider") {
            config.sharedProviderMode = static_cast<SharedProviderMode>(ParseConfigUint64(value));
        } else if (key == "router.type" || key == "routerType" || key == "hashTable.type" ||
                   key == "hash_table.type") {
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
        } else if (key == "hashTable.ringHash.virtualNodeCount" ||
                   key == "ring_hash.virtual_node_count") {
            config.attrs["ring_hash.virtual_node_count"] = value;
        } else if (key == "hashTable.maglev.tableSize" || key == "maglev.table_size") {
            config.attrs["maglev.table_size"] = value;
        } else if (key == "hashTable.contiguousBlockAffinity.blockCount" ||
                   key == "contiguous_block_affinity.block_count") {
            config.attrs["contiguous_block_affinity.block_count"] = value;
        } else if (key == "hashTable.contiguousBlockAffinity.fullSpreadType" ||
                   key == "contiguous_block_affinity.full_spread_type") {
            config.attrs["contiguous_block_affinity.full_spread_type"] = value;
        } else if (key == "hashTable.contiguousBlockAffinity.dynamicAdjustEnabled" ||
                   key == "contiguous_block_affinity.dynamic_adjust_enabled") {
            config.attrs["contiguous_block_affinity.dynamic_adjust_enabled"] = value;
        } else if (key == "hashTable.batchTopKAffinity.topK" ||
                   key == "batch_topk_affinity.top_k") {
            config.attrs["batch_topk_affinity.top_k"] = value;
        } else if (key == "hashTable.batchTopKAffinity.dynamicAdjustEnabled" ||
                   key == "batch_topk_affinity.dynamic_adjust_enabled") {
            config.attrs["batch_topk_affinity.dynamic_adjust_enabled"] = value;
        } else if (key == "transport.asuIds" || key == "transport.asu_ids" || key == "asuIds" ||
                   key == "asu_ids") {
            for (const auto& asuIdText : SplitConfigValue(value, ',')) {
                TransportConfig transportConfig;
                transportConfig.asuId = ParseConfigUint64(asuIdText);
                config.transportConfigs.emplace_back(std::move(transportConfig));
            }
        } else {
            AsuId asuId{0};
            std::string attrKey;
            if (TryParseAsuInfoKey(key, asuId)) {
                asuInfos[asuId] = ParseAsuInfo(value);
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

        auto iter = asuInfos.find(transportConfig.asuId);
        if (iter == asuInfos.end()) { continue; }
        ApplyAsuInfoToTransportConfig(iter->second, transportConfig);
    }
    return Status::OK();
}

void ApplyAsuInfoToTransportConfig(const AsuInfo& info, TransportConfig& config)
{
    if (info.endpoints.empty()) { return; }

    config.endpoints = info.endpoints;
}

}  // namespace kv