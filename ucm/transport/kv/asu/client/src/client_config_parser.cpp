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
#include "client_config_parser.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>
#include "asu_client/asu_client.h"
#include "view_server.h"

namespace UC::ASU {
namespace {

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> Split(const std::string& value, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream stream{value};
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        part = Trim(part);
        if (!part.empty()) { parts.emplace_back(std::move(part)); }
    }
    return parts;
}

std::uint64_t ParseUint64(const std::string& value) { return std::stoull(value, nullptr, 0); }

Protocol ToTransportProtocol(const std::string& value)
{
    auto protocol = value;
    std::transform(protocol.begin(), protocol.end(), protocol.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (protocol == "UB" || protocol == "UBOE") { return Protocol::UB; }
    if (protocol == "ROCE") { return Protocol::ROCE; }
    if (protocol == "TCP") { return Protocol::TCP; }
    return Protocol::TCP;
}

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

void SetEndpointAttr(AsuEndpoint& endpoint, const std::string& key, const std::string& value)
{
    endpoint.attrs[key] = value;
}

AsuEndpoint ParseAsuEndpoint(const std::string& value)
{
    AsuEndpoint endpoint;
    if (value.find('=') == std::string::npos) {
        auto parts = Split(value, ':');
        if (!parts.empty()) { endpoint.ip = parts[0]; }
        if (parts.size() > 1) { endpoint.port = static_cast<std::uint16_t>(ParseUint64(parts[1])); }
        if (parts.size() > 2) {
            endpoint.protocol = ToTransportProtocol(parts[2]);
            SetEndpointAttr(endpoint, "protocol", parts[2]);
        }
        return endpoint;
    }

    for (const auto& item : Split(value, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = Trim(item.substr(0, pos));
        const auto fieldValue = Trim(item.substr(pos + 1));
        if (key == "protocol") {
            endpoint.protocol = ToTransportProtocol(fieldValue);
            SetEndpointAttr(endpoint, "protocol", fieldValue);
        } else if (key == "placement") {
            SetEndpointAttr(endpoint, "placement", fieldValue);
        } else if (key == "port") {
            endpoint.port = static_cast<std::uint16_t>(ParseUint64(fieldValue));
        } else if (key == "local.comm_id" || key == "localCommId") {
            endpoint.ip = fieldValue;
        } else if (key == "local.phy_device_id" || key == "localPhyDeviceId") {
            endpoint.deviceId = static_cast<std::int32_t>(ParseUint64(fieldValue));
        } else if (key == "tc") {
            SetEndpointAttr(endpoint, "tc", fieldValue);
        } else if (key == "sl") {
            SetEndpointAttr(endpoint, "sl", fieldValue);
        } else if (key == "send_size" || key == "sendSize") {
            SetEndpointAttr(endpoint, "send_size", fieldValue);
        } else if (key == "flag_size" || key == "flagSize") {
            SetEndpointAttr(endpoint, "flag_size", fieldValue);
        } else if (key == "remote_send_addr" || key == "remoteSendAddr") {
            SetEndpointAttr(endpoint, "remote_send_addr", fieldValue);
        } else if (key == "remote_flag_addr" || key == "remoteFlagAddr") {
            SetEndpointAttr(endpoint, "remote_flag_addr", fieldValue);
        }
    }
    return endpoint;
}

AsuInfo ParseAsuInfo(const std::string& value)
{
    AsuInfo info;
    for (const auto& endpointValue : Split(value, ';')) {
        info.endpoints.emplace_back(ParseAsuEndpoint(endpointValue));
    }
    return info;
}

}  // namespace

Status LoadAsuClientConfig(const std::string& configPath, AsuClientConfig& config)
{
    std::ifstream configFile{configPath};
    if (!configFile.is_open()) {
        return Status::Error(StatusCode::NOT_FOUND,
                             "failed to open asu client config, path=" + configPath);
    }

    config = AsuClientConfig{};
    std::unordered_map<AsuId, AsuInfo> asuInfos;
    std::unordered_map<std::string, std::string> transportAttrs;
    std::string line;
    while (std::getline(configFile, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') { continue; }

        const auto pos = line.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = Trim(line.substr(0, pos));
        const auto value = Trim(line.substr(pos + 1));
        if (key == "clientId" || key == "client_id") {
            config.clientId = value;
        } else if (key == "viewServiceAddrs" || key == "view_service_addrs") {
            config.viewServiceAddrs = Split(value, ',');
        } else if (key == "defaultWaitTimeoutMs" || key == "default_wait_timeout_ms") {
            config.defaultWaitTimeoutMs = ParseUint64(value);
        } else if (key == "hashTable.type" || key == "hash_table.type") {
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
        } else if (key == "transport.asuIds" || key == "asuIds" || key == "asu_ids") {
            for (const auto& asuIdText : Split(value, ',')) {
                TransportConfig transportConfig;
                transportConfig.asuId = ParseUint64(asuIdText);
                config.transportConfigs.emplace_back(std::move(transportConfig));
            }
        } else {
            AsuId asuId{0};
            std::string attrKey;
            if (TryParseAsuInfoKey(key, asuId)) {
                asuInfos[asuId] = ParseAsuInfo(value);
            } else if (TryGetTransportAttrKey(key, attrKey)) {
                transportAttrs[attrKey] = value;
            }
        }
    }

    for (auto& transportConfig : config.transportConfigs) {
        for (const auto& attr : transportAttrs) { transportConfig.attrs.emplace(attr); }

        auto iter = asuInfos.find(transportConfig.asuId);
        if (iter == asuInfos.end()) { continue; }
        ApplyAsuInfoToTransportConfig(iter->second, transportConfig);
    }
    return Status::OK();
}

}  // namespace UC::ASU
