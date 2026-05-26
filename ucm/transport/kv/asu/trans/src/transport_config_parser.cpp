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
#include "transport_config_parser.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>
#include "asu_transport/asu_transport.h"

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

Protocol ParseProtocol(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (value == "UB" || value == "UBOE") { return Protocol::UB; }
    if (value == "ROCE") { return Protocol::ROCE; }
    return Protocol::TCP;
}

void ApplyEndpointField(AsuEndpoint& endpoint, const std::string& key, const std::string& value)
{
    if (key == "ip" || key == "local.comm_id" || key == "localCommId") {
        endpoint.ip = value;
    } else if (key == "port") {
        endpoint.port = static_cast<std::uint16_t>(ParseUint64(value));
    } else if (key == "protocol") {
        endpoint.protocol = ParseProtocol(value);
    } else if (key == "numa_node" || key == "numaNode") {
        endpoint.numaNode = static_cast<std::int32_t>(ParseUint64(value));
    } else if (key == "device_id" || key == "deviceId" || key == "local.phy_device_id" ||
               key == "localPhyDeviceId") {
        endpoint.deviceId = static_cast<std::int32_t>(ParseUint64(value));
    } else if (key == "hca_name" || key == "hcaName") {
        endpoint.hcaName = value;
    } else if (key == "hca_port" || key == "hcaPort") {
        endpoint.hcaPort = static_cast<std::uint8_t>(ParseUint64(value));
    } else {
        endpoint.attrs[key] = value;
    }
}

AsuEndpoint ParseEndpoint(const std::string& value)
{
    AsuEndpoint endpoint;
    if (value.find('=') == std::string::npos) {
        auto parts = Split(value, ':');
        if (!parts.empty()) { endpoint.ip = parts[0]; }
        if (parts.size() > 1) { endpoint.port = static_cast<std::uint16_t>(ParseUint64(parts[1])); }
        if (parts.size() > 2) { endpoint.protocol = ParseProtocol(parts[2]); }
        return endpoint;
    }
    for (const auto& item : Split(value, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) { continue; }
        ApplyEndpointField(endpoint, Trim(item.substr(0, pos)), Trim(item.substr(pos + 1)));
    }
    return endpoint;
}

}  // namespace

Status LoadTransportConfig(const std::string& configPath, TransportConfig& config)
{
    std::ifstream configFile{configPath};
    if (!configFile.is_open()) {
        return Status::Error(StatusCode::NOT_FOUND,
                             "failed to open asu transport config, path=" + configPath);
    }

    config = TransportConfig{};
    std::string line;
    while (std::getline(configFile, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') { continue; }

        const auto pos = line.find('=');
        if (pos == std::string::npos) { continue; }

        const auto key = Trim(line.substr(0, pos));
        const auto value = Trim(line.substr(pos + 1));
        if (key == "asuName" || key == "asu_name") {
            config.asuName = value;
        } else if (key == "asuId" || key == "asu_id") {
            config.asuId = ParseUint64(value);
        } else if (key == "endpoint" || key == "endpoints") {
            config.endpoints.clear();
            for (const auto& endpointValue : Split(value, ';')) {
                config.endpoints.emplace_back(ParseEndpoint(endpointValue));
            }
        } else if (key == "queryTimeoutMs" || key == "query_timeout_ms") {
            config.queryTimeoutMs = ParseUint64(value);
        } else if (key == "loadTimeoutMs" || key == "load_timeout_ms") {
            config.loadTimeoutMs = ParseUint64(value);
        } else if (key == "storeTimeoutMs" || key == "store_timeout_ms") {
            config.storeTimeoutMs = ParseUint64(value);
        } else if (key == "maxInflightTasks" || key == "max_inflight_tasks") {
            config.maxInflightTasks = static_cast<std::uint32_t>(ParseUint64(value));
        } else if (key == "maxInflightBytes" || key == "max_inflight_bytes") {
            config.maxInflightBytes = ParseUint64(value);
        } else {
            config.attrs[key] = value;
        }
    }
    return Status::OK();
}

}  // namespace UC::ASU
